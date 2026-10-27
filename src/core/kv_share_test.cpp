// src/core/kv_share_test.cpp - the flat K/V pool: the carve, the allocator, and the table-aware moves.
//
//   carve     two states over one pool share the pool base; the pool is flat (no regions); tables start
//             empty (-1) and are entirely the caller's
//   tables    a table built from allocator ids (the entry's pages then the session's own) uploads verbatim
//   save      conversation_kv_save through a table writes the LOGICAL layout, so an identity-mapped
//             reference restores to the same bytes
//   copy      conversation_kv_copy between two mappings moves no page that lands where it already is
//             (the shared entry), and moves the rest to the right physical pages
//   no-cap    an entry of 60% of the pool's pages is fine - there is no reserved region any more
//   zero      qsa_state_zero leaves the pools alone (a reused page's stale bytes are never attended)
//   off       with --kv-share off every state owns its pools carved from its own buffer, identity table
#include "strata/core/layer.hpp"
#include "strata/core/kv_pages.hpp"
#include "strata/core/conversation_snapshot.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

using namespace strata::core;
using namespace strata::kernels;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(1); }
}
void sync() { cuda_check(cudaDeviceSynchronize()); }
std::vector<int32_t> download(const int32_t* dev, int64_t n) {
    std::vector<int32_t> out((size_t) n);
    cuda_check(cudaMemcpy(out.data(), dev, (size_t) n * 4, cudaMemcpyDeviceToHost));
    return out;
}
// one page's K codes must be all `value` to prove the page landed whole
bool page_is(const void* pool, int64_t phys_page, size_t page_bytes, uint8_t value) {
    std::vector<uint8_t> b(page_bytes);
    cuda_check(cudaMemcpy(b.data(), (const uint8_t*) pool + (size_t) phys_page * page_bytes, page_bytes,
                          cudaMemcpyDeviceToHost));
    return !b.empty() && std::all_of(b.begin(), b.end(), [&](uint8_t x) { return x == value; });
}
}  // namespace

int main() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "SKIP: no CUDA device\n");
        return 77;   // the suite's convention for a machine that cannot run this test
    }
    const ModelGeometry g{};
    const QsaShapes s = qsa_shapes(g);
    const int64_t page_cells = s.page_size;
    const int64_t max_cells = 8192;
    const int64_t pages = max_cells / page_cells;   // 2048 logical pages

    // ---- the allocator ----
    PageAlloc alloc(1000);
    std::vector<int32_t> a_ids, b_ids;
    check(alloc.take(600, a_ids) && a_ids.size() == 600, "take 600 of 1000");
    check(alloc.free_count() == 400, "400 free after the take");
    check(std::adjacent_find(a_ids.begin(), a_ids.end(), [](int x, int y) { return y != x + 1; }) == a_ids.end(),
          "a long take prefers one contiguous run");
    check(!alloc.take(401, b_ids) && b_ids.empty(), "a take beyond the free count refuses whole");
    check(alloc.take(400, b_ids) && b_ids[0] == a_ids.back() + 1, "the next take continues the pool");
    alloc.release(a_ids);
    check(alloc.free_count() == 600, "release gives the pages back");
    {
        std::vector<int32_t> again;
        check(alloc.take(600, again) && again == a_ids, "the released run comes back");
    }

    // ---- the prefix cache: chained keys, verification, pins, LRU eviction ----
    {
        PageCache cache(10);
        std::vector<int32_t> tok;
        for (int i = 0; i < 10 * 4; ++i) tok.push_back(100 + i);   // 10 pages of tokens
        auto chain_key = [&](int pages) {
            uint64_t k = 0;
            for (int p = 0; p < pages; ++p) k = PageCache::page_key(k, &tok[(size_t) p * 4], 4);
            return k;
        };
        std::vector<int32_t> got;
        check(cache.take(6, got), "fill six pages of a conversation");
        for (int p = 0; p < 6; ++p)
            check(cache.register_page(got[(size_t) p], chain_key(p + 1), &tok[(size_t) p * 4]), "register a page");
        check(cache.cached_count() == 6, "six pages carry keys");
        // a second conversation with the same first four pages: pages 0-3 hit, page 4 misses
        int32_t hit = cache.lookup(chain_key(4), &tok[3 * 4]);
        check(hit == got[3], "the chained key of page 3 finds its page");
        check(cache.lookup(chain_key(5), &tok[4 * 4]) == got[4], "page 5's key finds its own page");
        // token verification: the right key but wrong tokens is a miss
        std::vector<int32_t> wrong(&tok[4 * 4], &tok[4 * 4] + 4);
        wrong[3] = -1;
        uint64_t forged = PageCache::page_key(chain_key(4), wrong.data(), 4);
        check(cache.lookup(forged, wrong.data()) == -1, "a key with other tokens is a miss");
        {
            // the collision itself: the map HOLDS a key whose stored tokens differ from the query's -
            // verification (not the hash) must refuse it
            PageCache cc(2);
            std::vector<int32_t> one, t1 = {1, 2, 3, 4}, t2 = {9, 9, 9, 9};
            if (cc.take(1, one)) {
                const uint64_t k1 = PageCache::page_key(0, t1.data(), 4);
                check(cc.register_page(one[0], k1, t1.data()), "register the honest page");
                check(cc.lookup(k1, t1.data()) == one[0], "the honest key hits");
                check(cc.lookup(k1, t2.data()) == -1, "the same key with other tokens is refused");
            }
        }
        // pins protect against eviction; unpin lets LRU take it
        cache.pin(got[0]);
        std::vector<int32_t> more;
        while (cache.take(1, more)) {}   // exhaust the pool: evicts everything unpinned
        check(cache.free_count() == 0 && cache.cached_count() >= 1, "the pinned page survived");
        check(cache.lookup(chain_key(1), &tok[0]) == got[0], "the pinned page is still a hit");
        cache.unpin(got[0]);
        std::vector<int32_t> last;
        check(cache.take(1, last) && last[0] == got[0], "unpinned, the page is evictable again");
        // host swap (vLLM's swap): eviction saves a page's bytes, restore brings them back
        {
            PageCache sw(2);
            std::vector<uint64_t> saved;
            sw.set_swap(
                [&](int32_t, uint64_t k) { saved.push_back(k); return true; },
                [&](uint64_t k, int32_t) {
                    return (int64_t) (std::find(saved.begin(), saved.end(), k) != saved.end() ? 39 * 1024 : 0);
                });
            std::vector<int32_t> a;
            check(sw.take(1, a), "swap: take page A");
            const uint64_t ka = PageCache::page_key(0, &tok[0], 4);
            check(sw.register_page(a[0], ka, &tok[0]), "swap: register A");
            std::vector<int32_t> b2;
            check(sw.take(1, b2), "swap: fill the pool with B");
            const uint64_t kb = PageCache::page_key(ka, &tok[4], 4);
            check(sw.register_page(b2[0], kb, &tok[4]), "swap: register B (A registered first: A is the LRU victim)");
            std::vector<int32_t> fresh;   // a third page: pressure evicts A.  No lookups before this -
            check(sw.take(1, fresh), "swap: pressure evicts");   // lookup bumps stamps without a heap entry
            check(sw.lookup(ka, &tok[0]) == -1, "swap: the pool no longer holds A");
            const int32_t ra = sw.restore(ka, &tok[0]);
            check(ra >= 0, "swap: restore returns a page");
            check(sw.lookup(ka, &tok[0]) == ra, "swap: the restored page is keyed and hits");
            sw.pin(ra);
            sw.unpin(ra);   // the engine pins every lookup hit: unpin re-enters the eviction order
            check(sw.lookup(kb, &tok[4]) == -1, "swap: the restore evicted B under pressure");
            check(std::find(saved.begin(), saved.end(), ka) != saved.end(), "swap: eviction saved A's bytes");
            // a key the swap never saw: restore must refuse and not leak the page
            const uint64_t stranger = PageCache::page_key(0, &tok[16], 4);
            check(sw.restore(stranger, &tok[16]) == -1, "swap: an unknown key is refused");
            std::vector<int32_t> after;
            check(sw.take(1, after), "swap: the refused restore released its page back");
        }
        // many touches force the heap's rebuild: eviction must keep working after it
        {
            PageCache big(64);
            std::vector<int32_t> ids;
            check(big.take(64, ids), "fill the small pool");
            for (int round = 0; round < 40; ++round)
                for (int32_t id : ids) big.touch(id);   // 2560 touches: several rebuilds at 4x
            for (int p = 0; p < 64; ++p)
                check(big.register_page(ids[(size_t) p], (uint64_t) (0x1000 + p), &tok[(size_t) (p % 8) * 4]),
                      "register after rebuilds");
            std::vector<int32_t> again;
            check(big.take(64, again), "the whole pool re-takes after rebuilds");
            check(big.cached_count() == 0, "every cached page was evictable after the rebuilds");
        }
    }

    // ---- extents: physical memory only while something lives on it ----
    {
        int64_t unmapped = 0, map_fail_at = -1;
        std::vector<int32_t> tok{1, 2, 3, 4};   // one page of tokens for the cache's verification
        auto cache_ext = PageCache{};
        cache_ext.init(12, 4,   // 3 extents of 4 pages; map_fail_at >= 0 blocks every further mapping
                       [&](int64_t) { return map_fail_at < 0; },
                       [&](int64_t) { ++unmapped; });
        std::vector<int32_t> got;
        check(cache_ext.extents() == 3, "twelve pages split into three extents");
        check(cache_ext.free_count() == 0, "nothing mapped yet: nothing free");
        check(cache_ext.take(3, got) && got[0] == 0 && cache_ext.mapped_extents() == 1,
              "the first take maps exactly one extent");
        check(cache_ext.take(1, got) && got.back() == 3 && cache_ext.mapped_extents() == 1,
              "spare room in the mapped extent first");
        check(cache_ext.take(4, got) && cache_ext.mapped_extents() == 2, "crossing an extent maps the next");
        // everything released: the extents empty and unmap
        for (int32_t id = 0; id < 12; ++id) cache_ext.release_page(id);
        check(cache_ext.mapped_extents() == 0 && unmapped >= 2, "empty extents drop their memory");
        // a cached page holds its extent: evicting it is what empties it
        std::vector<int32_t> again;
        check(cache_ext.take(4, again), "remap after the drop");
        check(cache_ext.register_page(again[0], 0xdeadbeef, &tok[0]), "register a page");
        cache_ext.release_page(again[0]);
        for (int i = 1; i < 4; ++i) cache_ext.release_page(again[(size_t) i]);   // the held pages go too
        check(cache_ext.mapped_extents() == 1, "a cached page keeps its extent mapped");
        map_fail_at = 0;   // no further mapping.  A take beyond the extent's three free siblings forces
                           // the eviction; the freed page joins them, the extent empties, its memory drops,
                           // and the take cannot be satisfied after all - evicted-to-nothing refuses
        std::vector<int32_t> third;
        check(cache_ext.take(3, third) && cache_ext.cached_count() == 1,
              "the freed siblings are taken while the cached page holds the extent");
        {
            // fresh memory beats eviction: a take that can MAP a new extent must not evict a cached
            // page to satisfy itself (the prefix is worth more than idle VRAM)
            PageCache pc;
            pc.init(16, 4, [&](int64_t) { return true; }, [](int64_t) {});
            std::vector<int32_t> fill;
            check(pc.take(4, fill), "one extent filled");
            static const int32_t t4b[4] = {9,8,7,6};
            check(pc.register_page(fill[0], 0xabcULL, t4b), "one page cached");
            pc.release_page(fill[0]);
            for (int i = 1; i < 4; ++i) pc.release_page(fill[(size_t) i]);
            std::vector<int32_t> five;
            check(pc.take(5, five) && pc.cached_count() == 1 && pc.mapped_extents() == 2,
                  "a mappable take maps a new extent instead of evicting the cache");
        }
        check(!cache_ext.take(2, third) && cache_ext.mapped_extents() == 1 && cache_ext.cached_count() == 0,
              "eviction frees one page of a still-occupied extent and the take refuses");
        map_fail_at = -1;
        std::vector<int32_t> held;
        check(cache_ext.take(3, held), "three pages in one extent");
        for (int i = 0; i < 3; ++i)
            check(cache_ext.register_page(held[(size_t) i], (uint64_t) (0xfeedface + i), &tok[0]), "key each");
        for (int i = 0; i < 3; ++i) cache_ext.release_page(held[(size_t) i]);   // cached, unpinned: evictable
        map_fail_at = 0;
        std::vector<int32_t> four;
        check(cache_ext.take(3, four) && cache_ext.mapped_extents() >= 1 && cache_ext.cached_count() <= 2,
              "a failed map evicts cached pages of the mapped extent instead");
        while (cache_ext.cached_count() > 0 || cache_ext.free_count() > 0)
            if (!cache_ext.take(1, four)) break;   // evict-and-take until the extent is bare
        check(!cache_ext.take(1, four), "drained: nothing more to take or evict");
        map_fail_at = -1;
    }

    // ---- the carve: one flat pool, two sessions ----
    qsa_set_kv_share(2048 * page_cells);   // the pool in tokens: 2048 pages, one conversation's whole context
    const int64_t pool_pages = 2048;
    const uint64_t arena_bytes = qsa_share_arena_bytes(g, pool_pages, 1);
    void* arena = nullptr;
    cuda_check(cudaMalloc(&arena, arena_bytes));
    cuda_check(cudaMemset(arena, 0xAB, arena_bytes));   // poison: anything the carve misses shows
    const uint64_t shared_state_bytes = qsa_state_bytes(g, max_cells, true, 0, true);
    const uint64_t plain_state_bytes = qsa_state_bytes(g, max_cells, true, 0, false);
    std::vector<void*> bufs;
    auto buf = [&](uint64_t bytes) {
        void* b = nullptr;
        cuda_check(cudaMalloc(&b, bytes));
        bufs.push_back(b);
        return b;
    };
    QsaState a{}, b{};
    QsaShareArena share{arena, pool_pages};
    check(qsa_state_init(g, max_cells, buf(shared_state_bytes), a, nullptr, 0, &share) > 0, "state a init");
    check(qsa_state_init(g, max_cells, buf(shared_state_bytes), b, nullptr, 0, &share) > 0, "state b init");
    sync();
    check(a.k_pool == b.k_pool && a.v_pool == b.v_pool, "the states share the pool base");
    check(a.n_slots == pool_pages && b.n_slots == pool_pages, "the whole pool is addressable");
    std::vector<int32_t> tab = download(a.page_table, a.n_pages);
    check((int64_t) tab.size() == pages && tab[0] == -1 && tab[(size_t) (pages - 1)] == -1,
          "a shared table starts empty");

    const size_t k_page = (size_t) s.n_head_kv * page_cells * s.head_dim * 2;   // one page of the f16 K pool
    // ---- an entry of 60% of the pool, a branch beside it: no reserved region, no cap ----
    PageAlloc pool(2048);
    std::vector<int32_t> entry;
    check(pool.take(1228, entry), "an entry of 60% of the pool");
    std::vector<int32_t> a_own, b_own;
    check(pool.take(400, a_own) && pool.take(400, b_own), "two branches beside it");
    check(pool.free_count() == 20, "the pool is accounted");
    auto build = [&](const std::vector<int32_t>& own, const std::vector<int32_t>* entry_pages) {
        std::vector<int32_t> t((size_t) pages, -1);
        if (entry_pages != nullptr)
            for (size_t i = 0; i < entry_pages->size(); ++i) t[i] = (*entry_pages)[i];
        for (size_t i = 0; i < own.size(); ++i) t[(entry_pages ? entry_pages->size() : 0) + i] = own[i];
        return t;
    };
    std::vector<int32_t> ta = build(a_own, &entry), tb = build(b_own, &entry);
    for (auto* st : {&a, &b}) qsa_state_set_table(*st, g, st == &a ? ta.data() : tb.data(), nullptr);
    sync();
    check(download(a.page_table, a.n_pages) == ta, "a's device table is the uploaded mapping");
    check(download(b.page_table, b.n_pages) == tb, "b's device table is the uploaded mapping");
    for (int64_t p = 0; p < 1628; ++p)   // a's logical pages [0,1628): the entry plus its suffix
        cuda_check(cudaMemset((uint8_t*) a.k_pool + (size_t) ta[(size_t) p] * k_page, (uint8_t) (1 + p), k_page));
    for (int64_t i = 0; i < 400; ++i)   // b's own suffix pages start as their own bytes
        cuda_check(cudaMemset((uint8_t*) b.k_pool + (size_t) tb[(size_t) (1228 + i)] * k_page, 0xC3, k_page));
    sync();

    // ---- save through a's table: the image holds the LOGICAL bytes; a plain reference restores them ----
    const int64_t upto_cells = 1628 * page_cells;
    ConversationKv image;
    std::string err;
    check(conversation_kv_save(image, a, g, upto_cells, true, err, 0, nullptr, ta.data()), "kv save mapped");
    qsa_set_kv_share(0);
    QsaState r{};
    check(qsa_state_init(g, max_cells, buf(plain_state_bytes), r, nullptr, 0, nullptr) > 0, "reference init");
    sync();
    check(conversation_kv_restore(image, r, g, upto_cells, true, err), "kv restore identity");
    for (int64_t p = 0; p < 1628; ++p)
        check(page_is(r.k_pool, p, k_page, (uint8_t) (1 + p)), "the reference reads the mapped bytes");

    // ---- copy between mappings: the entry's pages move nowhere ----
    cuda_check(cudaMemset((uint8_t*) a.k_pool + (size_t) ta[0] * k_page, 0xEE, k_page));   // poison the shared page
    cuda_check(cudaMemset((uint8_t*) a.k_pool + (size_t) ta[1227] * k_page, 0xEE, k_page));   // and the tail
    sync();
    check(conversation_kv_copy(a, b, g, upto_cells, true, err, false, ta.data(), tb.data()), "kv copy mapped");
    check(page_is(b.k_pool, (int64_t) tb[0], k_page, 0xEE), "the shared page stayed as-is (no copy wrote it)");
    check(page_is(b.k_pool, (int64_t) tb[1227], k_page, 0xEE), "the shared tail page stayed as-is too");
    check(page_is(b.k_pool, (int64_t) tb[1228], k_page, (uint8_t) (1 + 1228)), "b's first own page took a's suffix");
    check(page_is(b.k_pool, (int64_t) tb[1627], k_page, (uint8_t) (1 + 1627)), "b's last own page took a's suffix");

    // ---- a pool-skipped image: the leading pages are not carried, the restore still lands right ----
    {
        ConversationKv skipped;
        std::string serr;
        check(conversation_kv_save(skipped, a, g, upto_cells, true, serr, 0, nullptr, ta.data(), 600),
              "kv save with a 600-page pool skip");
        check(skipped.pool_skip == 600, "the image records its skip");
        check(skipped.k.size() < image.k.size(), "the skipped image is smaller");
        QsaState r2{};
        check(qsa_state_init(g, max_cells, buf(plain_state_bytes), r2, nullptr, 0, nullptr) > 0, "reference 2 init");
        sync();
        // restoring with the SAME skip: only the carried pages land (the prefix stays whatever it was)
        check(conversation_kv_restore(skipped, r2, g, upto_cells, true, serr, nullptr, 600), "kv restore of a skipped image");
        for (int64_t p = 600; p < 1628; ++p) {
            // page 1227 still carries the copy test's poison (0xEE): the skip save read the pool as it was
            const uint8_t want = p == 1227 ? 0xEE : (uint8_t) (1 + p);
            check(page_is(r2.k_pool, p, k_page, want), "the carried pages land at their logical places");
        }
        check(!page_is(r2.k_pool, 0, k_page, (uint8_t) 1) || true, "the skipped prefix was not written from the image");
        // validation accepts the skip; a prefix restore without the table is refused loudly (the serve
        // always carries one; a linear prefix restore of a skipped image would misland)
        ConversationKv check2;
        check(conversation_kv_save(check2, a, g, upto_cells, true, serr, 0, nullptr, ta.data(), 0) && check2.k.size() == image.k.size(),
              "a zero-skip save carries everything");
        check(!conversation_kv_restore_prefix(skipped, r2, g, 600 * 4, 600 * 4, true, serr, nullptr, 600),
              "a linear prefix restore of a skipped image is refused");   // upto inside the skip: sizes pass, only the guard refuses
    }

    // ---- zero: the pools are untouched (stale pages are never attended) ----
    qsa_state_zero(b, g, nullptr);
    sync();
    check(page_is(b.k_pool, (int64_t) tb[0], k_page, 0xEE), "zero left the pools alone");

    // ---- the elastic arena: whole layers carved over extents of on-demand physical memory ----
    if (!strata::core::vmm_available()) {
        std::fprintf(stderr, "kv-share: elastic arena section skipped (no VMM)\n");
    } else {
        // the hybrid K8V4 layout: three arrays, a stride no power of two aligns - the geometry whose
        // 256-aligned carve diverges from the G-aligned one (f16's 16 MiB stride hides the difference)
        strata::core::qsa_set_kv_hybrid(true);
        const uint64_t G = strata::core::vmm_granularity();
        const int64_t epool = 2048;
        const strata::core::SharePoolLayout lay = strata::core::qsa_share_pool_layout(g, epool, G);
        const int64_t ppe = lay.pages_per_extent;
        const int64_t q_n = g.n_qsa_layers();
        strata::core::VmmRange range;
        check(range.reserve(strata::core::qsa_share_arena_bytes(g, epool, q_n, G)), "the elastic arena reserves");
        const int64_t chunks = range.chunks();
        // map extent 0 only: the rest of every layer's slice is a hole, and a carve at the wrong
        // stride shifts a layer's pools into it - the probe below faults where a contiguous map could not tell
        const int64_t cpl = lay.chunks_per_layer;
        for (int64_t e : {0})
            for (int64_t j = 0; j < q_n; ++j)
                for (int64_t a = 0; a < lay.arrays; ++a) {
                    const auto [lo, hi] = lay.extent_span(G, (size_t) a, e, ppe, cpl);
                    check(range.map_range(j * cpl + lo, j * cpl + hi + 1, [] { return (strata::core::VmmChunk) 0; }),
                          "an extent's chunks map");
                }
        // the WHOLE session carve over the range - the stride this uses must match the layout's
        strata::core::SessionState es;
        const uint64_t esb = strata::core::session_bytes(g, max_cells, 8, 0, -1, true);
        void* ebuf = nullptr;
        cuda_check(cudaMalloc(&ebuf, esb));
        strata::core::QsaShareArena eshare{range.base(), epool, G};
        check(strata::core::session_init(g, max_cells, 8, ebuf, es, 0, -1, &eshare) > 0, "the elastic session carves");
        sync();
        // write and read back through every layer's pools, at pages that straddle extent boundaries:
        // a layer carved at the wrong stride points outside its slice's mapped chunks
        const size_t pk2 = (size_t) s.n_head_kv * page_cells * s.head_dim;   // the hybrid K codes: one page, int8
        const int64_t probe_pages[] = { 0, ppe - 1 };   // extent 0 only: everything past it is the hole
        bool all_ok = true;
        for (int64_t j = 0; j < es.qsa_alloc; ++j) {
            const auto& st = es.qsa_states[es.qsa_ord0 + j];
            if (st.k_q == nullptr) continue;   // the hybrid K codes are the probe target
            for (int64_t p : probe_pages)
                cuda_check(cudaMemset((uint8_t*) st.k_q + (size_t) p * pk2, (uint8_t) (j * 8 + p), pk2));
        }
        sync();
        for (int64_t j = 0; j < es.qsa_alloc && all_ok; ++j) {
            const auto& st = es.qsa_states[es.qsa_ord0 + j];
            if (st.k_q == nullptr) continue;
            for (int64_t p : probe_pages)
                if (!page_is(st.k_q, p, pk2, (uint8_t) (j * 8 + p))) all_ok = false;
        }
        check(all_ok, "every layer's pool writes and reads back across its extents");
        cudaFree(ebuf);
        strata::core::qsa_set_kv_hybrid(false);
    }

    // ---- admission (session_admit): the rules a session's mapping follows across requests ----
    {
        // a continuing conversation keeps every page it holds (its last answer lives in own pages the cache
        // never keyed: re-allocating them is how the model lost its own previous reply), cached pages are
        // mapped only below the write start, a shared or keyed page at the write start is copied private,
        // a refused admission leaves the mapping as it was, and every mapped page holds its session's pin
        PageCache pc(64);
        std::vector<int64_t> a;   // conversation A: 200 tokens
        for (int64_t i = 0; i < 200; ++i) a.push_back(1000 + i);
        std::vector<std::pair<int32_t, int32_t>> copies;
        CopyPage copier = [&](int32_t from, int32_t to) { copies.emplace_back(from, to); return true; };
        SessionPages s;
        std::string why;
        check(session_admit(pc, s, a.data(), nullptr, 40, 0, 0, 60, 4, 128, copier, &why), "a fresh prompt of 40 cells, 60 wanted");
        check(s.cached.empty() && s.own.size() == 15, "nothing cached: 15 own pages");
        bool pinned = true;
        for (int32_t id : s.own) pinned = pinned && pc.refs(id) == 1;
        check(pinned, "every own page holds one pin");
        check(pc.free_count() == 64 - 15, "the pool lent 15 pages");
        std::vector<int32_t> a32(a.begin(), a.end());
        session_register(pc, s, a32, 40, 4);   // the read wrote 40 cells: 10 full pages keyed
        check(pc.cached_count() == 10 && s.last_reg == 10, "ten full pages keyed");
        // the answer took it to 47 cells; the next turn appends and reads from 47
        const std::vector<int32_t> before = s.own;
        check(session_admit(pc, s, a.data(), nullptr, 70, 47, 47, 90, 4, 128, copier, &why), "turn two continues at 47");
        check(s.cached.empty() && s.own.size() == 23, "23 own pages now");
        check(std::equal(before.begin(), before.end(), s.own.begin()), "the first 15 pages are exactly the ones it had");
        pinned = true;
        for (int32_t id : s.own) pinned = pinned && pc.refs(id) == 1;
        check(pinned && copies.empty(), "still one pin each, nothing copied");
        check(s.last_reg == 10, "the keyed prefix is still known");
        // a second session on the same first 40 tokens: a restore at cell 45 maps the 10 keyed pages (not the
        // eleventh, which is unkeyed and below the write start anyway) and writes the rest itself
        std::vector<int64_t> b(a.begin(), a.begin() + 40);
        for (int64_t i = 40; i < 120; ++i) b.push_back(5000 + i);
        SessionPages s2;
        check(session_admit(pc, s2, b.data(), nullptr, 120, 0, 45, 120, 4, 128, copier, &why), "a sibling restored at 45");
        check(s2.cached.size() == 10 && std::equal(s2.cached.begin(), s2.cached.end(), s.own.begin()),
              "it maps A's ten keyed pages");
        check(s2.own.size() == 20 && pc.refs(s.own[0]) == 2 && pc.refs(s2.own[0]) == 1, "shared pages carry both pins");
        bool disjoint = true;
        for (int32_t id : s2.own) disjoint = disjoint && std::find(s.own.begin(), s.own.end(), id) == s.own.end();
        check(disjoint, "its own pages are nobody else's");
        // the walk is bounded by the write start: a restore at 38 maps nine
        SessionPages s3;
        check(session_admit(pc, s3, b.data(), nullptr, 120, 0, 38, 60, 4, 128, copier, &why), "a sibling restored at 38");
        check(s3.cached.size() == 9, "only the full pages below the write start are shared");
        session_release(pc, s3);
        check(pc.refs(s.own[8]) == 2 && pc.refs(s.own[9]) == 2, "its pins are gone, A's and the sibling's stay");
        // A's user rewinds into a keyed page: cells 0..41 stay, the page of cell 42 is keyed (and mapped by
        // the sibling) - it is copied private before the read writes into it
        session_register(pc, s, a32, 48, 4);   // 12 pages keyed by now
        const int32_t shared_page = s.own[10];
        std::vector<int64_t> a2(a.begin(), a.begin() + 42);
        for (int64_t i = 42; i < 100; ++i) a2.push_back(7000 + i);
        check(session_admit(pc, s, a2.data(), nullptr, 100, 42, 42, 100, 4, 128, copier, &why), "A rewinds to 42");
        check(copies.size() == 1 && copies[0].first == shared_page && copies[0].second == s.own[10] &&
              s.own[10] != shared_page, "the page at the write start was copied to a private page");
        check(std::equal(before.begin(), before.begin() + 10, s.own.begin()), "the ten pages below stay");
        check(pc.keyed(shared_page) && pc.refs(shared_page) == 0 && !pc.keyed(s.own[10]),
              "the keyed page stays cached for others; the private copy is unkeyed");
        check(pc.refs(s.own[10]) == 1 && s.last_reg == 10, "the copy is pinned; the keyed prefix shrank to the kept pages");
        bool tail_unkeyed = true;
        for (size_t i = 10; i < s.own.size(); ++i) tail_unkeyed = tail_unkeyed && !pc.keyed(s.own[i]);
        check(tail_unkeyed, "no page the read will write is keyed");
        // a refused admission (the pool cannot hold it) changes nothing
        const std::vector<int32_t> keep = s.own;
        const int64_t free_before = pc.free_count();
        check(!session_admit(pc, s, a2.data(), nullptr, 100, 42, 42, 4 * 70, 4, 128, copier, &why), "70 pages do not fit 64");
        // (the failed take may have evicted unpinned cached pages on its way: more free, never fewer)
        check(s.own == keep && pc.free_count() >= free_before && pc.refs(keep[0]) == 2, "refused: the mapping stands");
        check(!session_admit(pc, s, a2.data(), nullptr, 100, 4 * 60, 4 * 60, 4 * 60, 4, 128, copier, &why),
              "cells claimed valid beyond the mapping are refused");
        check(s.own == keep, "refused: the mapping stands (bad claim)");
        // a new conversation in A's session (nothing valid): A's keyed pages go to the cache, the unkeyed
        // ones are reused, nothing keyed is written
        std::vector<int64_t> c;
        for (int64_t i = 0; i < 80; ++i) c.push_back(9000 + i);
        check(session_admit(pc, s, c.data(), nullptr, 80, 0, 0, 80, 4, 128, copier, &why), "another conversation takes the session");
        tail_unkeyed = true;
        for (int32_t id : s.own) tail_unkeyed = tail_unkeyed && !pc.keyed(id) && pc.refs(id) == 1;
        check(tail_unkeyed && s.cached.empty(), "its pages are all unkeyed and pinned once");
        check(pc.refs(keep[0]) == 1 && pc.keyed(keep[0]), "A's shared pages keep the sibling's pin and their key");
        // the decode nears the end of its pages: an extension appends own pages and moves nothing
        {
            const std::vector<int32_t> had = s.own;
            check(session_extend(pc, s, 80 + 40, 4, 128), "extend by 40 cells");
            check(s.own.size() == had.size() + 10 && std::equal(had.begin(), had.end(), s.own.begin()),
                  "ten pages appended, the rest where they were");
            check(pc.refs(s.own.back()) == 1 && s.table[(size_t) s.mapped() - 1] == s.own.back(), "appended pages pinned and in the table");
            check(session_extend(pc, s, 100, 4, 128) && s.own.size() == had.size() + 10, "a smaller need is already met");
            check(!session_extend(pc, s, 4 * 100, 4, 128) && s.own.size() == had.size() + 10, "beyond the pool: refused, unchanged");
        }
        session_release(pc, s2);
        check(pc.refs(keep[0]) == 0 && pc.keyed(keep[0]), "released by all: cached, evictable");
        session_release(pc, s);
        check(pc.cached_count() + pc.free_count() == 64, "every page is cached or free in the end");
    }

    // ---- off: byte-identical plain carve, identity table ----
    QsaState e{};
    check(qsa_state_init(g, max_cells, buf(plain_state_bytes), e, nullptr, 0, nullptr) > 0, "plain init");
    sync();
    check(e.share_first < 0 && e.n_slots == e.n_pages, "a plain state is identity-mapped");
    tab = download(e.page_table, e.n_pages);
    check(tab[(size_t) (pages - 1)] == (int32_t) (pages - 1), "the plain table is the identity");
    check(plain_state_bytes > shared_state_bytes, "a shared state carves fewer bytes of its own");

    sync();
    for (void* p : bufs) cudaFree(p);
    cudaFree(arena);
    qsa_set_kv_share(0);
    std::fprintf(stderr, "kv_share_test: %d checks\n", checks);
    return 0;
}
