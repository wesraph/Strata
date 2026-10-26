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

    // ---- zero: the pools are untouched (stale pages are never attended) ----
    qsa_state_zero(b, g, nullptr);
    sync();
    check(page_is(b.k_pool, (int64_t) tb[0], k_page, 0xEE), "zero left the pools alone");

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
