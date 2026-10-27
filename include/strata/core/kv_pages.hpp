// include/strata/core/kv_pages.hpp - the page allocator for --kv-share's dynamic K/V pool.
//
// The pool is flat: every page is the same, no reserved regions, no per-session runs.  The allocator hands
// pages out as conversations need them and takes them back when the conversation leaves.  A pinned prefix is
// simply pages the registry holds; a branch maps them instead of allocating - no copy, no reservation.  The
// only limit is the free count: a conversation is admitted when its whole page need fits beside what is
// already allocated (vLLM's conservative full-sequence fit; no preemption here).
//
// HOST-side only: the allocator never touches the device.  Its output is physical page ids, which the serve
// loop writes into the sessions' page tables (qsa_state_set_table).
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <unordered_map>
#include <queue>
#include <set>
#include <utility>
#include <vector>

namespace strata::core {

class PageAlloc {
public:
    PageAlloc() = default;
    explicit PageAlloc(int64_t pages) { init(pages); }

    void init(int64_t pages) {
        free_.clear();
        for (int64_t p = 0; p < pages; ++p) free_.insert(p);
        total_ = pages;
    }

    /// The pool's page count with none free yet (an extent-mapped cache hands pages in as extents map).
    void init_empty(int64_t pages) {
        free_.clear();
        total_ = pages;
    }

    /// n free pages as ids, preferring a contiguous run (a long prefix keeps the table's shared part
    /// compact; correctness never depends on contiguity).  false: not enough free.
    bool take(int64_t n, std::vector<int32_t>& out) {
        if (n < 0 || free_count() < n) return false;
        out.clear();
        out.reserve((size_t) n);
        auto it = free_.end();
        int64_t run = 0;
        for (auto cur = free_.begin(); cur != free_.end(); ++cur) {
            run = run > 0 && *cur == *std::prev(cur) + 1 ? run + 1 : 1;
            if (run >= n) { it = std::next(cur, 1 - (int) run); break; }   // the run's first page
        }
        if (run >= n && it != free_.end()) {
            for (int64_t i = 0; i < n; ++i) out.push_back((int32_t) *it++);
            free_.erase(std::prev(it, (int) n), it);
        } else {
            for (int64_t i = 0; i < n; ++i) out.push_back((int32_t) *free_.begin()), free_.erase(free_.begin());
        }
        return true;
    }

    void release(const std::vector<int32_t>& ids) {
        for (int32_t id : ids)
            if (id >= 0 && id < total_) free_.insert(id);
    }

    /// Pages leaving availability entirely (their extent's memory dropped): out of the free set.
    void erase(const std::vector<int32_t>& ids) {
        for (int32_t id : ids) free_.erase(id);
    }

    bool is_free(int32_t id) const { return free_.find(id) != free_.end(); }

    int64_t free_count() const { return (int64_t) free_.size(); }
    int64_t total() const { return total_; }

private:
    std::set<int32_t> free_;
    int64_t total_ = 0;
};

/// THE PREFIX CACHE over the pool - vLLM's hash-chained blocks, in pages.  A page's key is the chained
/// hash of every token up to and including it, so a key hit means the WHOLE PREFIX to that page matches.
/// Keys are verified against the tokens on every hit (a hash collision is impossible to act on), pages a
/// session maps are pinned (refcounted), and pressure evicts the least recently used cached pages first.
/// A page enters the cache when a conversation that filled it releases it; it leaves by eviction.
class PageCache {
public:
    static constexpr int page_cells = 4;   // the QSA page granule (qsa_real_shapes().page_size)
    /// Map/unmap an EXTENT of pages (the VMM chunk granularity, pages_per_extent at init): the elastic pool
    /// backs extents with physical memory only while they hold something.  Returning false from map fails
    /// the allocation (out of VRAM); the callbacks run on a synchronized device.
    using MapExtent = std::function<bool(int64_t)>;
    using UnmapExtent = std::function<void(int64_t)>;
    /// HOST SWAP (vLLM's swap): eviction hands the page's device bytes to `swap_out` keyed by its chain
    /// hash; a walk that misses a key asks `swap_in` to restore it into a fresh page.  Returning false /
    /// -1 leaves today's behavior (the bytes die, a later request re-reads them).
    using SwapOut = std::function<bool(int32_t, uint64_t)>;
    using SwapIn = std::function<int64_t(uint64_t, int32_t)>;
    /// The BATCHED swap (preferred: the engine copies runs of consecutive pages in one transfer each).
    /// out: (page, key) pairs, stores what it can; in: (key, page) pairs, all or nothing; has: the swap holds
    /// the key (a walk gathers a run of held keys and restores them together).
    using SwapOutMany = std::function<void(const std::vector<std::pair<int32_t, uint64_t>>&)>;
    using SwapInMany = std::function<bool(const std::vector<std::pair<uint64_t, int32_t>>&)>;
    using SwapHas = std::function<bool(uint64_t)>;
    PageCache() = default;
    explicit PageCache(int64_t pages) { init(pages); }

    void init(int64_t pages) { init(pages, 0, nullptr, nullptr); }

    /// `pages_per_extent` 0 or > pages: one extent (a plain always-mapped pool).
    void set_swap(SwapOut out, SwapIn in) { swap_out_ = std::move(out); swap_in_ = std::move(in); }
    void set_swap_many(SwapOutMany out, SwapInMany in, SwapHas has) {
        swap_out_many_ = std::move(out); swap_in_many_ = std::move(in); swap_has_ = std::move(has);
    }
    bool swap_has(uint64_t key) const { return swap_has_ ? swap_has_(key) : false; }

    void init(int64_t pages, int64_t pages_per_extent, MapExtent map_extent, UnmapExtent unmap_extent) {
        alloc_.init_empty(pages);
        total_pages_ = pages;
        per_extent_ = pages_per_extent > 0 && pages_per_extent <= pages ? pages_per_extent : pages;
        extents_ = (pages + per_extent_ - 1) / per_extent_;
        extent_mapped_.assign((size_t) extents_, extents_ == 1);   // a single extent is the plain pool: mapped
        map_extent_ = std::move(map_extent);
        unmap_extent_ = std::move(unmap_extent);
        key_to_page_.clear();
        page_key_.assign((size_t) pages, 0);
        page_tokens_.assign((size_t) pages * page_cells, 0);
        page_refs_.assign((size_t) pages, 0);
        page_stamp_.assign((size_t) pages, 0);
        heap_ = {};
        stamp_ = 0;
        if (extents_ == 1) alloc_.init(pages);
    }

    /// The chained key of page p over the tokens: fnv(parent-key, the page's tokens).  `tokens` holds at
    /// least p*page_size + page_size entries (the caller full-pages only).
    static uint64_t page_key(uint64_t parent, const int32_t* tokens, int page_size) {
        uint64_t h = parent == 0 ? 1469598103934665603ull : parent;
        for (int i = 0; i < page_size; ++i) {
            h ^= (uint64_t) (uint32_t) tokens[i];
            h *= 1099511628211ull;
        }
        return h ? h : 1;   // 0 marks "no key" on a page
    }

    /// A cached page for `key`, when its tokens really match (`tokens` = that page's ids).  Pinned on a
    /// hit: the page cannot be evicted until `unpin` (the mapping session releases it).
    int32_t lookup(uint64_t key, const int32_t* tokens) {
        auto it = key_to_page_.find(key);
        if (it == key_to_page_.end()) return -1;
        const int32_t page = it->second;
        if (page < 0 || (size_t) page >= page_key_.size()) return -1;
        for (int i = 0; i < page_cells; ++i)
            if (page_tokens_[(size_t) page * page_cells + (size_t) i] != tokens[i]) return -1;   // collision or stale: a miss
        page_stamp_[(size_t) page] = ++stamp_;
        return page;
    }

    void pin(int32_t id) {
        if (id >= 0 && (size_t) id < page_key_.size()) page_refs_[(size_t) id] += 1;
    }

    /// The page holds a keyed (hashed, verified) prefix page: immutable while it is keyed, since any session
    /// whose chain reaches it may map it.
    bool keyed(int32_t id) const {
        return id >= 0 && (size_t) id < page_key_.size() && page_key_[(size_t) id] != 0;
    }
    int32_t refs(int32_t id) const {
        return id >= 0 && (size_t) id < page_refs_.size() ? page_refs_[(size_t) id] : 0;
    }

    /// A walk missed `key`: if the swap holds its bytes, take a fresh page (evicting under pressure -
    /// eviction may swap OTHER pages out), copy the bytes in, key it, and hand the page back.  -1: the
    /// key is nowhere; the walk stops as before.  `tokens` are the page's own ids, for verification.
    int32_t restore(uint64_t key, const int32_t* tokens) {
        if (!swap_in_) return -1;
        std::vector<int32_t> got;
        if (!take(1, got) || got.empty()) return -1;
        const int32_t page = got[0];
        if (swap_in_(key, page) <= 0) {
            alloc_.release({page});
            return -1;
        }
        ++swap_in_count_;
        register_page(page, key, tokens);
        return page;
    }

    /// A walk missed a RUN of keys the swap holds: take that many pages (evicting under pressure), copy the
    /// run in with one batched transfer, key them.  `toks`: page_cells ids per key.  Returns the pages (in
    /// key order), empty when the pool has no room or the copy failed (nothing is kept then).
    std::vector<int32_t> restore_many(const std::vector<uint64_t>& keys, const int32_t* toks) {
        std::vector<int32_t> got;
        if (keys.empty() || (!swap_in_many_ && !swap_in_)) return got;
        if (!take((int64_t) keys.size(), got) || got.size() != keys.size()) { alloc_.release(got); return {}; }
        bool ok = true;
        if (swap_in_many_) {
            std::vector<std::pair<uint64_t, int32_t>> pairs;
            pairs.reserve(keys.size());
            for (size_t i = 0; i < keys.size(); ++i) pairs.emplace_back(keys[i], got[i]);
            ok = swap_in_many_(pairs);
        } else {
            for (size_t i = 0; i < keys.size() && ok; ++i) ok = swap_in_(keys[i], got[i]) > 0;
        }
        if (!ok) { alloc_.release(got); return {}; }
        swap_in_count_ += (int64_t) keys.size();
        for (size_t i = 0; i < keys.size(); ++i) register_page(got[i], keys[i], toks + i * page_cells);
        return got;
    }

    void unpin(int32_t id) {
        if (id < 0 || (size_t) id >= page_key_.size() || page_refs_[(size_t) id] <= 0) return;
        if (--page_refs_[(size_t) id] == 0 && page_key_[(size_t) id] != 0)
            touch(id);   // its heap entries were skipped while pinned: re-enter the eviction order
        extent_release(id);
    }

    /// Remember page `id` as holding the tokens whose chained key is `key` (a conversation filled it).
    /// The tokens are kept for verification on every later hit.  Returns false when the page is already
    /// registered under another key (one key per page: the older registration stands).
    bool register_page(int32_t id, uint64_t key, const int32_t* tokens) {
        if (id < 0 || (size_t) id >= page_key_.size() || key == 0) return false;
        if (page_key_[(size_t) id] != 0 && page_key_[(size_t) id] != key) return false;
        if (page_key_[(size_t) id] != 0) key_to_page_.erase(page_key_[(size_t) id]);
        page_key_[(size_t) id] = key;
        for (int i = 0; i < page_cells; ++i)
            page_tokens_[(size_t) page_cells * (size_t) id + (size_t) i] = tokens[i];
        key_to_page_[key] = id;
        touch(id);
        return true;
    }

    /// Refresh recency (every hit and re-registration); the eviction heap reads the stamp lazily.
    void touch(int32_t id) {
        if (id < 0 || (size_t) id >= page_key_.size()) return;
        page_stamp_[(size_t) id] = ++stamp_;
        heap_.emplace(page_stamp_[(size_t) id], id);
        if (heap_.size() > (size_t) (4 * (alloc_.total() + 1)))
            rebuild_heap_();   // lazy entries from repeated touches would grow forever
    }

    /// n pages: mapped-free first, then a freshly mapped extent while the device has memory for one
    /// (a cached prefix is worth more than idle VRAM is), then evicted cached pages, then failure.
    /// Each step repeats until enough or impossible, so a pool with no unmapped extent still evicts.
    bool take(int64_t n, std::vector<int32_t>& out) {
        while (alloc_.free_count() < n) {
            int64_t before = alloc_.free_count();
            if (map_one_more() && alloc_.free_count() > before) continue;
            if (evict_batch(n - alloc_.free_count()) > 0) continue;
            break;
        }
        return alloc_.take(n, out);
    }

    /// A session's page goes back: it loses this session's pin, hashed pages stay cached (evictable once
    /// nobody pins them), the rest free outright.  EVERY page a session maps holds one pin of that session
    /// - the cached prefix pages its walk hit and the own pages it took alike (session_admit pins what
    /// take() hands out) - so the eviction never takes a page some table still maps.
    void release_page(int32_t id) {
        if (id < 0 || (size_t) id >= page_key_.size()) { alloc_.release({id}); return; }
        unpin(id);
        if (page_key_[(size_t) id] == 0 || page_refs_[(size_t) id] > 0) {
            if (page_key_[(size_t) id] == 0 && extent_mapped_[(size_t) (id / per_extent_)]) {
                alloc_.release({id});
                extent_release(id);   // an uncached page leaving may empty its extent
            }
            return;   // unmapped (nowhere to free), still pinned elsewhere, or cached: stays resident
        }
        // hashed and unreferenced: the page stays resident with its key - reusable, and evictable
    }

    /// Drop a page's key and free it (the content is gone or being rewritten).
    void forget_page(int32_t id) {
        if (id < 0 || (size_t) id >= page_key_.size()) return;
        if (page_key_[(size_t) id] != 0) key_to_page_.erase(page_key_[(size_t) id]);
        page_key_[(size_t) id] = 0;
        page_refs_[(size_t) id] = 0;
        alloc_.release({id});
    }

    /// Evict the least recently used cached page nobody maps.  false: nothing evictable.
    bool evict_one() { return evict_batch(1) == 1; }

    /// Evict up to `n` of the least recently used cached pages nobody maps, their bytes swapped out in ONE
    /// batched transfer (or one at a time without the batched swap).  Returns how many.  The heap is lazy:
    /// stale entries (touched, re-keyed, freed since) are skipped until live ones surface.
    int64_t evict_batch(int64_t n) {
        if (std::getenv("STRATA_KV_SWAP_TRACE") != nullptr)
            std::fprintf(stderr, "evict: n=%lld heap=%zu swapset=%d\n", (long long) n, heap_.size(),
                         swap_out_many_ || swap_out_ ? 1 : 0);
        std::vector<std::pair<int32_t, uint64_t>> victims;
        while (!heap_.empty() && (int64_t) victims.size() < n) {
            const auto [stamp, id] = heap_.top();
            heap_.pop();
            if (id < 0 || (size_t) id >= page_key_.size()) continue;
            if (page_key_[(size_t) id] == 0 || page_refs_[(size_t) id] > 0) continue;   // freed or pinned: stale
            if (page_stamp_[(size_t) id] != stamp) continue;   // touched later: a fresher entry is in the heap
            victims.emplace_back(id, page_key_[(size_t) id]);
            page_stamp_[(size_t) id] = -1;   // out of the heap's reach while the batch forms
        }
        if (victims.empty()) return 0;
        // A refused swap (budget full, copy failed) lets the bytes die - the pages still free, as before.
        if (swap_out_many_) swap_out_many_(victims);
        else if (swap_out_)
            for (const auto& [id, key] : victims) if (swap_out_(id, key)) ++swap_out_count_;
        for (const auto& [id, key] : victims) {
            key_to_page_.erase(key);
            page_key_[(size_t) id] = 0;
            alloc_.release({id});
            extent_release(id);
        }
        return (int64_t) victims.size();
    }
    void count_swapped_out(int64_t n) { swap_out_count_ += n; }

    int64_t free_count() const { return alloc_.free_count(); }
    int64_t total() const { return total_pages_; }
    /// Pages some session maps (pinned), and cached pages nobody maps (what an eviction may take).
    int64_t pinned_count() const {
        int64_t n = 0;
        for (int32_t r : page_refs_) n += r > 0 ? 1 : 0;
        return n;
    }
    int64_t evictable_count() const {
        int64_t n = 0;
        for (size_t i = 0; i < page_key_.size(); ++i) n += page_key_[i] != 0 && page_refs_[i] == 0 ? 1 : 0;
        return n;
    }
    int64_t cached_count() const { return (int64_t) key_to_page_.size(); }
    int64_t swap_out_count() const { return swap_out_count_; }
    int64_t swap_in_count() const { return swap_in_count_; }
    int64_t mapped_extents() const {
        int64_t m = 0;
        for (bool b : extent_mapped_)
            if (b) ++m;
        return m;
    }
    int64_t extents() const { return extents_; }

    /// After a page stopped being pinned/cached: an extent whose every page is free of keys and pins
    /// holds nothing - its pages go back and its physical memory drops.  (Counting only pins was wrong:
    /// a cached page's extent must stay mapped or a later hit would read unmapped memory.)
    void extent_release(int32_t id) {
        const int64_t e = id / per_extent_;
        if (e < 0 || e >= extents_ || !extent_mapped_[(size_t) e] || !unmap_extent_) return;
        for (int64_t p = e * per_extent_; p < (e + 1) * per_extent_ && p < total_pages_; ++p)
            if (page_key_[(size_t) p] != 0 || page_refs_[(size_t) p] > 0 || !alloc_.is_free(p))
                return;   // keyed, pinned, or held by a session mid-read: someone lives there
        std::vector<int32_t> drop;
        for (int64_t p = e * per_extent_; p < (e + 1) * per_extent_ && p < total_pages_; ++p)
            drop.push_back((int32_t) p);
        alloc_.erase(drop);   // the extent's pages leave availability: their memory is gone
        extent_mapped_[(size_t) e] = false;
        unmap_extent_(e);
    }

    /// Unmap mapped extents that hold nothing at all (every page free, unkeyed, unpinned), highest first,
    /// until `target` extents stay mapped.  Returns how many were dropped.  (An extent empties through
    /// extent_release only as a page leaves it: the ones the eager floor mapped and nobody ever used stay
    /// mapped otherwise.)
    int64_t drop_free_extents(int64_t target) {
        int64_t dropped = 0;
        if (!unmap_extent_) return 0;
        for (int64_t e = extents_ - 1; e >= 0 && mapped_extents() > target; --e) {
            if (!extent_mapped_[(size_t) e]) continue;
            bool empty = true;
            for (int64_t p = e * per_extent_; p < (e + 1) * per_extent_ && p < total_pages_ && empty; ++p)
                empty = page_key_[(size_t) p] == 0 && page_refs_[(size_t) p] == 0 && alloc_.is_free((int32_t) p);
            if (!empty) continue;
            std::vector<int32_t> drop;
            for (int64_t p = e * per_extent_; p < (e + 1) * per_extent_ && p < total_pages_; ++p) drop.push_back((int32_t) p);
            alloc_.erase(drop);
            extent_mapped_[(size_t) e] = false;
            unmap_extent_(e);
            ++dropped;
        }
        return dropped;
    }

    /// An extent the caller mapped itself (the eager floor at startup): known as mapped, its pages free.
    void mark_mapped(int64_t e) {
        if (e < 0 || e >= extents_ || extent_mapped_[(size_t) e]) return;
        extent_mapped_[(size_t) e] = true;
        std::vector<int32_t> add;
        for (int64_t p = e * per_extent_; p < (e + 1) * per_extent_ && p < total_pages_; ++p) add.push_back((int32_t) p);
        alloc_.release(add);
    }

    /// Map the next unmapped extent (lowest first) into physical memory; its pages become free.
    bool map_one_more() {
        for (int64_t e = 0; e < extents_; ++e) {
            if (extent_mapped_[(size_t) e]) continue;
            if (map_extent_ && !map_extent_(e)) return false;
            extent_mapped_[(size_t) e] = true;
            std::vector<int32_t> add;
            for (int64_t p = e * per_extent_; p < (e + 1) * per_extent_ && p < total_pages_; ++p)
                add.push_back((int32_t) p);
            alloc_.release(add);
            return true;
        }
        return false;   // every extent is mapped already
    }

    PageAlloc alloc_;
    std::unordered_map<uint64_t, int32_t> key_to_page_;
    std::vector<uint64_t> page_key_;
    std::vector<int32_t> page_tokens_;   // 4 ids per page: what every hit is verified against
    std::vector<int32_t> page_refs_;
    std::vector<int64_t> page_stamp_;
    using HeapEntry = std::pair<int64_t, int32_t>;   // (stamp, page), min-stamp first
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> heap_;
    int64_t stamp_ = 0;
    int64_t total_pages_ = 0, per_extent_ = 1, extents_ = 1;
    std::vector<char> extent_mapped_;
    MapExtent map_extent_;
    UnmapExtent unmap_extent_;
    SwapOut swap_out_;
    SwapIn swap_in_;
    SwapOutMany swap_out_many_;
    SwapInMany swap_in_many_;
    SwapHas swap_has_;
    int64_t swap_out_count_ = 0, swap_in_count_ = 0;

    void rebuild_heap_() {
        heap_ = {};
        for (size_t id = 0; id < page_key_.size(); ++id)
            if (page_key_[id] != 0) heap_.emplace(page_stamp_[id], (int32_t) id);
    }
};


/// ONE SESSION'S MAPPING of its conversation's logical pages onto the pool: `cached` - the leading pages the
/// prefix cache held (filled by whoever read them first; shared, read-only) - then `own`, the pages this
/// session filled or is about to fill.  `table` is the host copy of what the device table holds (n_pages
/// entries: cached, own, then -1).  Every page in either list holds one pin of this session.
struct SessionPages {
    std::vector<int32_t> cached;   ///< the prefix pages mapped from the cache (pinned, never written)
    std::vector<int32_t> own;      ///< this conversation's pages past the cached prefix (pinned)
    std::vector<int32_t> table;    ///< the session's host table (n_pages); empty while unmapped
    bool shares = false;           ///< the table maps cached pages at the front
    int64_t last_reg = 0;          ///< logical pages already keyed into the cache this conversation

    int64_t mapped() const { return (int64_t) (cached.size() + own.size()); }
    int32_t at(int64_t p) const {
        return p < (int64_t) cached.size() ? cached[(size_t) p] : own[(size_t) (p - (int64_t) cached.size())];
    }
    void build(int64_t n_pages) {
        shares = !cached.empty();
        table.assign((size_t) n_pages, -1);
        for (size_t i = 0; i < cached.size() && i < table.size(); ++i) table[i] = cached[i];
        for (size_t i = 0; i < own.size() && cached.size() + i < table.size(); ++i) table[cached.size() + i] = own[i];
    }
};

/// EVERYTHING a session maps goes back (its conversation leaves): the cached pins open, the own pages to the
/// cache when keyed and to the free set otherwise.
inline void session_release(PageCache& cache, SessionPages& s) {
    for (int32_t id : s.cached) cache.unpin(id);
    for (int32_t id : s.own) cache.release_page(id);
    s.cached.clear();
    s.own.clear();
    s.last_reg = 0;
}

/// Copies one page's bytes on the device (every range, layer and array): (from, to).  false: it failed.
using CopyPage = std::function<bool(int32_t, int32_t)>;

/// ADMIT a conversation of `n_ids` ids (int64 or int32, one of the two) into `s`.
///   `valid`  - cells of s's CURRENT mapping that already hold this conversation's K/V (a live continuation:
///              the read start; a restore or a slot copy that writes the cells itself: 0).  Those pages stay
///              exactly where they are - re-allocating them is how a conversation loses its own last answer.
///   `resume` - the first cell this session WRITES.  Below it the cells come from the kept mapping, from the
///              prefix cache (full pages below floor(resume/page) whose chained key hits), or from the restore
///              the caller runs next.  At or past it nothing shared is ever mapped: a page the session writes
///              is private, so the page the write start falls into is copied (`copy`) when it is shared or
///              keyed - a keyed page is immutable for as long as its key may be hit.
///   `cells`  - the whole need, generation included (the table's -1 past it would drop the decode's writes).
/// Own pages the conversation no longer needs go back (keyed ones stay cached).  false: the pool cannot hold
/// it even after eviction, or the mapping cannot be what the caller claims - then s is unchanged when
/// `valid` > 0, and empty when `valid` == 0 (the old mapping was released first).
inline bool session_admit(PageCache& cache, SessionPages& s, const int64_t* ids64, const int32_t* ids32,
                          int64_t n_ids, int64_t valid, int64_t resume, int64_t cells, int page_size,
                          int64_t n_pages, const CopyPage& copy, std::string* why = nullptr) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    const int64_t ps = page_size > 0 ? page_size : 1;
    if (valid < 0) valid = 0;
    if (valid > n_ids) valid = n_ids;
    if (resume < valid) resume = valid;
    if (cells < resume) cells = resume;
    // nothing of the current mapping is valid: it goes first (its keyed pages stay cached, unpinned), so
    // the pool holds the old conversation's pages only as cache, not as pins blocking the new one - a
    // give-back or a restore into a session still pinning its previous 120K-token conversation could not
    // fit beside it.  A refusal then leaves the mapping EMPTY (the caller drops the session's live state).
    if (valid == 0) session_release(cache, s);
    const int64_t have = s.mapped();
    const int64_t total = (cells + ps - 1) / ps;
    if (total > n_pages) return fail("the conversation does not fit the session's page table");
    if (valid / ps > have || (valid % ps != 0 && valid / ps >= have))
        return fail("the mapping does not cover the cells claimed valid");
    const int64_t keep_full = valid / ps;                      // pages kept untouched
    const int64_t part_idx = valid % ps != 0 ? valid / ps : -1;   // the page the write start falls into
    auto toks_of = [&](int64_t p, int32_t* toks) {
        for (int64_t i = 0; i < ps && i < (int64_t) PageCache::page_cells; ++i)
            toks[i] = (int32_t) (ids64 ? ids64[p * ps + i] : ids32[p * ps + i]);
    };
    std::vector<int32_t> ncached, nown;
    for (int64_t p = 0; p < keep_full; ++p) (p < (int64_t) s.cached.size() ? ncached : nown).push_back(s.at(p));
    // THE WALK: full pages below the write start, keyed by the chain over every token up to them.  It runs
    // from page 0 so the chain is right, skipping what is kept; it stops at the first miss (later pages
    // cannot match), and never past kept own pages (the cached pages are a PREFIX of the table).
    std::vector<int32_t> walked;
    const int64_t walk_end = std::min<int64_t>(resume / ps, n_ids / ps);
    if (nown.empty() && walk_end > keep_full) {
        uint64_t key = 0;
        std::array<int32_t, PageCache::page_cells> toks{};
        // keys the swap holds are gathered into a run and restored together (one transfer per run of pages)
        std::vector<uint64_t> run_keys;
        std::vector<int32_t> run_toks;
        auto flush_run = [&]() -> bool {
            if (run_keys.empty()) return true;
            const std::vector<int32_t> got = cache.restore_many(run_keys, run_toks.data());
            run_keys.clear();
            run_toks.clear();
            if (got.empty()) return false;
            for (int32_t id : got) { cache.pin(id); ncached.push_back(id); walked.push_back(id); }
            return true;
        };
        for (int64_t p = 0; p < walk_end; ++p) {
            toks_of(p, toks.data());
            key = PageCache::page_key(key, toks.data(), (int) ps);
            if (p < keep_full) continue;
            const int32_t page = cache.lookup(key, toks.data());
            if (page >= 0) {
                if (!flush_run()) break;   // a gap before it: the walk ends at the gap
                cache.pin(page);
                ncached.push_back(page);
                walked.push_back(page);
                continue;
            }
            if (cache.swap_has(key)) {
                run_keys.push_back(key);
                run_toks.insert(run_toks.end(), toks.begin(), toks.end());
                continue;
            }
            if (run_keys.empty()) {   // the single-page swap path (no batched swap installed)
                const int32_t sp = cache.restore(key, toks.data());
                if (sp < 0) break;
                cache.pin(sp);
                ncached.push_back(sp);
                walked.push_back(sp);
                continue;
            }
            break;
        }
        (void) flush_run();
    }
    // the partial page: own and unkeyed stays; shared or keyed is copied to a private page below
    int32_t cow_from = -1;
    if (part_idx >= 0) {
        const int32_t entry = s.at(part_idx);
        if (part_idx < (int64_t) s.cached.size() || cache.keyed(entry)) cow_from = entry;
        else nown.push_back(entry);
    }
    // what else of the old mapping can serve: own pages neither kept nor re-found, unkeyed (a keyed page may be
    // mapped by another chain: it goes back to the cache, never rewritten)
    std::set<int32_t> used(ncached.begin(), ncached.end());
    used.insert(nown.begin(), nown.end());
    std::vector<int32_t> reusable, refound, leftover;
    for (int32_t id : s.own) {
        if (std::find(nown.begin(), nown.end(), id) != nown.end()) continue;
        if (used.count(id)) { refound.push_back(id); continue; }   // re-found by the walk: now a cached page
        (cache.keyed(id) ? leftover : reusable).push_back(id);
    }
    const int64_t want = total - (int64_t) ncached.size();   // own pages the new mapping holds
    int64_t missing = want - (int64_t) nown.size() - (int64_t) reusable.size();
    if (missing < 0) missing = 0;
    std::vector<int32_t> fresh;
    if (missing > 0 && !cache.take(missing, fresh)) {
        for (int32_t id : walked) cache.unpin(id);   // nothing else has changed: the old mapping stands
        return fail("the K/V pool is full");
    }
    for (int32_t id : fresh) cache.pin(id);
    // the copy first: a failure here still leaves the old mapping intact
    if (cow_from >= 0) {
        std::vector<int32_t>& from = reusable.empty() ? fresh : reusable;
        const int32_t dest = from.front();
        from.erase(from.begin());
        if (!copy || !copy(cow_from, dest)) {
            for (int32_t id : walked) cache.unpin(id);
            for (int32_t id : fresh) cache.release_page(id);
            cache.release_page(dest);
            return fail("copying the page at the write start failed");
        }
        nown.push_back(dest);
    }
    for (int32_t id : reusable) {
        if ((int64_t) nown.size() < want) nown.push_back(id);
        else cache.release_page(id);   // past the need: freed
    }
    for (int32_t id : fresh) nown.push_back(id);
    // the old mapping's pins this one does not carry
    for (int64_t p = keep_full; p < (int64_t) s.cached.size(); ++p) cache.unpin(s.cached[(size_t) p]);
    for (int32_t id : refound) cache.release_page(id);    // its own pin goes; the walk's pin keeps it
    for (int32_t id : leftover) cache.release_page(id);   // keyed: stays cached for whoever hits it
    s.cached = std::move(ncached);
    s.own = std::move(nown);
    s.last_reg = std::min(s.last_reg, keep_full);
    s.build(n_pages);
    return true;
}

/// EXTEND a session's mapping to `cells` (the decode nears the end of what it holds): own pages are appended,
/// nothing else moves.  false: the pool has none to give even after eviction; the mapping is unchanged.
inline bool session_extend(PageCache& cache, SessionPages& s, int64_t cells, int page_size, int64_t n_pages) {
    const int64_t ps = page_size > 0 ? page_size : 1;
    const int64_t total = std::min<int64_t>((cells + ps - 1) / ps, n_pages);
    const int64_t missing = total - s.mapped();
    if (missing <= 0) return true;
    std::vector<int32_t> fresh;
    if (!cache.take(missing, fresh)) return false;
    for (int32_t id : fresh) { cache.pin(id); s.own.push_back(id); }
    s.build(n_pages);
    return true;
}

/// The conversation's pages up to `filled` cells hold its K/V: key them into the cache (full pages only, from
/// where the last call stopped).  Pages the walk mapped are keyed already.
template <class Ids>
inline void session_register(PageCache& cache, SessionPages& s, const Ids& ids, int64_t filled, int page_size) {
    const int64_t ps = page_size > 0 ? page_size : 1;
    const int64_t upto = std::min<int64_t>(std::min<int64_t>(filled, (int64_t) ids.size()) / ps, s.mapped());
    if (upto <= s.last_reg) return;
    uint64_t key = 0;
    std::array<int32_t, PageCache::page_cells> toks{};
    for (int64_t p = 0; p < upto; ++p) {
        for (int64_t i = 0; i < ps && i < (int64_t) PageCache::page_cells; ++i) toks[(size_t) i] = (int32_t) ids[(size_t) (p * ps + i)];
        key = PageCache::page_key(key, toks.data(), (int) ps);
        if (p < (int64_t) s.cached.size() || p < s.last_reg) continue;
        cache.register_page(s.own[(size_t) (p - (int64_t) s.cached.size())], key, toks.data());
    }
    s.last_reg = upto;
}

}  // namespace strata::core
