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

#include <cstdint>
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
    PageCache() = default;
    explicit PageCache(int64_t pages) { init(pages); }

    void init(int64_t pages) {
        alloc_.init(pages);
        key_to_page_.clear();
        page_key_.assign((size_t) pages, 0);
        page_tokens_.assign((size_t) pages * page_cells, 0);
        page_refs_.assign((size_t) pages, 0);
        page_stamp_.assign((size_t) pages, 0);
        stamp_ = 0;
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

    void unpin(int32_t id) {
        if (id < 0 || (size_t) id >= page_key_.size() || page_refs_[(size_t) id] <= 0) return;
        if (--page_refs_[(size_t) id] == 0 && page_key_[(size_t) id] != 0)
            touch(id);   // its heap entries were skipped while pinned: re-enter the eviction order
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
    }

    bool take(int64_t n, std::vector<int32_t>& out) {
        while (alloc_.free_count() < n && evict_one()) {}
        return alloc_.take(n, out);
    }

    /// A session's page goes back: pinned pages lose their pin, hashed pages stay cached (evictable),
    /// the rest free outright.
    void release_page(int32_t id) {
        if (id < 0 || (size_t) id >= page_key_.size()) { alloc_.release({id}); return; }
        unpin(id);
        if (page_key_[(size_t) id] == 0 || page_refs_[(size_t) id] > 0) {
            if (page_key_[(size_t) id] == 0) alloc_.release({id});
            return;   // still pinned elsewhere, or cached: stays resident
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

    /// Evict the least recently used cached page nobody maps.  false: nothing evictable.  The heap is
    /// lazy: stale entries (touched, re-keyed, freed since) are skipped until a live one surfaces.
    bool evict_one() {
        while (!heap_.empty()) {
            const auto [stamp, id] = heap_.top();
            heap_.pop();
            if (id < 0 || (size_t) id >= page_key_.size()) continue;
            if (page_key_[(size_t) id] == 0 || page_refs_[(size_t) id] > 0) continue;   // freed or pinned: stale
            if (page_stamp_[(size_t) id] != stamp) continue;   // touched later: a fresher entry is in the heap
            key_to_page_.erase(page_key_[(size_t) id]);
            page_key_[(size_t) id] = 0;
            alloc_.release({id});
            return true;
        }
        return false;
    }

    int64_t free_count() const { return alloc_.free_count(); }
    int64_t total() const { return alloc_.total(); }
    int64_t cached_count() const { return (int64_t) key_to_page_.size(); }

private:
    PageAlloc alloc_;
    std::unordered_map<uint64_t, int32_t> key_to_page_;
    std::vector<uint64_t> page_key_;
    std::vector<int32_t> page_tokens_;   // 4 ids per page: what every hit is verified against
    std::vector<int32_t> page_refs_;
    std::vector<int64_t> page_stamp_;
    using HeapEntry = std::pair<int64_t, int32_t>;   // (stamp, page), min-stamp first
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> heap_;
    int64_t stamp_ = 0;
};

}  // namespace strata::core
