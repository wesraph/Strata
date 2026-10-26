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
#include <set>
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

}  // namespace strata::core
