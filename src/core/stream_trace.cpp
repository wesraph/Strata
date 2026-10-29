// src/core/stream_trace.cpp - the shadow log of stream waits, event records and synchronizes (stream_trace.hpp).
#include "strata/core/stream_trace.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <chrono>
#include <map>
#include <thread>
#include <vector>

namespace strata::core {

namespace {
struct Entry {
    uint64_t t_us = 0;
    uint64_t tid = 0;
    const char* where = nullptr;
    const void* stream = nullptr;
    const void* event = nullptr;
    int dev = -1;
    StKind kind = StKind::wait;
    uint32_t repeats = 0;
    const void* done = nullptr;   // STRATA_STREAM_TRACE=2: the event recorded right after this launch (its pool's)
    uint32_t seq = 0;             // the launch's number on its stream (the pool reuses events: valid while recent)
};
constexpr size_t kRing = 65536;
Entry g_ring[kRing];
std::atomic<uint64_t> g_next{0};
uint64_t now_us() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
uint64_t tid_of() { return (uint64_t) std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xffff; }
const char* kind_name(StKind k) {
    switch (k) {
        case StKind::wait: return "WAIT";
        case StKind::record: return "RECORD";
        case StKind::sync_begin: return "SYNC-BEGIN";
        case StKind::sync_end: return "SYNC-END";
        case StKind::esync_begin: return "ESYNC-BEGIN";
        case StKind::esync_end: return "ESYNC-END";
        case StKind::launch: return "LAUNCH";
        case StKind::graph: return "GRAPH";
    }
    return "?";
}
}  // namespace

// every stream ever logged (the ring forgets; a stream that is busy at a stall is worth a query even when nothing
// was enqueued on it lately)
constexpr size_t kStreams = 512;
struct SeenStream { std::atomic<const void*> stream{nullptr}; int dev = -1; };
SeenStream g_streams[kStreams];
std::atomic<size_t> g_n_streams{0};
void note_stream(const void* stream, int dev) {
    if (stream == nullptr) return;
    const size_t n = g_n_streams.load(std::memory_order_acquire);
    for (size_t i = 0; i < n; ++i) if (g_streams[i].stream.load(std::memory_order_relaxed) == stream) return;
    const size_t i = g_n_streams.fetch_add(1, std::memory_order_acq_rel);
    if (i < kStreams) { g_streams[i].dev = dev; g_streams[i].stream.store(stream, std::memory_order_release); }
}

const char* kernel_name(const void* func) {
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && defined(CUDART_VERSION) && CUDART_VERSION >= 12030
    const char* name = nullptr;
    if (func != nullptr && cudaFuncGetName(&name, func) == cudaSuccess && name != nullptr) return name;
    (void) cudaGetLastError();
#else
    (void) func;
#endif
    return "?";
}

void st_log_launch(StKind kind, const void* stream, const void* func, int dev, const void* done, uint32_t seq) {
    note_stream(stream, dev);
    const uint64_t t = now_us();
    const uint64_t i = g_next.fetch_add(1, std::memory_order_relaxed);
    Entry& e = g_ring[i % kRing];
    e.t_us = t; e.tid = tid_of(); e.repeats = 0; e.where = nullptr; e.stream = stream; e.event = func; e.dev = dev; e.kind = kind;
    e.done = done; e.seq = seq;
}
void st_log(StKind kind, const char* where, const void* stream, const void* event, int dev) {
    note_stream(stream, dev);
    // a polling loop (a synchronize begun and ended a microsecond apart, over and over) must not wash the ring: a
    // sync_begin that repeats the previous entry's thread and site within 200 us is counted, not appended
    // (the begin/end pair stays; the pair's last repetition keeps the time)
    const uint64_t tid = tid_of();
    const uint64_t t = now_us();
    if (kind == StKind::sync_begin) {
        const uint64_t n = g_next.load(std::memory_order_acquire);
        if (n >= 2) {
            Entry& b = g_ring[(n - 2) % kRing];
            Entry& e = g_ring[(n - 1) % kRing];
            if (b.kind == StKind::sync_begin && e.kind == StKind::sync_end && b.tid == tid && b.where == where &&
                t - e.t_us < 200) {   // (the stream may differ: a loop over the stages' streams is one loop)
                b.repeats++; b.t_us = t;
                return;
            }
        }
    } else if (kind == StKind::sync_end) {
        const uint64_t n = g_next.load(std::memory_order_acquire);
        if (n >= 2) {
            Entry& b = g_ring[(n - 2) % kRing];
            Entry& e = g_ring[(n - 1) % kRing];
            if (b.repeats > 0 && b.kind == StKind::sync_begin && e.kind == StKind::sync_end && b.tid == tid &&
                e.where == where && e.tid == tid) {
                e.t_us = t;
                return;
            }
        }
    }
    const uint64_t i = g_next.fetch_add(1, std::memory_order_relaxed);
    Entry& e = g_ring[i % kRing];
    e.t_us = t;
    e.tid = tid;
    e.repeats = 0;
    e.where = where;
    e.stream = stream;
    e.event = event;
    e.dev = dev;
    e.kind = kind;
    e.done = nullptr;
    e.seq = 0;
}

void stream_trace_dump(std::FILE* f, size_t last_n) {
    const uint64_t end = g_next.load(std::memory_order_acquire);
    const uint64_t n = std::min<uint64_t>(end, kRing);
    const uint64_t begin = end - n;
    std::vector<Entry> es;
    es.reserve((size_t) n);
    for (uint64_t i = begin; i < end; ++i) es.push_back(g_ring[i % kRing]);   // (a racing writer may tear the oldest)
    const uint64_t t_last = es.empty() ? 0 : es.back().t_us;
    std::fprintf(f, "strata stream trace: %llu entries logged, the last %zu (us before the last; tid = low 16 bits of the thread):\n",
                 (unsigned long long) end, std::min<size_t>(last_n, es.size()));
    const size_t from = es.size() > last_n ? es.size() - last_n : 0;
    for (size_t i = from; i < es.size(); ++i) {
        const Entry& e = es[i];
        if (e.kind == StKind::launch)
            std::fprintf(f, "  %10.1f %-11s tid %04llx dev %d stream %p kernel %s\n", (double) (t_last - e.t_us), kind_name(e.kind),
                         (unsigned long long) e.tid, e.dev, e.stream, kernel_name(e.event));
        else
            std::fprintf(f, "  %10.1f %-11s tid %04llx dev %d stream %p event %p  %s%s\n",
                         (double) (t_last - e.t_us), kind_name(e.kind), (unsigned long long) e.tid, e.dev, e.stream, e.event,
                         e.where ? e.where : "?", e.repeats ? " (repeated)" : "");
    }
    // STRATA_STREAM_TRACE=2: for each stream still busy, the boundary between its launches that finished and the ones
    // that did not - the first launch not done is the kernel the card sits on (the host runs far ahead of the GPU, so
    // the last 40 entries above are usually all "not done")
    {
        std::map<const void*, std::vector<const Entry*>> launches;
        for (const Entry& e : es) if ((e.kind == StKind::launch || e.kind == StKind::graph) && e.done != nullptr) launches[e.stream].push_back(&e);
        for (auto& [st, v] : launches) {
            if (cudaStreamQuery((cudaStream_t) st) != cudaErrorNotReady) { (void) cudaGetLastError(); continue; }
            (void) cudaGetLastError();
            // from the newest back: the first launch whose event is done
            size_t first_not_done = v.size();
            for (size_t i = v.size(); i-- > 0;) {
                const cudaError_t q = cudaEventQuery((cudaEvent_t) v[i]->done);
                (void) cudaGetLastError();
                if (q == cudaSuccess) break;
                first_not_done = i;
            }
            std::fprintf(f, "strata stream trace: stream %p (dev %d) is busy: of its %zu traced launches %zu finished; the first not done and its neighbours:\n",
                         st, v.back()->dev, v.size(), first_not_done);
            const size_t a = first_not_done >= 4 ? first_not_done - 4 : 0, b = std::min(v.size(), first_not_done + 6);
            for (size_t i = a; i < b; ++i)
                std::fprintf(f, "   %s %10.1f %s %s\n", i == first_not_done ? ">>" : "  ", (double) (t_last - v[i]->t_us),
                             v[i]->kind == StKind::graph ? "graph" : "kernel", v[i]->kind == StKind::graph ? "(a captured window or commit)" : kernel_name(v[i]->event));
            if (first_not_done == v.size()) std::fprintf(f, "   (every traced launch finished: the stream waits on something else - a copy, an event, a graph)\n");
        }
    }
    std::fflush(f);
    // per stream, its last entries in order: the enqueue order on each stream is what a cycle reads off
    {
        std::map<const void*, std::vector<const Entry*>> per;
        for (const Entry& e : es) if (e.stream != nullptr && e.kind != StKind::esync_begin && e.kind != StKind::esync_end) per[e.stream].push_back(&e);
        std::fprintf(f, "strata stream trace: the last 40 entries per stream (kernel launches included):\n");
        for (const auto& [st, v] : per) {
            std::fprintf(f, " stream %p (dev %d), %zu entries in the ring:\n", st, v.back()->dev, v.size());
            const size_t from = v.size() > 40 ? v.size() - 40 : 0;
            for (size_t i = from; i < v.size(); ++i) {
                const Entry& e = *v[i];
                if (e.kind == StKind::launch) {
                    const char* state = "";
                    if (e.done != nullptr) {   // STRATA_STREAM_TRACE=2: the first launch not done is the one the card sits on
                        const cudaError_t q = cudaEventQuery((cudaEvent_t) e.done);
                        (void) cudaGetLastError();
                        state = q == cudaSuccess ? "  [done]" : q == cudaErrorNotReady ? "  [NOT DONE]" : "  [query failed]";
                    }
                    std::fprintf(f, "   %10.1f %-11s tid %04llx kernel %s%s\n", (double) (t_last - e.t_us), kind_name(e.kind),
                                 (unsigned long long) e.tid, kernel_name(e.event), state);
                } else
                    std::fprintf(f, "   %10.1f %-11s tid %04llx event %p  %s\n", (double) (t_last - e.t_us), kind_name(e.kind),
                                 (unsigned long long) e.tid, e.event, e.where ? e.where : "?");
            }
        }
    }
    // replay: each stream's entries in order; a wait binds to its event's last record before it (CUDA's semantics:
    // the record in force when the wait was enqueued; none in the ring = taken as complete); a record is complete when
    // its stream's cursor has passed it.  Cursors advance until nothing moves: a stream whose cursor rests on a wait
    // is blocked by that wait's record, and the streams that block each other are the cycle.
    {
        std::map<const void*, std::vector<const Entry*>> per;
        for (const Entry& e : es) if (e.stream != nullptr && (e.kind == StKind::wait || e.kind == StKind::record)) per[e.stream].push_back(&e);
        std::map<const Entry*, const Entry*> bound;   // wait -> the record it captured
        std::map<const void*, const Entry*> last;
        for (const Entry& e : es) {
            if (e.kind == StKind::record) last[e.event] = &e;
            else if (e.kind == StKind::wait) bound[&e] = last.count(e.event) ? last[e.event] : nullptr;
        }
        std::map<const void*, size_t> cur;
        for (const auto& [st, v] : per) cur[st] = 0;
        auto done = [&](const Entry* rec) {
            if (rec == nullptr) return true;
            const auto& v = per[rec->stream];
            const size_t c = cur[rec->stream];
            for (size_t i = 0; i < v.size(); ++i) if (v[i] == rec) return i < c;
            return true;
        };
        bool moved = true;
        while (moved) {
            moved = false;
            for (auto& [st, v] : per) {
                size_t& c = cur[st];
                while (c < v.size() && (v[c]->kind == StKind::record || done(bound[v[c]]))) { ++c; moved = true; }
            }
        }
        std::fprintf(f, "strata stream trace: replay (a wait binds to the record in force when it was enqueued; a stream resting on a wait is blocked):\n");
        for (const auto& [st, v] : per) {
            const size_t c = cur[st];
            if (c >= v.size()) { std::fprintf(f, "  stream %p (dev %d): all %zu entries complete\n", st, v.back()->dev, v.size()); continue; }
            const Entry* w = v[c];
            const Entry* r = bound[w];
            std::fprintf(f, "  stream %p (dev %d): BLOCKED at entry %zu of %zu, the wait at %s (%.1f us before the last) on event %p, bound to the record at %s on stream %p (dev %d), entry %zu of that stream\n",
                         st, v.back()->dev, c, v.size(), w->where ? w->where : "?", (double) (t_last - w->t_us), w->event,
                         r ? (r->where ? r->where : "?") : "?", r ? r->stream : nullptr, r ? r->dev : -1,
                         r ? (size_t) (std::find(per[r->stream].begin(), per[r->stream].end(), r) - per[r->stream].begin()) : 0);
        }
    }
    // which streams still have work: the one that never finishes carries the kernel that spins
    {
        std::map<const void*, int> seen;
        const size_t ns = std::min(g_n_streams.load(std::memory_order_acquire), kStreams);
        for (size_t i = 0; i < ns; ++i) if (const void* st = g_streams[i].stream.load(std::memory_order_acquire)) seen[st] = g_streams[i].dev;
        std::fprintf(f, "strata stream trace: the streams' state now (cudaStreamQuery):\n");
        for (const auto& [st, dev] : seen) {
            const cudaError_t q = cudaStreamQuery((cudaStream_t) st);
            std::fprintf(f, "  stream %p (dev %d): %s\n", st, dev,
                         q == cudaSuccess ? "idle" : q == cudaErrorNotReady ? "BUSY (work still running)" : cudaGetErrorString(q));
            (void) cudaGetLastError();
        }
    }
    // derived: open synchronizes (a begin with no end from the same thread on the same stream/event)
    std::map<std::pair<uint64_t, const void*>, const Entry*> open_sync, open_esync;
    std::map<const void*, const Entry*> last_record;              // event -> its last record
    std::map<const void*, std::vector<const Entry*>> waits_on;      // event -> waits logged after its last record
    for (const Entry& e : es) {
        switch (e.kind) {
            case StKind::sync_begin: open_sync[{e.tid, e.stream}] = &e; break;
            case StKind::sync_end: open_sync.erase({e.tid, e.stream}); break;
            case StKind::esync_begin: open_esync[{e.tid, e.event}] = &e; break;
            case StKind::esync_end: open_esync.erase({e.tid, e.event}); break;
            case StKind::record: last_record[e.event] = &e; waits_on[e.event].clear(); break;
            case StKind::wait: waits_on[e.event].push_back(&e); break;
        }
    }
    std::fprintf(f, "strata stream trace: synchronizes still open:\n");
    for (const auto& [k, e] : open_sync)
        std::fprintf(f, "  stream %p on dev %d by tid %04llx since %.1f us before the last: %s\n", k.second, e->dev,
                     (unsigned long long) k.first, (double) (t_last - e->t_us), e->where ? e->where : "?");
    for (const auto& [k, e] : open_esync)
        std::fprintf(f, "  event %p on dev %d by tid %04llx since %.1f us before the last: %s\n", k.second, e->dev,
                     (unsigned long long) k.first, (double) (t_last - e->t_us), e->where ? e->where : "?");
    std::fprintf(f, "strata stream trace: waits logged after their event's last record (the ones that can still be pending):\n");
    for (const auto& [ev, ws] : waits_on) {
        if (ws.empty()) continue;
        const Entry* r = last_record.count(ev) ? last_record[ev] : nullptr;
        for (const Entry* w : ws)
            std::fprintf(f, "  stream %p (dev %d) waits on event %p at %s; the event's last record: %s on stream %p (dev %d), %.1f us before the wait\n",
                         w->stream, w->dev, ev, w->where ? w->where : "?", r ? (r->where ? r->where : "?") : "NEVER RECORDED",
                         r ? r->stream : nullptr, r ? r->dev : -1, r ? (double) (w->t_us - r->t_us) : 0.0);
    }
}

}  // namespace strata::core

#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
// Every kernel launch of the process (the runtime's <<<>>> stubs call cudaLaunchKernel) and every graph launch, logged
// with its stream: the dump's per-stream tails then name the last kernels enqueued before a synchronize that never
// returned.  The executable's definition binds before libcudart's; the real one is found with RTLD_NEXT.
#include <dlfcn.h>
#include <mutex>
// STRATA_STREAM_TRACE=1 logs the launches too (off: the entry points forward at once; the waits, records and
// synchronizes are always logged - they are few and cheap)
static int st_trace_level() {
    static const int v = [] { const char* e = std::getenv("STRATA_STREAM_TRACE"); return e != nullptr ? std::atoi(e) : 0; }();
    return v;
}
static bool st_launches_on() { return st_trace_level() != 0; }
// STRATA_STREAM_TRACE=2: an event recorded behind every launch, from a pool of 64 per stream (round robin), so the
// stall report can query which of a busy stream's last launches finished - the first one not done is the kernel the
// card sits on.  Not inside a graph capture (the record would become a node of the graph).
namespace {
constexpr uint32_t kPoolEvents = 1024;
struct LaunchPool { const void* stream = nullptr; int dev = -1; cudaEvent_t ev[kPoolEvents] = {}; uint32_t next = 0; };
constexpr size_t kPools = 48;
LaunchPool g_pools[kPools];
std::atomic<size_t> g_n_pools{0};
std::mutex g_pool_mu;
cudaEvent_t launch_event(cudaStream_t stream, int dev, uint32_t& seq) {
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) { (void) cudaGetLastError(); return nullptr; }
    std::lock_guard<std::mutex> lk(g_pool_mu);
    const size_t n = g_n_pools.load(std::memory_order_acquire);
    LaunchPool* p = nullptr;
    for (size_t i = 0; i < n; ++i) if (g_pools[i].stream == (const void*) stream) { p = &g_pools[i]; break; }
    if (p == nullptr) {
        if (n >= kPools) return nullptr;
        p = &g_pools[n]; p->stream = stream; p->dev = dev;
        for (cudaEvent_t& e : p->ev) if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) { (void) cudaGetLastError(); e = nullptr; }
        g_n_pools.store(n + 1, std::memory_order_release);
    }
    cudaEvent_t e = p->ev[p->next % kPoolEvents];
    seq = ++p->next;
    if (e != nullptr && cudaEventRecord(e, stream) != cudaSuccess) { (void) cudaGetLastError(); return nullptr; }
    return e;
}
}  // namespace
static void log_launch(cudaStream_t stream, const void* func) {
    int dev = -1; (void) cudaGetDevice(&dev);
    const void* done = nullptr; uint32_t seq = 0;
    if (st_trace_level() >= 2) done = launch_event(stream, dev, seq);
    strata::core::st_log_launch(strata::core::StKind::launch, stream, func, dev, done, seq);
}
extern "C" cudaError_t cudaLaunchKernel(const void* func, dim3 grid, dim3 block, void** args, size_t shmem, cudaStream_t stream) {
    using Fn = cudaError_t (*)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
    static const Fn real = (Fn) dlsym(RTLD_NEXT, "cudaLaunchKernel");
    const cudaError_t r = real(func, grid, block, args, shmem, stream);
    if (st_launches_on()) log_launch(stream, func);
    return r;
}
// CUDA 13's <<<>>> stubs call __cudaLaunchKernel (and some paths cudaLaunchKernelExC), not cudaLaunchKernel: without
// these two the per-stream tails showed no kernel at all (2x RTX 3090, 2026-10-29, the concurrent read's stall).
extern "C" cudaError_t __cudaLaunchKernel(const void* func, dim3 grid, dim3 block, void** args, size_t shmem, cudaStream_t stream) {
    using Fn = cudaError_t (*)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
    static const Fn real = (Fn) dlsym(RTLD_NEXT, "__cudaLaunchKernel");
    const cudaError_t r = real(func, grid, block, args, shmem, stream);
    if (st_launches_on()) log_launch(stream, func);
    return r;
}
extern "C" cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* config, const void* func, void** args) {
    using Fn = cudaError_t (*)(const cudaLaunchConfig_t*, const void*, void**);
    static const Fn real = (Fn) dlsym(RTLD_NEXT, "cudaLaunchKernelExC");
    const cudaError_t r = real(config, func, args);
    if (st_launches_on() && config != nullptr) log_launch(config->stream, func);
    return r;
}
extern "C" cudaError_t cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream) {
    using Fn = cudaError_t (*)(cudaGraphExec_t, cudaStream_t);
    static const Fn real = (Fn) dlsym(RTLD_NEXT, "cudaGraphLaunch");
    const cudaError_t r = real(exec, stream);
    if (st_launches_on()) {
        int dev = -1; (void) cudaGetDevice(&dev);
        const void* done = nullptr; uint32_t seq = 0;
        if (st_trace_level() >= 2) done = launch_event(stream, dev, seq);
        strata::core::st_log_launch(strata::core::StKind::graph, stream, exec, dev, done, seq);
    }
    return r;
}
#endif
