// include/strata/core/stream_trace.hpp - a shadow log of the engine's stream waits, event records and synchronizes.
//
// Why: a hang with both cards idle and a stream synchronize that never returns is a dependency cycle between streams
// (a wait on an event that will only be recorded after the waiter moves on).  The host backtraces cannot name it: the
// waits sit inside the driver.  Every cudaStreamWaitEvent / cudaEventRecord / cudaStreamSynchronize /
// cudaEventSynchronize of the prompt path, the verifier and the drafter goes through the wrappers below, which log
// (time, thread, kind, stream, event, where) into a ring, and the watchdog's stall report dumps the ring plus the
// derived state: which stream is inside a synchronize, on which event each pending wait sits, and where that event
// was last recorded.  The cycle then reads off the dump.  The wrappers are the calls they replace (same return).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace strata::core {

enum class StKind : uint8_t { wait = 0, record = 1, sync_begin = 2, sync_end = 3, esync_begin = 4, esync_end = 5, launch = 6, graph = 7 };

/// Log one entry (lock-free, a fixed ring; the oldest entries are overwritten).
void st_log(StKind kind, const char* where, const void* stream, const void* event, int dev);
/// A launch entry with its completion event (STRATA_STREAM_TRACE=2) and its number on the stream.
void st_log_launch(StKind kind, const void* stream, const void* func, int dev, const void* done, uint32_t seq);

inline cudaError_t st_wait(cudaStream_t stream, cudaEvent_t event, const char* where) {
    int dev = -1; (void) cudaGetDevice(&dev);
    st_log(StKind::wait, where, stream, event, dev);
    return cudaStreamWaitEvent(stream, event, 0);
}
inline cudaError_t st_record(cudaEvent_t event, cudaStream_t stream, const char* where) {
    int dev = -1; (void) cudaGetDevice(&dev);
    st_log(StKind::record, where, stream, event, dev);
    return cudaEventRecord(event, stream);
}
inline cudaError_t st_stream_sync(cudaStream_t stream, const char* where) {
    int dev = -1; (void) cudaGetDevice(&dev);
    st_log(StKind::sync_begin, where, stream, nullptr, dev);
    const cudaError_t e = cudaStreamSynchronize(stream);
    st_log(StKind::sync_end, where, stream, nullptr, dev);
    return e;
}
inline cudaError_t st_event_sync(cudaEvent_t event, const char* where) {
    int dev = -1; (void) cudaGetDevice(&dev);
    st_log(StKind::esync_begin, where, nullptr, event, dev);
    const cudaError_t e = cudaEventSynchronize(event);
    st_log(StKind::esync_end, where, nullptr, event, dev);
    return e;
}

inline cudaError_t st_device_sync(const char* where) {
    int dev = -1; (void) cudaGetDevice(&dev);
    st_log(StKind::sync_begin, where, nullptr, nullptr, dev);   // stream null = the whole device
    const cudaError_t e = cudaDeviceSynchronize();
    st_log(StKind::sync_end, where, nullptr, nullptr, dev);
    return e;
}

/// The ring's last entries (newest last) and the derived state: the synchronizes still open (begun, not ended),
/// per event its last record, and the waits since that record whose waiter is a stream that was synchronized on
/// or waited on since.  Written by the watchdog's stall report.
void stream_trace_dump(std::FILE* f, size_t last_n = 160);

#define STRATA_ST_STR2(x) #x
#define STRATA_ST_STR(x) STRATA_ST_STR2(x)
/// "file:line" for the wrappers' `where`
#define STRATA_ST_HERE (__FILE__ ":" STRATA_ST_STR(__LINE__))

}  // namespace strata::core
