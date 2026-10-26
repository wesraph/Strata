// src/core/kv_share_test.cpp - the shared K/V arena's carve, tables and table-aware moves.
//
// What is pinned here, in order:
//   carve     two states over one arena share the pool base, sit at their own private runs, and their
//             default (no-prefix) tables map exactly those runs with -1 past them
//   tables    a branch table maps the shared prefix at its own ids and the suffix at the private run
//   save      conversation_kv_save through a table writes the LOGICAL layout, so an identity-mapped
//             reference restores to the same bytes (a branch saves what a private session reads)
//   skip      restore_prefix with skip_pages leaves the shared pages untouched (the branch keeps them)
//   copy      conversation_kv_copy between two mappings moves no page that lands where it already is
//             (the shared prefix), and moves the rest to the right physical pages
//   zero      qsa_state_zero clears a shared state's private run only - the shared pages and the other
//             session's run keep their bytes
//   off       with --kv-share off every state owns its pools carved from its own buffer, as before
#include "strata/core/layer.hpp"
#include "strata/core/conversation_snapshot.hpp"

#include <cuda_runtime.h>

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
std::vector<uint8_t> read_pool(const void* pool, size_t bytes) {
    std::vector<uint8_t> out(bytes);
    cuda_check(cudaMemcpy(out.data(), pool, bytes, cudaMemcpyDeviceToHost));
    return out;
}
// a per-page pattern: page p's first byte is p, so a page landing in the wrong place is a different number
void write_page(void* pool, int64_t phys_page, size_t page_bytes, uint8_t value) {
    cuda_check(cudaMemset((uint8_t*) pool + (size_t) phys_page * page_bytes, value, page_bytes));
}
bool page_is(const void* pool, int64_t phys_page, size_t page_bytes, uint8_t value) {
    std::vector<uint8_t> b = read_pool((const uint8_t*) pool + (size_t) phys_page * page_bytes, page_bytes);
    return !b.empty() && b[0] == value && std::all_of(b.begin(), b.end(), [&](uint8_t x) { return x == value; });
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
    const int64_t pages = max_cells / page_cells;   // 2048

    // ---- the carve: one arena, two sessions, the shared front ----
    qsa_set_kv_share(2048);   // 512 shared pages; the private runs hold 1536
    const int64_t shared = qsa_kv_share_pages(g, max_cells);
    check(shared == 512, "share pages");
    const int64_t priv = pages - shared;
    const uint64_t arena_bytes = qsa_share_arena_bytes(g, max_cells, 1, 2);
    void* arena = nullptr;
    cuda_check(cudaMalloc(&arena, arena_bytes));
    cuda_check(cudaMemset(arena, 0xAB, arena_bytes));   // poison: anything the carve misses shows
    const uint64_t shared_state_bytes = qsa_state_bytes(g, max_cells, true, 0, true);
    const uint64_t plain_state_bytes = qsa_state_bytes(g, max_cells, true, 0, false);
    std::vector<void*> bufs;
    auto buf = [&](bool external) {
        void* b = nullptr;
        cuda_check(cudaMalloc(&b, external ? shared_state_bytes : plain_state_bytes));
        bufs.push_back(b);
        return b;
    };
    QsaState a{}, b{};
    QsaShareArena share0{arena, 2, 0}, share1{arena, 2, 1};
    check(qsa_state_init(g, max_cells, buf(true), a, nullptr, 0, &share0) > 0, "state a init");
    check(qsa_state_init(g, max_cells, buf(true), b, nullptr, 0, &share1) > 0, "state b init");
    sync();
    check(a.k_pool == b.k_pool && a.v_pool == b.v_pool, "the states share the pool base");
    check(a.share_pages == shared && b.share_pages == shared, "both know the shared front");
    check(a.share_first == shared && b.share_first == shared + priv, "each at its own private run");

    const size_t k_page = (size_t) s.n_head_kv * page_cells * s.head_dim * 2;   // one page of the f16 K pool
    // the default table is the private mapping: p -> S + p, -1 past the run
    std::vector<int32_t> tab = download(a.page_table, a.n_pages);
    check((int64_t) tab.size() == pages, "table size");
    check(tab[0] == (int32_t) shared && tab[(size_t) (priv - 1)] == (int32_t) (shared + priv - 1),
          "the default table maps the private run");
    check(tab[(size_t) priv] == -1, "the default table is -1 past the run");

    // ---- the branch table: 10 pages of shared prefix ----
    const int64_t pfx = 10;
    std::vector<int32_t> host_tab((size_t) pages);
    qsa_state_share_table(a, g, pfx, host_tab.data(), nullptr);
    sync();
    tab = download(a.page_table, a.n_pages);
    check(tab[0] == 0 && tab[(size_t) (pfx - 1)] == (int32_t) (pfx - 1), "the prefix maps at its own ids");
    check(tab[(size_t) pfx] == (int32_t) (shared + 0), "the suffix maps at the private run");
    check(std::equal(host_tab.begin(), host_tab.end(), tab.begin()), "the host table matches the device one");
    check(qsa_state_share_capacity(a, g, pfx) == (pfx + priv) * page_cells, "the branch capacity");

    // fill: pages [0, pfx) of the prefix, [pfx, pfx + 10) of the suffix, distinct patterns
    const int64_t suffix_pages = 10;
    const int64_t upto_cells = (pfx + suffix_pages) * page_cells;
    for (int64_t p = 0; p < pfx + suffix_pages; ++p)
        write_page(a.k_pool, tab[(size_t) p], k_page, (uint8_t) (1 + p));
    sync();

    // ---- save through the table: the image holds the LOGICAL bytes ----
    ConversationKv image;
    std::string err;
    check(conversation_kv_save(image, a, g, upto_cells, true, err, 0, nullptr, host_tab.data()), "kv save mapped");
    // an identity-mapped (private, unbranch) reference restores to the same logical bytes
    qsa_set_kv_share(0);
    QsaState r{};
    check(qsa_state_init(g, max_cells, buf(false), r, nullptr, 0, nullptr) > 0, "reference init");
    check(r.share_first < 0, "the reference owns its pools");
    sync();
    check(conversation_kv_restore(image, r, g, upto_cells, true, err), "kv restore identity");
    for (int64_t p = 0; p < pfx + suffix_pages; ++p)
        check(page_is(r.k_pool, p, k_page, (uint8_t) (1 + p)), "the reference reads the branch's bytes");

    // ---- skip_pages: the shared pages stay where they are ----
    write_page(a.k_pool, 0, k_page, 0xEE);   // poison the first shared page
    sync();
    check(conversation_kv_restore_prefix(image, a, g, upto_cells, upto_cells, true, err, host_tab.data(), pfx),
          "kv restore_prefix skip");
    check(page_is(a.k_pool, 0, k_page, 0xEE), "the skipped shared page was not written");
    check(page_is(a.k_pool, (int64_t) tab[(size_t) (pfx + suffix_pages - 1)], k_page,
                  (uint8_t) (1 + pfx + suffix_pages - 1)), "the suffix pages were restored");

    // ---- copy between mappings: the shared pages move nowhere ----
    std::vector<int32_t> host_b((size_t) pages);
    qsa_state_share_table(b, g, pfx, host_b.data(), nullptr);
    sync();
    check(conversation_kv_copy(a, b, g, upto_cells, true, err, false, host_tab.data(), host_b.data()),
          "kv copy mapped");
    // the shared page still holds the poison (no copy wrote it back) and both sides read it as theirs
    check(page_is(b.k_pool, 0, k_page, 0xEE), "the copy left the shared page alone");
    std::vector<int32_t> tab_b = download(b.page_table, b.n_pages);
    check(tab_b[0] == 0 && tab_b[(size_t) pfx] == (int32_t) (shared + priv), "b maps prefix and its own run");
    check(page_is(b.k_pool, (int64_t) tab_b[(size_t) pfx], k_page, (uint8_t) (1 + pfx)), "b's suffix page moved in");

    // ---- zero: a shared state clears only its private run ----
    qsa_state_zero(b, g, nullptr);
    sync();
    check(page_is(b.k_pool, 0, k_page, 0xEE), "zero left the shared page");
    check(page_is(a.k_pool, (int64_t) tab[(size_t) pfx], k_page, (uint8_t) (1 + pfx)), "zero left a's run");
    check(page_is(b.k_pool, (int64_t) tab_b[(size_t) pfx], k_page, 0), "zero cleared b's own run");

    // ---- off: byte-identical carve, pools inside the state's own buffer ----
    const uint64_t plain = plain_state_bytes;
    check(plain > shared_state_bytes, "a shared state carves fewer bytes of its own");
    QsaState e{};
    check(qsa_state_init(g, max_cells, buf(false), e, nullptr, 0, nullptr) > 0, "plain init");
    sync();
    check(e.share_first < 0 && e.n_slots == e.n_pages, "a plain state is identity-mapped");
    tab = download(e.page_table, e.n_pages);
    check(tab[(size_t) (pages - 1)] == (int32_t) (pages - 1), "the plain table is the identity");

    sync();
    for (void* p : bufs) cudaFree(p);
    cudaFree(arena);
    qsa_set_kv_share(0);
    std::fprintf(stderr, "kv_share_test: %d checks\n", checks);
    return 0;
}
