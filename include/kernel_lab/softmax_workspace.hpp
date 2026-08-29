#pragma once
#include <cstddef>
#include <memory>
#include <vector>

namespace kernel_lab {
enum class SoftmaxSyncMode {
    Staged,          // original H2D + kernel device-wide waits
    KernelBoundary,  // only explicit post-kernel wait
    CopyCompletion   // same-stream ordering; blocking D2H completes the result
};
// Diagnostic CPU-wall stages, NOT GPU-event kernel time. Synchronous boundaries.
// total_ms covers execute only, excluding construction/destruction of workspace.
struct SoftmaxStageTimes {
    double validation_ms = 0, reserve_ms = 0, h2d_ms = 0;
    double kernel_sync_ms = 0, host_output_ms = 0, d2h_ms = 0, total_ms = 0;
};

// Finite FP32 input; same numerical contract and kernel dispatch as warp wrapper.
// Retains two device buffers up to the largest successful input size. Every call
// still validates input, transfers H2D/D2H, synchronizes and returns a new vector.
// Single-thread use only. Keep the creating CUDA device current for its lifetime.
// Copy/move disabled to make ownership unambiguous. No pinned memory/async overlap.
class SoftmaxWorkspace {
public:
    SoftmaxWorkspace();
    ~SoftmaxWorkspace();
    SoftmaxWorkspace(const SoftmaxWorkspace&) = delete;
    SoftmaxWorkspace& operator=(const SoftmaxWorkspace&) = delete;
    std::vector<float> run(const std::vector<float>& input, std::size_t rows, std::size_t cols);
    // All modes return completed CPU output. CopyCompletion is NOT an async API.
    // Uses this library's legacy default stream; no cross-stream ordering contract.
    std::vector<float> run(const std::vector<float>& input, std::size_t rows, std::size_t cols,
                           SoftmaxSyncMode mode);
    // Diagnostic timings always use Staged, never timings for CopyCompletion.
    std::vector<float> run_profiled(const std::vector<float>& input, std::size_t rows,
                                  std::size_t cols, SoftmaxStageTimes& times);
    std::size_t capacity() const noexcept;
    std::size_t allocation_events() const noexcept; // successful buffer-pair growths
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace kernel_lab
