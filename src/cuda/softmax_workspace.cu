#include "kernel_lab/softmax_workspace.hpp"
#include "kernel_lab/validation.hpp"
#include "device_ops.cuh"
#include <array>
#include <chrono>

namespace kernel_lab {
struct SoftmaxWorkspace::Impl {
    std::unique_ptr<device::Buffer> input_buffer, output_buffer;
    std::size_t capacity = 0, allocations = 0;
    int owner_device = 0;
    Impl() { device::check(cudaGetDevice(&owner_device), "workspace device"); }
    void reserve(std::size_t count) {
        int current = 0;
        device::check(cudaGetDevice(&current), "workspace current device");
        if (current != owner_device) throw std::runtime_error("workspace CUDA device changed");
        if (count <= capacity) return;
        // Strong allocation guarantee: leave existing buffers intact if either fails.
        auto next_input = std::make_unique<device::Buffer>(count);
        auto next_output = std::make_unique<device::Buffer>(count);
        input_buffer.swap(next_input);
        output_buffer.swap(next_output);
        capacity = count;
        ++allocations;
    }
    template<bool Timed, SoftmaxSyncMode Mode = SoftmaxSyncMode::Staged>
    std::vector<float> execute(const std::vector<float>& input, std::size_t rows,
                               std::size_t cols, SoftmaxStageTimes* times) {
        static_assert(!Timed || Mode == SoftmaxSyncMode::Staged, "stages require explicit boundaries");
        using Clock = std::chrono::steady_clock;
        std::array<Clock::time_point, 7> ticks;
        if constexpr (Timed) { *times = {}; ticks[0] = Clock::now(); }
        validate_softmax_shape(input.size(), rows, cols);
        require_finite_input(input);
        if constexpr (Timed) ticks[1] = Clock::now();
        reserve(input.size());
        if constexpr (Timed) ticks[2] = Clock::now();
        const auto bytes = input.size() * sizeof(float);
        device::check(cudaMemcpy(input_buffer->data(), input.data(), bytes, cudaMemcpyHostToDevice), "workspace H2D");
        // All copies/launches use the library's same legacy default stream.
        // Staged needs a host boundary for diagnostic attribution; other modes
        // rely on stream ordering even if pageable H2D returns before final DMA.
        if constexpr (Mode == SoftmaxSyncMode::Staged)
            device::check(cudaDeviceSynchronize(), "workspace H2D complete");
        if constexpr (Timed) ticks[3] = Clock::now();
        device::launch_softmax_warp_register(input_buffer->data(), output_buffer->data(), rows, cols);
        if constexpr (Mode != SoftmaxSyncMode::CopyCompletion)
            device::check(cudaDeviceSynchronize(), "workspace kernel complete");
        if constexpr (Timed) ticks[4] = Clock::now();
        std::vector<float> output(input.size());
        if constexpr (Timed) ticks[5] = Clock::now();
        // Blocking D2H is ordered after the kernel and completes CPU output.
        // In CopyCompletion, host output allocation above may overlap kernel work.
        device::check(cudaMemcpy(output.data(), output_buffer->data(), bytes, cudaMemcpyDeviceToHost), "workspace D2H");
        if constexpr (Timed) {
            ticks[6] = Clock::now();
            const auto ms = [&](int end, int begin) {
                return std::chrono::duration<double, std::milli>(ticks[end] - ticks[begin]).count();
            };
            *times = {ms(1,0), ms(2,1), ms(3,2), ms(4,3), ms(5,4), ms(6,5), ms(6,0)};
        }
        return output;
    }
};
SoftmaxWorkspace::SoftmaxWorkspace() : impl_(std::make_unique<Impl>()) {}
SoftmaxWorkspace::~SoftmaxWorkspace() = default;
std::size_t SoftmaxWorkspace::capacity() const noexcept { return impl_->capacity; }
std::size_t SoftmaxWorkspace::allocation_events() const noexcept { return impl_->allocations; }
std::vector<float> SoftmaxWorkspace::run(const std::vector<float>& input, std::size_t rows, std::size_t cols) {
    return impl_->execute<false>(input, rows, cols, nullptr);
}
std::vector<float> SoftmaxWorkspace::run(const std::vector<float>& input, std::size_t rows,
                                        std::size_t cols, SoftmaxSyncMode mode) {
    switch (mode) {
        case SoftmaxSyncMode::Staged: return impl_->execute<false>(input, rows, cols, nullptr);
        case SoftmaxSyncMode::KernelBoundary:
            return impl_->execute<false, SoftmaxSyncMode::KernelBoundary>(input, rows, cols, nullptr);
        case SoftmaxSyncMode::CopyCompletion:
            return impl_->execute<false, SoftmaxSyncMode::CopyCompletion>(input, rows, cols, nullptr);
        default: throw std::invalid_argument("invalid SoftmaxSyncMode");
    }
}
std::vector<float> SoftmaxWorkspace::run_profiled(const std::vector<float>& input, std::size_t rows,
                                                std::size_t cols, SoftmaxStageTimes& times) {
    return impl_->execute<true>(input, rows, cols, &times);
}
} // namespace kernel_lab
