#include "kernel_lab/softmax_workspace.hpp"
#include "kernel_lab/validation.hpp"
#include "operator_cases.hpp"
#include "../src/cuda/device_ops.cuh"
#include <type_traits>

int main() {
    try {
        using namespace kernel_lab;
        static_assert(!std::is_copy_constructible_v<SoftmaxWorkspace>);
        static_assert(!std::is_move_constructible_v<SoftmaxWorkspace>);
        device::check(cudaSetDevice(0), "test device");
        SoftmaxWorkspace workspace;
        if (workspace.capacity() || workspace.allocation_events()) throw std::runtime_error("eager allocation");
        int checked = 0;
        const auto run = [&](std::size_t rows, std::size_t cols, float bias) {
            std::vector<float> input(rows * cols);
            for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 37) - 18) * bias;
            const auto expected = cases::softmax_oracle(input, rows, cols);
            const auto capacity = workspace.capacity(), allocations = workspace.allocation_events();
            auto output = workspace.run(input, rows, cols);
            require_close(output, expected, 2e-6, 2e-5, "workspace");
            require_probability_rows(output, rows, cols);
            if (workspace.capacity() != std::max(capacity, input.size()) ||
                workspace.allocation_events() != allocations + (input.size() > capacity ? 1 : 0))
                throw std::runtime_error("incorrect capacity growth/reuse");
            SoftmaxStageTimes times;
            auto profiled = workspace.run_profiled(input, rows, cols, times);
            require_close(profiled, expected, 2e-6, 2e-5, "workspace profiled");
            if (workspace.allocation_events() != allocations + (input.size() > capacity ? 1 : 0))
                throw std::runtime_error("profiled call reallocated");
            double sum = 0;
            for (double stage : {times.validation_ms, times.reserve_ms, times.h2d_ms,
                                 times.kernel_sync_ms, times.host_output_ms, times.d2h_ms}) {
                if (!std::isfinite(stage) || stage < 0) throw std::runtime_error("invalid stage time");
                sum += stage;
            }
            if (!std::isfinite(times.total_ms) || times.total_ms <= 0 || std::abs(sum - times.total_ms) > 1e-8)
                throw std::runtime_error("inconsistent stages");
            ++checked;
        };
        // Grow, shrink, change shape at same element count, then grow again.
        run(1, 1, 1); run(7, 37, 0.25f); run(1, 1, 0); run(1, 259, 1000);
        for (std::size_t rows : {1U, 3U, 4U, 5U, 7U})
            for (std::size_t cols : {31U, 32U, 33U, 127U, 128U, 129U, 511U, 512U,
                                     513U, 1023U, 1024U, 1025U, 2048U, 4097U}) run(rows, cols, 0.125f);
        const auto before = workspace.allocation_events();
        cases::rejects<std::invalid_argument>([&] { workspace.run({}, 0, 1); });
        cases::rejects<std::invalid_argument>([&] { workspace.run({1}, 1, 2); });
        cases::rejects<std::invalid_argument>([&] { workspace.run({}, std::numeric_limits<std::size_t>::max(), 2); });
        cases::rejects<std::invalid_argument>([&] { workspace.run({NAN}, 1, 1); });
        cases::rejects<std::invalid_argument>([&] { workspace.run({INFINITY}, 1, 1); });
        if (workspace.allocation_events() != before) throw std::runtime_error("invalid input allocated");
        run(7, 37, 0.5f); // workspace remains usable after validation errors
        std::cout << "WORKSPACE_PASS cases=" << checked << " profiled=" << checked
                  << " invalid=5 capacity_reuse=PASS recovery=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n'; return 1;
    }
}
