#include "kernel_lab/softmax_workspace.hpp"
#include "kernel_lab/validation.hpp"
#include "operator_cases.hpp"
#include "../src/cuda/device_ops.cuh"

int main() {
    try {
        using namespace kernel_lab;
        device::check(cudaSetDevice(0), "sync tests device");
        const SoftmaxSyncMode modes[] = {SoftmaxSyncMode::Staged, SoftmaxSyncMode::KernelBoundary,
                                         SoftmaxSyncMode::CopyCompletion};
        for (int m = 0; m < 3; ++m) {
            SoftmaxWorkspace workspace;
            int checked = 0;
            const auto run = [&](std::size_t rows, std::size_t cols, float bias) {
                std::vector<float> input(rows * cols);
                for (std::size_t i = 0; i < input.size(); ++i)
                    input[i] = float(int((i + checked) % 37) - 18) * bias;
                const auto expected = cases::softmax_oracle(input, rows, cols);
                const auto capacity = workspace.capacity(), allocations = workspace.allocation_events();
                const auto actual = workspace.run(input, rows, cols, modes[m]);
                // Intentionally NO caller-side CUDA synchronize before CPU consumption.
                require_close(actual, expected, 2e-6, 2e-5, "sync immediate CPU result");
                require_probability_rows(actual, rows, cols);
                if (workspace.capacity() != std::max(capacity,input.size()) ||
                    workspace.allocation_events() != allocations + (input.size() > capacity ? 1 : 0))
                    throw std::runtime_error("sync capacity reuse broken");
                ++checked;
            };
            run(1,1,1); run(7,37,0.25f); run(1,1,0); run(1,259,1000);
            for (std::size_t rows : {1U,3U,4U,5U,7U})
                for (std::size_t cols : {31U,32U,33U,127U,128U,129U,511U,512U,513U,1023U,1024U,1025U,2048U,4097U})
                    run(rows,cols,0.125f);
            const auto before = workspace.allocation_events();
            cases::rejects<std::invalid_argument>([&] { workspace.run({},0,1,modes[m]); });
            cases::rejects<std::invalid_argument>([&] { workspace.run({1},1,2,modes[m]); });
            cases::rejects<std::invalid_argument>([&] { workspace.run({},std::numeric_limits<std::size_t>::max(),2,modes[m]); });
            cases::rejects<std::invalid_argument>([&] { workspace.run({NAN},1,1,modes[m]); });
            cases::rejects<std::invalid_argument>([&] { workspace.run({INFINITY},1,1,modes[m]); });
            if (workspace.allocation_events() != before) throw std::runtime_error("invalid input allocated");
            run(7,37,0.5f);
            std::cout << "SYNC_MODE_PASS mode=" << m << " cases=" << checked << " invalid=5 immediate_cpu=PASS\n";
        }
        SoftmaxWorkspace workspace;
        cases::rejects<std::invalid_argument>([&] { workspace.run({1},1,1,static_cast<SoftmaxSyncMode>(99)); });
        if (workspace.allocation_events()) throw std::runtime_error("invalid mode allocated");
        // Same object, changing data and policies on every consecutive call.
        for (int i = 0; i < 300; ++i) {
            const std::vector<float> input = {float(i % 11), -float(i % 7), float(i % 3)};
            const auto actual = workspace.run(input,1,3,modes[i%3]);
            require_close(actual,cases::softmax_oracle(input,1,3),2e-6,2e-5,"alternating sync policies");
        }
        if (workspace.allocation_events() != 1) throw std::runtime_error("stress reallocated");
        std::cout << "SYNC_PASS modes=3 cases_per_mode=75 invalid_per_mode=5 invalid_mode=1 alternating_calls=300\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "FAIL " << error.what() << '\n'; return 1; }
}
