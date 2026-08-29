#include "kernel_lab/convolution.hpp"
#include "kernel_lab/softmax.hpp"

#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}  // namespace

int main() {
    const std::vector<float> logits{1000.0F, 1001.0F, 1002.0F, -1.0F, 0.0F, 1.0F};
    const auto probabilities = kernel_lab::softmax_reference(logits, 2, 3);
    require(probabilities.size() == logits.size(), "softmax output size mismatch");
    for (std::size_t row = 0; row < 2; ++row) {
        const float sum = std::accumulate(
            probabilities.begin() + static_cast<std::ptrdiff_t>(row * 3),
            probabilities.begin() + static_cast<std::ptrdiff_t>((row + 1) * 3),
            0.0F);
        require(std::abs(sum - 1.0F) < 1.0e-6F, "softmax row does not sum to one");
    }

    const std::vector<float> signal{1.0F, 2.0F, 3.0F};
    const std::vector<float> kernel{4.0F, 5.0F};
    const std::vector<float> expected{4.0F, 13.0F, 22.0F, 15.0F};
    const auto direct = kernel_lab::convolution_direct(signal, kernel);
    const auto blocked = kernel_lab::convolution_blocked(signal, kernel, 2);
    require(kernel_lab::max_abs_error(direct, expected) == 0.0F, "direct convolution mismatch");
    require(kernel_lab::max_abs_error(blocked, expected) == 0.0F, "blocked convolution mismatch");

    std::cout << "all CPU correctness tests passed\n";
    return 0;
}
