#include "kernel_lab/convolution.hpp"

#include <algorithm>
#include <stdexcept>

namespace kernel_lab {

std::vector<float> convolution_direct(
    const std::vector<float>& signal,
    const std::vector<float>& kernel) {
    if (signal.empty() || kernel.empty()) {
        throw std::invalid_argument("signal and kernel must be non-empty");
    }
    std::vector<float> output(signal.size() + kernel.size() - 1, 0.0F);
    for (std::size_t i = 0; i < signal.size(); ++i) {
        for (std::size_t j = 0; j < kernel.size(); ++j) {
            output[i + j] += signal[i] * kernel[j];
        }
    }
    return output;
}

std::vector<float> convolution_blocked(
    const std::vector<float>& signal,
    const std::vector<float>& kernel,
    const std::size_t block_size) {
    if (signal.empty() || kernel.empty() || block_size == 0) {
        throw std::invalid_argument("blocked convolution arguments are invalid");
    }
    std::vector<float> output(signal.size() + kernel.size() - 1, 0.0F);
    for (std::size_t base = 0; base < signal.size(); base += block_size) {
        const std::size_t block_end = std::min(signal.size(), base + block_size);
        for (std::size_t i = base; i < block_end; ++i) {
            for (std::size_t j = 0; j < kernel.size(); ++j) {
                output[i + j] += signal[i] * kernel[j];
            }
        }
    }
    return output;
}

}  // namespace kernel_lab
