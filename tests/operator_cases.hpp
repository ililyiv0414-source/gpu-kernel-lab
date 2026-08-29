#pragma once
#include "kernel_lab/validation.hpp"
#include <functional>
#include <iostream>
#include <random>
#include <utility>

namespace cases {
using Softmax = std::function<std::vector<float>(const std::vector<float>&, std::size_t, std::size_t)>;
using Convolution = std::function<std::vector<float>(const std::vector<float>&, const std::vector<float>&)>;
inline std::vector<float> softmax_oracle(const std::vector<float>& x, std::size_t rows, std::size_t cols) {
    std::vector<float> y(x.size());
    for (std::size_t r = 0; r < rows; ++r) {
        double maximum = x[r * cols], sum = 0;
        for (std::size_t c = 0; c < cols; ++c) maximum = std::max(maximum, double(x[r * cols + c]));
        for (std::size_t c = 0; c < cols; ++c) sum += std::exp(double(x[r * cols + c]) - maximum);
        for (std::size_t c = 0; c < cols; ++c) y[r * cols + c] = float(std::exp(double(x[r * cols + c]) - maximum) / sum);
    }
    return y;
}
inline std::vector<float> convolution_oracle(const std::vector<float>& x, const std::vector<float>& k) {
    std::vector<float> y(x.size() + k.size() - 1);
    // Output-centric double accumulation, independent of the CPU scatter loop.
    for (std::size_t o = 0; o < y.size(); ++o) {
        double sum = 0;
        for (std::size_t j = 0; j < k.size(); ++j) {
            if (o >= j && o - j < x.size()) sum += double(x[o - j]) * double(k[j]);
        }
        y[o] = float(sum);
    }
    return y;
}
inline int run(const Softmax& softmax, const Convolution& convolution) {
    std::mt19937 generator(20260825U);
    std::uniform_real_distribution<float> random(-2, 2);
    int passed = 0;
    auto soft = [&](const std::vector<float>& x, std::size_t rows, std::size_t cols) {
        const auto actual = softmax(x, rows, cols);
        const auto name = "softmax_" + std::to_string(rows) + "x" + std::to_string(cols);
        kernel_lab::require_close(actual, softmax_oracle(x, rows, cols), 2e-6, 2e-5, name);
        kernel_lab::require_probability_rows(actual, rows, cols);
        ++passed;
        std::cout << "PASS " << name << '\n';
    };
    for (std::size_t rows : {1U, 7U}) {
        for (std::size_t cols : {1U, 3U, 31U, 32U, 255U, 256U, 257U, 511U, 1024U, 4097U}) {
            std::vector<float> x(rows * cols);
            for (auto& v : x) v = random(generator);
            soft(x, rows, cols);
        }
    }
    soft(std::vector<float>(3 * 513, 1000), 3, 513);
    soft({1000, 1001, 1002, -1000, -1001, -1002}, 2, 3);
    soft({10000, -10000, 0, -10000, 10000, 0}, 2, 3);
    soft(std::vector<float>(257, 0), 1, 257);
    auto conv = [&](const std::vector<float>& x, const std::vector<float>& k) {
        const auto name = "convolution_" + std::to_string(x.size()) + "x" + std::to_string(k.size());
        kernel_lab::require_close(convolution(x, k), convolution_oracle(x, k), 2e-4, 2e-4, name);
        ++passed;
        std::cout << "PASS " << name << '\n';
    };
    // Give every pair an explicit element type; nvcc cannot deduce the mixed
    // typed/braced initializer-list form accepted by some host compilers.
    const std::pair<std::size_t, std::size_t> convolution_shapes[] = {
        {1, 1}, {3, 2}, {5, 37}, {255, 15}, {256, 63}, {257, 1}, {1025, 255}, {4096, 63}
    };
    for (const auto& shape : convolution_shapes) {
        std::vector<float> x(shape.first), k(shape.second);
        for (auto& v : x) v = random(generator);
        for (auto& v : k) v = random(generator);
        conv(x, k);
    }
    conv({1, 2, 3}, {4, 5});
    conv(std::vector<float>(31, 0), {1, -2, 3});
    conv({1, -1, 1, -1}, {1, 1, 1});
    conv({1, 0, 0, 0}, {2, -3, 4});
    // Repeated multi-warp runs exercise shared-memory reuse.
    for (int i = 0; i < 8; ++i) {
        std::vector<float> x(17 * 513);
        for (auto& v : x) v = random(generator);
        soft(x, 17, 513);
    }
    return passed;
}
template<class Expected = std::exception, class Function>
inline void rejects(Function&& function) {
    bool rejected = false;
    try { function(); } catch (const Expected&) { rejected = true; }
    if (!rejected) throw std::runtime_error("expected invalid data to be rejected");
}
}  // namespace cases
