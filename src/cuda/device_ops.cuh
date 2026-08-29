#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace kernel_lab::device {
inline void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
class Buffer {
public:
    explicit Buffer(std::size_t count) : count_(count) {
        if (!count || count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::invalid_argument("invalid allocation size");
        }
        check(cudaMalloc(reinterpret_cast<void**>(&data_), bytes()), "cudaMalloc");
    }
    ~Buffer() { if (data_) cudaFree(data_); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    float* data() const { return data_; }
    std::size_t bytes() const { return count_ * sizeof(float); }
    void upload(const std::vector<float>& input) {
        if (input.size() != count_) throw std::invalid_argument("upload size mismatch");
        check(cudaMemcpy(data_, input.data(), bytes(), cudaMemcpyHostToDevice), "H2D");
    }
    std::vector<float> download() const {
        std::vector<float> output(count_);
        check(cudaMemcpy(output.data(), data_, bytes(), cudaMemcpyDeviceToHost), "D2H");
        return output;
    }
private:
    float* data_ = nullptr;
    std::size_t count_;
};
class Event {
public:
    Event() { check(cudaEventCreate(&event_), "cudaEventCreate"); }
    ~Event() { if (event_) cudaEventDestroy(event_); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    cudaEvent_t get() const { return event_; }
private:
    cudaEvent_t event_ = nullptr;
};
// Internal device-pointer entry points: validated dimensions, caller-owned
// buffers, caller synchronization. Same kernels as the end-to-end wrappers.
void launch_softmax(const float* input, float* output, std::size_t rows, std::size_t cols);
void launch_softmax_block_shuffle(const float* input, float* output, std::size_t rows, std::size_t cols);
void launch_softmax_warp_register(const float* input, float* output, std::size_t rows, std::size_t cols);
void launch_softmax_block_shuffle_on_stream(const float* input, float* output,
                                            std::size_t rows, std::size_t cols,
                                            cudaStream_t stream);
void launch_softmax_warp_register_on_stream(const float* input, float* output,
                                            std::size_t rows, std::size_t cols,
                                            cudaStream_t stream);
void launch_convolution(const float* signal, const float* kernel, float* output,
                        std::size_t n, std::size_t k);
}  // namespace kernel_lab::device
