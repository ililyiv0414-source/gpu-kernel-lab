#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include "device_ops.cuh"
#include <climits>

namespace {
at::Tensor softmax_cuda(const at::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "kernel_lab_ext::softmax requires a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "kernel_lab_ext::softmax supports FP32 only");
    TORCH_CHECK(input.dim() == 2, "kernel_lab_ext::softmax requires a 2D [rows, cols] tensor");
    TORCH_CHECK(input.is_contiguous(), "kernel_lab_ext::softmax requires contiguous input");
    TORCH_CHECK(input.size(0) > 0 && input.size(1) > 0, "kernel_lab_ext::softmax rejects empty dimensions");
    TORCH_CHECK(input.size(0) <= INT_MAX && input.size(1) <= INT_MAX,
                "kernel_lab_ext::softmax dimensions exceed supported integer range");
    c10::cuda::CUDAGuard guard(input.device());
    auto output = at::empty_like(input);
    const auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
    kernel_lab::device::launch_softmax_warp_register_on_stream(
        input.data_ptr<float>(), output.data_ptr<float>(),
        static_cast<std::size_t>(input.size(0)),
        static_cast<std::size_t>(input.size(1)), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
} // namespace

TORCH_LIBRARY_IMPL(kernel_lab_ext, CUDA, m) {
    m.impl("softmax", &softmax_cuda);
}
