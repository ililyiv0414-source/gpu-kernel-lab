"""Load/register the CUDA operator plus FakeTensor and autograd formulas."""
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

_REGISTERED = False

def load_operator(project, build_directory, verbose=False):
    global _REGISTERED
    project, build_directory = Path(project), Path(build_directory)
    build_directory.mkdir(parents=True, exist_ok=True)
    load(name='kernel_lab_pytorch_ext',
         sources=[str(project/'framework/pytorch/softmax_op.cpp'),
                  str(project/'framework/pytorch/softmax_op_cuda.cu'),
                  str(project/'src/cuda/softmax_optimized.cu')],
         extra_include_paths=[str(project/'include'), str(project/'src/cuda')],
         extra_cflags=['-O3', '-std=c++17'],
         extra_cuda_cflags=['-O3', '-std=c++17', '-lineinfo'],
         build_directory=str(build_directory), with_cuda=True,
         is_python_module=False, verbose=verbose, keep_intermediates=True)
    if not _REGISTERED:
        @torch.library.register_fake('kernel_lab_ext::softmax')
        def _fake(input):
            torch._check(input.dim() == 2)
            torch._check(input.dtype == torch.float32)
            torch._check(input.is_contiguous())
            return torch.empty_like(input)

        def _setup_context(ctx, inputs, output):
            ctx.save_for_backward(output)

        def _backward(ctx, grad_output):
            (output,) = ctx.saved_tensors
            return output * (grad_output - (grad_output * output).sum(dim=-1, keepdim=True))

        torch.library.register_autograd('kernel_lab_ext::softmax', _backward,
                                        setup_context=_setup_context)
        _REGISTERED = True
    return torch.ops.kernel_lab_ext.softmax.default
