# Validation Record

## Local CPU Validation

The CPU-only path was built and tested with MSVC x64 in both Release and Debug configurations.

Verified items:

- Numerically stable CPU Softmax reference.
- Direct and blocked one-dimensional convolution implementations.
- Independent reference checks for valid operator inputs.
- Validation checks for invalid dimensions, non-finite values, and overflow-prone sizes.
- Python unit tests for GPU runner configuration and generated-result parsers.
- Release benchmark CSV generation.

Debug results are used for diagnostics only and are not reported as performance evidence.

## NVIDIA A10 Validation

The CUDA project was compiled and executed on an NVIDIA A10 on 2026-08-29.

Verified stages:

- Native CUDA build and CTest suite.
- Naive, block-shuffle, and warp-register Softmax comparison.
- Reusable workspace comparison in two independent processes.
- Synchronization strategy comparison in three independent processes.
- Compute Sanitizer `memcheck` and `racecheck` for the relevant native kernels.
- Registered PyTorch C++/CUDA custom operator.
- Current-stream execution and a non-default-stream test.
- Autograd, FakeTensor, `torch.library.opcheck`, and `torch.compile` checks.
- PyTorch profiler table and Chrome trace generation.
- Framework sanitizer smoke checks.

The final framework run recorded:

- 18 native summary rows and 198 native raw samples.
- 32 PyTorch summary rows and 672 PyTorch raw samples.
- PASS status for forward shapes, stream behavior, autograd, `opcheck`, compilation, and input rejection cases.

## Evidence Boundaries

- The `3.77x` number applies to `1024 x 128` FP32 data already resident on the GPU and compares the warp-register kernel with the in-project naive CUDA kernel.
- The `1.48x-1.51x` number applies to the full `1024 x 257` call and compares workspace reuse with the one-shot warp wrapper across two processes.
- Neither number represents end-to-end model inference speedup.
- The custom operator is not claimed to outperform `torch.softmax` across all shapes.
- Nsight Systems and Nsight Compute analysis has not been added to the published evidence.

Selected raw and summary evidence is available in `evidence/a10`.
