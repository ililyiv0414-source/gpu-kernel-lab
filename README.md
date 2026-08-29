# GPU Softmax Kernel Lab

A reproducible C++17/CUDA project for studying GPU reduction kernels, memory management, synchronization, and PyTorch custom operator integration.

The project implements a numerically stable CPU reference, several CUDA Softmax kernels, reusable device workspaces, synchronization variants, correctness tests, raw benchmark collection, Compute Sanitizer checks, and a registered PyTorch C++/CUDA operator.

## Highlights

- Three CUDA implementations: block-shared baseline, block shuffle, and warp-register reduction.
- Explicit separation of resident-kernel timing, end-to-end timing, and framework timing.
- Reusable device workspace managed with C++17 RAII.
- PyTorch dispatcher registration on the caller's current CUDA stream.
- Autograd, FakeTensor, `torch.library.opcheck`, and `torch.compile` validation.
- Reproducible commands, source hashes, raw samples, and correctness metadata.
- Compute Sanitizer `memcheck` and `racecheck` validation.

## Verified A10 Results

All figures below were measured on an NVIDIA A10. They describe specific test scopes and must not be interpreted as whole-application speedups.

| Experiment | Shape | Baseline | Optimized | Result |
| --- | ---: | ---: | ---: | ---: |
| Resident CUDA kernel, batched CUDA Events | 1024 x 128 FP32 | Naive: 10.886 us | Warp-register: 2.888 us | **3.77x** vs. the in-project naive kernel |
| Full call with validation, H2D, kernel, D2H | 1024 x 257 FP32 | One-shot warp wrapper | Reused workspace | **1.48x-1.51x** across two independent processes |

The resident-kernel result uses 5 warm-up rounds, 21 measured rounds, and 100 launches per sample. The end-to-end result includes host validation, transfers, synchronization, and output construction.

The custom PyTorch operator was also validated on the A10, including eight forward shapes, a non-default stream, autograd, FakeTensor, `opcheck`, and `torch.compile`. It does **not** consistently outperform `torch.softmax`; the framework comparison is retained as an engineering baseline rather than a superiority claim.

Selected result files are stored under [`evidence/a10`](evidence/a10). The directory includes summary and raw CSV data, run status, correctness metadata, and sanitizer smoke status.

## Implementation Overview

### Softmax kernels

1. **Block-shared baseline** assigns one row to a CUDA block and performs max/sum reductions through shared memory.
2. **Block shuffle** reduces shared-memory traffic by combining warp shuffle operations with a small block-level exchange.
3. **Warp-register kernel** keeps short-row values in registers and performs the reduction through warp communication.

All implementations apply the stable Softmax transformation:

```text
softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
```

### Workspace and synchronization

`SoftmaxWorkspace` reuses device input/output buffers and grows capacity only when needed. It is intentionally scoped to one CUDA device and one host thread.

Three synchronization modes are available for controlled experiments:

- `Staged`: synchronize after H2D and after the kernel.
- `KernelBoundary`: synchronize after the kernel.
- `CopyCompletion`: rely on ordered operations in the legacy default stream and the final blocking D2H copy.

The modes are experimental interfaces, not universal recommendations. Their behavior must not be generalized to arbitrary streams, devices, or concurrent callers.

### PyTorch custom operator

The extension registers `kernel_lab_ext::softmax` through the PyTorch dispatcher. It accepts finite, contiguous, two-dimensional FP32 CUDA tensors. The CUDA implementation runs on the caller's current stream.

Backward propagation is expressed with PyTorch tensor operations; this project does not claim a custom CUDA backward kernel.

## Repository Layout

```text
.
|-- include/kernel_lab/       Public C++/CUDA interfaces
|-- src/cpu/                  CPU references and benchmark
|-- src/cuda/                 CUDA kernels and native benchmarks
|-- framework/pytorch/        PyTorch custom operator and benchmark
|-- tests/                    CPU, CUDA, result-parser, and integration tests
|-- scripts/                  Reproducible build and benchmark runners
|-- experiments/              Experiment record template
|-- evidence/a10/             Selected verified A10 result files
|-- CMakeLists.txt
`-- CMakePresets.json
```

## Requirements

### CPU-only path

- CMake 3.24+
- C++17 compiler
- Python 3.9+ for runner tests

### CUDA path

- NVIDIA GPU and compatible driver
- CUDA Toolkit with `nvcc`
- Host compiler supported by the installed CUDA Toolkit
- CMake 3.24+
- Python 3.9+
- Optional: Compute Sanitizer

### PyTorch path

- PyTorch with CUDA support
- Python development environment capable of JIT-building a C++/CUDA extension

The verified framework run used PyTorch `2.10.0+cu128` on an NVIDIA A10.

## Build and Test

### CPU-only build

```bash
cmake -S . -B build/cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu --config Release
ctest --test-dir build/cpu -C Release --output-on-failure
```

On Windows with Visual Studio Build Tools, the convenience runner is:

```powershell
& .\scripts\run-cpu.ps1
```

### CUDA build and verification

```bash
python3 scripts/run_gpu.py --check
python3 scripts/run_gpu.py --sanitizer both
```

Additional experiments:

```bash
python3 scripts/run_gpu.py --sanitizer both --softmax-compare
python3 scripts/run_gpu.py --sanitizer both --workspace-compare
python3 scripts/run_gpu.py --sanitizer both --sync-compare
python3 scripts/run_gpu.py --sanitizer both --pytorch-compare
```

Each run creates a timestamped result directory with commands, environment metadata, correctness logs, source hashes, summary CSV files, and raw samples. Missing GPU tools or failed checks produce a non-zero exit status instead of silently falling back to CPU execution.

See [`GPU_RUNBOOK.md`](GPU_RUNBOOK.md) for measurement definitions and detailed validation rules.

## Correctness Contract

- Inputs must be finite FP32 values.
- CUDA Softmax outputs are compared against an independent double-accumulation reference.
- Element tolerance: `abs(actual - reference) <= 2e-6 + 2e-5 * abs(reference)`.
- Row sums must be within `2e-5` of one.
- NaN/Inf outputs, length mismatches, invalid ranks, unsupported dtypes, and non-contiguous PyTorch tensors are rejected.

## Measurement Rules

- `kernel_event` or `batched_event_per_launch` measures device-resident CUDA work only.
- `end_to_end_wall` includes validation, allocation or workspace handling, H2D, kernel execution, synchronization, D2H, and output construction.
- `resident_cuda_event_per_call` measures framework calls on resident CUDA tensors.
- `cpu_to_cuda_to_cpu_wall` includes tensor transfers and framework invocation.

Results from different scopes are never combined into one speedup claim. Profiler runs are kept separate from benchmark sampling.

## Current Limitations

- The optimized kernel targets row-wise Softmax for finite, contiguous FP32 data.
- The custom operator is not faster than `torch.softmax` for every tested shape.
- No custom CUDA backward kernel is implemented.
- Nsight Systems and Nsight Compute analysis is not included in the verified evidence set.
- Triton, ROCm/HIP, cuFFT, and production packaging are outside the current scope.

## Reproducing the Published Numbers

1. Build on a compatible NVIDIA CUDA machine.
2. Run the Softmax comparison with Compute Sanitizer enabled.
3. Run the workspace comparison in at least two independent processes.
4. Preserve generated raw samples and `status.json` files.
5. Compare the generated summaries with the files under `evidence/a10`.

Random seed, warm-up count, sample count, launch count, tolerance, and measurement scope are recorded in every summary row.
