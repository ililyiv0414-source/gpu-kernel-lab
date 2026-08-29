# GPU Validation and Benchmark Runbook

This document defines the execution order, correctness gates, timing scopes, and evidence requirements for the CUDA and PyTorch paths.

## 1. Prerequisites

- NVIDIA CUDA GPU allocated to the current user.
- Compatible NVIDIA driver and CUDA Toolkit.
- CMake 3.24+, Python 3.9+, and a CUDA-compatible host compiler.
- Ninja or Make on Linux; a supported MSVC toolchain on Windows.
- Compute Sanitizer for memory and shared-memory race checks.

The runner never installs software, changes driver settings, requests extra GPUs, or modifies scheduler allocations.

## 2. Recommended Execution Order

Run from the repository root:

```bash
# Inspect tools and the visible device without building.
python3 scripts/run_gpu.py --check

# Native CUDA build, correctness tests, and baseline benchmark.
python3 scripts/run_gpu.py --sanitizer both

# Compare the native Softmax implementations.
python3 scripts/run_gpu.py --sanitizer both --softmax-compare

# Measure reusable device buffers in independent processes.
python3 scripts/run_gpu.py --sanitizer both --workspace-compare

# Compare synchronization strategies.
python3 scripts/run_gpu.py --sanitizer both --sync-compare

# Build and validate the PyTorch custom operator.
python3 scripts/run_gpu.py --sanitizer both --pytorch-compare
```

Use `python` instead of `python3` when appropriate. The default architecture is the currently visible GPU; use `--architectures` only when a known deployment target requires it.

## 3. Pass/Fail Rules

Every run writes a timestamped directory under `results/`.

| File | Purpose |
| --- | --- |
| `status.json` | Final PASS/FAIL state and completed validation stages |
| `SUCCESS.txt` or `FAILURE.txt` | Human-readable terminal status |
| `configure.log`, `build.log`, `ctest.log` | Build and test output |
| `gpu_summary.csv`, `gpu_samples.csv` | Native CUDA summary and raw samples |
| `commands.jsonl` | Executed commands |
| `environment.json`, `runtime-device.log` | Toolchain and runtime device metadata |
| `source-sha256.json` | Source snapshot integrity data |
| `sanitizer-*.log` | Requested Compute Sanitizer results |
| `pytorch/*` | Framework correctness, benchmark, and profiler outputs |

A run is publishable only when:

- `status.json` reports `status=PASS`.
- GPU execution was verified.
- Correctness tests passed.
- Requested benchmark rows and raw samples are complete.
- Requested sanitizer checks completed successfully.

Missing tools, incomplete CSV data, non-finite timings, test failures, or sanitizer errors produce a non-zero exit code.

## 4. Correctness Coverage

Native GPU tests include regular, boundary, non-power-of-two, and large-row shapes. An independent double-accumulation reference is used for comparison.

- Softmax tolerance: `2e-6 + 2e-5 * abs(reference)`.
- Softmax row-sum tolerance: `2e-5`.
- Convolution tolerance: `2e-4 + 2e-4 * abs(reference)`.
- Non-finite inputs are rejected.
- Non-finite outputs, size mismatches, and out-of-tolerance elements fail the run.

The PyTorch integration additionally checks:

- Eight forward shapes.
- A non-default CUDA stream.
- Autograd registration.
- FakeTensor behavior.
- `torch.library.opcheck`.
- `torch.compile` with a full eager graph.
- Rejection of CPU tensors, unsupported dtype/rank, non-contiguous tensors, and empty dimensions.

## 5. Timing Scopes

| Scope | Included | Excluded |
| --- | --- | --- |
| `kernel_event` | Device execution between CUDA Events | Allocation, transfers, host validation |
| `batched_event_per_launch` | Batched resident launches divided by launch count | Allocation, transfers, result comparison |
| `end_to_end_wall` | Validation, allocation/workspace, H2D, kernel, synchronization, D2H, output construction | Random input generation and result comparison |
| `resident_cuda_event_per_call` | PyTorch call on resident CUDA tensors | Host-device transfers |
| `cpu_to_cuda_to_cpu_wall` | Host tensor creation/transfer, framework call, and return transfer | Input generation |

Each benchmark records warm-up count, measured rounds, launches per sample, random seed, min/median/max, tolerance, and correctness status.

Do not compare values from different timing scopes. In particular, a resident-kernel speedup is not a whole-application speedup.

## 6. Workspace Experiment

`SoftmaxWorkspace` retains input and output device buffers and grows capacity when required. The comparison includes:

- `warp_wrapper`: the one-shot public wrapper.
- `workspace_fresh`: construct and destroy a workspace for each call.
- `workspace_reuse`: keep the allocated capacity after warm-up.

Run at least two independent processes. The published `1.48x-1.51x` result at `1024 x 257` compares `workspace_reuse` with `warp_wrapper` for the full call scope.

## 7. Synchronization Experiment

The synchronization variants share the same numerical contract and workspace:

- `Staged` waits after H2D and after the kernel.
- `KernelBoundary` waits after the kernel.
- `CopyCompletion` uses ordered legacy-default-stream operations and the final blocking D2H copy.

`CopyCompletion` is still a synchronous host API. Its assumptions do not apply automatically to non-default streams, multiple devices, or concurrent calls.

## 8. PyTorch Comparison

The custom operator uses the caller's current CUDA stream. Benchmark rows alternate between `torch.softmax` and the custom operator to reduce order bias.

The comparison may pass even when the custom operator is slower. Correctness and performance are separate validation dimensions, and no test changes PASS/FAIL to force an optimization claim.

## 9. Compute Sanitizer

`memcheck` checks invalid device memory access. `racecheck` checks shared-memory hazards. These tools do not prove the absence of every CUDA defect; `initcheck` and `synccheck` are outside the current verified run.

Profiler collection must be executed separately from benchmark sampling because instrumentation changes timing behavior. Nsight reports are not part of the currently published evidence.

## 10. Evidence Publication

Only sanitized, relevant evidence should be committed. Do not publish build trees, absolute local paths, cloud instance identifiers, source archives, credentials, or unrelated logs.

The checked-in `evidence/a10` directory contains the files required to verify the headline results without exposing environment-specific build artifacts.
