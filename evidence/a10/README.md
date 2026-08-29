# NVIDIA A10 Evidence Set

This directory contains a compact, sanitized subset of the verified result files used by the project README and resume.

## Native Softmax comparison

- softmax_comparison_summary.csv: 48 summary rows covering naive, block-shuffle, and warp-register implementations.
- softmax_comparison_samples.csv: 1,008 raw timing samples.
- softmax_comparison_status.json: PASS state, GPU execution confirmation, and sanitizer configuration.

The 1024 x 128 resident-data rows contain the published 10.886 us -> 2.888 us result and the corresponding 3.77x speedup against the in-project naive CUDA baseline.

## Reusable workspace

- workspace_process_1_summary.csv and workspace_process_2_summary.csv: independent-process summaries.
- workspace_process_1_samples.csv and workspace_process_2_samples.csv: raw samples for both runs.

The 1024 x 257 rows contain the published 1.48x-1.51x full-call result for workspace reuse versus the one-shot warp wrapper.

## PyTorch integration

- pytorch_run_status.json: PASS state and sample counts.
- pytorch_correctness.json: forward, stream, autograd, FakeTensor, opcheck, compile, and rejection-case results.
- pytorch_summary.csv: 32 framework comparison rows.
- pytorch_samples.csv: 672 raw framework timing samples.
- pytorch_sanitizer_smoke.json: framework sanitizer smoke status.

## Interpretation rules

- Resident CUDA Event timings exclude host-device transfers.
- Full-call workspace timings include validation, transfers, synchronization, and output construction.
- PyTorch comparison rows are reported without claiming universal superiority over torch.softmax.
- These files are evidence for the checked-in source state recorded by the original run artifacts; build directories, cloud instance details, source archives, and unrelated logs are intentionally excluded from the public repository.
