#!/usr/bin/env python3
"""GPU run orchestration; Python 3.9+ standard library only. No installs/payments."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import zipfile

# Keep a Windows ASCII junction path instead of resolving back to Unicode.
ROOT = Path(os.path.abspath(__file__)).parents[1]
SEED = "20260825"
SHAPES = [("softmax", "1024", k) for k in ("128", "512", "1024")]
SHAPES += [("convolution", "32768", k) for k in ("15", "63", "255")]
SCOPES = ("cpu_wall", "kernel_event", "end_to_end_wall")


def expected_keys():
    return {(op, n, k, scope) for op, n, k in SHAPES for scope in SCOPES}


def validate_csv(directory, repeats, warmup):
    """Fail closed: verify shape/scope coverage, all raw repeats and statistics."""
    directory = Path(directory)
    with (directory / "gpu_summary.csv").open(newline="", encoding="utf-8") as f:
        summary = list(csv.DictReader(f))
    with (directory / "gpu_samples.csv").open(newline="", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))
    if len(summary) != 18 or len(raw) != 18 * repeats:
        raise ValueError("incomplete benchmark row count")
    by_key, samples = {}, {}
    for row in summary:
        key = tuple(row[x] for x in ("operator", "n", "cols_or_kernel", "scope"))
        if key not in expected_keys() or key in by_key:
            raise ValueError("unexpected or duplicate summary key")
        impl = ("cpu_reference" if key[0] == "softmax" else "cpu_direct") if key[3] == "cpu_wall" else "cuda_naive"
        atol, rtol = (2e-6, 2e-5) if key[0] == "softmax" else (2e-4, 2e-4)
        if (row["correctness"] != "PASS" or row["seed"] != SEED or
                row["implementation"] != impl or int(row["repeats"]) != repeats or
                int(row["warmup"]) != warmup or float(row["atol"]) != atol or float(row["rtol"]) != rtol):
            raise ValueError("benchmark metadata mismatch")
        for field in ("median_ms", "min_ms", "max_ms", "output_elements_per_second", "speedup_vs_cpu", "max_abs_error"):
            value = float(row[field])
            if not math.isfinite(value) or (value < 0 if field == "max_abs_error" else value <= 0):
                raise ValueError("invalid numeric field: " + field)
        by_key[key] = row
        samples[key] = {}
    if set(by_key) != expected_keys():
        raise ValueError("missing cases")
    for row in raw:
        key = tuple(row[x] for x in ("operator", "n", "cols_or_kernel", "scope"))
        if key not in samples:
            raise ValueError("unexpected raw key")
        index, value = int(row["sample"]), float(row["elapsed_ms"])
        if (index in samples[key] or not 0 <= index < repeats or not math.isfinite(value) or value <= 0 or
                row["seed"] != SEED or row["implementation"] != by_key[key]["implementation"]):
            raise ValueError("invalid raw sample")
        samples[key][index] = value
    for key, row in by_key.items():
        values = list(samples[key].values())
        if len(values) != repeats:
            raise ValueError("missing repeats")
        for field, expected in (("min_ms", min(values)), ("max_ms", max(values)), ("median_ms", statistics.median(values))):
            if not math.isclose(float(row[field]), expected, rel_tol=1e-8, abs_tol=1e-10):
                raise ValueError("summary disagrees with raw samples: " + field)
        n, k = int(key[1]), int(key[2])
        elements = n * k if key[0] == "softmax" else n + k - 1
        ms = float(row["median_ms"])
        cpu_ms = float(by_key[(*key[:3], "cpu_wall")]["median_ms"])
        if not math.isclose(float(row["speedup_vs_cpu"]), cpu_ms / ms, rel_tol=1e-8):
            raise ValueError("speedup mismatch")
        if not math.isclose(float(row["output_elements_per_second"]), elements * 1000 / ms, rel_tol=1e-8):
            raise ValueError("throughput mismatch")
    return {"summary_rows": len(summary), "raw_rows": len(raw)}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="tool/driver preflight only; no build or GPU correctness claim")
    p.add_argument("--architectures", default="native", help="native or e.g. 80;86 (quote semicolon lists)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=11)
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--timeout", type=int, default=1800, help="per-command seconds")
    p.add_argument("--profile", choices=("none", "nsys", "ncu", "both"), default="none")
    p.add_argument("--sanitizer", choices=("none", "memcheck", "racecheck", "both"), default="none")
    p.add_argument("--softmax-compare", action="store_true", help="run the baseline/shuffle/register comparison")
    p.add_argument("--compare-repeats", type=int, default=21)
    p.add_argument("--compare-launches", type=int, default=100)
    p.add_argument("--workspace-compare", action="store_true", help="run two independent-process workspace benchmarks")
    p.add_argument("--sync-compare", action="store_true", help="run three independent-process synchronization ablations")
    p.add_argument("--pytorch-compare", action="store_true",
                   help="build a registered PyTorch CUDA op and compare it with torch.softmax")
    return p


def required_tools(args):
    names = ["cmake", "ctest", "nvcc", "nvidia-smi"]
    names += ["ninja"] if os.name == "nt" or shutil.which("ninja") else ["make"]
    if args.profile in ("nsys", "both"):
        names.append("nsys")
    if args.profile in ("ncu", "both"):
        names.append("ncu")
    if args.sanitizer != "none":
        names.append("compute-sanitizer")
    return names


def preflight(args):
    found = {name: shutil.which(name) for name in required_tools(args)}
    missing = [name for name, path in found.items() if not path]
    if missing:
        raise RuntimeError("missing tools: " + ", ".join(missing) + "; use a configured NVIDIA GPU machine")
    return found


def logged_run(command, directory, label, cwd, timeout):
    command = [str(value) for value in command]
    with (directory / "commands.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"stage": label, "argv": command, "cwd": str(cwd)}, ensure_ascii=False) + "\n")
    print("RUN " + label, flush=True)
    with (directory / (label + ".log")).open("w", encoding="utf-8") as log:
        # No shell interpolation. Preserve tool failures and timeouts as failures.
        result = subprocess.run(command, cwd=cwd, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"{label} failed ({result.returncode}); see {directory / (label + '.log')}")


def snapshot(project, directory):
    paths = []
    for folder in ("src", "include", "tests", "scripts", "framework"):
        paths.extend(p for p in (project / folder).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    paths += [project / name for name in ("CMakeLists.txt", "CMakePresets.json", ".gitignore",
                                         "README.md", "GPU_RUNBOOK.md", "LOCAL_VALIDATION.md")]
    hashes = {}
    with zipfile.ZipFile(directory / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            relative = path.relative_to(project).as_posix()
            data = path.read_bytes()
            hashes[relative] = hashlib.sha256(data).hexdigest()
            archive.writestr(relative, data)
    (directory / "source-sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")


def execute(args, project=ROOT):
    # Use an ASCII server workspace. Do not change global locale/toolchain.
    if any(ord(c) > 127 for c in str(project)):
        raise RuntimeError("use an ASCII project path on the GPU machine (Windows: E:/KernelLabDev/gpu-kernel-lab)")
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    directory = project / "results" / ("gpu-" + run_id)
    directory.mkdir(parents=True, exist_ok=False)
    # Unique build tree prevents reusing stale CPU-only / other-toolchain caches.
    build = project / "build" / ("gpu-" + run_id)
    status = {"status": "RUNNING", "benchmark_valid": False, "profile": args.profile,
              "sanitizer": args.sanitizer, "gpu_execution_verified": False}
    if args.softmax_compare:
        status["softmax_comparison_valid"] = False
    if args.workspace_compare:
        status["workspace_comparison_valid"] = False
    if args.sync_compare:
        status["sync_comparison_valid"] = False
    if args.pytorch_compare:
        status["pytorch_comparison_valid"] = False
    def run(command, label):
        logged_run(command, directory, label, project, args.timeout)
    try:
        snapshot(project, directory)
        tools = preflight(args)
        metadata = {"utc": run_id, "platform": platform.platform(), "python": sys.version,
                    "tools": tools, "arguments": vars(args),
                    "environment": {k: os.environ.get(k, "") for k in
                                    ("CUDA_VISIBLE_DEVICES", "CXX", "CC", "CUDAHOSTCXX", "CUDA_HOME")}}
        (directory / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        for tool in ("cmake", "ctest", "nvcc"):
            run([tools[tool], "--version"], tool + "-version")
        cmake_text = (directory / "cmake-version.log").read_text(encoding="utf-8", errors="replace")
        version = re.search(r"cmake version (\d+)\.(\d+)", cmake_text)
        if not version or tuple(map(int, version.groups())) < (3, 24):
            raise RuntimeError("CMake >= 3.24 is required")
        run([tools["nvidia-smi"]], "nvidia-smi-before")
        run([tools["nvidia-smi"], "--query-gpu=index,name,uuid,driver_version,memory.total", "--format=csv"], "gpu-inventory")
        for tool in ("nsys", "ncu", "compute-sanitizer"):
            if tool in tools:
                run([tools[tool], "--version"], tool + "-version")
        generator = "Ninja" if "ninja" in tools else "Unix Makefiles"
        command = [tools["cmake"], "-S", project, "-B", build, "-G", generator,
                   "-DENABLE_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release",
                   "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                   "-DCMAKE_CUDA_COMPILER=" + tools["nvcc"],
                   "-DCMAKE_CUDA_ARCHITECTURES=" + args.architectures]
        if os.environ.get("CXX"):
            command += ["-DCMAKE_CXX_COMPILER=" + os.environ["CXX"]]
        if os.environ.get("CUDAHOSTCXX") or os.environ.get("CXX"):
            command += ["-DCMAKE_CUDA_HOST_COMPILER=" + (os.environ.get("CUDAHOSTCXX") or os.environ["CXX"])]
        run(command, "configure")
        run([tools["cmake"], "--build", build, "--parallel", args.jobs], "build")
        suffix = ".exe" if os.name == "nt" else ""
        benchmark = build / ("cuda_benchmark" + suffix)
        tests = build / ("kernel_lab_cuda_tests" + suffix)
        run([benchmark, "--device-info"], "runtime-device")
        run([tools["ctest"], "--test-dir", build, "--verbose", "--output-on-failure", "--no-tests=error"], "ctest")
        status["gpu_execution_verified"] = True
        if args.sanitizer != "none":
            modes = ("memcheck", "racecheck") if args.sanitizer == "both" else (args.sanitizer,)
            for mode in modes:
                run([tools["compute-sanitizer"], "--tool", mode, "--error-exitcode", "9", tests], "sanitizer-" + mode)
                if args.softmax_compare or args.workspace_compare or args.sync_compare:
                    run([tools["compute-sanitizer"], "--tool", mode, "--error-exitcode", "9",
                         build / ("softmax_optimized_tests" + suffix)], "sanitizer-optimized-" + mode)
                if args.workspace_compare or args.sync_compare:
                    run([tools["compute-sanitizer"], "--tool", mode, "--error-exitcode", "9",
                         build / ("softmax_workspace_tests" + suffix)], "sanitizer-workspace-" + mode)
                if args.sync_compare:
                    run([tools["compute-sanitizer"], "--tool", mode, "--error-exitcode", "9",
                         build / ("softmax_sync_tests" + suffix)], "sanitizer-sync-" + mode)
        run([benchmark, "--output-dir", directory, "--warmup", args.warmup, "--repeats", args.repeats], "benchmark")
        status.update(validate_csv(directory, args.repeats, args.warmup))
        status["benchmark_valid"] = True
        if args.softmax_compare:
            from softmax_compare_results import validate as validate_comparison
            run([build / ("softmax_compare" + suffix), "--output-dir", directory,
                 "--warmup", "5", "--repeats", args.compare_repeats,
                 "--launches", args.compare_launches], "softmax-compare")
            status.update(validate_comparison(directory, repeats=args.compare_repeats,
                                             warmup=5, launches=args.compare_launches))
            status["softmax_comparison_valid"] = True
        if args.workspace_compare:
            from workspace_results import validate as validate_workspace
            verified_runs = []
            for index in (1, 2):
                trial = directory / ("workspace-repeat-" + str(index))
                trial.mkdir(exist_ok=False)
                run([build / ("softmax_workspace_benchmark" + suffix), "--output-dir", trial],
                    "workspace-repeat-" + str(index))
                verified_runs.append(validate_workspace(trial))
            status["workspace_runs"] = verified_runs
            status["workspace_comparison_valid"] = True
        if args.sync_compare:
            from sync_results import validate as validate_sync
            verified_runs = []
            for index in (1, 2, 3):
                trial = directory / ("sync-repeat-" + str(index))
                trial.mkdir(exist_ok=False)
                run([build / ("softmax_sync_benchmark" + suffix), "--output-dir", trial],
                    "sync-repeat-" + str(index))
                verified_runs.append(validate_sync(trial))
            status["sync_runs"] = verified_runs
            status["sync_comparison_valid"] = True
        if args.pytorch_compare:
            from pytorch_results import validate as validate_pytorch
            framework_output = directory / "pytorch"
            framework_output.mkdir(exist_ok=False)
            framework_build = build / "pytorch-extension"
            framework_script = project / "framework" / "pytorch" / "benchmark.py"
            framework_command = [sys.executable, framework_script,
                                 "--project", project, "--build-dir", framework_build,
                                 "--output-dir", framework_output,
                                 "--warmup", "10", "--repeats", "21",
                                 "--resident-launches", "100"]
            run(framework_command, "pytorch-framework")
            status.update(validate_pytorch(framework_output, repeats=21, warmup=10,
                                           resident_launches=100))
            if args.sanitizer != "none":
                modes = ("memcheck", "racecheck") if args.sanitizer == "both" else (args.sanitizer,)
                for mode in modes:
                    run([tools["compute-sanitizer"], "--tool", mode, "--error-exitcode", "9",
                         sys.executable, framework_script,
                         "--project", project, "--build-dir", framework_build,
                         "--output-dir", framework_output, "--sanitizer-smoke"],
                        "sanitizer-pytorch-" + mode)
            status["pytorch_comparison_valid"] = True
        for op in ("softmax", "convolution"):
            if args.profile in ("nsys", "both"):
                run([tools["nsys"], "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
                     "--output", directory / ("nsys-" + op), benchmark, "--profile-case", op], "nsys-" + op)
                if not (directory / ("nsys-" + op + ".nsys-rep")).is_file():
                    raise RuntimeError("Nsight Systems report missing")
            if args.profile in ("ncu", "both"):
                run([tools["ncu"], "--set", "full", "--launch-skip", "3", "--launch-count", "1",
                     "--export", directory / ("ncu-" + op), benchmark, "--profile-case", op], "ncu-" + op)
                if not (directory / ("ncu-" + op + ".ncu-rep")).is_file():
                    raise RuntimeError("Nsight Compute report missing")
        run([tools["nvidia-smi"]], "nvidia-smi-after")
        shutil.copy2(build / "CMakeCache.txt", directory / "CMakeCache.txt")
        if (build / "compile_commands.json").is_file():
            shutil.copy2(build / "compile_commands.json", directory / "compile_commands.json")
        status["status"] = "PASS"
        (directory / "SUCCESS.txt").write_text(
            "CUDA build, tests, CSV validation and explicitly requested optional steps passed.\n"
            "Profiler=none does not prove profiling; sanitizer=none does not prove sanitizer validation.\n",
            encoding="utf-8")
        print("PASS evidence: " + str(directory))
        return 0
    except Exception as error:
        status["status"] = "FAIL"
        status["error"] = str(error)
        (directory / "FAILURE.txt").write_text(str(error) + "\n", encoding="utf-8")
        print("FAIL: " + str(error), file=sys.stderr)
        print("Evidence: " + str(directory), file=sys.stderr)
        return 1
    finally:
        (directory / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


def main(argv=None):
    args = parser().parse_args(argv)
    if (not 0 <= args.warmup <= 10000 or not 1 <= args.repeats <= 10000 or args.jobs < 1 or args.timeout < 1
            or not 1 <= args.compare_repeats <= 1000 or not 1 <= args.compare_launches <= 1000):
        raise SystemExit("invalid warmup/repeats/jobs/timeout")
    if not re.fullmatch(r"native|[0-9]+(?:-(?:real|virtual))?(?:;[0-9]+(?:-(?:real|virtual))?)*", args.architectures):
        raise SystemExit("invalid architecture list")
    try:
        if args.check:
            tools = preflight(args)
            for name in ("cmake", "nvcc", "nvidia-smi"):
                command = [tools[name]] + ([] if name == "nvidia-smi" else ["--version"])
                subprocess.run(command, check=True, timeout=min(args.timeout, 60))
            print("PREFLIGHT_ONLY: tools/driver responding; CUDA build and execution not yet verified.")
            return 0
        return execute(args)
    except Exception as error:
        print("FAIL: " + str(error), file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
