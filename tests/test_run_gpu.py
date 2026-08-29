"""Synthetic fixtures only: these tests never claim or simulate a GPU measurement."""
import contextlib
import csv
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("run_gpu", Path(__file__).parents[1] / "scripts" / "run_gpu.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kernel-lab-test-")
        self.directory = Path(self.temp.name)
        self.summary, self.raw = [], []
        for op, n, k in runner.SHAPES:
            for scope in runner.SCOPES:
                impl = ("cpu_reference" if op == "softmax" else "cpu_direct") if scope == "cpu_wall" else "cuda_naive"
                elements = int(n) * int(k) if op == "softmax" else int(n) + int(k) - 1
                self.summary.append(dict(
                    operator=op, n=n, cols_or_kernel=k, implementation=impl, scope=scope,
                    median_ms=2, min_ms=1, max_ms=3, output_elements_per_second=elements * 500,
                    speedup_vs_cpu=1, max_abs_error=0, atol=2e-6 if op == "softmax" else 2e-4,
                    rtol=2e-5 if op == "softmax" else 2e-4, warmup=1, repeats=3, seed=runner.SEED, correctness="PASS"))
                for index, value in enumerate((1, 2, 3)):
                    self.raw.append(dict(operator=op, n=n, cols_or_kernel=k, implementation=impl,
                                         scope=scope, sample=index, elapsed_ms=value, seed=runner.SEED))
        self.summary_fields = list(self.summary[0])
        self.raw_fields = list(self.raw[0])

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        for name, rows, fields in (("gpu_summary.csv", self.summary, self.summary_fields),
                                   ("gpu_samples.csv", self.raw, self.raw_fields)):
            with (self.directory / name).open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

    def assert_rejected(self):
        self.save()
        with self.assertRaises((ValueError, KeyError)):
            runner.validate_csv(self.directory, 3, 1)

    def test_valid_synthetic_fixture(self):
        self.save()
        self.assertEqual(runner.validate_csv(self.directory, 3, 1), {"summary_rows": 18, "raw_rows": 54})

    def test_missing_summary(self):
        self.summary.pop()
        self.assert_rejected()

    def test_duplicate_case(self):
        self.summary[-1] = dict(self.summary[0])
        self.assert_rejected()

    def test_missing_sample(self):
        self.raw.pop()
        self.assert_rejected()

    def test_duplicate_sample(self):
        self.raw[1] = dict(self.raw[0])
        self.assert_rejected()

    def test_raw_nan(self):
        self.raw[0]["elapsed_ms"] = math.nan
        self.assert_rejected()

    def test_summary_inf(self):
        self.summary[0]["median_ms"] = math.inf
        self.assert_rejected()

    def test_zero_timing(self):
        self.raw[0]["elapsed_ms"] = 0
        self.assert_rejected()

    def test_incorrect_median(self):
        self.summary[0]["median_ms"] = 2.1
        self.assert_rejected()

    def test_wrong_speedup(self):
        self.summary[0]["speedup_vs_cpu"] = 100
        self.assert_rejected()

    def test_wrong_throughput(self):
        self.summary[0]["output_elements_per_second"] = 1
        self.assert_rejected()

    def test_not_pass(self):
        self.summary[0]["correctness"] = "FAIL"
        self.assert_rejected()

    def test_changed_tolerance(self):
        self.summary[0]["atol"] = 1
        self.assert_rejected()

    def test_wrong_seed(self):
        self.raw[0]["seed"] = "0"
        self.assert_rejected()

    def test_missing_tools_fail_closed(self):
        with mock.patch.object(runner.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "missing tools"):
                runner.preflight(runner.parser().parse_args([]))

    def test_command_failure_keeps_log(self):
        with self.assertRaises(RuntimeError):
            runner.logged_run([sys.executable, "-c", "print('expected test failure'); raise SystemExit(7)"],
                              self.directory, "synthetic-failure", self.directory, 10)
        self.assertIn("expected test failure", (self.directory / "synthetic-failure.log").read_text())
        self.assertFalse((self.directory / "SUCCESS.txt").exists())

    def test_main_no_gpu_does_not_pass(self):
        with mock.patch.object(runner.shutil, "which", return_value=None):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(runner.main(["--check"]), 1)

    def test_requested_profiler_is_required(self):
        args = runner.parser().parse_args(["--profile", "both", "--sanitizer", "both"])
        names = runner.required_tools(args)
        self.assertTrue({"nsys", "ncu", "compute-sanitizer"}.issubset(names))

    def test_bad_cli_arguments(self):
        with self.assertRaises(SystemExit):
            runner.main(["--repeats", "0"])
        with self.assertRaises(SystemExit):
            runner.main(["--architectures", "80;untrusted"])

    def test_status_records_failure(self):
        # execute() deliberately requires an ASCII path on the actual GPU host.
        # Patch only the temporary path guard input if the test user's temp is Unicode.
        if any(ord(c) > 127 for c in str(self.directory)):
            self.skipTest("run this case from the configured ASCII TEMP dev environment")
        args = runner.parser().parse_args([])
        with mock.patch.object(runner, "snapshot"), mock.patch.object(runner, "preflight", side_effect=RuntimeError("synthetic preflight failure")):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(runner.execute(args, self.directory), 1)
        outputs = list((self.directory / "results").glob("gpu-*"))
        self.assertEqual(len(outputs), 1)
        self.assertEqual(json.loads((outputs[0] / "status.json").read_text())["status"], "FAIL")
        self.assertFalse((outputs[0] / "SUCCESS.txt").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
