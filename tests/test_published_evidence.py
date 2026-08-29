import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "a10"


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select(records, **expected):
    matches = [
        record
        for record in records
        if all(record.get(key) == str(value) for key, value in expected.items())
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one row for {expected}, found {len(matches)}")
    return matches[0]


class PublishedEvidenceTests(unittest.TestCase):
    def test_resident_softmax_speedup(self):
        records = rows(EVIDENCE / "softmax_comparison_summary.csv")
        baseline = select(
            records,
            rows=1024,
            cols=128,
            implementation="cuda_naive",
            scope="batched_event_per_launch",
        )
        optimized = select(
            records,
            rows=1024,
            cols=128,
            implementation="cuda_warp_register",
            scope="batched_event_per_launch",
        )
        baseline_us = float(baseline["median_ms"]) * 1000.0
        optimized_us = float(optimized["median_ms"]) * 1000.0
        speedup = baseline_us / optimized_us
        self.assertAlmostEqual(baseline_us, 10.8864, places=3)
        self.assertAlmostEqual(optimized_us, 2.88832, places=3)
        self.assertGreater(speedup, 3.76)
        self.assertLess(speedup, 3.78)
        self.assertEqual(baseline["correctness"], "PASS")
        self.assertEqual(optimized["correctness"], "PASS")

    def test_workspace_independent_processes(self):
        observed = []
        for process in (1, 2):
            records = rows(EVIDENCE / f"workspace_process_{process}_summary.csv")
            wrapper = select(records, rows=1024, cols=257, implementation="warp_wrapper")
            reused = select(records, rows=1024, cols=257, implementation="workspace_reuse")
            observed.append(float(wrapper["median_ms"]) / float(reused["median_ms"]))
            self.assertEqual(reused["correctness"], "PASS")
        self.assertTrue(all(1.47 < value < 1.52 for value in observed))

    def test_run_status_and_framework_correctness(self):
        native = json.loads((EVIDENCE / "softmax_comparison_status.json").read_text(encoding="utf-8"))
        framework = json.loads((EVIDENCE / "pytorch_run_status.json").read_text(encoding="utf-8"))
        correctness = json.loads((EVIDENCE / "pytorch_correctness.json").read_text(encoding="utf-8"))
        sanitizer = json.loads((EVIDENCE / "pytorch_sanitizer_smoke.json").read_text(encoding="utf-8"))

        self.assertEqual(native["status"], "PASS")
        self.assertTrue(native["gpu_execution_verified"])
        self.assertEqual(native["sanitizer"], "both")
        self.assertEqual(framework["status"], "PASS")
        self.assertTrue(framework["pytorch_comparison_valid"])
        self.assertEqual(framework["pytorch_summary_rows"], 32)
        self.assertEqual(framework["pytorch_raw_rows"], 672)
        self.assertEqual(correctness["status"], "PASS")
        self.assertTrue(all(case["status"] == "PASS" for case in correctness["cases"]))
        self.assertEqual(sanitizer["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
