"""Synthetic validator tests, not GPU performance evidence."""
import copy
import csv
import importlib.util
from pathlib import Path
import statistics
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('compare', Path(__file__).parents[1] / 'scripts/softmax_compare_results.py')
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def fixture():
    summary, raw = [], []
    for rows, cols in compare.SHAPES:
        for impl_index, impl in enumerate(compare.IMPLEMENTATIONS):
            for scope in compare.SCOPES:
                times = [0.1 * (impl_index + 1) + sample * 0.001 for sample in range(3)]
                common = dict(rows=str(rows), cols=str(cols), implementation=impl,
                              dispatch=compare.dispatch(impl, cols), scope=scope,
                              launches='7' if scope == compare.SCOPES[0] else '1', seed=compare.SEED)
                summary.append(dict(common, median_ms=str(statistics.median(times)), min_ms=str(min(times)),
                                    max_ms=str(max(times)), baseline_speedup=str(0.101/statistics.median(times)),
                                    max_abs_error='1e-8', atol='2e-6', rtol='2e-5', warmup='5', repeats='3', correctness='PASS'))
                for sample, elapsed in enumerate(times):
                    raw.append(dict(common, sample=str(sample), order_slot=str((impl_index-sample)%3), elapsed_ms=str(elapsed)))
    return summary, raw


class ComparisonTests(unittest.TestCase):
    def validate(self, summary, raw):
        with tempfile.TemporaryDirectory() as temp:
            for suffix, rows in (('summary', summary), ('samples', raw)):
                with (Path(temp) / ('softmax_compare_' + suffix + '.csv')).open('w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            return compare.validate(temp, repeats=3, warmup=5, launches=7)

    def test_valid_including_regression(self):
        self.assertEqual(self.validate(*fixture()), {'comparison_summary_rows': 48, 'comparison_raw_rows': 144})

    def test_missing_summary(self):
        summary, raw = fixture()
        with self.assertRaises(ValueError): self.validate(summary[:-1], raw)

    def test_missing_raw(self):
        summary, raw = fixture()
        with self.assertRaises(ValueError): self.validate(summary, raw[:-1])

    def test_duplicate_summary(self):
        summary, raw = fixture()
        summary[-1] = copy.deepcopy(summary[0])
        with self.assertRaises(ValueError): self.validate(summary, raw)

    def test_duplicate_raw(self):
        summary, raw = fixture()
        raw[-1] = copy.deepcopy(raw[0])
        with self.assertRaises(ValueError): self.validate(summary, raw)

    def test_bad_summary_fields(self):
        for field, value in (('median_ms', 'nan'), ('min_ms', '-1'), ('max_ms', 'inf'),
                             ('baseline_speedup', '999'), ('correctness', 'FAIL'), ('max_abs_error', '0.1'),
                             ('atol', '1'), ('rtol', '1'), ('warmup', '1'), ('repeats', '99'),
                             ('seed', '0'), ('dispatch', 'wrong'), ('launches', '2')):
            with self.subTest(field=field):
                summary, raw = fixture()
                summary[0][field] = value
                with self.assertRaises(ValueError): self.validate(summary, raw)

    def test_bad_raw_fields(self):
        for field, value in (('elapsed_ms', '0'), ('elapsed_ms', 'inf'), ('elapsed_ms', 'nan'),
                             ('sample', '9'), ('order_slot', '9'), ('dispatch', 'wrong'),
                             ('seed', '0'), ('launches', '1'), ('cols', '100')):
            with self.subTest(field=field, value=value):
                summary, raw = fixture()
                raw[0][field] = value
                with self.assertRaises(ValueError): self.validate(summary, raw)

    def test_fallback_label_required(self):
        summary, raw = fixture()
        row = next(r for r in summary if r['implementation'] == 'cuda_warp_register' and r['cols'] == '4097')
        row['dispatch'] = 'warp_reg32'
        with self.assertRaises(ValueError): self.validate(summary, raw)

    def test_summary_raw_must_agree(self):
        summary, raw = fixture()
        raw[0]['elapsed_ms'] = '123'
        with self.assertRaises(ValueError): self.validate(summary, raw)


if __name__ == '__main__':
    unittest.main(verbosity=2)
