"""Synthetic CSV tests only; never GPU execution evidence."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import pytorch_results as checker

class PytorchResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='pytorch-results-test-')
        self.path = Path(self.temp.name)
        self.summary, self.raw = [], []
        for rows, cols in checker.SHAPES:
            for scope in checker.SCOPES:
                for variant, impl in enumerate(checker.IMPLEMENTATIONS):
                    self.summary.append(dict(rows=rows, cols=cols, implementation=impl, scope=scope,
                        median_ms=2.5, min_ms=1, max_ms=4, relative_to_torch=1,
                        max_abs_error=0, atol=2e-6, rtol=2e-5, warmup=10, repeats=4,
                        launches_per_sample=100 if scope == checker.SCOPES[0] else 1,
                        seed=checker.SEED, correctness='PASS'))
                    for i in range(4):
                        self.raw.append(dict(rows=rows, cols=cols, implementation=impl, scope=scope,
                            sample=i, order_slot=(variant-i)%2, elapsed_ms=i+1, seed=checker.SEED))
    def tearDown(self): self.temp.cleanup()
    def save(self):
        for name, rows in [('summary', self.summary), ('samples', self.raw)]:
            with (self.path / ('pytorch_' + name + '.csv')).open('w', newline='', encoding='utf-8') as f:
                w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    def reject(self):
        self.save()
        with self.assertRaises((ValueError, KeyError)): checker.validate(self.path, repeats=4)
    def test_valid_even_median(self):
        self.save(); self.assertEqual(checker.validate(self.path, repeats=4)['pytorch_raw_rows'], 128)
    def test_missing(self):
        for rows in (self.summary, self.raw):
            old=rows.pop(); self.reject(); rows.append(old)
    def test_duplicate(self):
        for rows in (self.summary, self.raw):
            old=rows[-1]; rows[-1]=dict(rows[0]); self.reject(); rows[-1]=old
    def test_bad_metadata(self):
        for field,bad in [('seed','0'),('correctness','FAIL'),('warmup',0),('repeats',9),
                          ('launches_per_sample',1),('atol',1),('max_abs_error',1)]:
            old=self.summary[0][field]; self.summary[0][field]=bad; self.reject(); self.summary[0][field]=old
    def test_bad_raw(self):
        for field,bad in [('order_slot',1),('seed','0'),('elapsed_ms',0)]:
            old=self.raw[0][field]; self.raw[0][field]=bad; self.reject(); self.raw[0][field]=old
    def test_bad_statistics(self):
        for field in ('median_ms','min_ms','max_ms','relative_to_torch'):
            old=self.summary[0][field]; self.summary[0][field]=9; self.reject(); self.summary[0][field]=old
    def test_regression_accepted(self):
        for row in self.raw:
            if row['implementation'] == checker.IMPLEMENTATIONS[1]: row['elapsed_ms'] *= 2
        for row in self.summary:
            if row['implementation'] == checker.IMPLEMENTATIONS[1]:
                for field in ('median_ms','min_ms','max_ms'): row[field] *= 2
                row['relative_to_torch'] = .5
        self.save(); checker.validate(self.path, repeats=4)

if __name__ == '__main__': unittest.main(verbosity=2)
