"""Synthetic validation fixtures only; no GPU-performance claims."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('workspace_results', Path(__file__).parents[1] / 'scripts/workspace_results.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

class WorkspaceResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='workspace-results-test-')
        self.path = Path(self.temp.name)
        self.data = {k: [] for k in ('summary', 'samples', 'stages', 'setup')}
        for rows, cols in checker.SHAPES:
            self.data['setup'].append(dict(rows=rows, cols=cols, first_call_ms=4,
                capacity=rows*cols, allocation_events=1, seed=checker.SEED))
            for v, variant in enumerate(checker.VARIANTS):
                self.data['summary'].append(dict(rows=rows, cols=cols, implementation=variant,
                    median_ms=2, min_ms=1, max_ms=3, speedup_vs_wrapper=1, speedup_vs_fresh=1,
                    max_abs_error=0, atol=2e-6, rtol=2e-5, warmup=5, repeats=3, seed=checker.SEED, correctness='PASS'))
                for sample in range(3):
                    self.data['samples'].append(dict(rows=rows, cols=cols, implementation=variant,
                        sample=sample, order_slot=(v-sample)%3, elapsed_ms=sample+1, seed=checker.SEED))
                    if v:
                        self.data['stages'].append(dict(rows=rows, cols=cols, implementation=variant,
                            sample=sample, order_slot=(v-1-sample)%2, **{p: 1 for p in checker.PHASES},
                            total_ms=6, lifecycle_ms=1, wall_total_ms=7,
                            allocation_events=1 if v == 1 else 0, seed=checker.SEED))
        self.fields = {k: list(v[0]) for k,v in self.data.items()}
    def tearDown(self):
        self.temp.cleanup()
    def save(self):
        for k, values in self.data.items():
            with (self.path / ('workspace_' + k + '.csv')).open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fields[k]); writer.writeheader(); writer.writerows(values)
    def reject(self):
        self.save()
        with self.assertRaises((ValueError, KeyError)):
            checker.validate(self.path, repeats=3)
    def test_valid(self):
        self.save()
        self.assertEqual(checker.validate(self.path, repeats=3)['workspace_stage_rows'], 48)
    def test_missing_rows(self):
        for name in self.data:
            with self.subTest(name=name):
                old = self.data[name].pop(); self.reject(); self.data[name].append(old)
    def test_duplicate_rows(self):
        for name in self.data:
            with self.subTest(name=name):
                old = self.data[name][-1]; self.data[name][-1] = dict(self.data[name][0]); self.reject(); self.data[name][-1] = old
    def test_numeric_rejected(self):
        for name, field in [('summary','median_ms'), ('samples','elapsed_ms'), ('stages','validation_ms'), ('setup','first_call_ms')]:
            for bad in (float('nan'), float('inf'), -1):
                with self.subTest(name=name, bad=bad):
                    old = self.data[name][0][field]; self.data[name][0][field] = bad; self.reject(); self.data[name][0][field] = old
    def test_seed_rejected(self):
        for name in self.data:
            with self.subTest(name=name):
                self.data[name][0]['seed'] = '0'; self.reject(); self.data[name][0]['seed'] = checker.SEED
    def test_statistics_rejected(self):
        for field in ('min_ms','median_ms','max_ms','speedup_vs_wrapper','speedup_vs_fresh'):
            old = self.data['summary'][0][field]; self.data['summary'][0][field] = 999; self.reject(); self.data['summary'][0][field] = old
    def test_stage_sum_rejected(self):
        self.data['stages'][0]['total_ms'] = 100; self.reject()
    def test_lifecycle_sum_rejected(self):
        self.data['stages'][0]['wall_total_ms'] = 100; self.reject()
    def test_unexpected_allocation_rejected(self):
        row = next(r for r in self.data['stages'] if r['implementation'] == 'workspace_reuse')
        row['allocation_events'] = 1; self.reject()
    def test_setup_capacity_rejected(self):
        self.data['setup'][0]['capacity'] = 1; self.reject()
    def test_order_rejected(self):
        for name in ('samples','stages'):
            old = self.data[name][0]['order_slot']; self.data[name][0]['order_slot'] = 99; self.reject(); self.data[name][0]['order_slot'] = old
    def test_regression_accepted(self):
        for row in self.data['samples']:
            if row['implementation'] == 'workspace_reuse': row['elapsed_ms'] *= 2
        for row in self.data['summary']:
            if row['implementation'] == 'workspace_reuse':
                for field in ('median_ms','min_ms','max_ms'): row[field] *= 2
                row['speedup_vs_wrapper'] = row['speedup_vs_fresh'] = 0.5
        self.save(); checker.validate(self.path, repeats=3)

if __name__ == '__main__':
    unittest.main(verbosity=2)
