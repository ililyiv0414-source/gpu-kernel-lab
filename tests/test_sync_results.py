"""Synthetic CSV fixtures, never GPU execution evidence."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import sync_results as checker

class SyncResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='sync-results-test-')
        self.path=Path(self.temp.name)
        self.data={k:[] for k in ('summary','samples','setup')}
        for rows,cols in checker.SHAPES:
            self.data['setup'].append(dict(rows=rows,cols=cols,first_call_ms=4,capacity=rows*cols,allocation_events=1,seed=checker.SEED))
            for v,variant in enumerate(checker.VARIANTS):
                self.data['summary'].append(dict(rows=rows,cols=cols,implementation=variant,scope='end_to_end_wall',
                    median_ms=2.5,min_ms=1,max_ms=4,speedup_vs_wrapper=1,speedup_vs_staged=1,max_abs_error=0,
                    atol=2e-6,rtol=2e-5,warmup=5,repeats=4,seed=checker.SEED,explicit_device_waits=checker.WAITS[v],correctness='PASS'))
                for i in range(4):
                    self.data['samples'].append(dict(rows=rows,cols=cols,implementation=variant,sample=i,
                        order_slot=(v-i)%4,elapsed_ms=i+1,seed=checker.SEED))
        self.fields={k:list(v[0]) for k,v in self.data.items()}
    def tearDown(self): self.temp.cleanup()
    def save(self):
        for k,rows in self.data.items():
            with (self.path/('sync_'+k+'.csv')).open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=self.fields[k]);w.writeheader();w.writerows(rows)
    def reject(self):
        self.save()
        with self.assertRaises((ValueError,KeyError)): checker.validate(self.path,repeats=4)
    def test_valid_even_median(self):
        self.save();self.assertEqual(checker.validate(self.path,repeats=4)['sync_raw_rows'],160)
    def test_missing(self):
        for k in self.data:
            with self.subTest(k=k):
                old=self.data[k].pop();self.reject();self.data[k].append(old)
    def test_duplicates(self):
        for k in self.data:
            with self.subTest(k=k):
                old=self.data[k][-1];self.data[k][-1]=dict(self.data[k][0]);self.reject();self.data[k][-1]=old
    def test_bad_numbers(self):
        for k,field in [('summary','median_ms'),('samples','elapsed_ms'),('setup','first_call_ms')]:
            for bad in (0,-1,float('nan'),float('inf')):
                with self.subTest(k=k,bad=bad):
                    old=self.data[k][0][field];self.data[k][0][field]=bad;self.reject();self.data[k][0][field]=old
    def test_bad_metadata(self):
        for field,bad in [('scope','kernel_event'),('correctness','FAIL'),('seed','0'),('explicit_device_waits',99),
                          ('atol',1),('rtol',1),('warmup',0),('repeats',24),('max_abs_error',1)]:
            old=self.data['summary'][0][field];self.data['summary'][0][field]=bad;self.reject();self.data['summary'][0][field]=old
    def test_wrong_even_median(self):
        self.data['summary'][0]['median_ms']=3;self.reject()
    def test_wrong_ratios(self):
        for field in ('speedup_vs_wrapper','speedup_vs_staged'):
            self.data['summary'][0][field]=2;self.reject();self.data['summary'][0][field]=1
    def test_wrong_order(self):
        self.data['samples'][0]['order_slot']=1;self.reject()
    def test_wrong_raw_seed(self):
        self.data['samples'][0]['seed']='0';self.reject()
    def test_setup_capacity(self):
        self.data['setup'][0]['capacity']=99;self.reject()
    def test_setup_allocations(self):
        self.data['setup'][0]['allocation_events']=0;self.reject()
    def test_regression_accepted(self):
        for row in self.data['samples']:
            if row['implementation']==checker.VARIANTS[-1]:row['elapsed_ms']*=2
        for row in self.data['summary']:
            if row['implementation']==checker.VARIANTS[-1]:
                for field in ('median_ms','min_ms','max_ms'):row[field]*=2
                row['speedup_vs_wrapper']=row['speedup_vs_staged']=0.5
        self.save();checker.validate(self.path,repeats=4)

if __name__=='__main__':unittest.main(verbosity=2)
