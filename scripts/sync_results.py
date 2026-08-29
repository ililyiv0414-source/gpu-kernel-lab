"""Validate full-call synchronization ablation, including regressions and even medians."""
import csv
from pathlib import Path
import statistics
from workspace_results import require, number, same

SHAPES = [(1,1),(1,128),(7,37),(32,128),(1024,128),(1024,257),(1024,512),(1024,1024),(1024,2048),(1024,4097)]
VARIANTS = ('warp_wrapper','workspace_staged','workspace_kernel_sync','workspace_copy_sync')
WAITS = (1,2,1,0)
SEED = '20260829'

def key(row): return (int(row['rows']),int(row['cols']),row['implementation'])

def validate(directory,repeats=24,warmup=5):
    def read(name):
        with (Path(directory)/('sync_'+name+'.csv')).open(newline='',encoding='utf-8') as f:
            return list(csv.DictReader(f))
    summary,raw,setups=(read(name) for name in ('summary','samples','setup'))
    expected={(r,c,v) for r,c in SHAPES for v in VARIANTS}
    require(len(summary)==40 and len(raw)==40*repeats and len(setups)==10,'incomplete sync row counts')
    lookup,samples={},{}
    for row in summary:
        k=key(row)
        require(k in expected and k not in lookup,'duplicate/unexpected summary')
        require(row['scope']=='end_to_end_wall' and row['correctness']=='PASS' and row['seed']==SEED
                and int(row['repeats'])==repeats and int(row['warmup'])==warmup
                and float(row['atol'])==2e-6 and float(row['rtol'])==2e-5
                and int(row['explicit_device_waits'])==WAITS[VARIANTS.index(k[2])],'sync metadata mismatch')
        for field in ('median_ms','min_ms','max_ms','speedup_vs_wrapper','speedup_vs_staged'):
            number(row,field,True)
        require(number(row,'max_abs_error')<=2.2e-5,'error too large')
        lookup[k],samples[k]=row,{}
    for row in raw:
        k,i=key(row),int(row['sample'])
        require(k in expected and 0<=i<repeats,'unexpected raw case')
        require(i not in samples[k] and row['seed']==SEED and
                int(row['order_slot'])==(VARIANTS.index(k[2])-i)%4,'duplicate sample or incorrect order/seed')
        samples[k][i]=number(row,'elapsed_ms',True)
    for k,row in lookup.items():
        values=list(samples[k].values())
        require(len(values)==repeats,'missing raw repeats')
        for field,value in [('median_ms',statistics.median(values)),('min_ms',min(values)),('max_ms',max(values))]:
            require(same(float(row[field]),value),'summary/sample mismatch')
        for field,variant in [('speedup_vs_wrapper',VARIANTS[0]),('speedup_vs_staged',VARIANTS[1])]:
            require(same(float(row[field]),float(lookup[(*k[:2],variant)]['median_ms'])/float(row['median_ms'])),
                    'speedup mismatch')
    seen=set()
    for row in setups:
        k=(int(row['rows']),int(row['cols']))
        require(k in SHAPES and k not in seen and row['seed']==SEED,'invalid setup case')
        require(int(row['capacity'])==k[0]*k[1] and int(row['allocation_events'])==1,'invalid initial allocation')
        number(row,'first_call_ms',True); seen.add(k)
    return {'sync_summary_rows':len(summary),'sync_raw_rows':len(raw),'sync_setup_rows':len(setups)}
