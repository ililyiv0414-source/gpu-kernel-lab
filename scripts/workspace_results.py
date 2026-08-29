"""Validate synchronous workspace experiments without assuming improvement."""
import csv
import math
from pathlib import Path
import statistics

SHAPES = [(1,128), (7,37), (1024,128), (1024,257), (1024,512), (1024,1024), (1024,2048), (1024,4097)]
VARIANTS = ('warp_wrapper', 'workspace_fresh', 'workspace_reuse')
PHASES = ('validation_ms', 'reserve_ms', 'h2d_ms', 'kernel_sync_ms', 'host_output_ms', 'd2h_ms')
SEED = '20260829'

def require(condition, message):
    if not condition:
        raise ValueError(message)

def number(row, name, positive=False):
    value = float(row[name])
    require(math.isfinite(value) and (value > 0 if positive else value >= 0), 'invalid ' + name)
    return value

def same(a, b):
    return math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-10)

def key(row):
    return (int(row['rows']), int(row['cols']), row['implementation'])

def validate(directory, repeats=21, warmup=5):
    def read(name):
        with (Path(directory) / ('workspace_' + name + '.csv')).open(newline='', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    summary, raw, stages, setups = (read(n) for n in ('summary', 'samples', 'stages', 'setup'))
    expected = {(r,c,v) for r,c in SHAPES for v in VARIANTS}
    require(len(summary) == 24 and len(raw) == 24 * repeats and len(stages) == 16 * repeats and len(setups) == 8,
            'incomplete row counts')
    lookup, samples = {}, {}
    for row in summary:
        k = key(row)
        require(k in expected and k not in lookup, 'unexpected/duplicate summary')
        require(row['seed'] == SEED and row['correctness'] == 'PASS' and int(row['repeats']) == repeats
                and int(row['warmup']) == warmup and float(row['atol']) == 2e-6 and float(row['rtol']) == 2e-5,
                'summary metadata mismatch')
        for name in ('median_ms', 'min_ms', 'max_ms', 'speedup_vs_wrapper', 'speedup_vs_fresh'):
            number(row, name, True)
        require(number(row, 'max_abs_error') <= 2.2e-5, 'error exceeds probability tolerance upper bound')
        lookup[k], samples[k] = row, {}
    for row in raw:
        k, i = key(row), int(row['sample'])
        require(k in expected and 0 <= i < repeats, 'unknown raw sample')
        require(i not in samples[k] and row['seed'] == SEED and
                int(row['order_slot']) == (VARIANTS.index(k[2]) - i) % 3, 'duplicate sample or incorrect order/seed')
        samples[k][i] = number(row, 'elapsed_ms', True)
    for k, row in lookup.items():
        values = list(samples[k].values())
        require(len(values) == repeats, 'missing samples')
        for field, value in [('median_ms', statistics.median(values)), ('min_ms', min(values)), ('max_ms', max(values))]:
            require(same(float(row[field]), value), 'summary/raw mismatch')
        for field, variant in [('speedup_vs_wrapper', VARIANTS[0]), ('speedup_vs_fresh', VARIANTS[1])]:
            require(same(float(row[field]), float(lookup[(*k[:2], variant)]['median_ms']) / float(row['median_ms'])),
                    'speedup mismatch')
    stage_keys = set()
    for row in stages:
        k, i = key(row), int(row['sample'])
        require(k in expected and k[2] != VARIANTS[0] and 0 <= i < repeats and (*k,i) not in stage_keys,
                'invalid/duplicate stage case')
        require(row['seed'] == SEED and int(row['order_slot']) == (VARIANTS.index(k[2]) - 1 - i) % 2,
                'stage order/seed mismatch')
        require(int(row['allocation_events']) == (1 if k[2] == VARIANTS[1] else 0), 'unexpected device allocation')
        total = number(row, 'total_ms', True)
        require(same(sum(number(row, field) for field in PHASES), total), 'stage sum mismatch')
        require(same(total + number(row, 'lifecycle_ms'), number(row, 'wall_total_ms', True)), 'lifecycle sum mismatch')
        stage_keys.add((*k,i))
    setup_keys = set()
    for row in setups:
        k = (int(row['rows']), int(row['cols']))
        require(k in SHAPES and k not in setup_keys and row['seed'] == SEED, 'invalid setup metadata')
        require(int(row['capacity']) == k[0]*k[1] and int(row['allocation_events']) == 1, 'invalid setup allocation')
        number(row, 'first_call_ms', True)
        setup_keys.add(k)
    return {'workspace_summary_rows': len(summary), 'workspace_raw_rows': len(raw),
            'workspace_stage_rows': len(stages), 'workspace_setup_rows': len(setups)}
