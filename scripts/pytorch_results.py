"""Validate PyTorch comparison artifacts without assuming the custom op wins."""
import csv
import math
from pathlib import Path
import statistics

SHAPES = [(1,128), (7,37), (1024,128), (1024,257), (1024,512),
          (1024,1024), (1024,2048), (1024,4097)]
IMPLEMENTATIONS = ('torch_softmax', 'kernel_lab_custom')
SCOPES = ('resident_cuda_event_per_call', 'cpu_to_cuda_to_cpu_wall')
SEED = '20260829'

def require(condition, message):
    if not condition:
        raise ValueError(message)

def number(row, field, positive=False):
    value = float(row[field])
    require(math.isfinite(value) and (value > 0 if positive else value >= 0),
            'invalid ' + field)
    return value

def same(a, b):
    return math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-10)

def key(row):
    return (int(row['rows']), int(row['cols']), row['implementation'], row['scope'])

def validate(directory, repeats=21, warmup=10, resident_launches=100):
    directory = Path(directory)
    def read(name):
        with (directory / ('pytorch_' + name + '.csv')).open(newline='', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    summary, raw = read('summary'), read('samples')
    expected = {(r,c,i,s) for r,c in SHAPES for i in IMPLEMENTATIONS for s in SCOPES}
    require(len(summary) == len(expected) and len(raw) == len(expected) * repeats,
            'incomplete PyTorch comparison row counts')
    lookup, samples = {}, {}
    for row in summary:
        k = key(row)
        require(k in expected and k not in lookup, 'duplicate/unexpected PyTorch summary')
        launches = resident_launches if k[3] == SCOPES[0] else 1
        require(row['seed'] == SEED and row['correctness'] == 'PASS'
                and int(row['repeats']) == repeats and int(row['warmup']) == warmup
                and int(row['launches_per_sample']) == launches
                and float(row['atol']) == 2e-6 and float(row['rtol']) == 2e-5,
                'PyTorch summary metadata mismatch')
        for field in ('median_ms', 'min_ms', 'max_ms', 'relative_to_torch'):
            number(row, field, True)
        require(number(row, 'max_abs_error') <= 2.2e-5, 'PyTorch comparison error too large')
        lookup[k], samples[k] = row, {}
    for row in raw:
        k, index = key(row), int(row['sample'])
        require(k in expected and 0 <= index < repeats, 'unexpected PyTorch raw case')
        require(index not in samples[k] and row['seed'] == SEED
                and int(row['order_slot']) == (IMPLEMENTATIONS.index(k[2]) - index) % 2,
                'duplicate sample or incorrect PyTorch rotation/seed')
        samples[k][index] = number(row, 'elapsed_ms', True)
    for k, row in lookup.items():
        values = list(samples[k].values())
        require(len(values) == repeats, 'missing PyTorch repeats')
        for field, value in (('median_ms', statistics.median(values)),
                             ('min_ms', min(values)), ('max_ms', max(values))):
            require(same(float(row[field]), value), 'PyTorch summary/raw mismatch')
        torch_ms = float(lookup[(k[0], k[1], IMPLEMENTATIONS[0], k[3])]['median_ms'])
        require(same(float(row['relative_to_torch']), torch_ms / float(row['median_ms'])),
                'relative-to-torch mismatch')
    return {'pytorch_summary_rows': len(summary), 'pytorch_raw_rows': len(raw)}
