"""Validate same-run CUDA-to-CUDA comparison; never substitutes CPU speedups."""
import csv
import math
from pathlib import Path
import statistics

SHAPES = [(1, 128), (7, 37), (1024, 128), (1024, 257), (1024, 512),
          (1024, 1024), (1024, 2048), (1024, 4097)]
IMPLEMENTATIONS = ('cuda_naive', 'cuda_block_shuffle', 'cuda_warp_register')
SCOPES = ('batched_event_per_launch', 'end_to_end_wall')
SEED = '20260829'


def dispatch(implementation, cols):
    if implementation == IMPLEMENTATIONS[0]:
        return 'block_shared256'
    if implementation == IMPLEMENTATIONS[1]:
        return 'block_shuffle256'
    for limit, items in ((32, 1), (128, 4), (512, 16), (1024, 32)):
        if cols <= limit:
            return 'warp_reg' + str(items)
    return 'block_shuffle256_fallback'


def expected_keys():
    return {(str(rows), str(cols), impl, scope) for rows, cols in SHAPES
            for impl in IMPLEMENTATIONS for scope in SCOPES}


def key(row):
    return tuple(row[name] for name in ('rows', 'cols', 'implementation', 'scope'))


def validate(directory, repeats=21, warmup=5, launches=100):
    directory = Path(directory)
    with (directory / 'softmax_compare_summary.csv').open(newline='', encoding='utf-8') as f:
        summary = list(csv.DictReader(f))
    with (directory / 'softmax_compare_samples.csv').open(newline='', encoding='utf-8') as f:
        raw = list(csv.DictReader(f))
    expected = expected_keys()
    if len(summary) != len(expected) or len(raw) != len(expected) * repeats:
        raise ValueError('comparison row count mismatch')
    lookup, samples = {}, {}
    for row in summary:
        k = key(row)
        count = launches if k[3] == SCOPES[0] else 1
        if k not in expected or k in lookup:
            raise ValueError('unexpected or duplicate comparison case')
        if (row['correctness'] != 'PASS' or row['seed'] != SEED or
                row['dispatch'] != dispatch(k[2], int(k[1])) or
                int(row['repeats']) != repeats or int(row['warmup']) != warmup or
                int(row['launches']) != count or float(row['atol']) != 2e-6 or float(row['rtol']) != 2e-5):
            raise ValueError('comparison metadata mismatch')
        for field in ('median_ms', 'min_ms', 'max_ms', 'baseline_speedup', 'max_abs_error'):
            value = float(row[field])
            if not math.isfinite(value) or (value < 0 if field == 'max_abs_error' else value <= 0):
                raise ValueError('invalid comparison numeric field')
        # Every softmax probability is in [0,1]; this is a necessary upper bound
        # for the claimed tolerance. Detailed elementwise checks are in CUDA.
        if float(row['max_abs_error']) > 2e-6 + 2e-5:
            raise ValueError('comparison error exceeds tolerance upper bound')
        lookup[k], samples[k] = row, {}
    for row in raw:
        k = key(row)
        if k not in samples:
            raise ValueError('unknown raw comparison key')
        sample = int(row['sample'])
        elapsed = float(row['elapsed_ms'])
        if (not 0 <= sample < repeats or sample in samples[k] or
                not math.isfinite(elapsed) or elapsed <= 0 or row['seed'] != SEED or
                row['dispatch'] != lookup[k]['dispatch'] or
                int(row['launches']) != int(lookup[k]['launches']) or
                int(row['order_slot']) != (IMPLEMENTATIONS.index(k[2]) - sample) % 3):
            raise ValueError('invalid raw comparison sample/order')
        samples[k][sample] = elapsed
    for k, row in lookup.items():
        values = list(samples[k].values())
        if len(values) != repeats:
            raise ValueError('comparison samples incomplete')
        for field, value in (('median_ms', statistics.median(values)),
                             ('min_ms', min(values)), ('max_ms', max(values))):
            if not math.isclose(float(row[field]), value, rel_tol=1e-8, abs_tol=1e-12):
                raise ValueError('comparison summary/raw mismatch')
        baseline = lookup[(k[0], k[1], IMPLEMENTATIONS[0], k[3])]
        speedup = float(baseline['median_ms']) / float(row['median_ms'])
        if not math.isclose(float(row['baseline_speedup']), speedup, rel_tol=1e-8):
            raise ValueError('comparison baseline ratio mismatch')
    return {'comparison_summary_rows': len(summary), 'comparison_raw_rows': len(raw)}
