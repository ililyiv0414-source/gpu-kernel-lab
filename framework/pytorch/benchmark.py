#!/usr/bin/env python3
"""Build, verify and benchmark the registered PyTorch CUDA softmax operator."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ops import load_operator

SEED = 20260829
SHAPES = [(1,128), (7,37), (1024,128), (1024,257), (1024,512),
          (1024,1024), (1024,2048), (1024,4097)]
IMPLEMENTATIONS = ('torch_softmax', 'kernel_lab_custom')
SCOPES = ('resident_cuda_event_per_call', 'cpu_to_cuda_to_cpu_wall')
ATOL, RTOL = 2e-6, 2e-5

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--build-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--repeats', type=int, default=21)
    p.add_argument('--resident-launches', type=int, default=100)
    p.add_argument('--sanitizer-smoke', action='store_true')
    p.add_argument('--verbose-build', action='store_true')
    return p

def require(condition, message):
    if not condition: raise RuntimeError(message)

def call(implementation, custom, tensor):
    return torch.softmax(tensor, dim=-1) if implementation == IMPLEMENTATIONS[0] else custom(tensor)

def max_error(actual, expected):
    return float((actual-expected).abs().max().cpu())

def verify(custom, output_dir):
    generator=torch.Generator(device='cpu').manual_seed(SEED)
    cases=[]
    with torch.inference_mode():
        for rows,cols in SHAPES:
            cpu=torch.randn((rows,cols), generator=generator, dtype=torch.float32)
            if (rows,cols)==(1024,257): cpu=cpu*40.0
            x=cpu.cuda()
            expected=torch.softmax(x,dim=-1)
            actual=custom(x)
            error=max_error(actual,expected)
            require(torch.allclose(actual,expected,atol=ATOL,rtol=RTOL), f'forward mismatch {(rows,cols)}: {error}')
            cases.append({'case':'forward','shape':[rows,cols],'max_abs_error':error,'status':'PASS'})
        stream=torch.cuda.Stream()
        x=torch.randn((64,257),device='cuda')
        with torch.cuda.stream(stream):
            actual=custom(x); expected=torch.softmax(x,dim=-1)
        stream.synchronize()
        error=max_error(actual,expected)
        require(torch.allclose(actual,expected,atol=ATOL,rtol=RTOL),'non-default stream mismatch')
        cases.append({'case':'non_default_stream','shape':[64,257],'max_abs_error':error,'status':'PASS'})

    x1=torch.randn((8,257),device='cuda',requires_grad=True)
    x2=x1.detach().clone().requires_grad_(True)
    grad=torch.randn_like(x1)
    custom(x1).backward(grad); torch.softmax(x2,dim=-1).backward(grad)
    error=max_error(x1.grad,x2.grad)
    require(torch.allclose(x1.grad,x2.grad,atol=ATOL,rtol=RTOL),'autograd mismatch')
    cases.append({'case':'autograd','shape':[8,257],'max_abs_error':error,'status':'PASS'})

    opcheck_result=torch.library.opcheck(custom,(torch.randn((4,37),device='cuda',requires_grad=True),),
                                        raise_exception=True)
    require(all(value == 'SUCCESS' for value in opcheck_result.values()),'torch.library.opcheck did not fully pass')
    cases.append({'case':'torch_library_opcheck','details':opcheck_result,'status':'PASS'})
    compiled=torch.compile(lambda value: custom(value),backend='eager',fullgraph=True)
    x=torch.randn((4,37),device='cuda')
    error=max_error(compiled(x),torch.softmax(x,dim=-1))
    require(error <= 2.2e-5,'torch.compile result mismatch')
    cases.append({'case':'torch_compile_fullgraph_eager','max_abs_error':error,'status':'PASS'})

    invalid=[('cpu',torch.randn(2,3)),('float64',torch.randn(2,3,device='cuda',dtype=torch.float64)),
             ('rank1',torch.randn(3,device='cuda')),('noncontiguous',torch.randn(3,2,device='cuda').t()),
             ('empty',torch.empty(0,3,device='cuda'))]
    for name,value in invalid:
        try: custom(value)
        except (RuntimeError, NotImplementedError) as error_value:
            cases.append({'case':'reject_'+name,'error':str(error_value)[:300],'status':'PASS'})
        else: raise RuntimeError('invalid input accepted: '+name)
    result={'status':'PASS','seed':SEED,'atol':ATOL,'rtol':RTOL,'cases':cases,
            'finite_input_precondition':True,
            'note':'The custom kernel is benchmarked only on finite FP32 contiguous 2D CUDA tensors.'}
    (output_dir/'framework_correctness.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return {(item['shape'][0],item['shape'][1]):item['max_abs_error'] for item in cases
            if item['case']=='forward'}

def resident_sample(implementation, custom, x, launches):
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(launches): call(implementation,custom,x)
    end.record(); end.synchronize()
    return start.elapsed_time(end)/launches

def e2e_sample(implementation, custom, cpu):
    torch.cuda.synchronize(); start=time.perf_counter()
    output=call(implementation,custom,cpu.cuda()).cpu()
    elapsed=(time.perf_counter()-start)*1000.0
    require(output.isfinite().all().item(),'non-finite end-to-end output')
    return elapsed

def benchmark(custom,args,errors):
    args.output_dir.mkdir(parents=True,exist_ok=True)
    generator=torch.Generator(device='cpu').manual_seed(SEED)
    raw=[]
    with torch.inference_mode():
        for rows,cols in SHAPES:
            cpu=torch.randn((rows,cols),generator=generator,dtype=torch.float32)
            gpu=cpu.cuda()
            for implementation in IMPLEMENTATIONS:
                for _ in range(args.warmup):
                    call(implementation,custom,gpu)
                torch.cuda.synchronize()
            for scope in SCOPES:
                for sample in range(args.repeats):
                    order=IMPLEMENTATIONS[sample%2:]+IMPLEMENTATIONS[:sample%2]
                    for order_slot,implementation in enumerate(order):
                        elapsed=(resident_sample(implementation,custom,gpu,args.resident_launches)
                                 if scope==SCOPES[0] else e2e_sample(implementation,custom,cpu))
                        require(elapsed>0,'non-positive elapsed time')
                        raw.append({'rows':rows,'cols':cols,'implementation':implementation,'scope':scope,
                                    'sample':sample,'order_slot':order_slot,'elapsed_ms':format(elapsed,'.12g'),
                                    'seed':SEED})
    fields=list(raw[0])
    with (args.output_dir/'pytorch_samples.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(raw)
    grouped={}
    for row in raw:
        key=(row['rows'],row['cols'],row['implementation'],row['scope'])
        grouped.setdefault(key,[]).append(float(row['elapsed_ms']))
    summary=[]
    for rows,cols in SHAPES:
        for scope in SCOPES:
            torch_median=statistics.median(grouped[(rows,cols,IMPLEMENTATIONS[0],scope)])
            for implementation in IMPLEMENTATIONS:
                values=grouped[(rows,cols,implementation,scope)]
                median=statistics.median(values)
                summary.append({'rows':rows,'cols':cols,'implementation':implementation,'scope':scope,
                    'median_ms':format(median,'.12g'),'min_ms':format(min(values),'.12g'),
                    'max_ms':format(max(values),'.12g'),'relative_to_torch':format(torch_median/median,'.12g'),
                    'max_abs_error':format(errors[(rows,cols)],'.12g'),'atol':ATOL,'rtol':RTOL,
                    'warmup':args.warmup,'repeats':args.repeats,
                    'launches_per_sample':args.resident_launches if scope==SCOPES[0] else 1,
                    'seed':SEED,'correctness':'PASS'})
    with (args.output_dir/'pytorch_summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)

def profile(custom,output_dir):
    x=torch.randn((1024,257),device='cuda')
    activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
    with torch.inference_mode(), torch.profiler.profile(activities=activities,record_shapes=True) as prof:
        with torch.profiler.record_function('torch_softmax_5_calls'):
            for _ in range(5): torch.softmax(x,dim=-1)
        with torch.profiler.record_function('kernel_lab_custom_5_calls'):
            for _ in range(5): custom(x)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(output_dir/'pytorch_trace.json'))
    (output_dir/'pytorch_profiler_table.txt').write_text(
        prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=30),encoding='utf-8')

def sanitizer_smoke(custom,args):
    generator=torch.Generator(device='cpu').manual_seed(SEED)
    cases=[]
    with torch.inference_mode():
        for shape in ((1,128),(7,37),(64,257),(32,4097)):
            cpu=torch.randn(shape,generator=generator)
            expected=torch.softmax(cpu,dim=-1)
            actual=custom(cpu.cuda()).cpu()
            error=max_error(actual,expected)
            require(torch.allclose(actual,expected,atol=ATOL,rtol=RTOL),f'sanitizer smoke mismatch {shape}')
            cases.append({'shape':shape,'max_abs_error':error})
    (args.output_dir/'pytorch_sanitizer_smoke.json').write_text(
        json.dumps({'status':'PASS','cases':cases},indent=2),encoding='utf-8')

def main():
    args=parser().parse_args()
    require(torch.cuda.is_available(),'PyTorch CUDA is unavailable')
    require(args.warmup>=0 and args.repeats>0 and args.resident_launches>0,'invalid benchmark counts')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    custom=load_operator(args.project,args.build_dir,args.verbose_build)
    environment={'torch_version':torch.__version__,'torch_cuda':torch.version.cuda,
                 'device':torch.cuda.get_device_name(),'device_capability':torch.cuda.get_device_capability(),
                 'seed':SEED,'argv':sys.argv}
    (args.output_dir/'pytorch_environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    if args.sanitizer_smoke: sanitizer_smoke(custom,args); return
    errors=verify(custom,args.output_dir)
    benchmark(custom,args,errors)
    profile(custom,args.output_dir)
    print('PASS PyTorch custom-op correctness, comparison and profiler evidence:',args.output_dir)

if __name__=='__main__': main()
