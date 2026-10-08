#!/usr/bin/env python3
"""Small deterministic CUDA workload for checking Nsight Compute counters.

This validates the profiler path and teaches roofline interpretation. It is not a
replacement for a vLLM production kernel.
"""
import argparse, json, time
from pathlib import Path
import torch

p=argparse.ArgumentParser(); p.add_argument('--size',type=int,default=4096); p.add_argument('--iterations',type=int,default=8)
a=p.parse_args(); assert torch.cuda.is_available()
torch.manual_seed(7); device='cuda'; dtype=torch.bfloat16
x=torch.randn((a.size,a.size),device=device,dtype=dtype); w=torch.randn_like(x)
for _ in range(2): torch.mm(x,w)
torch.cuda.synchronize(); start=time.perf_counter()
torch.cuda.nvtx.range_push('decode_like_bf16_gemm')
for _ in range(a.iterations): y=torch.mm(x,w)
torch.cuda.nvtx.range_pop(); torch.cuda.synchronize()
elapsed=time.perf_counter()-start
print(json.dumps({'workload':'controlled_bf16_gemm_not_vllm_kernel','size':a.size,'iterations':a.iterations,
 'elapsed_s':elapsed,'approx_tflops':2*a.size**3*a.iterations/elapsed/1e12},indent=2))
