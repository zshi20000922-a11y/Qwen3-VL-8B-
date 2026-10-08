#!/usr/bin/env python3
import argparse, csv, json, sys
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--require-ncu',action='store_true'); args=p.parse_args()
ROOT=Path(__file__).resolve().parents[1]; errors=[]
nsys=ROOT/'profiles/summary/nsys_analysis.json'
if not nsys.exists(): errors.append('缺少 nsys_analysis.json')
else:
    for x in json.loads(nsys.read_text()):
        if x['kernel_rows'] < 1: errors.append(f"{x['profile']}: 无 CUDA kernel 数据")
        if x['cuda_api_rows'] < 1: errors.append(f"{x['profile']}: 无 CUDA API 数据")
for p in (ROOT/'profiles/workloads').glob('*.json'):
    d=json.loads(p.read_text());
    if d.get('total_output_tokens',d.get('output_tokens',0))<=0: errors.append(f'{p.name}: 无输出 token')
if args.require_ncu and not list((ROOT/'profiles/raw').glob('*.ncu-rep')):
    errors.append('缺少 .ncu-rep；需要交互式 sudo 解锁 GPU Performance Counters')
if errors:
    print('\n'.join('FAIL '+x for x in errors)); sys.exit(1)
print('PASS: Nsight 报告包含有效 CUDA trace，工作负载元数据完整')
