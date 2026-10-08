#!/usr/bin/env python3
"""Summarize exported Nsight Systems CSV files and workload metrics."""
import csv, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(path):
    if not path.exists() or path.stat().st_size == 0: return []
    with path.open(encoding='utf-8-sig') as f: return list(csv.DictReader(f))
def value(row, needle, default=''):
    return next((row[k] for k in row if needle in k), default)
def category(name):
    n=name.lower()
    if 'flash' in n and ('attn' in n or 'fwd' in n): return 'Attention'
    if 'float_e4m3' in n or 'scaled_mm' in n: return 'FP8 GEMM/Quant'
    if 'gemm' in n or 'gemv' in n: return 'BF16/Other GEMM'
    if 'triton' in n or 'rms' in n or 'rope' in n: return 'Fused/Triton'
    if 'elementwise' in n or 'fillfunctor' in n: return 'Elementwise'
    return 'Other'

rows=[]
for workload in sorted((ROOT/'profiles/workloads').glob('*.json')):
    name=workload.stem; data=json.loads(workload.read_text())
    kernels=read(ROOT/'profiles/export'/f'{name}_cuda_gpu_kern_sum.csv')
    apis=read(ROOT/'profiles/export'/f'{name}_cuda_api_sum.csv')
    def top(items,n=10):
        return items[:n]
    cats={}
    for k in kernels:
        c=category(value(k,'Name')); cats[c]=cats.get(c,0.0)+float(value(k,'Time (%)','0') or 0)
    item={'profile':name,'workload':data,'kernel_rows':len(kernels),'cuda_api_rows':len(apis),'category_time_pct':{k:round(v,1) for k,v in sorted(cats.items(),key=lambda x:-x[1])},
          'top_kernels':top(kernels),'top_cuda_apis':top(apis)}
    rows.append(item)
(ROOT/'profiles/summary').mkdir(parents=True,exist_ok=True)
(ROOT/'profiles/summary/nsys_analysis.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
md=['# Nsight Systems 自动分析','']
for x in rows:
    w=x['workload']; concurrency=w.get('concurrency',1)
    ttft=w.get('mean_ttft_ms','offline 未单独测量'); throughput=w.get('aggregate_output_tok_s',w.get('output_tok_s'))
    md += [f"## {x['profile']}",'',f"- 场景：{w['scenario']}，并发 {concurrency}",
      f"- TTFT：{ttft} ms；输出吞吐：{throughput} tok/s",
      f"- 有效 Kernel 类别：{x['kernel_rows']}；CUDA API 类别：{x['cuda_api_rows']}",
      f"- Kernel 类别占比：{x['category_time_pct']}",'', '### Top kernels','']
    for k in x['top_kernels'][:8]:
        name=value(k,'Name','unknown'); pct=value(k,'Time (%)','?')
        md.append(f"- {pct}% — `{name}`")
    md.append('')
(ROOT/'profiles/summary/NSYS_AUTO_REPORT.md').write_text('\n'.join(md)+'\n')
print(f'分析 {len(rows)} 份 profile')
