#!/usr/bin/env python3
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
analysis=ROOT/'profiles/summary/nsys_analysis.json'; out=ROOT/'profiles/summary/ab_comparison.md'
rows=json.loads(analysis.read_text()) if analysis.exists() else []
md=['# Nsight A/B 对比','', '| Profile | 场景 | 并发 | TTFT ms | E2E P50 ms | 聚合 tok/s |',
    '|---|---|---:|---:|---:|---:|']
for x in rows:
    w=x['workload']; md.append(f"| {x['profile']} | {w['scenario']} | {w.get('concurrency',1)} | {w.get('mean_ttft_ms','N/A')} | {w.get('p50_e2e_ms',w.get('e2e_ms'))} | {w.get('aggregate_output_tok_s',w.get('output_tok_s'))} |")
md += ['', '> 只有在输入、缓存状态和采样设置一致时才比较 Profile；低于 5% 的差异不写成优化收益。','']
out.parent.mkdir(parents=True,exist_ok=True); out.write_text('\n'.join(md)); print(out)
