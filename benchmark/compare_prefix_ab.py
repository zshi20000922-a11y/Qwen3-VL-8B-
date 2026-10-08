#!/usr/bin/env python3
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name):
 d=json.loads((ROOT/f'results/ab/{name}/summary.json').read_text()); return {x['scenario']:x for x in d['scenarios']}
on,off=load('prefix_on'),load('prefix_off'); rows=[]
for c in (1,8):
 k=f'concurrency_{c}'; a,b=on[k],off[k]
 rows.append({'concurrency':c,'prefix_on_ttft_p50_ms':a['ttft_p50_ms'],'prefix_off_ttft_p50_ms':b['ttft_p50_ms'],
  'ttft_reduction_pct':round((1-a['ttft_p50_ms']/b['ttft_p50_ms'])*100,1),
  'throughput_gain_pct':round((a['aggregate_tok_s_mean']/b['aggregate_tok_s_mean']-1)*100,1)})
out={'comparison':'prefix cache on vs off','rows':rows,'warning':'Repeated-image/template workload; do not generalize to unrelated prompts.'}
(ROOT/'results/ab/prefix_comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(out,ensure_ascii=False,indent=2))
