#!/usr/bin/env python3
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name):
    d=json.loads((ROOT/f'results/ab/{name}/summary.json').read_text()); return {x['scenario']:x for x in d['scenarios']}
graph,eager=load('cuda_graph'),load('eager')
rows=[]
for c in (1,8):
    k=f'concurrency_{c}'; g,e=graph[k],eager[k]
    gain=(g['aggregate_tok_s_mean']/e['aggregate_tok_s_mean']-1)*100
    latency=(1-g['e2e_p95_ms']/e['e2e_p95_ms'])*100
    rows.append({'concurrency':c,'graph_tok_s':g['aggregate_tok_s_mean'],'eager_tok_s':e['aggregate_tok_s_mean'],
                 'throughput_gain_pct':round(gain,1),'p95_latency_reduction_pct':round(latency,1),
                 'resume_eligible':abs(gain)>=5})
out={'comparison':'CUDA Graph vs enforce-eager','rows':rows,'claim_policy':'Only >=5% changes are resume-eligible.'}
(ROOT/'results/ab/comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
