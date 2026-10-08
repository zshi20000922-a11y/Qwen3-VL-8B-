#!/usr/bin/env python3
"""Focused, five-repeat c1/c8 A/B workload for one running service config."""
import argparse, asyncio, csv, json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import httpx
from run_suite import run_group, summarize

async def run(a):
    root=Path(__file__).resolve().parents[1]; image=root/'images/test.jpg'
    cfg=SimpleNamespace(model='Qwen3-VL-8B-Instruct',prompt='请客观描述图片中的主要目标、场景、文字和目标状态。',
        max_tokens=a.max_tokens,base_url=a.base_url,timeout=300)
    rows=[]
    async with httpx.AsyncClient(timeout=httpx.Timeout(300)) as client:
        (await client.get(a.base_url+'/health')).raise_for_status()
        await run_group(client,cfg,'warmup',image,1,-1,f'{a.label}-warmup')
        for c in (1,8):
            for repeat in range(a.repeats):
                rows += await run_group(client,cfg,f'concurrency_{c}',image,c,repeat,f'{a.label}-c{c}-r{repeat}')
    summary=summarize(rows); a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'requests.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    with (a.output_dir/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)
    meta={'label':a.label,'generated_at':datetime.now(timezone.utc).isoformat(),'repeats':a.repeats,'max_tokens':a.max_tokens}
    (a.output_dir/'summary.json').write_text(json.dumps({'metadata':meta,'scenarios':summary},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--label',required=True); p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--base-url',default='http://127.0.0.1:8000'); p.add_argument('--repeats',type=int,default=5); p.add_argument('--max-tokens',type=int,default=256)
    asyncio.run(run(p.parse_args()))
