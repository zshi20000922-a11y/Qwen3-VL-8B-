#!/usr/bin/env python3
"""Deterministic HTTP workload used while profiling a vLLM server."""
import argparse, asyncio, json, subprocess
from datetime import datetime, timezone
from pathlib import Path


async def main(args):
    root = Path(__file__).resolve().parents[1]
    client = root / "scripts/benchmark_vllm.py"
    out = []
    warmups = 0 if args.scenario == "cold" else args.warmups
    if args.scenario == "prefill":
        image = root / "images/resolution_test/3840x2160.jpg"
        tokens, concurrency = min(args.output_tokens, 32), 1
    elif args.scenario == "decode":
        image = root / "images/resolution_test/640x360.jpg"
        tokens, concurrency = args.output_tokens, 1
    else:
        image = root / "images/test.jpg"
        tokens, concurrency = args.output_tokens, args.concurrency
    for i in range(warmups):
        cmd = [str(root/'.venv-vllm/bin/python'), str(client), '--base-url', args.base_url,
               '--image', str(image), '--max-tokens', '16', '--concurrency', '1', '--no-warmup']
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
    cmd = [str(root/'.venv-vllm/bin/python'), str(client), '--base-url', args.base_url,
           '--image', str(image), '--max-tokens', str(tokens), '--concurrency', str(concurrency), '--no-warmup']
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
    result = json.loads(proc.stdout)
    result.update({'scenario': args.scenario, 'warmups': warmups, 'image': str(image.relative_to(root)),
                   'generated_at': datetime.now(timezone.utc).isoformat()})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--base-url',default='http://127.0.0.1:8002')
    p.add_argument('--scenario',choices=['cold','steady','prefill','decode','concurrency'],default='steady')
    p.add_argument('--concurrency',type=int,default=1); p.add_argument('--output-tokens',type=int,default=256)
    p.add_argument('--warmups',type=int,default=1); p.add_argument('--output',required=True)
    asyncio.run(main(p.parse_args()))
