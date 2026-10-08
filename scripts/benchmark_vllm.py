#!/usr/bin/env python3
"""对 OpenAI 兼容的 Qwen3-VL 服务测量 TTFT、吞吐和并发扩展性。"""

import argparse
import asyncio
import base64
import json
import mimetypes
import statistics
import time
from pathlib import Path

import httpx


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


async def one_request(client, url, payload, request_id):
    started = time.perf_counter()
    first_token_at = None
    text_parts = []
    usage = {}
    async with client.stream("POST", url, json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            raw = line[6:]
            if raw == "[DONE]":
                break
            event = json.loads(raw)
            usage = event.get("usage") or usage
            choices = event.get("choices") or []
            if not choices:
                continue
            content = choices[0].get("delta", {}).get("content")
            if content:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                text_parts.append(content)
    ended = time.perf_counter()
    completion_tokens = int(usage.get("completion_tokens", 0))
    decode_seconds = max(ended - (first_token_at or ended), 1e-9)
    return {
        "id": request_id,
        "ttft_ms": ((first_token_at or ended) - started) * 1000,
        "e2e_ms": (ended - started) * 1000,
        "completion_tokens": completion_tokens,
        "decode_tok_s": completion_tokens / decode_seconds,
        "text": "".join(text_parts),
    }


async def run(args):
    image = image_data_url(args.image.resolve())
    payload = {
        "model": args.model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image}},
                {"type": "text", "text": args.prompt},
            ],
        }],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        health = await client.get(f"{args.base_url}/health")
        health.raise_for_status()
        if args.warmup:
            warm = dict(payload)
            warm["max_tokens"] = min(16, args.max_tokens)
            await one_request(client, f"{args.base_url}/v1/chat/completions", warm, -1)
        wall_start = time.perf_counter()
        results = await asyncio.gather(*[
            one_request(client, f"{args.base_url}/v1/chat/completions", payload, i)
            for i in range(args.concurrency)
        ])
        wall_seconds = time.perf_counter() - wall_start

    total_tokens = sum(x["completion_tokens"] for x in results)
    summary = {
        "concurrency": args.concurrency,
        "requests": len(results),
        "wall_seconds": round(wall_seconds, 4),
        "total_output_tokens": total_tokens,
        "aggregate_output_tok_s": round(total_tokens / wall_seconds, 2),
        "mean_ttft_ms": round(statistics.mean(x["ttft_ms"] for x in results), 2),
        "p50_e2e_ms": round(statistics.median(x["e2e_ms"] for x in results), 2),
        "mean_per_request_decode_tok_s": round(statistics.mean(x["decode_tok_s"] for x in results), 2),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.show_output:
        print("\n首个回答：\n" + results[0]["text"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--image", type=Path, default=Path("images/test.jpg"))
    parser.add_argument("--prompt", default="请详细描述图片中的主要目标、场景和目标状态。")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-output", action="store_true")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"图片不存在：{args.image}")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
