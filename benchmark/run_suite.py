#!/usr/bin/env python3
"""可复现的 Qwen3-VL 在线推理实验：输出逐请求 JSONL 和场景汇总 CSV/JSON。"""

import argparse
import asyncio
import base64
import csv
import json
import mimetypes
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


def percentile(values, p):
    values = sorted(values)
    if not values:
        return 0.0
    pos = (len(values) - 1) * p / 100
    lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def image_url(path):
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


async def sample_gpu(stop, samples):
    while not stop.is_set():
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits", stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        try:
            util, memory, power = [float(x.strip()) for x in stdout.decode().split(",")]
            samples.append({"gpu_util_pct": util, "memory_mib": memory, "power_w": power})
        except (ValueError, IndexError):
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.2)
        except asyncio.TimeoutError:
            pass


async def request(client, url, payload, scenario, repeat, slot):
    started = time.perf_counter()
    first = None
    usage = {}
    output = []
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
            content = choices[0].get("delta", {}).get("content") if choices else None
            if content:
                first = first or time.perf_counter()
                output.append(content)
    ended = time.perf_counter()
    output_tokens = int(usage.get("completion_tokens") or 0)
    decode_s = max(ended - (first or ended), 1e-9)
    return {
        "scenario": scenario, "repeat": repeat, "slot": slot,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "ttft_ms": ((first or ended) - started) * 1000,
        "e2e_ms": (ended - started) * 1000,
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": output_tokens,
        "decode_tok_s": output_tokens / decode_s,
        "output_chars": len("".join(output)),
    }


async def run_group(client, args, scenario, image, concurrency, repeat, unique_prompt):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url(image)}},
            {"type": "text", "text": f"{args.prompt}\n实验标识：{unique_prompt}"},
        ]}],
        "temperature": 0, "max_tokens": args.max_tokens, "stream": True,
        "stream_options": {"include_usage": True},
    }
    gpu_samples, stop = [], asyncio.Event()
    monitor = asyncio.create_task(sample_gpu(stop, gpu_samples))
    wall_start = time.perf_counter()
    rows = await asyncio.gather(*[
        request(client, f"{args.base_url}/v1/chat/completions", payload, scenario, repeat, slot)
        for slot in range(concurrency)
    ])
    wall_s = time.perf_counter() - wall_start
    stop.set()
    await monitor
    total_tokens = sum(row["output_tokens"] for row in rows)
    for row in rows:
        row.update({
            "image": str(image), "width": scenario.split("_")[-1] if scenario.startswith("resolution_") else "",
            "concurrency": concurrency, "group_wall_ms": wall_s * 1000,
            "aggregate_tok_s": total_tokens / wall_s,
            "gpu_util_mean_pct": statistics.mean(x["gpu_util_pct"] for x in gpu_samples) if gpu_samples else 0,
            "gpu_memory_peak_mib": max((x["memory_mib"] for x in gpu_samples), default=0),
            "gpu_power_mean_w": statistics.mean(x["power_w"] for x in gpu_samples) if gpu_samples else 0,
        })
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["scenario"], []).append(row)
    summaries = []
    for scenario, values in groups.items():
        group_keys = {(x["repeat"], x["group_wall_ms"], x["aggregate_tok_s"]) for x in values}
        aggregate = [x[2] for x in sorted(group_keys)]
        summaries.append({
            "scenario": scenario,
            "concurrency": values[0]["concurrency"],
            "request_count": len(values),
            "input_tokens_mean": round(statistics.mean(x["input_tokens"] for x in values), 2),
            "output_tokens_total": sum(x["output_tokens"] for x in values),
            "ttft_mean_ms": round(statistics.mean(x["ttft_ms"] for x in values), 2),
            "ttft_p50_ms": round(percentile([x["ttft_ms"] for x in values], 50), 2),
            "ttft_p95_ms": round(percentile([x["ttft_ms"] for x in values], 95), 2),
            "e2e_p50_ms": round(percentile([x["e2e_ms"] for x in values], 50), 2),
            "e2e_p95_ms": round(percentile([x["e2e_ms"] for x in values], 95), 2),
            "decode_tok_s_mean": round(statistics.mean(x["decode_tok_s"] for x in values), 2),
            "aggregate_tok_s_mean": round(statistics.mean(aggregate), 2),
            "gpu_util_mean_pct": round(statistics.mean(x["gpu_util_mean_pct"] for x in values), 2),
            "gpu_memory_peak_mib": round(max(x["gpu_memory_peak_mib"] for x in values), 1),
            "gpu_power_mean_w": round(statistics.mean(x["gpu_power_mean_w"] for x in values), 2),
        })
    return summaries


async def main_async(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout)) as client:
        (await client.get(f"{args.base_url}/health")).raise_for_status()
        warm_image = Path("images/test.jpg")
        await run_group(client, args, "warmup", warm_image, 1, -1, "warmup")
        rows = []
        for image in sorted(Path("images/resolution_test").glob("*.jpg")):
            for repeat in range(args.repeats):
                rows += await run_group(client, args, f"resolution_{image.stem}", image, 1, repeat, f"resolution-{image.stem}-{repeat}")
        for concurrency in args.concurrency:
            for repeat in range(args.repeats):
                rows += await run_group(client, args, f"concurrency_{concurrency}", warm_image, concurrency, repeat, f"concurrency-{concurrency}-{repeat}")

    summaries = summarize(rows)
    raw_path, csv_path, json_path = (args.output_dir / "requests.jsonl", args.output_dir / "summary.csv", args.output_dir / "summary.json")
    raw_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    json_path.write_text(json.dumps({"metadata": {
        "generated_at": datetime.now(timezone.utc).isoformat(), "model": args.model,
        "max_tokens": args.max_tokens, "repeats": args.repeats,
        "command": "benchmark/run_suite.py",
    }, "scenarios": summaries}, ensure_ascii=False, indent=2))
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"\n原始数据：{raw_path}\n汇总数据：{csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--prompt", default="请客观描述图片中的主要目标、场景、文字和目标状态。")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output-dir", type=Path, default=Path("results/vllm_bf16"))
    args = parser.parse_args()
    if args.repeats < 2 or any(x < 1 for x in args.concurrency):
        parser.error("repeats 至少为 2，concurrency 必须为正整数")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
