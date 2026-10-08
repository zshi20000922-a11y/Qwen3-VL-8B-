#!/usr/bin/env python3
"""Qwen3-VL 长对话 TTFT 基准：保留逐请求证据并区分冷/热文本前缀。"""

import argparse
import asyncio
import base64
import csv
import json
import mimetypes
import platform
import statistics
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from transformers import AutoTokenizer


def percentile(values, p):
    values = sorted(values)
    pos = (len(values) - 1) * p / 100
    lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def command_output(command):
    try:
        return subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def as_data_url(path):
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def token_material(tokenizer, count):
    if count == 0:
        return []
    seed = (
        "在多轮视觉问答中，用户逐步补充目标、约束、空间关系和历史结论。"
        "助手需要保留有效事实，忽略重复描述，并基于图片与上下文给出一致回答。"
    )
    seed_ids = tokenizer.encode(seed, add_special_tokens=False)
    return (seed_ids * (count // len(seed_ids) + 1))[:count]


def build_messages(tokenizer, image, history_tokens, cache_mode, nonce, rounds):
    # 冷模式把唯一值放在第一个 token block 内，使后续文本 KV block 不能复用。
    # 两种模式都保留相同格式和长度的编号；冷模式逐请求改变，热模式固定。
    system = f"实验请求编号 {nonce}。你是严谨的多模态助手，回答应简洁一致。"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image}},
            {"type": "text", "text": "请观察图片，后续对话会引用其中的内容。"},
        ]},
        {"role": "assistant", "content": "好的，我会结合图片和后续上下文回答。"},
    ]
    ids = token_material(tokenizer, history_tokens)
    parts = 2 * rounds
    for index in range(rounds):
        u0, u1 = (2 * index * len(ids)) // parts, ((2 * index + 1) * len(ids)) // parts
        a0, a1 = ((2 * index + 1) * len(ids)) // parts, ((2 * index + 2) * len(ids)) // parts
        messages.append({"role": "user", "content": f"第{index + 1}轮补充：{tokenizer.decode(ids[u0:u1])}"})
        messages.append({"role": "assistant", "content": f"第{index + 1}轮记录：{tokenizer.decode(ids[a0:a1])}"})
    messages.append({"role": "user", "content": "请用一句话总结图片主题，并说明回答参考了历史对话。"})
    return messages


async def one_request(client, args, messages, cache_mode, history_tokens, repeat, warmup):
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first = None
    usage, output = {}, []
    async with client.stream("POST", f"{args.base_url}/v1/chat/completions", json=payload) as response:
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
    if first is None:
        raise RuntimeError("流式响应未返回可见 token，无法计算 TTFT")
    return {
        "cache_mode": cache_mode,
        "target_history_tokens": history_tokens,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "history_rounds": args.rounds,
        "repeat": repeat,
        "warmup": warmup,
        "ttft_ms": (first - started) * 1000,
        "e2e_ms": (ended - started) * 1000,
        "output_chars": len("".join(output)),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["cache_mode"], row["target_history_tokens"]), []).append(row)
    result = []
    for (mode, target), values in groups.items():
        ttft = [x["ttft_ms"] for x in values]
        e2e = [x["e2e_ms"] for x in values]
        prompts = [x["prompt_tokens"] for x in values]
        result.append({
            "cache_mode": mode,
            "target_history_tokens": target,
            "prompt_tokens_mean": round(statistics.mean(prompts), 2),
            "prompt_tokens_min": min(prompts),
            "prompt_tokens_max": max(prompts),
            "samples": len(values),
            "ttft_mean_ms": round(statistics.mean(ttft), 2),
            "ttft_p50_ms": round(percentile(ttft, 50), 2),
            "ttft_p95_ms": round(percentile(ttft, 95), 2),
            "ttft_stdev_ms": round(statistics.stdev(ttft), 2) if len(ttft) > 1 else 0,
            "e2e_p50_ms": round(percentile(e2e, 50), 2),
            "e2e_p95_ms": round(percentile(e2e, 95), 2),
        })
    return sorted(result, key=lambda x: (x["cache_mode"], x["target_history_tokens"]))


def build_report(metadata, summaries):
    lines = [
        "# Qwen3-VL 不同对话长度 TTFT 实测", "",
        f"- 模型：`{metadata['model']}`；服务：vLLM OpenAI-compatible API",
        f"- GPU：{metadata['gpu']}",
        f"- 每点：1 次预热 + {metadata['repeats']} 次有效重复；输出上限 {metadata['max_tokens']} tokens",
        "- `cold_text_prefix`：首个文本 block 含唯一 nonce，不复用文本 KV；多模态 processor/image cache 仍可能命中。",
        "- `warm_prefix`：预热后重复完全相同的对话，用于展示 Prefix Cache 对长对话续问的效果。", "",
        "|缓存模式|目标历史 tokens|API 实际 prompt tokens|TTFT P50 (ms)|TTFT P95 (ms)|E2E P50 (ms)|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for x in summaries:
        lines.append(f"|{x['cache_mode']}|{x['target_history_tokens']}|{x['prompt_tokens_mean']:.0f}|{x['ttft_p50_ms']:.2f}|{x['ttft_p95_ms']:.2f}|{x['e2e_p50_ms']:.2f}|")
    lines += ["", "## 口径与限制", "", "TTFT 从客户端发起 HTTP 请求计时，到收到首个非空流式文本 token；包含 HTTP、调度、视觉处理和 Prefill。P95 仅由少量重复样本插值得到，适合作为本机对比证据，不代表线上 SLA。横轴以服务返回的 `prompt_tokens` 为准。", ""]
    return "\n".join(lines)


async def main_async(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    image = as_data_url(args.image)
    effective, warmup_rows = [], []
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout)) as client:
        (await client.get(f"{args.base_url}/health")).raise_for_status()
        model_info = (await client.get(f"{args.base_url}/v1/models")).json()
        for target in args.history_tokens:
            for mode in ("cold_text_prefix", "warm_prefix"):
                warm_nonce = uuid.uuid4().hex
                warm_messages = build_messages(tokenizer, image, target, mode, warm_nonce, args.rounds)
                warmup_rows.append(await one_request(client, args, warm_messages, mode, target, -1, True))
                if warmup_rows[-1]["prompt_tokens"] + args.max_tokens > model_info["data"][0]["max_model_len"]:
                    raise ValueError(f"上下文 {warmup_rows[-1]['prompt_tokens']} + 输出超过服务 max_model_len")
                for repeat in range(args.repeats):
                    nonce = uuid.uuid4().hex if mode == "cold_text_prefix" else warm_nonce
                    messages = build_messages(tokenizer, image, target, mode, nonce, args.rounds)
                    effective.append(await one_request(client, args, messages, mode, target, repeat, False))

    summaries = summarize(effective)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_path": str(args.model_path.resolve()),
        "model_endpoint": model_info,
        "gpu": command_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        "python": platform.python_version(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "base_url": args.base_url,
        "image": str(args.image.resolve()),
        "history_tokens": args.history_tokens,
        "rounds": args.rounds,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "ttft_definition": "HTTP request start to first non-empty streamed text token",
    }
    (args.output_dir / "requests.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in effective))
    (args.output_dir / "warmups.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in warmup_rows))
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    (args.output_dir / "summary.json").write_text(json.dumps({"metadata": metadata, "scenarios": summaries}, ensure_ascii=False, indent=2))
    (args.output_dir / "REPORT.md").write_text(build_report(metadata, summaries))
    print(build_report(metadata, summaries))
    print(f"原始记录与报告：{args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="不同长度多轮对话的 TTFT 基准")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--model-path", type=Path, default=Path("models/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--image", type=Path, default=Path("images/test.jpg"))
    parser.add_argument("--history-tokens", type=int, nargs="+", default=[0, 512, 1024, 2048, 4096, 8192, 12000])
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output-dir", type=Path, default=Path("results/context_ttft"))
    args = parser.parse_args()
    if args.repeats < 2 or args.rounds < 1 or any(x < 0 for x in args.history_tokens):
        parser.error("repeats 至少为 2、rounds 至少为 1，history-tokens 不能为负")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
