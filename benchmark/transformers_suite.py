#!/usr/bin/env python3
"""Transformers BF16 单请求基线；与 vLLM 使用相同图片和输出长度。"""

import argparse
import csv
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_vl_baseline import build_messages, prepare_inputs, run_benchmark, warmup_model


def percentile(values, p):
    values = sorted(values)
    pos = (len(values) - 1) * p / 100
    lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("results/transformers_bf16"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用")

    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda").eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    raw, summary = [], []
    prompt = "请客观描述图片中的主要目标、场景、文字和目标状态。"
    for image in sorted(Path("images/resolution_test").glob("*.jpg")):
        prepared = prepare_inputs(processor, build_messages(image.resolve(), prompt), model.device)
        inputs = prepared["inputs"]
        warmup_model(model, inputs, 16)
        rows = []
        for repeat in range(args.repeats):
            result = run_benchmark(model, processor, inputs, args.max_tokens)
            row = {
                "scenario": f"resolution_{image.stem}", "repeat": repeat,
                "input_tokens": result["prompt_tokens"], "output_tokens": result["output_tokens"],
                "ttft_ms": result["ttft"] * 1000, "e2e_ms": result["e2e_latency"] * 1000,
                "decode_tok_s": result["decode_throughput"], "e2e_tok_s": result["e2e_throughput"],
                "itl_p99_ms": result["itl_p99"] * 1000,
                "peak_memory_mib": result["peak_allocated"] / 1024**2,
                "preprocess_ms": prepared["preprocess_time"] * 1000,
                "transfer_ms": prepared["transfer_time"] * 1000,
            }
            rows.append(row); raw.append(row)
        summary.append({
            "scenario": rows[0]["scenario"], "request_count": len(rows),
            "input_tokens_mean": round(statistics.mean(x["input_tokens"] for x in rows), 2),
            "output_tokens_total": sum(x["output_tokens"] for x in rows),
            "ttft_mean_ms": round(statistics.mean(x["ttft_ms"] for x in rows), 2),
            "ttft_p50_ms": round(percentile([x["ttft_ms"] for x in rows], 50), 2),
            "ttft_p95_ms": round(percentile([x["ttft_ms"] for x in rows], 95), 2),
            "e2e_p50_ms": round(percentile([x["e2e_ms"] for x in rows], 50), 2),
            "e2e_p95_ms": round(percentile([x["e2e_ms"] for x in rows], 95), 2),
            "decode_tok_s_mean": round(statistics.mean(x["decode_tok_s"] for x in rows), 2),
            "e2e_tok_s_mean": round(statistics.mean(x["e2e_tok_s"] for x in rows), 2),
            "itl_p99_mean_ms": round(statistics.mean(x["itl_p99_ms"] for x in rows), 2),
            "peak_memory_mib": round(max(x["peak_memory_mib"] for x in rows), 1),
        })

    (args.output_dir / "requests.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in raw))
    with (args.output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    (args.output_dir / "summary.json").write_text(json.dumps({"metadata": {
        "generated_at": datetime.now(timezone.utc).isoformat(), "framework": "transformers",
        "torch": torch.__version__, "cuda": torch.version.cuda, "load_seconds": load_seconds,
        "repeats": args.repeats, "max_tokens": args.max_tokens,
    }, "scenarios": summary}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
