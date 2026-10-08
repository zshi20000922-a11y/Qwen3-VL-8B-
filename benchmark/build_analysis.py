#!/usr/bin/env python3
"""Build reproducible cross-backend comparison tables from benchmark artifacts."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def rows_by_scenario(summary_path: Path):
    payload = load_json(summary_path)
    rows = payload["scenarios"] if isinstance(payload, dict) else payload
    return {row["scenario"]: row for row in rows}


def field(row: dict, *names: str) -> float:
    for name in names:
        if name in row:
            return float(row[name])
    raise KeyError(f"missing any of {names}: {row.keys()}")


def pct(new: float, old: float) -> float:
    return (new / old - 1.0) * 100.0


def main() -> None:
    bf16 = rows_by_scenario(ROOT / "results/vllm_bf16/summary.json")
    fp8 = rows_by_scenario(ROOT / "results/vllm_fp8/summary.json")
    hf = rows_by_scenario(ROOT / "results/transformers_bf16/summary.json")

    concurrency = []
    for c in (1, 2, 4, 8):
        scenario = f"concurrency_{c}"
        for variant, source in (("vLLM BF16", bf16), ("vLLM FP8", fp8)):
            row = source[scenario]
            concurrency.append({
                "concurrency": c,
                "variant": variant,
                "aggregate_tok_s": round(field(row, "aggregate_tok_s_mean", "aggregate_tok_s"), 2),
                "ttft_p50_ms": round(field(row, "ttft_p50_ms"), 2),
                "e2e_p95_ms": round(field(row, "e2e_p95_ms"), 2),
            })

    resolution = []
    for size in ("640x360", "1280x720", "1920x1080", "3840x2160"):
        scenario = f"resolution_{size}"
        for variant, source in (("Transformers BF16", hf), ("vLLM BF16", bf16), ("vLLM FP8", fp8)):
            row = source[scenario]
            resolution.append({
                "resolution": size,
                "variant": variant,
                "input_tokens": int(round(field(row, "input_tokens_mean", "input_tokens"))),
                "e2e_p50_ms": round(field(row, "e2e_p50_ms"), 2),
                "decode_tok_s": round(field(row, "decode_tok_s_mean", "decode_tok_s"), 2),
            })

    cold_steady = []
    for variant, path in (
        ("vLLM BF16", ROOT / "results/vllm_bf16/requests.jsonl"),
        ("vLLM FP8", ROOT / "results/vllm_fp8/requests.jsonl"),
    ):
        records = load_jsonl(path)
        for size in ("640x360", "1280x720", "1920x1080", "3840x2160"):
            selected = [r for r in records if r["scenario"] == f"resolution_{size}"]
            cold = [r for r in selected if r["repeat"] == 0]
            steady = [r for r in selected if r["repeat"] > 0]
            cold_steady.append({
                "resolution": size,
                "variant": variant,
                "cold_ttft_ms": round(statistics.mean(r["ttft_ms"] for r in cold), 2),
                "steady_ttft_median_ms": round(statistics.median(r["ttft_ms"] for r in steady), 2),
                "steady_samples": len(steady),
            })

    metrics = {
        "model": "Qwen3-VL-8B-Instruct",
        "architecture": "dense",
        "hardware": "NVIDIA GeForce RTX 5090 32GB",
        "bf16_repeats": 5,
        "fp8_repeats": 3,
        "fixed_output_tokens": 256,
        "fp8_vs_bf16": {
            "c1_aggregate_throughput_gain_pct": round(pct(fp8["concurrency_1"]["aggregate_tok_s_mean"], bf16["concurrency_1"]["aggregate_tok_s_mean"]), 1),
            "c8_aggregate_throughput_gain_pct": round(pct(fp8["concurrency_8"]["aggregate_tok_s_mean"], bf16["concurrency_8"]["aggregate_tok_s_mean"]), 1),
            "weight_memory_reduction_pct": round((1 - 9.94 / 16.65) * 100, 1),
            "kv_capacity_gain_pct": round(pct(114128, 65712), 1),
        },
        "concurrency": concurrency,
        "resolution": resolution,
        "cold_vs_steady": cold_steady,
        "quality_smoke": {
            "cases": 4,
            "conclusion": "Both variants produced usable descriptions/OCR, but answers differ and the sample is too small for an accuracy claim.",
        },
    }

    output_dir = ROOT / "results/analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, rows in (("concurrency.csv", concurrency), ("resolution.csv", resolution), ("cold_vs_steady.csv", cold_steady)):
        with (output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(metrics["fp8_vs_bf16"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
