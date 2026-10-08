#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


REQUIRED = {"scenario", "concurrency", "request_count", "ttft_mean_ms", "ttft_p95_ms", "e2e_p50_ms", "decode_tok_s_mean", "aggregate_tok_s_mean"}


def main():
    parser = argparse.ArgumentParser(description="校验 Qwen3-VL 基准结果的完整性和合理性")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--raw", type=Path)
    args = parser.parse_args()
    with args.summary.open() as f:
        rows = list(csv.DictReader(f))
    if not rows or not REQUIRED.issubset(rows[0]):
        raise SystemExit(f"缺少汇总列：{sorted(REQUIRED - set(rows[0] if rows else []))}")
    errors = []
    for index, row in enumerate(rows, 2):
        for key in REQUIRED - {"scenario"}:
            try:
                value = float(row[key])
                if not math.isfinite(value) or value <= 0:
                    errors.append(f"第 {index} 行 {key} 非正有限值")
            except ValueError:
                errors.append(f"第 {index} 行 {key} 不是数字")
        if float(row["ttft_p95_ms"]) < float(row["ttft_mean_ms"]) * 0.5:
            errors.append(f"第 {index} 行 TTFT 分位数不合理")
    if args.raw:
        raw = [json.loads(line) for line in args.raw.read_text().splitlines() if line]
        expected = sum(int(float(row["request_count"])) for row in rows)
        if len(raw) != expected:
            errors.append(f"原始请求 {len(raw)} 条，与汇总 {expected} 条不一致")
        if any(x.get("output_tokens", 0) <= 0 for x in raw):
            errors.append("存在没有输出 Token 的请求")
    if errors:
        print("\n".join(f"ERROR: {x}" for x in errors))
        raise SystemExit(1)
    print(f"PASS: {len(rows)} 个场景通过结构、数值和原始行数校验")


if __name__ == "__main__":
    main()
