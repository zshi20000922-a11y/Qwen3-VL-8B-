#!/usr/bin/env python3
"""校验长对话 TTFT 原始证据与汇总是否一致。"""

import argparse
import csv
import json
import math
from pathlib import Path

from context_ttft import percentile


def close(actual, expected, tolerance=0.02):
    return math.isclose(float(actual), float(expected), abs_tol=tolerance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path, nargs="?", default=Path("results/context_ttft"))
    args = parser.parse_args()
    rows = [json.loads(x) for x in (args.output_dir / "requests.jsonl").read_text().splitlines() if x]
    summaries = list(csv.DictReader((args.output_dir / "summary.csv").open()))
    metadata = json.loads((args.output_dir / "summary.json").read_text())["metadata"]
    assert rows and summaries, "原始数据或汇总为空"
    assert all(x["ttft_ms"] > 0 and x["prompt_tokens"] > 0 for x in rows), "存在非正指标"
    assert all(x["completion_tokens"] > 0 for x in rows), "存在无输出请求"
    expected = len(metadata["history_tokens"]) * 2 * metadata["repeats"]
    assert len(rows) == expected, f"请求数 {len(rows)} != {expected}"
    for summary in summaries:
        group = [x for x in rows if x["cache_mode"] == summary["cache_mode"] and x["target_history_tokens"] == int(summary["target_history_tokens"])]
        assert len(group) == int(summary["samples"])
        assert close(summary["ttft_p50_ms"], round(percentile([x["ttft_ms"] for x in group], 50), 2))
        assert close(summary["ttft_p95_ms"], round(percentile([x["ttft_ms"] for x in group], 95), 2))
    for mode in ("cold_text_prefix", "warm_prefix"):
        ordered = sorted((x for x in summaries if x["cache_mode"] == mode), key=lambda x: int(x["target_history_tokens"]))
        actual = [float(x["prompt_tokens_mean"]) for x in ordered]
        assert all(b > a for a, b in zip(actual, actual[1:])), f"{mode} prompt tokens 未严格递增"
    print(f"PASS: {len(rows)} 条有效请求、{len(summaries)} 个汇总点；P50/P95 可由原始记录复算。")


if __name__ == "__main__":
    main()
