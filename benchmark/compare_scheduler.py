#!/usr/bin/env python3
import json
from pathlib import Path
root=Path("results/ab/scheduler"); rows=[]
for p in sorted(root.glob("tokens_*/summary.json")): rows.append(json.loads(p.read_text())["summary"])
best=min(rows,key=lambda x:x["short_ttft_p95_ms"]); base=next(x for x in rows if x["variant"]=="tokens_2048")
lines=["# Chunked Prefill 调度参数 A/B","","场景：1 个约 12K-token 长 Prefill 先到达，50 ms 后并发注入 4 个短输入、128-token Decode 请求；每组 1 次预热 + 5 次重复。","","|max-num-batched-tokens|短请求 TTFT P95 ms|短请求 TPOT P95 ms|长请求 TTFT P50 ms|相对 2048 的短 TTFT|","|---:|---:|---:|---:|---:|"]
for x in rows:
    n=x["variant"].split("_")[-1]; gain=100*(1-x["short_ttft_p95_ms"]/base["short_ttft_p95_ms"])
    lines.append(f"|{n}|{x['short_ttft_p95_ms']:.2f}|{x['short_tpot_p95_ms']:.2f}|{x['long_ttft_p50_ms']:.2f}|{gain:+.2f}%|")
lines += ["",f"当前矩阵中短请求尾延迟最低的是 `{best['variant']}`。最终配置需要同时考虑短请求保护、长请求 TTFT 和吞吐；低于 5% 的变化不写成优化成果。",""]
(root/"REPORT.md").write_text("\n".join(lines)); print("\n".join(lines))
