#!/usr/bin/env python3
import json
from pathlib import Path
root=Path("results/ab/max_num_seqs"); rows=[]
for p in sorted(root.glob("seqs_*/summary.json"),key=lambda p:int(p.parent.name.split("_")[-1])): rows.append(json.loads(p.read_text())["summary"])
base=next(x for x in rows if x["variant"]=="seqs_8")
lines=["# max-num-seqs 混部调度 A/B","","固定 `max-num-batched-tokens=4096`，负载为 1 个 12K 长 Prefill + 4 个短 Decode。","","|max-num-seqs|短 TTFT P95 ms|短 TPOT P95 ms|长 TTFT P50 ms|相对 seqs=8|","|---:|---:|---:|---:|---:|"]
for x in rows:
 n=x['variant'].split('_')[-1]; gain=100*(1-x['short_ttft_p95_ms']/base['short_ttft_p95_ms']); lines.append(f"|{n}|{x['short_ttft_p95_ms']:.2f}|{x['short_tpot_p95_ms']:.2f}|{x['long_ttft_p50_ms']:.2f}|{gain:+.2f}%|")
lines += ["","低于 5% 或未超出逐批波动的变化不作为成果。`max-num-seqs` 还受 KV 容量约束，不能仅按本场景延迟选择。",""]
(root/"REPORT.md").write_text("\n".join(lines)); print("\n".join(lines))
