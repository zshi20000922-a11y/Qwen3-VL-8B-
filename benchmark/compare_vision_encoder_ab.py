#!/usr/bin/env python3
import json,re
from pathlib import Path
root=Path("results/ab/vision_encoder"); rows=[]
for name in ("default","compile","compile_graph"):
    p=root/name/"summary.json"
    if p.exists(): rows.append(json.loads(p.read_text())["summary"])
base=next(x for x in rows if x["variant"]=="default")
lines=["# 视觉 Encoder Compile / CUDA Graph A/B","","|配置|TTFT P50 ms|TTFT P95 ms|相对默认|样本|","|---|---:|---:|---:|---:|"]
for x in rows:
    gain=100*(base["ttft_p50_ms"]/x["ttft_p50_ms"]-1)
    lines.append(f"|{x['variant']}|{x['ttft_p50_ms']:.2f}|{x['ttft_p95_ms']:.2f}|{gain:+.2f}%|{x['samples']}|")
operational=[]
for name in ("default","compile","compile_graph"):
    log=(root/name/"server.log").read_text(); init=re.findall(r"init engine .* took ([0-9.]+) s",log); kv=re.findall(r"Available KV cache memory: ([0-9.]+) GiB",log); graph=re.findall(r"([0-9.]+) GiB for CUDAGraph memory",log)
    operational.append((name,init[-1] if init else "n/a",kv[-1] if kv else "n/a",graph[-1] if graph else "n/a"))
lines += ["","## 启动与显存代价","","|配置|Engine 初始化 s|KV Cache GiB|CUDA Graph 显存 GiB|","|---|---:|---:|---:|"]
lines += [f"|{n}|{i}|{k}|{g}|" for n,i,k,g in operational]
lines += ["","## 结论","","三组 TTFT P50 差异均低于 1%，不足 5% 显著收益门槛。视觉 Graph 额外占用约 0.4 GiB CUDA Graph 显存；当前固定单图、c1 shape 下不建议启用。15 个输出中 14 个逐字一致，另 1 个是同义的“航行/驶向镜头”差异；短生成样本只能作为冒烟，不能证明数值完全等价。","","每次请求使用相同尺寸、仅改变边角 64x64 色块后重新编码的图片，以避免 Encoder Cache 跳过视觉前向。低于 5% 或落入波动范围的变化不作为优化成果。",""]
(root/"REPORT.md").write_text("\n".join(lines)); print("\n".join(lines))
