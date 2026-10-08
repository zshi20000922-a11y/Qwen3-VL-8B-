#!/usr/bin/env python3
"""从模型配置推导 KV Cache 容量，并与 vLLM 启动日志交叉验证。"""
import argparse,json,re
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=Path("models/Qwen3-VL-8B-Instruct/config.json"))
    p.add_argument("--log",type=Path,default=Path("logs/vllm-server.log")); p.add_argument("--output",type=Path,default=Path("results/kv_capacity/REPORT.md")); a=p.parse_args()
    cfg=json.loads(a.config.read_text())["text_config"]; dtype_bytes=2
    per_token=2*cfg["num_hidden_layers"]*cfg["num_key_value_heads"]*cfg["head_dim"]*dtype_bytes
    text=a.log.read_text(); memories=re.findall(r"Available KV cache memory: ([0-9.]+) GiB",text)
    tokens=re.findall(r"GPU KV cache size: ([0-9,]+) tokens",text)
    if not memories or not tokens: raise RuntimeError("日志中找不到 KV cache 容量")
    gib=float(memories[-1]); measured_tokens=int(tokens[-1].replace(",","")); derived_tokens=int(gib*1024**3/per_token)
    error=100*(derived_tokens-measured_tokens)/measured_tokens
    rows=[]
    for length in (2048,4096,8192,16384):
        req_gib=per_token*length/1024**3; rows.append((length,req_gib,gib/req_gib,measured_tokens//length))
    lines=["# Qwen3-VL KV Cache 容量模型","",f"配置：{cfg['num_hidden_layers']} 层、{cfg['num_key_value_heads']} KV heads、head_dim={cfg['head_dim']}、BF16。","",
      f"每 token KV = `2 × layers × kv_heads × head_dim × dtype_bytes` = **{per_token:,} bytes（{per_token/1024:.0f} KiB）**。",
      f"vLLM 日志：可用 **{gib:.2f} GiB**、**{measured_tokens:,} tokens**；理论为 {derived_tokens:,} tokens，差异 {error:+.2f}%。","",
      "|上下文长度|单请求理论 KV|按 GiB 理论并发|按 block tokens 安全并发|","|---:|---:|---:|---:|"]
    lines += [f"|{n:,}|{g:.3f} GiB|{c:.2f}×|{safe}|" for n,g,c,safe in rows]
    lines += ["","说明：安全并发只按 KV token 容量取整，未包含视觉 encoder cache、运行时临时张量、CUDA Graph 和碎片；实际服务还受 `max-num-seqs` 与调度策略限制。",""]
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text("\n".join(lines)); print("\n".join(lines))
if __name__=="__main__": main()
