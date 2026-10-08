#!/usr/bin/env python3
import json
from pathlib import Path
sat=json.loads(Path("results/saturation/summary.json").read_text())
conv=json.loads(Path("results/conversation_cache/summary.json").read_text())
s=sat["scenarios"]; turns=conv["turns"]
peak=max(s,key=lambda x:x["aggregate_output_tok_s"]); eligible=[x for x in s if x["slo_pass_rate_pct"]==100]
best=max(eligible,key=lambda x:x["concurrency"])
lines=["# AI Inference 工程案例：Qwen3-VL 单卡容量与缓存分析","",
"## 实验结论","",
f"- 在 RTX 5090 / vLLM BF16 上，冷文本多模态请求的输出吞吐从 c1 的 {s[0]['aggregate_output_tok_s']:.2f} tok/s 增长到 c{peak['concurrency']} 的 {peak['aggregate_output_tok_s']:.2f} tok/s。",
f"- c8 后吞吐接近饱和；c32 TTFT P95 达 {s[-1]['ttft_p95_ms']:.2f} ms，说明继续堆并发主要增加排队，而非有效吞吐。",
f"- 在 TTFT P95≤{sat['metadata']['ttft_slo_ms']:.0f} ms、TPOT P95≤{sat['metadata']['tpot_slo_ms']:.0f} ms 且 100% 请求达标的口径下，最大稳定档为 c{best['concurrency']}，Goodput {best['goodput_rps']:.3f} req/s。",
f"- 真实追加式会话第 2 轮命中 {turns[1]['cache_hit_rate_mean_pct']:.2f}% prompt tokens，TTFT 从首轮 {turns[0]['ttft_mean_ms']:.2f} ms 降至 {turns[1]['ttft_mean_ms']:.2f} ms；随着每轮新增上下文增多，第 6 轮命中率降至 {turns[-1]['cache_hit_rate_mean_pct']:.2f}%，TTFT 回升至 {turns[-1]['ttft_mean_ms']:.2f} ms。","",
"## 饱和曲线","","|并发|TTFT P95 ms|TPOT P95 ms|吞吐 tok/s|Goodput req/s|SLO 通过率|","|---:|---:|---:|---:|---:|---:|"]
lines += [f"|{x['concurrency']}|{x['ttft_p95_ms']:.2f}|{x['tpot_p95_ms']:.2f}|{x['aggregate_output_tok_s']:.2f}|{x['goodput_rps']:.3f}|{x['slo_pass_rate_pct']:.2f}%|" for x in s]
lines += ["","## 真实多轮缓存","","|轮次|Prompt tokens|缓存命中率|TTFT mean ms|","|---:|---:|---:|---:|"]
lines += [f"|{x['turn']}|{x['prompt_tokens_mean']:.0f}|{x['cache_hit_rate_mean_pct']:.2f}%|{x['ttft_mean_ms']:.2f}|" for x in turns]
lines += ["","## 工程判断","","1. 最大吞吐点不是推荐服务点；需要以业务 SLO 计算 Goodput。","2. 当前服务 `max-num-seqs=8`，c16/c32 产生明显排队；可用 admission control 将并发限制在容量范围内。","3. Prefix Cache 对追加式会话确实有效，但命中比例取决于旧前缀占总 prompt 的比例，不能用完全相同请求的 96.3% 收益代表真实聊天。","4. TPOT 在高并发下仍低于 30 ms，本次 SLO 首先由 TTFT/排队击穿，因此下一优化方向应是调度和 Prefill 隔离，而不是 Decode Kernel。","","## 口径与限制","","负载固定同一图片、唯一首段文本前缀和 128 个输出 tokens；图片 processor cache 可能命中。并发测试采用同步 closed-loop burst，适合寻找容量边界，但不等价于线上泊松到达流量。每档 3 批，尾延迟仍需更长时间压测后才能作为生产 SLA。",""]
Path("docs/INFERENCE_ENGINEERING_CASE_STUDY.md").write_text("\n".join(lines))
print("wrote docs/INFERENCE_ENGINEERING_CASE_STUDY.md")
