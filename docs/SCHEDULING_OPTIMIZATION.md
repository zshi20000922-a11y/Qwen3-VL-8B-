# vLLM 调度层优化案例

## 最终配置

```text
max-num-batched-tokens=4096
max-num-seqs=8
enable-chunked-prefill=true
enable-prefix-caching=true
```

## Chunked Prefill 粒度

在 1 个约 12K-token 长 Prefill 到达 50 ms 后注入 4 个短输入、128-token Decode 请求，每组 1 次预热、5 次重复。

|batch token 上限|短请求 TTFT P95|短请求 TPOT P95|长请求 TTFT P50|
|---:|---:|---:|---:|
|1024|1173.30 ms|11.79 ms|1221.52 ms|
|2048|1125.18 ms|11.93 ms|1161.59 ms|
|4096|1061.53 ms|11.49 ms|1101.36 ms|
|8192|1079.01 ms|11.55 ms|1081.20 ms|

4096 相对 2048 将短请求 TTFT P95 降低 5.66%，逐批均值降低 5.73%，组间 CV 0.52%；继续增至 8192 后短请求尾延迟回升。因此选择 4096。

## max-num-seqs

固定 batch token 上限 4096，并保持同一混部负载：

|max sequences|短请求 TTFT P95|短请求 TPOT P95|长请求 TTFT P50|
|---:|---:|---:|---:|
|4|1258.57 ms|11.56 ms|1098.11 ms|
|8|1056.86 ms|11.66 ms|1103.29 ms|
|16|1051.84 ms|11.61 ms|1101.79 ms|

seqs=4 无法同时容纳 1 长+4短，短请求 P95 较 8 恶化 19.09%；8→16 仅改善 0.47%，不值得扩大调度和 KV 风险，所以保留 8。

## Admission Control

以 32 路同步突发重复 5 批，共向每种入口提供 160 个请求，TTFT SLO 为 500 ms：

|入口|接受|429|已接受 TTFT P95|SLO Goodput|接受请求达标率|
|---|---:|---:|---:|---:|---:|
|直接 vLLM|160|0|4470.07 ms|1.354 req/s|25%|
|max-inflight=2，等待 50 ms|10|150|39.53 ms|1.417 req/s|100%|

限流将已接受请求的尾延迟稳定下来，但整体接受率只有 6.25%。它是过载保护而不是容量提升，生产环境必须结合客户端指数退避、多个副本、负载均衡和可观测的 429 指标。不能只报告 100% 达标率而隐藏拒绝率。

## 结论边界

负载为单机、同步 closed-loop burst；结果适合本项目配置选择，不代表泊松到达的线上 SLA。下一步可增加恒定到达率/open-loop 测试，并用 queue-time histogram 区分入口排队与 Engine 排队。
