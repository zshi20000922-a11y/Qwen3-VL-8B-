# Qwen3-VL KV Cache 容量模型

配置：36 层、8 KV heads、head_dim=128、BF16。

每 token KV = `2 × layers × kv_heads × head_dim × dtype_bytes` = **147,456 bytes（144 KiB）**。
vLLM 日志：可用 **9.03 GiB**、**65,712 tokens**；理论为 65,754 tokens，差异 +0.06%。

|上下文长度|单请求理论 KV|按 GiB 理论并发|按 block tokens 安全并发|
|---:|---:|---:|---:|
|2,048|0.281 GiB|32.11×|32|
|4,096|0.562 GiB|16.05×|16|
|8,192|1.125 GiB|8.03×|8|
|16,384|2.250 GiB|4.01×|4|

说明：安全并发只按 KV token 容量取整，未包含视觉 encoder cache、运行时临时张量、CUDA Graph 和碎片；实际服务还受 `max-num-seqs` 与调度策略限制。
