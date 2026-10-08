# Qwen3-VL 不同对话长度 TTFT 实测

- 模型：`Qwen3-VL-8B-Instruct`；服务：vLLM OpenAI-compatible API
- GPU：NVIDIA GeForce RTX 5090, 595.84
- 每点：1 次预热 + 5 次有效重复；输出上限 16 tokens
- `cold_text_prefix`：首个文本 block 含唯一 nonce，不复用文本 KV；多模态 processor/image cache 仍可能命中。
- `warm_prefix`：预热后重复完全相同的对话，用于展示 Prefix Cache 对长对话续问的效果。

|缓存模式|目标历史 tokens|API 实际 prompt tokens|TTFT P50 (ms)|TTFT P95 (ms)|E2E P50 (ms)|
|---|---:|---:|---:|---:|---:|
|cold_text_prefix|0|2227|190.52|192.14|356.67|
|cold_text_prefix|512|2738|232.01|234.09|400.35|
|cold_text_prefix|1024|3251|272.81|276.59|442.99|
|cold_text_prefix|2048|4275|370.02|370.31|539.70|
|cold_text_prefix|4096|6323|563.67|564.38|738.36|
|cold_text_prefix|8192|10421|998.98|1002.05|1179.29|
|cold_text_prefix|12000|14228|1442.67|1446.58|1627.93|
|warm_prefix|0|2230|27.27|28.07|194.17|
|warm_prefix|512|2738|27.94|28.99|197.65|
|warm_prefix|1024|3253|31.36|31.73|200.59|
|warm_prefix|2048|4276|32.67|33.44|202.93|
|warm_prefix|4096|6325|35.32|37.80|210.59|
|warm_prefix|8192|10419|45.32|45.88|224.83|
|warm_prefix|12000|14230|52.73|56.29|238.57|

## 口径与限制

TTFT 从客户端发起 HTTP 请求计时，到收到首个非空流式文本 token；包含 HTTP、调度、视觉处理和 Prefill。P95 仅由少量重复样本插值得到，适合作为本机对比证据，不代表线上 SLA。横轴以服务返回的 `prompt_tokens` 为准。
