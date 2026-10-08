# 项目定位：相对 NanoLLM 的差异

“NanoLLM”常指两类项目：一类是 dusty-nv 面向 Jetson Orin 的轻量本地推理库，另一类是从零实现 Transformer 推理的教学项目。两者都适合展示底层理解，但与本项目的主问题不同。

本项目的优势不是“代码行数更多”，而是完成 NVIDIA 桌面/服务器 GPU 上的生产推理与性能闭环：加载真实 Qwen3-VL-8B 多模态权重，以 vLLM 暴露 OpenAI 兼容流式 API，并验证并发调度、PagedAttention、FlashAttention、CUDA Graph、FP8、长上下文、Prefix/多模态缓存、健康检查和监控。

|维度|NanoLLM/从零教学项目|本项目|
|---|---|---|
|主要目标|理解推理引擎或 Jetson 边缘部署|单卡生产服务与性能工程|
|模型与工作负载|通常为小模型或边缘设备工作负载|Qwen3-VL-8B 图文、多轮、并发请求|
|交付接口|本地调用/示例应用|OpenAI-compatible 流式 API、systemd、监控|
|内存与调度|教学实现或设备特定后端|Paged KV Cache、Continuous Batching、Prefix Cache|
|性能证据|单点 demo/算子学习为主|逐请求 JSONL、P50/P95、A/B、Nsight Systems/Compute|
|量化与硬件|偏 Jetson/TensorRT 或自研实现|RTX 5090 Blackwell、BF16/FP8、SM120 Kernel 证据|

简历中应将它表述为“推理部署与性能分析项目”，而不是“从零手写推理框架”。若岗位强调 CUDA/C++ 引擎开发，NanoLLM 类项目在底层实现深度上可能更直接；本项目更匹配 AI Inference/Serving、部署和性能优化岗位。二者是互补关系，不应声称本项目在所有维度都更强。

## 长对话 TTFT 证据

运行：

```bash
make benchmark-context
make validate-context
```

测量口径是 HTTP 请求发起至首个非空流式文本 token。基准固定图片、4 轮历史结构、确定性解码和 16-token 输出上限；以服务实际返回的 `prompt_tokens` 为横轴。`cold_text_prefix` 用首 block 唯一 nonce 阻止文本 KV 复用，`warm_prefix` 在一次预热后重复完全相同对话。

当前 RTX 5090 / vLLM BF16 实测：不可复用文本前缀从约 2,227 tokens 的 190.52 ms 增长到 14,228 tokens 的 1,442.67 ms；完全缓存命中时，约 14,230 tokens 为 52.73 ms。原始记录、汇总和自动报告位于 `results/context_ttft/`。

限制：冷文本前缀仍可能命中 vLLM 的多模态图片/processor cache；热前缀是理想上界；每点 5 次的 P95 是描述性统计，不应当作生产 SLA。
