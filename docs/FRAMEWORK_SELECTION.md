# 推理框架选型：不是“云端 vs 单轮”的简单二分

你的理解抓住了常见用法，但需要修正：vLLM、SGLang、TensorRT-LLM 都能在线服务或离线推理；llama.cpp 也有 `llama-server`，支持 OpenAI 兼容接口和多轮会话。核心差异是优化目标、硬件生态与模型格式。

| 框架 | 主要优势 | 典型硬件/格式 | 对本项目的判断 |
|---|---|---|---|
| Transformers | 模型支持早、逻辑透明 | PyTorch + Safetensors | 正确性基线，不作为高并发首选 |
| vLLM | PagedAttention、continuous batching、低部署成本 | GPU、HF 权重/多种量化 | 已实测，当前主方案 |
| SGLang | RadixAttention、结构化生成、复杂多轮/Agent 调度 | GPU、通常直接读 HF 模型 | 共享前缀/Agent 流量值得 A/B |
| TensorRT-LLM | NVIDIA 专用优化、FP8/NVFP4、多 GPU | NVIDIA GPU，配置更复杂 | 官方支持 Qwen3VL；本项目未实测 |
| llama.cpp | GGUF、多后端、CPU/GPU 混合卸载、依赖轻 | CPU/CUDA/Metal/Vulkan | 适合边缘与跨硬件，也能做服务 |
| RKLLM | Rockchip NPU 转换和端侧运行 | 瑞芯微 NPU、RKLLM 格式 | 真正偏嵌入式，受芯片工具链约束 |

对单张 RTX 5090 的 Qwen3-VL-8B-Instruct，vLLM 是最平衡选择：Safetensors 无需转换，API 可直接接业务，且已有 BF16/FP8 证据。TensorRT-LLM 是下一阶段极限优化路线；SGLang 应拿真实多轮/Agent 流量验证；llama.cpp/RKLLM 属于不同约束下的路线。

本地模型是 Dense 8B。`Qwen3-VL-30B-A3B` 才是 MoE：部署会额外关注专家并行、路由负载与专家权重显存。

