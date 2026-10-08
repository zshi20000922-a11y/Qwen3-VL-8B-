# Qwen3-VL-8B-Instruct 单卡推理优化报告

## 结论摘要

本项目在 NVIDIA GeForce RTX 5090 32GB 上完成 Qwen3-VL-8B-Instruct 的 OpenAI 兼容 API 部署，并同机实测 Transformers BF16、vLLM BF16、vLLM 在线 FP8。默认保留 BF16 服务；FP8 是吞吐优先配置。

- 模型是 **Dense 多模态模型，不是 MoE**。本地 `config.json` 的架构为 `Qwen3VLForConditionalGeneration`，没有专家或 Router 配置；MoE 版本的架构名是 `Qwen3VLMoeForConditionalGeneration`。
- 1920×1080、固定输出 256 tokens 时，vLLM BF16 Decode 约 90.9 tok/s，Transformers BF16 为 71.9 tok/s，约提升 26.5%。
- vLLM FP8 在并发 1/8 下聚合吞吐分别为 136.14/948.56 tok/s，相对 BF16 的 89.53/619.63 tok/s 提升 52.1%/53.1%。
- FP8 权重加载显存由约 16.65 GiB 降至 9.94 GiB，下降 40.3%；KV Cache 容量从 65,712 增至 114,128 tokens，提升 73.7%。

## 实验范围与指标

vLLM 0.28.0 使用 PyTorch 2.13.0+cu130；Transformers 基线使用 PyTorch 2.10.0+cu128。测试覆盖 640×360、1280×720、1920×1080、3840×2160，以及并发 1/2/4/8，性能请求均固定生成 256 tokens。vLLM BF16 每场景 5 次，FP8 每场景 3 次，Transformers 每分辨率 5 次。

TTFT 为请求到首个流式 token；E2E 为请求到最后一个 token；Decode tok/s 排除首 token；Aggregate tok/s 为一批请求总输出 token 除以墙钟时间。原始记录在 `results/*/requests.jsonl`，汇总在 `summary.csv/json`；`benchmark/build_analysis.py` 生成跨后端对比，`tests/test_results.py` 校验样本完整性。

## 关键实测结果

| 配置 | 并发 | 聚合吞吐 tok/s | TTFT P50 ms | E2E P95 ms |
|---|---:|---:|---:|---:|
| vLLM BF16 | 1 | 89.53 | 32.30 | 2861.77 |
| vLLM BF16 | 8 | 619.63 | 73.09 | 3303.34 |
| vLLM FP8 | 1 | 136.14 | 27.37 | 1881.63 |
| vLLM FP8 | 8 | 948.56 | 62.84 | 2157.70 |

并发从 1 增至 8 时，BF16 聚合吞吐达到 6.92 倍，FP8 为 6.97 倍，说明 continuous batching 有效摊薄调度和内核开销；未达到线性 8 倍是因为算力/带宽饱和及请求阶段并非完全重合。

| 分辨率 | Transformers BF16 E2E P50 ms | vLLM BF16 E2E P50 ms | vLLM FP8 E2E P50 ms |
|---|---:|---:|---:|
| 640×360 | 3526.38 | 2802.23 | 1837.37 |
| 1280×720 | 3627.41 | 2815.90 | 1847.31 |
| 1920×1080 | 3857.05 | 2849.70 | 1880.81 |
| 3840×2160 | 5811.20 | 3112.66 | 2144.71 |

高分辨率主要扩大视觉 token 数与视觉编码/Prefill 成本。vLLM 对相同图片的多模态缓存会让第 2 次起明显更快，所以冷请求与缓存命中必须分开解释。例如 BF16 4K 首次 TTFT 约 2.01 秒，后续中位数约 73.1 ms；后者不能代表任意新图片。

## 优化设计

- PagedAttention/KV Cache：分页管理 KV，减少碎片并提高可承载并发。
- Continuous Batching：不同请求动态入批，提高 GPU 利用率和聚合吞吐。
- Prefix Cache 与多模态缓存：复用公共文本前缀及相同图片视觉特征。
- Chunked Prefill：长视觉/文本输入分块调度，降低长 Prefill 的阻塞。
- FlashAttention、CUDA Graph、`torch.compile`：减少注意力内存流量和 CPU/Kernel Launch 开销。
- 在线 FP8：降低权重带宽和显存，利用 Blackwell FP8 Tensor Core。

当前 FlashInfer 0.6.16 的采样器对 SM 12.0 检测异常，故设置 `VLLM_USE_FLASHINFER_SAMPLER=0`，仅让 top-k/top-p 采样回退；日志确认其余注意力与图优化仍启用。

## 数据质量与限制

- Transformers 与 vLLM 的视觉预处理存在差异；性能 prompt 还追加了实验标识，输入 token 相差约 14–16，因此跨框架 E2E 是工程近似，不是严格算子级对照。
- 分辨率实验首请求包含图像编码、JIT/shape warmup，后续命中缓存；小样本 P95 对冷请求敏感。
- FP8 仅重复 3 次，BF16 为 5 次，结果只代表本机与当前配置。
- 4 个 BF16/FP8 描述和 OCR 样例仅是冒烟检查。两种配置均有可用输出，也存在 OCR 差异；不能据此宣称“精度无损”。
- 功耗由 `nvidia-smi` 轮询，不等同于高频板级测量。

## 下一步与复现

TensorRT-LLM 官方矩阵已列出 Qwen3VL，支持 CUDA Graph、Chunked Prefill、KV Cache Reuse 及图像/视频输入；但本项目未安装实测，不能写成“TensorRT-LLM 更快”。下一阶段应在相同数据、token 预算和缓存策略下与 vLLM/SGLang A/B，并补充 TextVQA、DocVQA、MMBench 精度回归。

```bash
make install
make start
make health
make benchmark
make validate
make analysis
```

FP8 可用 `QUANTIZATION=fp8 PORT=8001 ./scripts/start_vllm.sh`。完整参数见 `VLLM_DEPLOYMENT.md`。

## 外部依据

- [vLLM 支持模型表](https://docs.vllm.ai/en/stable/models/supported_models/)：分别列出 Dense 与 MoE Qwen3-VL 架构，并注明视觉预处理差异。
- [TensorRT-LLM 支持矩阵](https://nvidia.github.io/TensorRT-LLM/latest/models/supported-models.html)：Qwen3VL 多模态功能矩阵。
- [llama.cpp 多模态文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)：`libmtmd`、GGUF 与独立 `mmproj` 的服务方式。

