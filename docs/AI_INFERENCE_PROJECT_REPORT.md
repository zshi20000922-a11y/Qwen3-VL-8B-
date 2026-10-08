# Qwen3-VL-8B 单卡多模态推理部署与性能优化项目报告

> 面向 AI Inference / AI 部署岗位的项目总报告  
> 实验平台：NVIDIA GeForce RTX 5090 32GB（Compute Capability 12.0）  
> 模型：Qwen3-VL-8B-Instruct（Dense，多模态）  
> 推理引擎：vLLM 0.28.0；默认生产配置 BF16  
> 报告口径：只引用仓库中可由原始 JSONL、CSV、JSON 或 Nsight 报告复算的结果

## 1. 项目摘要

本项目不是“把模型跑起来”的 Demo，而是一套围绕单卡多模态大模型建立的推理工程闭环：从模型文件和架构审计、引擎选型、OpenAI 兼容服务化，到量化、缓存、连续批处理、Paged KV Cache、CUDA Graph、调度和准入控制，再到 Nsight Systems 的算子级证据与质量回归。

在 RTX 5090 32GB 上，项目完成 Qwen3-VL-8B-Instruct 的 BF16/FP8 部署与可复现评测。FP8 相比 BF16，在相同 2,074-token 多模态输入与 256-token 输出下，并发 1/8 的聚合输出吞吐分别提高 52.1%/53.1%。通过 Nsight Systems 2026.2 定位 BF16 路径的 FlashAttention、BF16 GEMV/GEMM 以及 FP8 路径的 SM120 E4M3 CUTLASS Kernel，证明收益来自实际算子路径变化，而非只比较 API 延迟。

项目进一步把优化重心从“单请求速度”扩展到“线上 SLO”：长上下文实验量化 TTFT 随 Prefill 长度增长的斜率；饱和实验发现 c8 后吞吐几乎不再增长但 TTFT 急剧恶化；混部实验据此选择 `max-num-batched-tokens=4096`；过载实验用有界等待和快速拒绝保护已接收请求。最终默认配置为 BF16、16K 上下文、`max-num-seqs=8`、`max-num-batched-tokens=4096`、Prefix Cache 开启。

## 2. 项目目标与岗位能力映射

| 工程问题 | 项目回答 | 对应岗位能力 |
|---|---|---|
| 模型能否在单卡稳定部署 | vLLM OpenAI 兼容流式服务、健康检查、进程管理 | 服务部署与故障排查 |
| 算子实际走哪条路径 | Nsight Systems + Kernel 分类，区分视觉、Prefill、Decode | CUDA 性能分析 |
| FP8 是否真正加速 | BF16/FP8 同负载 A/B，结合 SM120 FP8 Kernel 证据 | 量化与硬件适配 |
| 高并发配置是否适合上线 | 用 TTFT/TPOT SLO 计算 Goodput，而非只看 tok/s | 容量规划与性能建模 |
| 长对话为什么变慢 | 测量 2.2K～14.2K prompt 的 TTFT 与 Prefix Cache 命中 | KV Cache / Prefill 理解 |
| 如何隔离长短请求干扰 | A/B Chunked Prefill 预算与最大序列数 | 调度优化 |
| 过载时如何保护尾延迟 | Admission Control、排队超时、429 快速失败 | 在线服务治理 |
| 如何保证结论可信 | warmup、重复实验、原始 JSONL、验证脚本、质量冒烟 | 实验设计与回归 |

## 3. 模型与系统架构

### 3.1 模型判断

Qwen3-VL-8B-Instruct 是 Dense Transformer，不是 MoE。判断依据是模型配置中没有专家数量、路由器、top-k expert 等 MoE 字段，FFN 使用普通稠密投影。不能因为 Qwen3 系列存在 MoE 变体，就把本模型描述成 MoE。

模型由视觉编码器、跨模态特征融合和语言模型组成。图片首先经过 resize/patch/视觉 Transformer，视觉 token 与文本 token 合并后进入语言模型 Prefill；随后按 token 进行 Decode。由此形成两类主要性能问题：视觉编码与 Prefill 决定 TTFT，逐 token Decode 决定 TPOT 和持续输出速度。

模型文件格式和职责详见 [MODEL_FILES.md](MODEL_FILES.md)。

### 3.2 服务请求路径

```text
Client
  -> OpenAI-compatible HTTP / streaming
  -> vLLM API Server
  -> EngineCore scheduler
  -> multimodal preprocessing / vision encoder
  -> chunked prefill + paged KV allocation
  -> continuous-batched decode
  -> streamed first token and subsequent tokens
```

核心机制：

- **PagedAttention / Paged KV Cache**：KV Cache 按 block 管理，降低连续大块显存预留和碎片问题，并支持前缀 block 复用。
- **FlashAttention v2**：用于视觉编码和注意力计算；Nsight Kernel 名称中可直接观察 `flash_fwd_kernel` 与 `flash_fwd_splitkv_kernel`。
- **Continuous Batching**：不同请求可以在 Decode step 动态进入和退出批次，提高 GPU 利用率。
- **Chunked Prefill**：长 Prefill 被切分并与 Decode 交错调度，避免长请求长期独占批次。
- **CUDA Graph**：复用稳定 Decode 路径的 launch 序列，减少 CPU launch overhead。
- **Prefix Cache**：复用相同 token 前缀的 KV blocks；实际收益取决于前缀命中比例。

### 3.3 为什么选择 vLLM

本项目目标是 NVIDIA 单卡上的生产式 API 和并发调度，因此以 vLLM 为主线。llama.cpp/RKLLM 更强调轻量、本地或特定端侧硬件；TensorRT-LLM 更接近 NVIDIA 深度定制和极限性能工程；SGLang 同样面向服务端，强项包括结构化生成与前缀复用调度。当前项目先把一套 vLLM 路径分析做深，避免用未经实测的跨框架结论包装成果。详细比较见 [FRAMEWORK_SELECTION.md](FRAMEWORK_SELECTION.md)。

## 4. 实验设计

### 4.1 指标定义

- **TTFT**：客户端发起 HTTP 请求到收到首个非空流式文本 token 的时间。
- **TPOT**：首 token 后每个输出 token 的平均时间，反映 Decode 体验。
- **Aggregate output tok/s**：一批请求总输出 token 除以批次 wall time。
- **Goodput**：同时满足 TTFT 与 TPOT SLO 的请求吞吐；用于容量决策。
- **P50/P95**：分别描述典型和尾延迟。
- **GPU 证据**：Kernel 时间占比、CUDA API、显存、利用率和功耗。

### 4.2 可复现原则

- 固定模型、图片、prompt、输出 token 数与采样参数。
- 每项 A/B 一次只改变一个核心变量。
- 至少 warmup 后再采有效样本；关键 A/B 使用 5 次重复。
- 保存逐请求 JSONL、汇总 CSV/JSON、服务日志与复算脚本。
- 小于 5% 或落在波动范围内的变化不写成优化成果。
- 性能变更后执行 API 健康检查与 BF16/FP8 多模态质量冒烟。

仓库目前保留 1,151 条逐请求记录；不同实验的数据口径不同，跨实验绝对值不直接拼接比较。

## 5. 核心结果

### 5.1 BF16 与 FP8

固定约 2,074 prompt tokens、输出 256 tokens：

| 精度 | 并发 | TTFT P95 | E2E P95 | 聚合输出吞吐 |
|---|---:|---:|---:|---:|
| BF16 | 1 | 35.89 ms | 2861.77 ms | 89.53 tok/s |
| FP8 | 1 | 29.65 ms | 1881.63 ms | 136.14 tok/s |
| BF16 | 8 | 85.68 ms | 3303.34 ms | 619.63 tok/s |
| FP8 | 8 | 75.68 ms | 2157.70 ms | 948.56 tok/s |

聚合吞吐提升：

- c1：`(136.14 / 89.53 - 1) = 52.1%`
- c8：`(948.56 / 619.63 - 1) = 53.1%`

FP8 还使权重加载显存下降约 40.3%，KV Cache 容量提升 73.7%。但当前只有描述/OCR 级冒烟验证，因此只能写“完成质量回归”，不能写“精度无损”。原始证据：[BF16 summary](../results/vllm_bf16/summary.json)、[FP8 summary](../results/vllm_fp8/summary.json)。

### 5.2 CUDA Graph

| 并发 | Eager | CUDA Graph | 吞吐变化 | P95 E2E 变化 | 是否可写成果 |
|---:|---:|---:|---:|---:|---|
| 1 | 92.34 tok/s | 95.43 tok/s | +3.3% | -3.2% | 否 |
| 8 | 604.78 tok/s | 635.67 tok/s | +5.1% | -3.2% | 是 |

c8 的提升达到预先设定的 5% 门槛，符合 Decode 阶段大量短 Kernel 的 launch overhead 可被图回放摊薄的预期；c1 变化不足 5%，不夸大。证据：[comparison.json](../results/ab/comparison.json)。

### 5.3 长上下文 TTFT 与 Prefix Cache

不可复用文本前缀下，Prompt 从约 2,227 增长至 14,228 tokens，TTFT P50 从 190.52 ms 增至 1,442.67 ms，拟合斜率约 104.3 ms/1K tokens。完全复用前缀时，14,230-token 请求 TTFT P50 为 52.73 ms，相对不可复用前缀下降 96.3%，约 27.4 倍。

| Prompt tokens | 不可复用前缀 TTFT P50 | 完全命中 TTFT P50 |
|---:|---:|---:|
| 2,227 | 190.52 ms | 27.27 ms |
| 4,275 | 370.02 ms | 32.67 ms |
| 6,323 | 563.67 ms | 35.32 ms |
| 10,421 | 998.98 ms | 45.32 ms |
| 14,228 | 1,442.67 ms | 52.73 ms |

完全命中是理想上界，不代表真实聊天每轮都能获得相同比例收益；不可复用文本实验仍可能命中图片 processor cache。证据：[context summary](../results/context_ttft/summary.json)、[70 条原始请求](../results/context_ttft/requests.jsonl)。

### 5.4 真实追加式多轮会话

| 轮次 | Prompt tokens | Cache hit | TTFT mean |
|---:|---:|---:|---:|
| 1 | 2,107 | 0.00% | 189.03 ms |
| 2 | 2,643 | 79.30% | 66.98 ms |
| 3 | 3,179 | 83.04% | 68.01 ms |
| 4 | 4,227 | 74.94% | 114.08 ms |
| 5 | 6,300 | 67.05% | 227.16 ms |
| 6 | 10,420 | 60.38% | 471.17 ms |

结果证明 Prefix Cache 对追加式会话有效，但随着新增历史占比增加，旧前缀在总 Prompt 中占比下降，命中率和收益随之变化。证据：[conversation summary](../results/conversation_cache/summary.json)。

### 5.5 饱和点与 SLO Goodput

SLO 定义为 TTFT P95 ≤ 500 ms、TPOT P95 ≤ 30 ms，并要求该档 100% 请求达标。

| 并发 | TTFT P95 | TPOT P95 | 输出吞吐 | Goodput | SLO 通过率 |
|---:|---:|---:|---:|---:|---:|
| 1 | 189.74 ms | 11.24 ms | 79.26 tok/s | 0.619 req/s | 100% |
| 2 | 362.06 ms | 11.72 ms | 139.11 tok/s | 1.087 req/s | 100% |
| 4 | 701.00 ms | 14.98 ms | 224.82 tok/s | 0.439 req/s | 25.00% |
| 8 | 1,402.40 ms | 20.89 ms | 329.19 tok/s | 0.321 req/s | 12.50% |
| 16 | 4,417.52 ms | 20.91 ms | 333.17 tok/s | 0.108 req/s | 4.17% |
| 32 | 10,414.73 ms | 20.98 ms | 334.67 tok/s | 0.082 req/s | 3.12% |

c8 之后输出吞吐只从 329.19 增长到 334.67 tok/s（c32），但 TTFT P95 从 1.40 s 恶化至 10.41 s；TPOT 仍低于 30 ms。由此判断瓶颈是排队和 Prefill，而非 Decode。最大稳定档是 c2，不是最大吞吐档 c32。证据：[saturation summary](../results/saturation/summary.json)。

### 5.6 KV Cache 容量规划

本模型语言层配置为 36 层、8 个 KV heads、head dimension 128，BF16 每元素 2 bytes：

```text
KV bytes/token = 2(K,V) × 36 × 8 × 128 × 2 bytes
               = 147,456 bytes
               = 144 KiB/token
```

vLLM 启动日志给出 9.03 GiB KV 池、65,712 tokens；理论值约 65,754 tokens，误差仅 0.06%。容量近似为：2K/4K/8K/16K 请求分别最多容纳 32/16/8/4 个满长序列。这里是 KV 池容量上界，不等于满足延迟 SLO 的并发量。证据：[KV capacity report](../results/kv_capacity/REPORT.md)。

## 6. Nsight 算子与瓶颈分析

### 6.1 当前算子由什么实现

| 阶段/算子 | 当前实现证据 | 主要语言/库 |
|---|---|---|
| 视觉 Attention | `flash_fwd_kernel` | FlashAttention CUDA/CUTLASS |
| 语言 Attention / Split-KV | `flash_fwd_splitkv_kernel` | FlashAttention CUDA |
| BF16 Linear | cuBLAS GEMV、CUTLASS/cuBLASLt GEMM | CUDA C++ 模板库 |
| FP8 Linear | SM120 E4M3 CUTLASS Tensor Core Kernel | CUTLASS/CUDA |
| RMSNorm/SwiGLU/点算子 | `triton_*_fused_*` | PyTorch Inductor + Triton |
| KV 管理与调度准备 | vLLM 自定义 Kernel | CUDA/Triton/Python 调度层 |

因此项目已实际使用 PagedAttention 思想和 FlashAttention Kernel；它们不是 README 中的配置名，而能从 vLLM 内部路径和 Nsight Kernel trace 中验证。完整审计见 [OPERATOR_AUDIT.md](OPERATOR_AUDIT.md)。

### 6.2 Nsight Systems 结果

BF16 稳态单请求中：视觉 `flash_fwd_kernel` 占 Kernel 时间 35.9%，主要 BF16 GEMV 合计约 28.5%，其余包括 CUTLASS BF16 GEMM、Split-KV Attention 和 Triton 融合算子。

FP8 稳态单请求中：视觉 `flash_fwd_kernel` 相对占比升至 42.3%；SM120 E4M3 CUTLASS GEMM 至少占 17.3% + 7.4%。这说明 Linear 被 FP8 加速后，瓶颈比例向未同步量化的视觉/Attention 路径迁移，下一步不应继续只盯着 Linear。

并发 8 trace 还观察到大量 Kernel launch、同步和批处理重叠；CUDA Graph 的 c8 A/B 收益与该证据一致。自动报告：[NSYS_AUTO_REPORT.md](../profiles/summary/NSYS_AUTO_REPORT.md)，结构化结果：[nsys_analysis.json](../profiles/summary/nsys_analysis.json)。

### 6.3 Nsight Compute 状态

Nsight Compute 2026.2.1 已配置，但硬件计数器采集仍被 `ERR_NVGPUCTRPERM` 阻塞，当前系统 `kernel.perf_event_paranoid=4`。因此项目目前没有可引用的 SM/Tensor Core utilization、DRAM/L2 bandwidth、occupancy、warp stall 或 Roofline 数字。报告不会用 Systems 时间占比冒充 Compute 硬件计数器结果。

解除权限后应按 Systems Top Kernel 定点采样：视觉 FlashAttention、BF16 GEMV、FP8 SM120 E4M3 GEMM 与 Decode Split-KV；先 `basic`，再对高占比 Kernel 使用 `roofline`/`detailed`。操作指南见 [NSIGHT_COMPUTE_GUIDE.md](NSIGHT_COMPUTE_GUIDE.md)。

## 7. 调度层优化

### 7.1 `max-num-batched-tokens`

构造一个约 12K-token 长 Prefill，并在 50 ms 后注入 4 个短 Decode 请求，每档 5 次重复：

| Token budget | 短请求 TTFT P95 | 短请求 TPOT P95 | 长请求 TTFT P50 |
|---:|---:|---:|---:|
| 1,024 | 1,173.30 ms | 11.79 ms | 1,221.52 ms |
| 2,048 | 1,125.18 ms | 11.93 ms | 1,161.59 ms |
| 4,096 | 1,061.53 ms | 11.49 ms | 1,101.36 ms |
| 8,192 | 1,079.01 ms | 11.55 ms | 1,081.20 ms |

4,096 相比 2,048 将短请求 TTFT P95 降低 5.66%，长请求 TTFT P50 同时降低 5.19%；8,192 对短请求略有回退。因此把 4,096 设为当前混合负载默认值。证据：[scheduler report](../results/ab/scheduler/REPORT.md)。

### 7.2 `max-num-seqs`

| 最大序列数 | 短请求 TTFT P95 | 相对 seqs=8 |
|---:|---:|---:|
| 4 | 1,258.57 ms | +19.09% |
| 8 | 1,056.86 ms | baseline |
| 16 | 1,051.84 ms | -0.47% |

4 限制过紧，16 相比 8 仅改善 0.47%，不足以构成收益，故保留 8。证据：[max-num-seqs report](../results/ab/max_num_seqs/REPORT.md)。

### 7.3 Admission Control

32 路并发、5 批、每种方案 160 个 offered requests：

| 方案 | 接受/拒绝 | 接受请求 TTFT P95 | 接受请求 SLO 通过率 | SLO Goodput |
|---|---:|---:|---:|---:|
| Direct | 160 / 0 | 4,470.07 ms | 25% | 1.354 req/s |
| Limit=2, wait=50ms | 10 / 150 | 39.53 ms | 100% | 1.417 req/s |

准入控制把已经接受的请求从秒级排队中保护出来，但以 93.75% 快速拒绝为代价。这不是“容量提升”，上线时必须配合 429、指数退避、客户端重试、队列预算或多副本扩容。证据：[admission report](../results/ab/admission/REPORT.md)。

## 8. 负实验与工程决策

负实验比“所有参数都有效”更能体现推理工程判断：

- **通用 Prefix Cache A/B**：c1 +0.1%、c8 -2.2%，当前工作负载无显著收益；保留功能用于真实追加式会话，不把该 A/B 写成性能成果。
- **Vision Encoder compile**：TTFT P50 335.22 → 333.15 ms，仅 +0.62%。
- **Vision Encoder compile + graph**：TTFT P50 335.22 → 334.61 ms，仅 +0.18%，额外约 0.4 GiB graph memory；不启用。
- **`max-num-seqs=16`**：相对 8 只改善 0.47%；没有证据支持扩大默认值。
- **CUDA Graph c1**：吞吐 +3.3%，低于 5% 门槛；只引用 c8 结果。
- **Nsight Compute**：权限未打通，不生成伪 Roofline 结论。

这些结论共同说明：优化必须由当前瓶颈、负载 shape 和 SLO 决定，而不是机械开启所有开关。

## 9. 默认部署配置与操作入口

当前默认服务参数：

```bash
--dtype bfloat16 \
--max-model-len 16384 \
--gpu-memory-utilization 0.90 \
--max-num-seqs 8 \
--max-num-batched-tokens 4096 \
--enable-prefix-caching
```

主要入口：

```bash
make start && make health
make benchmark
make benchmark-context
make benchmark-saturation
make benchmark-conversation
make ab-cuda-graph
make ab-scheduler
make ab-maxseq
make ab-admission
make profile-check
make profile-nsys
make profile-export
make profile-validate
```

## 10. 简历写法

### 10.1 推荐项目名

**Qwen3-VL-8B 单卡多模态推理服务与性能优化｜RTX 5090 / vLLM / FP8 / Nsight**

### 10.2 三条精简版

- 在 RTX 5090 32GB 上部署 Dense 多模态模型 Qwen3-VL-8B-Instruct，基于 vLLM 构建 OpenAI 兼容流式 API，集成 Paged KV Cache、Continuous Batching、Chunked Prefill、FlashAttention、CUDA Graph、Prefix Cache 与服务健康检查。
- 建立 BF16/FP8 可复现基准和 Nsight Systems 2026.2 分析链路；FP8 在并发 1/8 下将聚合输出吞吐提升 52.1%/53.1%，并从 trace 验证 SM120 E4M3 CUTLASS Kernel，定位瓶颈由 Linear 向视觉 Attention 迁移。
- 建立长上下文、SLO Goodput 与长短请求混部评测；识别 c2 为 100% SLO 最大稳定档，并将 `max-num-batched-tokens` 从 2048 调优至 4096，使短请求 TTFT P95 降低 5.66%、长请求 TTFT P50 降低 5.19%。

### 10.3 五条性能工程版

- 设计覆盖分辨率、并发 1～32、2.2K～14.2K 上下文、冷/热缓存与长短请求混部的基准体系，保存 1,151 条逐请求记录，并用验证脚本复算 TTFT、TPOT、P95、tok/s 与 Goodput。
- A/B BF16 与在线 FP8：在相同 2,074-token 多模态输入和 256-token 输出下，c1/c8 聚合吞吐提升 52.1%/53.1%；完成描述/OCR 质量冒烟并明确不宣称精度无损。
- 使用 Nsight Systems 定位 BF16 单请求中视觉 FlashAttention 占 35.9%、主要 BF16 GEMV 约 28.5%；FP8 trace 验证 SM120 E4M3 CUTLASS GEMM 后视觉 Attention 占比升至 42.3%，据此判断瓶颈迁移。
- 量化长上下文 TTFT：不可复用前缀下约 2.2K→14.2K tokens 的 TTFT P50 从 190.52 ms 增至 1,442.67 ms；真实追加会话第 2 轮 Prefix Cache 命中 79.30%，TTFT 降至 66.98 ms。
- 以 TTFT≤500 ms、TPOT≤30 ms 定义 SLO，发现 c8 后吞吐饱和而 TTFT 排队恶化；通过 Chunked Prefill 预算 A/B 和 Admission Control 完成调度优化与过载保护。

### 10.4 部署工程版替换条目

- 基于 vLLM 构建单卡多模态推理服务，提供 OpenAI 兼容流式 API、后台进程管理、端口冲突检查、健康检查、失败日志、质量回归及 Prometheus/Grafana 可观测性入口。
- 推导 BF16 KV Cache 为 144 KiB/token，并以 vLLM 启动日志 9.03 GiB/65,712 tokens 交叉验证（误差 0.06%），据此完成 2K～16K 上下文显存容量规划。
- 实现带 50 ms 排队预算的准入代理；在 32 路突发过载中将已接受请求 TTFT P95 从 4.47 s 降至 39.53 ms，同时披露 93.75% 快速拒绝率和重试/扩容要求。

## 11. 面试讲解脚本

### 11.1 90 秒版本

“我在 RTX 5090 单卡上做了 Qwen3-VL-8B 的生产式推理项目。先用 vLLM 建立 BF16 基线和 OpenAI 流式 API，再加入 FP8、Paged KV Cache、Continuous Batching、Chunked Prefill、Prefix Cache 和 CUDA Graph。项目的重点不是堆开关，而是做测量—分析—优化—回归闭环。我保存逐请求数据，用 Nsight Systems 验证实际 Kernel：BF16 中视觉 FlashAttention 和 GEMV 是主要开销，FP8 启用 SM120 E4M3 CUTLASS 后吞吐提升约 52%～53%，瓶颈转向 Attention。随后我从线上 SLO 出发测到 c8 以后吞吐饱和但 TTFT 排队爆炸，所以用长短请求混部 A/B 选择 4096 的 batch-token budget，并实现准入控制保护尾延迟。项目也保留了 Prefix Cache 通用 A/B、视觉 compile 等负结果，所有简历数字都能从 JSONL/CSV 复算。”

### 11.2 深挖问题

**为什么 c8 吞吐高但不推荐上线？**  
吞吐是资源效率，Goodput 才是满足用户体验的有效容量。c8 输出吞吐 329.19 tok/s，但 TTFT P95 1.40 s、仅 12.5% 请求满足 SLO；c2 虽只有 139.11 tok/s，却是 100% SLO 的最大稳定档。

**PagedAttention 和 FlashAttention 有什么区别？**  
PagedAttention主要解决 KV Cache 的分页管理与访问；FlashAttention主要通过 IO-aware tiling 降低 Attention 的显存读写。一个偏缓存管理，一个偏算子计算，本项目两者都使用。

**FP8 为什么加速后 Attention 占比上升？**  
占比是相对量。FP8 主要加速 Linear/GEMM，Linear 时间下降后，没有同比例加速的视觉 FlashAttention 成为更大的相对瓶颈，这是典型的 Amdahl 定律现象。

**为什么不直接把并发限制为 2？**  
`max-num-seqs` 是引擎内批容量，Admission Control 是入口流量治理，两者职责不同。真实系统还应根据到达分布、队列预算、重试和副本数动态控制，不能把一次 closed-loop burst 的 c2 当作所有业务的固定常数。

更多题目见 [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md)。

## 12. 局限与下一步

1. 当前硬件为消费级单卡，结论不能直接外推到 A100/H100/B200 或多卡张量并行。
2. BF16 与 FP8 有效重复数不完全相同；后续应在相同运行窗口、功耗和温度条件下重测。
3. 质量验证为小规模描述/OCR 冒烟，不足以证明量化精度等价；应引入 DocVQA、TextVQA、OCRBench 等数据集。
4. 饱和测试是 closed-loop burst，不等价于线上泊松到达；应增加 30～60 分钟 open-loop soak test。
5. Nsight Compute 硬件计数器权限尚未打通；解锁后补齐 Roofline、occupancy、带宽和 warp stall 证据。
6. Admission Control 使用固定阈值；下一阶段可实现基于 KV blocks、running/waiting queue、预计 Prefill tokens 的 token-aware admission。
7. 可进一步比较 prefill/decode 分离、优先级队列、aging 防饥饿和多副本负载均衡。

## 13. 证据索引

| 结论 | 原始/汇总数据 | 复算或验证入口 |
|---|---|---|
| BF16/FP8 性能 | `results/vllm_bf16`, `results/vllm_fp8` | `benchmark/run_suite.py`, `benchmark/validate_results.py` |
| CUDA Graph | `results/ab/cuda_graph`, `results/ab/eager` | `benchmark/compare_service_ab.py` |
| Prefix Cache A/B | `results/ab/prefix_on`, `results/ab/prefix_off` | `benchmark/compare_prefix_ab.py` |
| 长上下文 TTFT | `results/context_ttft` | `benchmark/context_ttft.py`, `benchmark/validate_context_ttft.py` |
| 真实多轮缓存 | `results/conversation_cache` | `benchmark/conversation_cache.py`, `benchmark/validate_conversation_cache.py` |
| SLO/饱和曲线 | `results/saturation` | `benchmark/saturation.py`, `benchmark/validate_saturation.py` |
| KV 容量 | `results/kv_capacity` | `benchmark/kv_capacity.py` |
| Vision compile | `results/ab/vision_encoder` | `benchmark/compare_vision_encoder_ab.py` |
| Scheduler token budget | `results/ab/scheduler` | `benchmark/compare_scheduler.py`, `benchmark/validate_scheduler.py` |
| max-num-seqs | `results/ab/max_num_seqs` | `benchmark/compare_maxseq.py`, `benchmark/validate_maxseq.py` |
| Admission Control | `results/ab/admission` | `benchmark/admission_overload.py`, `benchmark/validate_admission.py` |
| Nsight Systems | `profiles/raw`, `profiles/export`, `profiles/summary` | `make profile-export`, `make profile-validate` |
| Nsight Compute 权限 | `profiles/logs/ncu_permission_probe.log` | `make profile-check`, `make profile-ncu` |

## 14. 简历使用边界

可以写：FP8 吞吐提升、CUDA Graph c8 收益、长上下文 TTFT、真实缓存命中、SLO 容量点、调度 A/B、KV Cache 推导、Nsight Kernel 占比。

不要写：FP8 精度无损、TensorRT-LLM/SGLang 一定更快、模型是 MoE、并发提升等于 QPS 线性提升、Admission Control 提升系统容量、已经完成 Nsight Compute Roofline。

本项目最有价值的简历信号不是单个峰值数字，而是能够从 API 指标定位到调度行为，再定位到 CUDA Kernel，并用单变量实验和质量回归闭环验证工程决策。
