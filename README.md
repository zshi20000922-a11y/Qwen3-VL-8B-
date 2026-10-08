# Qwen3-VL-8B-Instruct 高性能推理部署

在单张 NVIDIA RTX 5090 32GB 上部署 Qwen3-VL-8B-Instruct，并对 Transformers 与 vLLM 的延迟、吞吐、并发扩展和视觉输入成本进行可复现实验。

该模型是 Dense 多模态模型，不是 MoE。判断依据是模型架构为 `Qwen3VLForConditionalGeneration`，配置中没有专家数量或 Router 字段。

## 项目成果入口

- `docs/AI_INFERENCE_PROJECT_REPORT.md`：完整项目总报告，包含架构、实验方法、性能结果、Nsight 算子证据、负实验、局限、证据索引和简历写法。
- `portfolio-report/`：基于同一批 reviewed 数据生成的本地交互式成果页；构建产物位于 `portfolio-report/dist/`。
- `docs/RESUME.md`：适合快速复制的简历条目与面试讲解顺序。

## 快速开始

```bash
make install
make start
make health
make client
```

服务地址为 `http://127.0.0.1:8000/v1`，兼容 OpenAI Chat Completions API。停止服务：

```bash
make stop
```

## 可复现基准

```bash
make benchmark
make validate
make benchmark-context
make validate-context
make benchmark-saturation
make validate-saturation
make benchmark-conversation
make validate-conversation
make kv-capacity
make inference-report
```

基准会执行：

- 640×360、1280×720、1920×1080、3840×2160 分辨率实验；
- 1、2、4、8 并发实验；
- 每个场景默认重复 5 次；
- 记录逐请求 TTFT、端到端延迟、输入/输出 Token、Decode 吞吐；
- 同时采样 GPU 利用率、显存和功耗；
- 保存原始 JSONL、汇总 CSV 和带实验元数据的 JSON。

长对话 TTFT 基准固定 4 轮历史对话和同一图片，覆盖 0～12K 附加历史 tokens，同时分别测量不可复用文本前缀与完全命中 Prefix Cache 的场景。实测结果和口径见 `results/context_ttft/REPORT.md`。

饱和基准覆盖并发 1～32，并同时报告 TTFT、TPOT、吞吐和 SLO Goodput；追加式会话基准直接读取 vLLM 的 token 级 Prefix Cache 计数器，避免用完全相同请求代替真实多轮聊天。

## 关键优化

- vLLM PagedAttention 与 9GiB KV Cache；
- Continuous Batching，提高多请求聚合吞吐；
- Prefix Cache，复用公共文本前缀；
- 多模态处理缓存，复用相同图片的视觉特征；
- Chunked Prefill，降低长输入阻塞；
- CUDA Graph 和 `torch.compile`；
- FlashAttention 用于语言和视觉注意力；
- BF16 保持原模型精度。

调度层通过长 Prefill/短 Decode 混部 A/B 将默认 `max-num-batched-tokens` 设为 4096；这是当前 RTX 5090 工作负载的实测平衡点，可用 `MAX_NUM_BATCHED_TOKENS` 覆盖，不视为其他模型或流量分布的通用最优值。

由于当前 FlashInfer 0.6.16 对 RTX 5090 SM 12.0 的采样 JIT 存在架构误判，启动脚本设置 `VLLM_USE_FLASHINFER_SAMPLER=0`，只回退 top-k/top-p 采样器，不关闭 FlashAttention、PagedAttention 或 CUDA Graph。

## 监控

vLLM 原生暴露 `/metrics`。可选启动 Prometheus 和 Grafana：

```bash
make monitoring-up
```

- Prometheus：<http://127.0.0.1:9090>
- Grafana：<http://127.0.0.1:3000>，用户名 `admin`，本地演示密码 `qwen-local`

生产环境请通过环境变量或 Secret 管理密码，不要使用演示凭据。

## 文档

- `docs/MODEL_FILES.md`：模型目录各文件格式、加载关系和 Dense/MoE 判断；
- `VLLM_DEPLOYMENT.md`：参数解释与初步结果；
- `docs/OPTIMIZATION_REPORT.md`：完整技术报告；
- `docs/RESUME.md`：简历表述与面试要点；
- `docs/FRAMEWORK_SELECTION.md`：框架选型与 TensorRT-LLM/量化路线。
- `docs/NSIGHT_SYSTEMS_GUIDE.md`：端到端 CUDA 时间线分析；
- `docs/NSIGHT_COMPUTE_GUIDE.md`：Kernel、Roofline 与硬件计数器；
- `docs/OPTIMIZATION_CASE_STUDY.md`：从假设、证据到优化决策；
- `docs/INTERVIEW_GUIDE.md`：AI 推理/部署面试要点。
- `docs/PROJECT_POSITIONING.md`：与 NanoLLM 类项目的定位差异与简历叙事。
- `docs/INFERENCE_ENGINEERING_CASE_STUDY.md`：并发饱和、SLO Goodput、真实多轮缓存与工程决策。
- `results/kv_capacity/REPORT.md`：KV Cache 理论模型与 vLLM 实测容量交叉验证。
- `docs/OPERATOR_AUDIT.md`：模型操作到 cuBLAS/CUTLASS/FlashAttention/Triton Kernel 的映射与优化优先级。
- `docs/SCHEDULING_OPTIMIZATION.md`：Chunked Prefill、max-num-seqs 与 Admission Control 的完整 A/B。

## Nsight 性能分析

```bash
make profile-check
PROFILE_VARIANT=bf16 PROFILE_SCENARIO=steady PROFILE_CONCURRENCY=1 make profile-nsys
make profile-export profile-compare profile-validate
```

Nsight Compute 计数器受 NVIDIA 权限控制，按 `docs/NSIGHT_COMPUTE_GUIDE.md` 使用交互式 sudo 采样。

服务参数单变量 A/B：

```bash
make ab-cuda-graph
make ab-prefix-cache
```

脚本会依次启动对照配置、重复测量并恢复默认 BF16/CUDA Graph 服务。结果保存在 `results/ab/`；低于 5% 的变化不会作为简历成果。
