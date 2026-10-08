# Qwen3-VL 推理优化案例：从吞吐数字到 Kernel 证据

## 问题

Transformers 能运行模型，但无法充分服务并发请求。目标是在 RTX 5090 32GB 上提高 Qwen3-VL-8B 的吞吐与容量，并解释优化来自哪里。

## 测量设计

服务基准固定图片、prompt 和 256 输出 tokens，记录 TTFT、E2E、Decode/聚合吞吐、显存和功耗。Nsight Systems 使用相同 vLLM 模型的单进程 offline 路径采 CUDA/NVTX，排除 HTTP 与 profiler 多进程注入差异。BF16/FP8 分别测 c1/c8。

## 发现

- 服务基准：FP8 相比 BF16 在 c1/c8 聚合吞吐提升 52.1%/53.1%；权重加载显存下降约 40.3%，KV 容量提升 73.7%。
- Nsight 同口径方向一致：c1 从 90.20 到 137.79 tok/s，c8 从 627.23 到 956.04 tok/s。
- BF16 c1 中 FlashAttention 占 Kernel 时间 35.9%，主要 BF16 GEMV 合计约 28.5%。
- FP8 c1 出现 SM120 E4M3 CUTLASS GEMM，首类占 17.3%；FlashAttention 占比升至 42.3%。
- CUDA Graph 单变量 A/B（每场景 5 次）显示：c1 吞吐提升 3.3%，未达到成果阈值；c8 从 604.78 提升至 635.67 tok/s，提升 5.1%。
- Prefix Cache 在当前同图、固定模板负载中没有显著收益：c1 仅 +0.1%，c8 为 -2.2%，所以不将其写成优化成果。

## 推断与决策

证据支持 FP8 通过减少权重存储/带宽并使用 Blackwell FP8 GEMM 加速 Linear 层。GEMM 加速后 Attention 成为更大比例的剩余成本，因此下一轮优化优先考虑请求合批、视觉输入治理、缓存和 Attention 路径，而不是盲目重写 GEMM。

并发从 1 到 8 使 BF16/FP8 聚合吞吐约提升 6.95/6.94 倍，说明 continuous batching 有效；不是 8 倍线性扩展，因为 GPU 已接近饱和且请求阶段不能完全重叠。

CUDA Graph 的收益集中在 c8，符合 Decode 阶段重复调度、CPU launch overhead 被摊薄的机制。Prefix Cache 的负结果说明当前请求的可复用前缀没有带来足以覆盖哈希和缓存管理的收益；保留开关是为了未来真实多轮流量验证，而不是把它当作默认必然加速项。

## 限制

Nsight 会扰动绝对时间；性能数字以非 profiling 服务基准为主。FP8 质量仅完成 4 个冒烟样例，不能声称精度无损。普通用户的 NCU 硬件计数器被 `ERR_NVGPUCTRPERM` 阻止，Roofline 需要执行文档中的交互式 sudo 命令后补齐。

## 可审核产物

- `profiles/raw/*.nsys-rep`：有效 CUDA Trace，不提交 Git。
- `profiles/export/*cuda_gpu_kern_sum.csv`：Kernel 明细。
- `profiles/summary/nsys_analysis.json`：机器可读分析。
- `profiles/summary/NSYS_AUTO_REPORT.md` 与 `ab_comparison.md`：可读报告。
- `results/ab/comparison.json` 与 `prefix_comparison.json`：服务参数单变量 A/B。
