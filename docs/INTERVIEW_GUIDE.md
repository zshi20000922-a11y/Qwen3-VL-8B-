# AI 推理/部署面试要点

1. **TTFT 与 TPOT 有何区别？** TTFT 覆盖排队、视觉编码和 Prefill；TPOT/Decode tok/s 描述首 token 后逐 token 速度。
2. **Prefill 和 Decode 谁更 compute-bound？** 不能绝对化；大 prompt Prefill 的矩阵更大、并行度高，Decode 小批 GEMV 常偏权重带宽受限。
3. **为什么并发提高吞吐？** Continuous batching 把多个序列同一步的计算合批，提高矩阵规模和 GPU 利用率，但会增加排队与单请求 TPOT。
4. **PagedAttention 优化什么？** 以页管理 KV Cache，降低碎片和复制成本，提高可承载序列数；它不直接减少模型权重。
5. **CUDA Graph 为什么有效？** 捕获重复 Kernel 图并回放，减少 Python/CPU 调度和 launch overhead，尤其适合 Decode 的重复 shape。
6. **Roofline 怎么读？** 用 Arithmetic Intensity 对照带宽斜线和计算平台，判断增加计算峰值还是减少内存流量更可能有效。
7. **Occupancy 越高越好吗？** 不是；足够隐藏延迟即可。为追求 occupancy 降低寄存器可能导致 spill，反而变慢。
8. **FP8 为什么快？** 降低权重容量和流量，并利用 Tensor Core；但缩放、量化 Kernel 与精度回归都是额外成本。
9. **为什么 FP8 后 Attention 占比上升？** GEMM 绝对时间下降导致剩余 Attention 比例上升，是瓶颈迁移，不等于 Attention 退化。
10. **Systems 与 Compute 如何分工？** Systems 选阶段和 Top Kernel、分析 CPU/GPU overlap；Compute 对选中的 Kernel 读取硬件计数器和 Roofline。
11. **如何保证压测可信？** 固定输入/输出、分开冷/热缓存、warmup、重复、报告分位数、保留原始记录、一次只改一个变量。
12. **这个模型是 MoE 吗？** 不是；架构为 `Qwen3VLForConditionalGeneration` 且无专家/router 字段，属于 Dense 8B。

