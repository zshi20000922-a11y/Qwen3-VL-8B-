# 简历项目写法与面试材料

## 推荐标题

**Qwen3-VL-8B 单卡高性能推理服务与量化评测（RTX 5090 / vLLM / FP8）**

## 可直接写进简历

在单张 NVIDIA RTX 5090 32GB 上完成 Dense 多模态模型 Qwen3-VL-8B-Instruct 的生产化推理部署，基于 vLLM 提供 OpenAI 兼容流式 API，集成 systemd 守护、健康检查及 Prometheus/Grafana 可观测性。

设计可复现基准，覆盖 4 档图像分辨率、1/2/4/8 并发、TTFT/P50/P95、Decode/聚合吞吐、显存、GPU 利用率和功耗；保留逐请求 JSONL 与汇总 CSV/JSON，并以自动化测试校验 172 条性能请求的完整性。

应用 PagedAttention、Continuous Batching、Prefix/多模态缓存、Chunked Prefill、FlashAttention、CUDA Graph 和在线 FP8；实测 FP8 相比 vLLM BF16 在并发 1/8 下聚合吞吐提升 52.1%/53.1%，权重加载显存下降约 40.3%，KV Cache 容量提升 73.7%。

建立 Transformers BF16 基线与 BF16/FP8 描述/OCR 冒烟回归，区分冷启动/JIT、图像缓存命中和稳态指标，并记录预处理、样本量与量化精度限制。

使用 Nsight Systems 2026.2 对 BF16/FP8、并发 1/8 的推理路径进行 CUDA/NVTX 分析；定位 BF16 稳态单请求中 FlashAttention 占 Kernel 时间 35.9%、主要 BF16 GEMV 约 28.5%，并验证 FP8 使用 SM120 E4M3 CUTLASS Kernel 后瓶颈向 Attention 迁移。

采用单变量、5 次重复 A/B 验证 CUDA Graph：并发 8 聚合吞吐由 604.78 提升至 635.67 tok/s（+5.1%）；同时验证 Prefix Cache 在当前工作负载中无显著收益并拒绝将其包装为优化成果。

构建多模态长对话 TTFT 基准，覆盖 API 实际约 2,227～14,230 prompt tokens、每点 5 次有效重复；不可复用文本前缀下 TTFT P50 从 190.52 ms 增至 1,442.67 ms（约 104.3 ms/1K tokens），完全命中 Prefix Cache 时 14K 上下文 TTFT 降至 52.73 ms（下降 96.3%），70 条原始请求均可由校验脚本复算。

构建并发 1/2/4/8/16/32 的容量与 SLO Goodput 测试：输出吞吐在 c8 后由 329.19 tok/s 逐步饱和至 c32 的 334.67 tok/s，而 TTFT P95 从 1.40 s 恶化至 10.41 s；在 TTFT P95≤500 ms、TPOT P95≤30 ms 且 100% 请求达标的口径下，将 c2（1.087 req/s）识别为最大稳定服务档，而非以最大吞吐配置上线。

基于 vLLM token 级计数器验证真实追加式多轮会话：第二轮 Prefix Cache 命中率 79.30%，TTFT 从 189.03 ms 降至 66.98 ms；推导 BF16 KV Cache 为 144 KiB/token，并以启动日志的 9.03 GiB、65,712 tokens 交叉验证，理论误差仅 0.06%。

构造 12K-token 长 Prefill 与 4 路短 Decode 混部干扰，A/B `max-num-batched-tokens=1024/2048/4096/8192`；将 4096 确定为当前负载最优点，短请求 TTFT P95 较 2048 从 1,125.18 ms 降至 1,061.53 ms（-5.66%），且长请求 TTFT 同步下降 5.19%，据此更新默认调度配置。

验证 `max-num-seqs=4/8/16`，定位 seqs=4 因容量不足使短请求 TTFT P95 恶化 19.09%，而 8→16 仅改善 0.47%，保留 seqs=8；实现带 50 ms 排队超时的 Admission Control，在 32 路×5 批过载测试中将已接受请求 TTFT P95 从 4.47 s 降至 39.53 ms，并完整披露 93.75% 快速拒绝率与客户端重试要求。

## 面试讲解顺序

1. 选 vLLM：HF 权重直载、API 完整、单卡并发强、交付成本低。
2. 两阶段推理：视觉编码/Prefill 决定 TTFT，Decode 决定持续输出速度；4K 首先伤害前者。
3. 并发收益：continuous batching 合并不同请求的 decode step，聚合吞吐提高但单请求延迟可能增加。
4. FP8 收益：权重带宽和占用降低，Blackwell Tensor Core 更适配；仍须任务级精度回归。
5. Dense 判断：架构名、无专家字段、普通 Dense FFN；不能因 Qwen3 有 MoE 变体而误判。
6. 主动讲限制：热图不代表新图性能，样本量小，OCR 冒烟不能证明精度等价。
7. 长上下文：冷文本前缀反映 Prefill 随序列增长的成本；热前缀体现 Paged KV block 复用，但不等价于所有真实续问都能达到完全命中。
8. 容量规划：吞吐最大点不等于线上最佳点；本次高并发首先击穿 TTFT，TPOT 仍满足阈值，因此瓶颈是排队/Prefill，不是 Decode。
9. KV Cache：从 GQA 配置推导每 token 144 KiB；16K 请求理论需要 2.25 GiB KV，当前 9.03 GiB 池最多约 4 个满长请求。

不要写“FP8 精度无损”“TensorRT-LLM 比 vLLM 快”“8B 使用 MoE”或“QPS 提升 8 倍”；当前证据分别不支持这些结论。
