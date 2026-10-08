# Nsight Systems 实战指南

## 目标与入口

Nsight Systems 回答“时间去哪了”，而不是回答单个 Kernel 为什么慢。本项目固定使用 2026.2.1：

```bash
make profile-check
PROFILE_VARIANT=bf16 PROFILE_SCENARIO=steady PROFILE_CONCURRENCY=1 make profile-nsys
PROFILE_VARIANT=fp8 PROFILE_SCENARIO=steady PROFILE_CONCURRENCY=8 make profile-nsys
make profile-export profile-compare profile-validate
```

生产 API 是多进程；本机 Nsight 注入无法跟踪 spawn 后的 EngineCore CUDA activity，所以 Kernel 主实验使用同一 vLLM、同一模型、同一调度器的 offline single-process 模式。服务层 TTFT/P95 来自 HTTP benchmark，Kernel 时间占比来自 offline trace，二者不混为一个口径。

## 时间线阅读顺序

1. 先看 NVTX 的 `bf16:steady` / `fp8:steady` 正式窗口，排除加载和 warmup。
2. 看 CUDA HW 行是否连续工作；大片空白再向上追 CUDA API 和 CPU 线程。
3. Prefill 是少量较宽的 GEMM/Attention；Decode 是反复出现的小批 Attention、GEMV、RMSNorm/RoPE。
4. 看 CUDA Graph：重复 decode step 如果由 graph launch 驱动，CPU launch 间隙会缩短。
5. 比较 c1/c8：c8 将多个序列合批，单请求 tok/s 会下降，但聚合 tok/s 大幅增加。

## 本机证据

- BF16 c1：约 90.20 tok/s；FP8 c1：约 137.79 tok/s。
- BF16 c8：约 627.23 tok/s；FP8 c8：约 956.04 tok/s。
- BF16 c1 Top Kernel：FlashAttention 35.9%，两类 BF16 GEMV 合计约 28.5%。
- FP8 c1：FlashAttention 42.3%，SM120 FP8 CUTLASS GEMM 首类 17.3%。

FP8 加速矩阵层后，Attention 占比上升，这是典型瓶颈迁移。不能解释为 Attention 自身变慢；应查看绝对时间后再判断。

## 冷图与热图

`PROFILE_SCENARIO=cold` 不做请求级 warmup；`steady` 先用同图生成 16 tokens。`prefill` 使用 4K 图和短输出，突出视觉编码/Prefill；`decode` 使用 640×360 与长输出，突出逐 token 阶段。热图缓存命中结果只适用于同图多轮，不代表新图片。

## 常见陷阱

- PATH 中 `/usr/local/cuda-12.4/bin/nsys` 是 2023.4，不用于 RTX 5090 报告。
- 只看到 `cuModuleLoad` 而没有 Kernel，说明 CUDA activity 没注入到 EngineCore。
- Profiling 会显著扰动绝对延迟；用普通 benchmark 报性能，用 Nsight 报组成和因果。
- `kernel.perf_event_paranoid=4` 会禁用 CPU sampling/context switch；需要时临时设为 2，结束恢复 4。

