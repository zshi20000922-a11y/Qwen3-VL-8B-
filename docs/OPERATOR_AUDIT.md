# Qwen3-VL-8B 推理算子审计与优化空间

## 结论

当前运行栈不是单一后端，而是 Python 模型编排、PyTorch/Inductor、vLLM 自定义 C++/CUDA、FlashAttention、cuBLAS 与 CUTLASS 的组合。下表以当前 vLLM 0.28.0 启动日志、已安装源码和 Nsight Systems Kernel 名称交叉确认；不能仅凭 Python 层的类名判断最终 Kernel。

|模型操作|Python/vLLM 表达|当前实际后端或 Kernel|证据与判断|
|---|---|---|---|
|语言 Attention|`vllm::unified_attention_with_output`，Paged KV Cache|vLLM 内置 FlashAttention v2；Decode 出现 `flash_fwd_splitkv`|日志明确选择 `FLASH_ATTN`/FA2；Nsight 有 split-KV Kernel|
|视觉 Attention|`MMEncoderAttention`|FlashAttention v2，CUTLASS C++/CUDA 模板|单请求报告中 `flash_fwd_kernel` 占 35.9%，54 次约对应 27 层视觉块 × 2 次运行|
|BF16 Linear|`QKVParallelLinear`、`MergedColumnParallelLinear`、`RowParallelLinear`|Decode 以 cuBLAS `gemvx` 为主；较大 shape 使用 CUTLASS/cuBLASLt Tensor Core GEMM|BF16 报告中 GEMV 23.5%+5.0%，多种 CUTLASS BF16 GEMM|
|FP8 Linear|vLLM online FP8 quantization linear|SM120 CUTLASS E4M3 Tensor Core GEMM|FP8 报告中 SM120 `float_e4m3_t` CUTLASS Kernel 占 17.3%+，不是模拟量化|
|Q/K RMSNorm|vLLM `RMSNorm` IR op|编译路径被 Inductor 生成/融合为 Triton reduction；eager 可走 vLLM C++/CUDA op|报告含 `triton_red_fused_*rms_norm*`；配置为 Inductor compile|
|Residual + RMSNorm|`fused_add_rms_norm`|Triton 融合 Kernel（当前编译图）|报告含 `triton_red_fused_fused_add_rms_norm_*`|
|RoPE/MRoPE|vLLM rotary embedding custom op|vLLM C++/CUDA 或被编译图包围；当前没有 QK-Norm+RoPE 全融合|启动配置 `enable_qk_norm_rope_fusion=False`|
|SwiGLU|`SiluAndMul`，原生实现为 vLLM `_C.silu_and_mul`|当前编译图中出现 Triton pointwise 融合|报告含 `triton_poi_fused_mul_silu_slice_*`|
|视觉 LayerNorm/GELU|PyTorch `nn.LayerNorm` / `nn.GELU`|ATen/Inductor 生成的 elementwise/reduction Kernel|视觉 encoder 当前未整体 compile，局部仍可能经公共路径融合|
|Patch Embedding|vLLM `Conv3dLayer`|PyTorch CUDA/cuDNN 或 GEMM lowering|当前占比不在 Top Kernels，优先级低于视觉 Attention/GEMM|
|Top-k/Top-p Sampling|vLLM/PyTorch sampler|未使用 FlashInfer sampler|因 SM120 JIT 兼容问题显式设置 `VLLM_USE_FLASHINFER_SAMPLER=0`|

`PagedAttention` 更准确地说是 KV Cache 的分页布局与调度机制，不一定出现一个名字就叫 `paged_attention` 的 Kernel。当前 FA2 unified attention 读取 vLLM 分页 KV blocks，Nsight 中 Decode 体现为 `flash_fwd_splitkv` 系列。

## 已观察到的时间构成

BF16 单请求稳态 Systems 汇总：

- 视觉 FlashAttention `flash_fwd_kernel`：35.9%。
- cuBLAS BF16 GEMV：23.5% + 5.0%，主要对应小 batch Decode Linear。
- 视觉/Prefill CUTLASS BF16 GEMM：多组 Kernel，单项最高 5.5%。
- Triton 生成 Kernel：包括 RMSNorm、SwiGLU 与其他 pointwise/reduction 融合，单项多在 0.4%～2.6%。

FP8 单请求稳态：

- 视觉 BF16 FlashAttention 升至 42.3% 的相对占比。
- SM120 E4M3 CUTLASS GEMM 至少占 17.3% + 7.4%；Linear 变快后瓶颈向未量化的视觉 Attention 转移。
- FP8 量化、scale、RMSNorm 和激活产生更复杂的 Triton fused Kernel，但单项占比小于主要 Attention/Linear。

这些是 Nsight Systems 的 GPU 时间占比，能够决定“先看谁”；在 Nsight Compute 计数器解锁前，不能据此声称 Kernel 是 compute-bound 或 memory-bound。

## 优化优先级

### P0：视觉 Encoder compile / CUDA Graph A/B

当前启动配置明确为：

```text
compile_mm_encoder=False
cudagraph_mm_encoder=False
encoder_cudagraph_token_budgets=[]
```

而视觉 FlashAttention 是 BF16/FP8 第一大 Kernel 类别。应对固定分辨率与混合分辨率分别测试：

1. 默认 eager vision encoder；
2. `--compilation-config` 开启 multimodal encoder compile；
3. 支持时增加 encoder CUDA Graph token budgets；
4. 比较冷 shape 编译成本、热图 TTFT、显存和 graph 命中率。

只有固定 shape/常用分辨率命中时才可能收益；输入尺寸高度动态时，重编译与 graph 数量可能抵消收益。

本机 A/B 已完成（固定 1920×1080 shape，每次改变 64×64 边角色块以避免 Encoder Cache，2 次预热 + 5 次有效请求）：默认、Encoder compile、Encoder compile+Graph 的 TTFT P50 分别为 335.22、333.15、334.61 ms，即仅 +0.62%/+0.18%，均不足 5% 门槛。Graph 显存由约 0.1 GiB 增至 0.5 GiB，因此当前工作负载保留默认配置。这是负结果，不写成性能提升。

### P0：用 Nsight Compute 判断 FA2 与 GEMV 的真实瓶颈

需要分别采一个视觉 `flash_fwd_kernel`、一个 Decode `gemvx`、一个 FP8 SM120 CUTLASS GEMM：

- 视觉 Attention：检查 Tensor Pipe、DRAM/L2、occupancy、长序列 tile 效率；
- Decode GEMV：通常受权重读取/小 M shape 限制，但必须用带宽和 Roofline 证明；
- FP8 GEMM：检查 Tensor Core 利用率、TMA、tile shape 和量化 epilogue 成本。

当前 `ERR_NVGPUCTRPERM`/系统权限尚未解除，所以这一项仍是“可执行实验”，不是已取得结论。

### P1：FP8 作为生产配置候选

已有服务基准显示 FP8 c1/c8 聚合吞吐较 BF16 提升 52.1%/53.1%，且 Nsight 确认运行了 SM120 E4M3 Tensor Core Kernel。下一步不是继续优化 CUTLASS 源码，而是扩充 OCR、图表、多轮一致性和长上下文召回质量集，决定是否能把 FP8 设为默认服务。

### P1：QK Norm + RoPE 融合实验

Qwen3 每层依次执行 Q RMSNorm、K RMSNorm 和 RoPE，当前 `enable_qk_norm_rope_fusion=False`。这是减少 launch 和中间读写的候选，尤其适合 Decode 的小 shape。但 vLLM 当前显式融合实现/开关可能只覆盖部分模型，必须先确认 Qwen3-VL 路径支持，再做正确性与 A/B；不能直接打开后写成收益。

### P1：Prefill/Decode 分流与调度参数

已有饱和实验中 c8 后吞吐约 329～335 tok/s，但 TTFT P95 从 1.40 s 上升到 10.41 s，TPOT P95 仍约 21 ms。说明当前更紧迫的是排队、Prefill 和 admission control，而不是优化占比不到 1% 的点算子。优先验证：

- `max-num-batched-tokens`；
- Chunked Prefill chunk size；
- `max-num-seqs` 与入口并发限制；
- 长 Prefill 与短 Decode 混部时的尾延迟。

混部 A/B 已完成：在 1 个约 12K-token 长 Prefill 到达后 50 ms 注入 4 个短 Decode 请求，`max-num-batched-tokens=1024/2048/4096/8192` 的短请求 TTFT P95 分别为 1173.30/1125.18/1061.53/1079.01 ms。4096 相对 2048 改善 5.66%，5 批组内均值的变异系数为 0.52%，同时长请求 TTFT P50 从 1161.59 降至 1101.36 ms。因此默认服务采用 4096；该结论只适用于当前单卡混部负载，不外推为通用最优值。

### P2：采样器与小算子

FlashInfer sampler 当前因 RTX 5090 SM120 兼容问题关闭。采样通常不是本工作负载的主要 GPU 时间，除非使用大词表、复杂 top-k/top-p 或高并发短输出。应等兼容版本后做独立 A/B，不建议为了“全用 FlashInfer”修改第三方 JIT 架构判断。

RMSNorm、SwiGLU 等单项占比较低，直接重写生产 Kernel 的预期收益有限。若为了展示 Kernel 能力，可独立实现 Triton RMSNorm/SwiGLU benchmark，验证 correctness、不同 shape 和 Roofline，但明确不替换 vLLM 生产算子。

## 不建议做的事情

- 看到 `triton_` 名称就声称所有模型算子由 Triton 编写；Linear/Attention 的主要时间实际来自 cuBLAS、CUTLASS 和 FlashAttention CUDA。
- 直接修改 vLLM/FlashAttention 第三方 Kernel；升级维护和多 shape 正确性成本高，且当前没有 NCU 证据支持。
- 优先优化占比小于 1% 的 RoPE、采样或 pointwise Kernel，而忽略视觉 Attention、Linear 与调度。
- 把 Systems 时间占比当成 Roofline 结论，或把理想 Prefix Cache 命中收益当成算子优化。

## 面试表述

> 对 Qwen3-VL 的 Python 图、vLLM 后端选择和 Nsight Systems Kernel 进行三层映射，确认视觉/语言 Attention 使用 FlashAttention v2 与分页 KV，BF16 Decode Linear 主要落到 cuBLAS GEMV，FP8 Linear 使用 Blackwell SM120 E4M3 CUTLASS Tensor Core Kernel，RMSNorm/SwiGLU 等由 Inductor 融合为 Triton Kernel。基于 GPU 时间占比将后续优化聚焦于视觉 Encoder graph/compile、FP8 质量闭环和 Prefill 调度，而非盲目重写低占比点算子。
