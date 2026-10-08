# Qwen3-VL-8B-Instruct：RTX 5090 单卡 vLLM 部署

## 启动

```bash
./scripts/start_vllm.sh
```

服务默认只监听 `127.0.0.1:8000`。需要局域网访问时使用：

```bash
HOST=0.0.0.0 ./scripts/start_vllm.sh
```

默认将上下文限制为 16K，给 32GB 单卡的权重、视觉编码、KV Cache 和并发留出空间。可按业务修改：

```bash
MAX_MODEL_LEN=32768 MAX_NUM_SEQS=4 GPU_MEMORY_UTILIZATION=0.92 ./scripts/start_vllm.sh
```

## 功能和性能测试

```bash
.venv-vllm/bin/python scripts/benchmark_vllm.py --concurrency 1 --show-output
.venv-vllm/bin/python scripts/benchmark_vllm.py --concurrency 4
.venv-vllm/bin/python scripts/benchmark_vllm.py --concurrency 8
```

关注三个指标：

- `mean_ttft_ms`：首 Token 延迟，主要受视觉编码和 Prefill 影响。
- `mean_per_request_decode_tok_s`：单请求生成速度。
- `aggregate_output_tok_s`：并发请求总吞吐；这是 continuous batching 的主要收益。

## 停止

```bash
./scripts/stop_vllm.sh
```

## 当前参数为什么这样设置

- BF16：不改变现有权重精度，RTX 5090 原生支持。
- 16K 上下文：模型声明支持 262K，但单卡实际显存不能同时承受超长 KV Cache 和较高并发。
- Prefix caching：重复的 system prompt 或相同前缀可以复用 KV Cache。
- `max-num-seqs=8`：允许 continuous batching 合并并发请求。
- 图片数限制为 1、禁用视频：让服务的显存上界更可控；需要视频时应重新做容量测试。
- `VLLM_USE_FLASHINFER_SAMPLER=0`：规避 FlashInfer 0.6.16 对 RTX 5090 SM 12.0 的架构误判；它只切换 top-k/top-p 采样器，不会关闭 FlashAttention 或 PagedAttention。

vLLM 会自动使用其可用的优化内核、PagedAttention、continuous batching、CUDA Graph 和 chunked prefill。首轮会包含模型加载、内核选择和图捕获，正式比较前必须预热。

## 本机实测（RTX 5090 32GB）

环境：vLLM 0.28.0、PyTorch 2.13.0、CUDA 13.0、BF16、输出 256 Token。

| 场景 | TTFT | 单请求 Decode | 聚合吞吐 | 端到端 |
|---|---:|---:|---:|---:|
| Transformers 基线，640×360 | 40.79 ms | 78.11 tok/s | 77.44 tok/s | 约 3.31 s |
| vLLM 首次 640×360 请求 | 46.52 ms | 95.16 tok/s | 93.54 tok/s | 2.74 s |
| vLLM 缓存后 640×360 请求 | 18.51 ms | 93.85 tok/s | 93.21 tok/s | 2.75 s |
| vLLM，1 并发，测试图 | 55.03 ms | 93.56 tok/s | 91.71 tok/s | 2.79 s |
| vLLM，4 并发，测试图 | 51.92 ms（平均） | 84.12 tok/s（平均） | 329.98 tok/s | 3.10 s（P50） |
| vLLM，8 并发，测试图 | 74.76 ms（平均） | 80.92 tok/s（平均） | 629.73 tok/s | 3.24 s（P50） |

和 640×360 Transformers 基线相比，vLLM 单请求 Decode 提升约 21.8%，端到端输出吞吐提升约 20.8%。8 并发时的聚合吞吐是 vLLM 单并发的约 6.87 倍；这是 continuous batching 最直观的收益。缓存后的 TTFT 从首次请求的 46.52 ms 降至 18.51 ms，但该收益只适用于相同图片/前缀的重复访问。

完整结果位于 `logs/vllm_*.txt`。不同 Prompt 可能提前生成 EOS，比较时应固定图片、Prompt、输出长度和采样参数。
