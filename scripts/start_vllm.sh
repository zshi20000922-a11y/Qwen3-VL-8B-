#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models/Qwen3-VL-8B-Instruct}"
VLLM_BIN="${VLLM_BIN:-${ROOT_DIR}/.venv-vllm/bin/vllm}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-}"

# FlashInfer 0.6.16 的采样器会在消费级 Blackwell（SM 12.0）上误判
# CUDA 架构。只禁用采样器；FlashAttention、PagedAttention 等仍由 vLLM 使用。
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

if [[ ! -x "${VLLM_BIN}" ]]; then
  echo "vLLM 不存在：${VLLM_BIN}" >&2
  echo "请先运行：uv venv --python 3.12 .venv-vllm && uv pip install --python .venv-vllm/bin/python vllm openai httpx" >&2
  exit 1
fi

args=(
  serve "${MODEL_DIR}"
  --served-model-name Qwen3-VL-8B-Instruct
  --host "${HOST}"
  --port "${PORT}"
  --dtype bfloat16
  --max-model-len "${MAX_MODEL_LEN:-16384}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
  --max-num-seqs "${MAX_NUM_SEQS:-8}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-4096}"
  --limit-mm-per-prompt '{"image":1,"video":0}'
)

if [[ "${ENABLE_PREFIX_CACHING:-1}" == "1" ]]; then
  args+=(--enable-prefix-caching)
fi
if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  args+=(--enforce-eager)
fi

if [[ -n "${API_KEY}" ]]; then
  args+=(--api-key "${API_KEY}")
fi
if [[ -n "${QUANTIZATION:-}" ]]; then
  args+=(--quantization "${QUANTIZATION}")
fi
if [[ -n "${KV_CACHE_DTYPE:-}" ]]; then
  args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi
if [[ -n "${COMPILATION_CONFIG:-}" ]]; then
  args+=(--compilation-config "${COMPILATION_CONFIG}")
fi

exec "${VLLM_BIN}" "${args[@]}"
