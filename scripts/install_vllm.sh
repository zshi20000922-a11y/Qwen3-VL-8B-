#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

command -v uv >/dev/null || { echo "需要先安装 uv" >&2; exit 1; }
uv venv --python 3.12 .venv-vllm
UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}" \
UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-1}" \
  uv pip install --python .venv-vllm/bin/python \
  'vllm==0.28.0' 'openai==3.7.0' 'httpx==0.28.1'
.venv-vllm/bin/python -c 'import torch, vllm; print("vLLM", vllm.__version__, "PyTorch", torch.__version__, "CUDA", torch.version.cuda)'
