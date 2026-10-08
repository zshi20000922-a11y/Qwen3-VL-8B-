#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PROFILE_PORT:-8002}"; VARIANT="${PROFILE_VARIANT:-bf16}"
SCENARIO="${PROFILE_SCENARIO:-steady}"; CONCURRENCY="${PROFILE_CONCURRENCY:-1}"
TOKENS="${PROFILE_OUTPUT_TOKENS:-256}"; RESULT="${PROFILE_WORKLOAD_OUTPUT:?}"
server_pid=""
cleanup(){ [[ -n "${server_pid}" ]] && kill -INT "${server_pid}" 2>/dev/null || true; wait "${server_pid}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
# Nsight injection does not follow vLLM's spawn-based EngineCore reliably on this
# host. Single-process V1 keeps the same kernels/scheduler while making CUDA
# activities observable. Production service remains multiprocess.
export VLLM_ENABLE_V1_MULTIPROCESSING="${PROFILE_VLLM_MULTIPROCESSING:-0}"
if [[ "${VARIANT}" == fp8 ]]; then QUANTIZATION=fp8 PORT="${PORT}" "${ROOT_DIR}/scripts/start_vllm.sh" & else PORT="${PORT}" "${ROOT_DIR}/scripts/start_vllm.sh" & fi
server_pid=$!
for _ in $(seq 1 180); do curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break; sleep 1; done
curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null
"${ROOT_DIR}/.venv-vllm/bin/python" "${ROOT_DIR}/profiling/workload.py" --base-url "http://127.0.0.1:${PORT}" \
  --scenario "${SCENARIO}" --concurrency "${CONCURRENCY}" --output-tokens "${TOKENS}" --output "${RESULT}"
kill -INT "${server_pid}" 2>/dev/null || true; wait "${server_pid}" || true; server_pid=""
