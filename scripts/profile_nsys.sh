#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NSYS="${PROFILE_NSYS_PATH:-/usr/local/NVIDIA-Nsight-Compute-2026.2/host/target-linux-x64/nsys}"
VARIANT="${PROFILE_VARIANT:-bf16}"; SCENARIO="${PROFILE_SCENARIO:-steady}"
CONCURRENCY="${PROFILE_CONCURRENCY:-1}"; TOKENS="${PROFILE_OUTPUT_TOKENS:-256}"; PORT="${PROFILE_PORT:-8002}"
STAMP="${PROFILE_TAG:-${VARIANT}_${SCENARIO}_c${CONCURRENCY}}"
OUT="${ROOT_DIR}/profiles/raw/${STAMP}"; LOG="${ROOT_DIR}/profiles/logs/${STAMP}.log"
mkdir -p "${ROOT_DIR}/profiles/raw" "${ROOT_DIR}/profiles/logs" "${ROOT_DIR}/profiles/workloads"
test -x "${NSYS}"; "${NSYS}" --version | grep -q '2026.2'
if ss -ltn | awk '{print $4}' | grep -qE ":${PORT}$"; then echo "端口 ${PORT} 已占用" >&2; exit 1; fi
cleanup(){ pkill -f "vllm serve.*--port ${PORT}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
PROFILE_PORT="${PORT}" PROFILE_VARIANT="${VARIANT}" PROFILE_SCENARIO="${SCENARIO}" \
PROFILE_CONCURRENCY="${CONCURRENCY}" PROFILE_OUTPUT_TOKENS="${TOKENS}" \
PROFILE_WORKLOAD_OUTPUT="${ROOT_DIR}/profiles/workloads/${STAMP}.json" PROFILE_VLLM_MULTIPROCESSING="${PROFILE_VLLM_MULTIPROCESSING:-0}" \
"${NSYS}" profile --trace=cuda,nvtx,osrt,cublas,cudnn --sample=process-tree --cpuctxsw=process-tree \
  --cuda-graph-trace=node --force-overwrite=true --output="${OUT}" \
  "${ROOT_DIR}/scripts/profile_session.sh" >"${LOG}" 2>&1
test -s "${OUT}.nsys-rep"
echo "生成 ${OUT}.nsys-rep"
