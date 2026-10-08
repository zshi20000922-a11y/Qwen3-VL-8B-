#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NSYS="${PROFILE_NSYS_PATH:-/usr/local/NVIDIA-Nsight-Compute-2026.2/host/target-linux-x64/nsys}"
VARIANT="${PROFILE_VARIANT:-bf16}"; SCENARIO="${PROFILE_SCENARIO:-steady}"; TAG="${PROFILE_TAG:-${VARIANT}_${SCENARIO}_offline}"
mkdir -p "${ROOT_DIR}/profiles/raw" "${ROOT_DIR}/profiles/logs" "${ROOT_DIR}/profiles/workloads"
"${NSYS}" --version | grep -q '2026.2'
result="${ROOT_DIR}/profiles/workloads/${TAG}.json"
"${NSYS}" profile --trace=cuda,nvtx,osrt,cublas,cudnn --sample=none --cuda-graph-trace=node \
 --force-overwrite=true --output="${ROOT_DIR}/profiles/raw/${TAG}" \
 "${ROOT_DIR}/.venv-vllm/bin/python" "${ROOT_DIR}/profiling/vllm_offline.py" --variant "${VARIANT}" \
 --scenario "${SCENARIO}" --concurrency "${PROFILE_CONCURRENCY:-1}" --output-tokens "${PROFILE_OUTPUT_TOKENS:-256}" \
 --output "${result}" >"${ROOT_DIR}/profiles/logs/${TAG}.log" 2>&1 &
nsys_pid=$!
# The vLLM/NCCL teardown can leave nsys waiting after the reportable workload is
# complete. Once metadata exists, allow teardown then request a clean finalize.
for _ in $(seq 1 "${PROFILE_MAX_WAIT_SECONDS:-240}"); do [[ -s "${result}" ]] && break; sleep 1; done
[[ -s "${result}" ]] || { kill -TERM "${nsys_pid}" 2>/dev/null || true; echo "工作负载超时" >&2; exit 1; }
sleep 8; kill -TERM "${nsys_pid}" 2>/dev/null || true; wait "${nsys_pid}" || true
test -s "${ROOT_DIR}/profiles/raw/${TAG}.nsys-rep"; echo "生成 profiles/raw/${TAG}.nsys-rep"
