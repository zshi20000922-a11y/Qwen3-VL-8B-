#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NCU="${PROFILE_NCU_PATH:-/usr/local/NVIDIA-Nsight-Compute-2026.2/ncu}"
SET="${PROFILE_NCU_SET:-basic}"; COUNT="${PROFILE_NCU_LAUNCH_COUNT:-1}"
FILTER="${PROFILE_KERNEL_REGEX:-regex:.*gemm.*|.*cutlass.*|.*_scaled_mm.*}"
TAG="${PROFILE_TAG:-controlled_bf16_gemm_${SET}}"
mkdir -p "${ROOT_DIR}/profiles/raw" "${ROOT_DIR}/profiles/logs" "${ROOT_DIR}/profiles/export"
test -x "${NCU}"; "${NCU}" --version | grep -q '2026.2'
set +e
"${NCU}" --target-processes all --set "${SET}" --kernel-name-base demangled \
  --kernel-name "${FILTER}" --launch-count "${COUNT}" --force-overwrite \
  --export "${ROOT_DIR}/profiles/raw/${TAG}" \
  "${ROOT_DIR}/.venv-vllm/bin/python" "${ROOT_DIR}/profiling/ncu_probe.py" \
  >"${ROOT_DIR}/profiles/logs/${TAG}.log" 2>&1
rc=$?
set -e
if (( rc != 0 )); then
  cat "${ROOT_DIR}/profiles/logs/${TAG}.log" >&2
  if grep -q 'ERR_NVGPUCTRPERM' "${ROOT_DIR}/profiles/logs/${TAG}.log"; then
    echo "GPU 计数器受限。请执行: sudo -E PROFILE_NCU_SET=${SET} make profile-ncu" >&2
  fi
  exit "${rc}"
fi
"${NCU}" --import "${ROOT_DIR}/profiles/raw/${TAG}.ncu-rep" --csv --page details \
  >"${ROOT_DIR}/profiles/export/${TAG}_details.csv" 2>"${ROOT_DIR}/profiles/logs/${TAG}_export.log"
echo "生成 profiles/raw/${TAG}.ncu-rep"
