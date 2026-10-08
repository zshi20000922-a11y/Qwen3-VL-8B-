#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NSYS="${PROFILE_NSYS_PATH:-/usr/local/NVIDIA-Nsight-Compute-2026.2/host/target-linux-x64/nsys}"
mkdir -p "${ROOT_DIR}/profiles/export" "${ROOT_DIR}/profiles/summary"
shopt -s nullglob
reports=("${ROOT_DIR}"/profiles/raw/*.nsys-rep)
(( ${#reports[@]} )) || { echo "没有 profiles/raw/*.nsys-rep" >&2; exit 1; }
for report in "${reports[@]}"; do
  name="$(basename "${report}" .nsys-rep)"
  "${NSYS}" stats --force-export=true --format csv --output "${ROOT_DIR}/profiles/export/${name}" \
    --report cuda_gpu_kern_sum,cuda_api_sum,cuda_kern_exec_sum,nvtx_pushpop_sum "${report}"
done
"${ROOT_DIR}/.venv-vllm/bin/python" "${ROOT_DIR}/profiling/analyze_profiles.py"
