#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NSIGHT_ROOT="${NSIGHT_ROOT:-/usr/local/NVIDIA-Nsight-Compute-2026.2}"
NSYS="${PROFILE_NSYS_PATH:-${NSIGHT_ROOT}/host/target-linux-x64/nsys}"
NCU="${PROFILE_NCU_PATH:-${NSIGHT_ROOT}/ncu}"

fail=0
check() { if "$@" >/dev/null 2>&1; then echo "PASS $*"; else echo "FAIL $*"; fail=1; fi; }

echo "== Nsight executables =="
check test -x "${NSYS}"
check test -x "${NCU}"
"${NSYS}" --version || true
"${NCU}" --version | tail -n 1 || true
if ! "${NSYS}" --version | grep -q '2026.2'; then
  echo "FAIL: 必须使用 Nsight Systems 2026.2，不能使用 PATH 中的旧版。" >&2
  fail=1
fi

echo "== GPU and runtime =="
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader
check test -x "${ROOT_DIR}/.venv-vllm/bin/vllm"
check test -d "${ROOT_DIR}/models/Qwen3-VL-8B-Instruct"
echo "kernel.perf_event_paranoid=$(sysctl -n kernel.perf_event_paranoid 2>/dev/null || echo unknown)"
echo "NVreg_RestrictProfilingToAdminUsers=$(cat /proc/driver/nvidia/params 2>/dev/null | awk -F: '/RestrictProfilingToAdminUsers/{gsub(/ /,"",$2);print $2}' || true)"
df -h "${ROOT_DIR}" | tail -n 1
systemctl --user is-active qwen3-vl-vllm.service 2>/dev/null || true
exit "${fail}"
