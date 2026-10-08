#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${ROOT_DIR}/run"
PID_FILE="${PID_FILE:-${RUN_DIR}/vllm.pid}"
LOG_FILE="${LOG_FILE:-${ROOT_DIR}/logs/vllm-server.log}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-180}"

mkdir -p "${RUN_DIR}" "$(dirname "${LOG_FILE}")"

if systemctl --user is-active --quiet qwen3-vl-vllm.service; then
  echo "vLLM 已由 systemd 用户服务运行。"
  exit 0
fi

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(<"${PID_FILE}")"
  if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "vLLM 已运行，PID=${old_pid}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

env_args=()
for name in QUANTIZATION ENFORCE_EAGER ENABLE_PREFIX_CACHING MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS GPU_MEMORY_UTILIZATION COMPILATION_CONFIG; do
  if [[ -n "${!name:-}" ]]; then env_args+=(--setenv="${name}=${!name}"); fi
done

systemd-run --user \
  --unit=qwen3-vl-vllm \
  --collect \
  --property=Restart=on-failure \
  --property=RestartSec=3 \
  --property="StandardOutput=append:${LOG_FILE}" \
  --property="StandardError=append:${LOG_FILE}" \
  --working-directory="${ROOT_DIR}" \
  "${env_args[@]}" \
  "${ROOT_DIR}/scripts/start_vllm.sh" >/dev/null
pid="$(systemctl --user show qwen3-vl-vllm.service --property=MainPID --value)"
echo "${pid}" >"${PID_FILE}"

for ((second=1; second<=STARTUP_TIMEOUT; second++)); do
  if ! systemctl --user is-active --quiet qwen3-vl-vllm.service; then
    echo "vLLM 启动失败，日志末尾：" >&2
    tail -n 80 "${LOG_FILE}" >&2
    rm -f "${PID_FILE}"
    exit 1
  fi
  if curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    echo "vLLM 已就绪：http://${HOST}:${PORT}（PID=${pid}）"
    exit 0
  fi
  sleep 1
done

echo "vLLM 在 ${STARTUP_TIMEOUT} 秒内未就绪，请检查 ${LOG_FILE}" >&2
exit 1
