#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="${PID_FILE:-${ROOT_DIR}/run/vllm.pid}"

if systemctl --user is-active --quiet qwen3-vl-vllm.service; then
  systemctl --user stop qwen3-vl-vllm.service
  rm -f "${PID_FILE}"
  echo "vLLM systemd 用户服务已停止。"
  exit 0
fi

if [[ -f "${PID_FILE}" ]]; then
  pid="$(<"${PID_FILE}")"
  if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
    kill -TERM "${pid}"
    for _ in {1..30}; do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${pid}" 2>/dev/null; then
      echo "服务在 30 秒内未退出，请检查进程 ${pid}。" >&2
      exit 1
    fi
    rm -f "${PID_FILE}"
    echo "vLLM 服务已停止。"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

if ! pgrep -f "vllm serve.*Qwen3-VL-8B-Instruct" >/dev/null; then
  echo "没有发现 Qwen3-VL vLLM 服务。"
  exit 0
fi

pkill -TERM -f "vllm serve.*Qwen3-VL-8B-Instruct"
echo "已向 Qwen3-VL vLLM 服务发送 SIGTERM。"
