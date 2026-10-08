#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT_DIR}"; OUT="results/ab/scheduler"; mkdir -p "${OUT}"
restore(){ ./scripts/stop_vllm.sh >/dev/null 2>&1 || true; MAX_NUM_BATCHED_TOKENS=4096 MAX_NUM_SEQS=8 ./scripts/start_vllm_background.sh >/dev/null 2>&1 || true; }
trap restore EXIT INT TERM
for tokens in 1024 2048 4096 8192; do
  name="tokens_${tokens}"; mkdir -p "${OUT}/${name}"; ./scripts/stop_vllm.sh
  MAX_NUM_BATCHED_TOKENS="${tokens}" MAX_NUM_SEQS=8 LOG_FILE="${ROOT_DIR}/${OUT}/${name}/server.log" ./scripts/start_vllm_background.sh
  .venv-vllm/bin/python benchmark/scheduler_interference.py --variant "${name}" --output-dir "${OUT}/${name}"
done
.venv-vllm/bin/python benchmark/compare_scheduler.py
