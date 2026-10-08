#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT_DIR}"; OUT="results/ab/max_num_seqs"; mkdir -p "${OUT}"
restore(){ ./scripts/stop_vllm.sh >/dev/null 2>&1 || true; MAX_NUM_BATCHED_TOKENS=4096 MAX_NUM_SEQS=8 ./scripts/start_vllm_background.sh >/dev/null 2>&1 || true; }
trap restore EXIT INT TERM
for seqs in 4 8 16; do
  name="seqs_${seqs}"; mkdir -p "${OUT}/${name}"; ./scripts/stop_vllm.sh
  MAX_NUM_BATCHED_TOKENS=4096 MAX_NUM_SEQS="${seqs}" LOG_FILE="${ROOT_DIR}/${OUT}/${name}/server.log" ./scripts/start_vllm_background.sh
  .venv-vllm/bin/python benchmark/scheduler_interference.py --variant "${name}" --output-dir "${OUT}/${name}"
done
.venv-vllm/bin/python benchmark/compare_maxseq.py
