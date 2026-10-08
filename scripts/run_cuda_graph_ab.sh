#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT_DIR}"
restore(){ ./scripts/stop_vllm.sh >/dev/null 2>&1 || true; ENFORCE_EAGER=0 ./scripts/start_vllm_background.sh >/dev/null 2>&1 || true; }
trap restore EXIT INT TERM
./scripts/stop_vllm.sh; ENFORCE_EAGER=0 ./scripts/start_vllm_background.sh
.venv-vllm/bin/python benchmark/service_ab.py --label cuda_graph --output-dir results/ab/cuda_graph
./scripts/stop_vllm.sh; ENFORCE_EAGER=1 ./scripts/start_vllm_background.sh
.venv-vllm/bin/python benchmark/service_ab.py --label eager --output-dir results/ab/eager
.venv-vllm/bin/python benchmark/compare_service_ab.py
