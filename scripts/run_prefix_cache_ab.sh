#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT_DIR}"
restore(){ ./scripts/stop_vllm.sh >/dev/null 2>&1 || true; ENABLE_PREFIX_CACHING=1 ENFORCE_EAGER=0 ./scripts/start_vllm_background.sh >/dev/null 2>&1 || true; }
trap restore EXIT INT TERM
./scripts/stop_vllm.sh; ENABLE_PREFIX_CACHING=1 ENFORCE_EAGER=0 ./scripts/start_vllm_background.sh
.venv-vllm/bin/python benchmark/service_ab.py --label prefix_on --output-dir results/ab/prefix_on
./scripts/stop_vllm.sh; ENABLE_PREFIX_CACHING=0 ENFORCE_EAGER=0 ./scripts/start_vllm_background.sh
.venv-vllm/bin/python benchmark/service_ab.py --label prefix_off --output-dir results/ab/prefix_off
.venv-vllm/bin/python benchmark/compare_prefix_ab.py
