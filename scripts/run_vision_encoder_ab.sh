#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT_DIR}"
OUT="results/ab/vision_encoder"; mkdir -p "${OUT}"
restore(){ ./scripts/stop_vllm.sh >/dev/null 2>&1 || true; COMPILATION_CONFIG='' ./scripts/start_vllm_background.sh >/dev/null 2>&1 || true; }
trap restore EXIT INT TERM
run_one(){
  local name="$1" config="$2"
  ./scripts/stop_vllm.sh
  COMPILATION_CONFIG="${config}" LOG_FILE="${ROOT_DIR}/${OUT}/${name}/server.log" ./scripts/start_vllm_background.sh
  .venv-vllm/bin/python benchmark/vision_encoder_ab.py --variant "${name}" --output-dir "${OUT}/${name}"
}
mkdir -p "${OUT}/default" "${OUT}/compile" "${OUT}/compile_graph"
run_one default ''
run_one compile '{"compile_mm_encoder":true}'
run_one compile_graph '{"compile_mm_encoder":true,"cudagraph_mm_encoder":true,"encoder_cudagraph_token_budgets":[2048],"encoder_cudagraph_max_vision_items_per_batch":1}'
.venv-vllm/bin/python benchmark/compare_vision_encoder_ab.py
