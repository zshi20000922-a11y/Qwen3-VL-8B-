#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
curl -fsS "${BASE_URL}/health" >/dev/null
curl -fsS "${BASE_URL}/v1/models"
echo
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
