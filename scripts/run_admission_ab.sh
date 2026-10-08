#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT_DIR}"; OUT="results/ab/admission"; mkdir -p "${OUT}/direct" "${OUT}/limited"
proxy_pid=''; cleanup(){ [[ -n "${proxy_pid}" ]] && kill "${proxy_pid}" 2>/dev/null || true; }; trap cleanup EXIT INT TERM
.venv-vllm/bin/python benchmark/admission_overload.py --variant direct --base-url http://127.0.0.1:8000 --output-dir "${OUT}/direct"
ADMISSION_MAX_INFLIGHT=2 ADMISSION_WAIT_MS=50 .venv-vllm/bin/python -m uvicorn server.admission_proxy:app --host 127.0.0.1 --port 8001 >"${OUT}/proxy.log" 2>&1 & proxy_pid=$!
for _ in {1..50}; do curl -fsS http://127.0.0.1:8001/health >/dev/null 2>&1 && break; sleep .1; done
.venv-vllm/bin/python benchmark/admission_overload.py --variant limited --base-url http://127.0.0.1:8001 --output-dir "${OUT}/limited"
.venv-vllm/bin/python - <<'PY'
import json
from pathlib import Path
r=Path('results/ab/admission'); a=json.loads((r/'direct/summary.json').read_text()); b=json.loads((r/'limited/summary.json').read_text())
lines=['# Admission Control 过载保护 A/B','',f"并发突发：{a['offered']} 请求，TTFT SLO≤500 ms。",'', '|入口|接受|拒绝|接受请求 TTFT P95|SLO Goodput|接受请求达标率|','|---|---:|---:|---:|---:|---:|',f"|直接 vLLM|{a['accepted']}|{a['rejected']}|{a['accepted_ttft_p95_ms']:.2f} ms|{a['slo_goodput_rps']:.3f} req/s|{a['accepted_slo_pass_pct']:.2f}%|",f"|限流 max=2|{b['accepted']}|{b['rejected']}|{b['accepted_ttft_p95_ms']:.2f} ms|{b['slo_goodput_rps']:.3f} req/s|{b['accepted_slo_pass_pct']:.2f}%|",'', '限流以快速 429 换取已接受请求的延迟可预测性；客户端必须退避重试。拒绝请求不算成功吞吐，也不能把接受请求的高达标率解释为整体可用性提升。','']
(r/'REPORT.md').write_text('\n'.join(lines)); print('\n'.join(lines))
PY
