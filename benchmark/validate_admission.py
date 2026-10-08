#!/usr/bin/env python3
import json
from pathlib import Path
root=Path("results/ab/admission")
direct=[json.loads(x) for x in (root/"direct/requests.jsonl").read_text().splitlines()]; limited=[json.loads(x) for x in (root/"limited/requests.jsonl").read_text().splitlines()]
assert len(direct)==len(limited)==160
assert all(x["status"]==200 and x["ttft_ms"] is not None for x in direct)
assert sum(x["status"]==200 for x in limited)==10 and sum(x["status"]==429 for x in limited)==150
assert all(x["ttft_ms"] is None for x in limited if x["status"]==429)
print("PASS: 320 offered requests; direct statuses and admission 200/429 counts are complete.")
