#!/usr/bin/env python3
import json
from pathlib import Path
root=Path("results/ab/scheduler"); count=0
for p in sorted(root.glob("tokens_*/summary.json")):
    doc=json.loads(p.read_text()); rows=[json.loads(x) for x in (p.parent/"requests.jsonl").read_text().splitlines()]; s=doc["summary"]
    assert len(rows)==s["repeats"]*(s["short_requests"]//s["repeats"]+1)
    assert all(x["output_tokens"] in (16,128) and x["ttft_ms"]>0 for x in rows); count+=len(rows)
assert count>0; print(f"PASS: {count} mixed-workload requests across scheduler variants.")
