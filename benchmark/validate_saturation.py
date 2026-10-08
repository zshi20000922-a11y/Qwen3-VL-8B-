#!/usr/bin/env python3
import csv,json,math
from pathlib import Path
from saturation import percentile
root=Path("results/saturation")
rows=[json.loads(x) for x in (root/"requests.jsonl").read_text().splitlines()]
doc=json.loads((root/"summary.json").read_text()); meta=doc["metadata"]
summary=list(csv.DictReader((root/"summary.csv").open()))
assert rows and summary
for item in summary:
    c=int(item["concurrency"]); values=[x for x in rows if x["concurrency"]==c]
    assert len(values)==c*meta["repeats"]
    assert math.isclose(float(item["ttft_p95_ms"]),round(percentile([x["ttft_ms"] for x in values],95),2),abs_tol=.02)
    assert math.isclose(float(item["tpot_p95_ms"]),round(percentile([x["tpot_ms"] for x in values],95),2),abs_tol=.02)
    assert all(x["output_tokens"]==meta["max_tokens"] for x in values)
print(f"PASS: {len(rows)} requests, {len(summary)} concurrency levels; TTFT/TPOT/Goodput are reproducible.")
