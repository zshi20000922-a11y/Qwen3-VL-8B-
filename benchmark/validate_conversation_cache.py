#!/usr/bin/env python3
import csv,json
from pathlib import Path
root=Path("results/conversation_cache")
rows=[json.loads(x) for x in (root/"requests.jsonl").read_text().splitlines()]
doc=json.loads((root/"summary.json").read_text()); meta=doc["metadata"]
summary=list(csv.DictReader((root/"summary.csv").open()))
assert len(rows)==meta["sessions"]*len(summary)
prompts=[float(x["prompt_tokens_mean"]) for x in summary]
assert all(b>a for a,b in zip(prompts,prompts[1:])),"prompt tokens must grow by turn"
assert float(summary[0]["cache_hit_rate_mean_pct"])==0
assert all(float(x["cache_hit_rate_mean_pct"])>0 for x in summary[1:])
assert all(0<=x["cache_hit_tokens"]<=x["cache_query_tokens"] for x in rows)
print(f"PASS: {len(rows)} requests across {len(summary)} append-only turns; cache counters are valid.")
