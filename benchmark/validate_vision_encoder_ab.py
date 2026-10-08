#!/usr/bin/env python3
import json,math
from pathlib import Path
root=Path("results/ab/vision_encoder"); docs=[]
for name in ("default","compile","compile_graph"):
    doc=json.loads((root/name/"summary.json").read_text()); rows=[json.loads(x) for x in (root/name/"requests.jsonl").read_text().splitlines()]
    assert len(rows)==doc["metadata"]["repeats"]==5
    assert all(x["prompt_tokens"]==2059 and x["ttft_ms"]>100 and x["output_tokens"]>0 for x in rows),"encoder may have been cached or request invalid"
    docs.append((doc,rows))
assert len({x["prompt_tokens"] for _,rows in docs for x in rows})==1
base=docs[0][0]["summary"]["ttft_p50_ms"]
for doc,_ in docs[1:]:
    gain=100*(base/doc["summary"]["ttft_p50_ms"]-1)
    assert abs(gain)<5,"result changed enough to require report review"
print("PASS: 15 uncached-encoder requests; equal prompt shape; no variant exceeds 5% threshold.")
