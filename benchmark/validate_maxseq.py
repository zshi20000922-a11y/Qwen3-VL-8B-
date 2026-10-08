#!/usr/bin/env python3
import json
from pathlib import Path
root=Path("results/ab/max_num_seqs"); total=0
for seqs in (4,8,16):
    p=root/f"seqs_{seqs}"; d=json.loads((p/"summary.json").read_text()); rows=[json.loads(x) for x in (p/"requests.jsonl").read_text().splitlines()]
    assert len(rows)==25 and d["summary"]["short_requests"]==20; total+=len(rows)
assert total==75
print("PASS: 75 mixed-workload requests across max-num-seqs 4/8/16.")
