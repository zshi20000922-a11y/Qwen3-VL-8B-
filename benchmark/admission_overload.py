#!/usr/bin/env python3
import argparse,asyncio,json,time,uuid
from pathlib import Path
import httpx
def pct(v,p):
    v=sorted(v); x=(len(v)-1)*p/100; lo=int(x); hi=min(lo+1,len(v)-1); return v[lo]+(v[hi]-v[lo])*(x-lo)
async def one(client,args,repeat,slot):
    payload={"model":"Qwen3-VL-8B-Instruct","messages":[{"role":"system","content":uuid.uuid4().hex},{"role":"user","content":"列出推理服务检查项并持续输出。"}],"temperature":0,"max_tokens":128,"ignore_eos":True,"stream":True,"stream_options":{"include_usage":True}}
    start=time.perf_counter(); first=None; status=0
    async with client.stream("POST",args.base_url+"/v1/chat/completions",json=payload) as r:
        status=r.status_code
        async for line in r.aiter_lines():
            if status!=200: continue
            if line.startswith("data: ") and line[6:]!="[DONE]":
                e=json.loads(line[6:]); c=e.get("choices") or []; content=c[0].get("delta",{}).get("content") if c else None
                if content: first=first or time.perf_counter()
    end=time.perf_counter(); return {"repeat":repeat,"slot":slot,"status":status,"ttft_ms":((first-start)*1000 if first else None),"e2e_ms":(end-start)*1000}
async def run(args):
    async with httpx.AsyncClient(timeout=httpx.Timeout(300)) as c:
        rows=[]; wall=0
        for repeat in range(args.repeats):
            start=time.perf_counter(); rows += await asyncio.gather(*[one(c,args,repeat,i) for i in range(args.concurrency)]); wall += time.perf_counter()-start
    ok=[x for x in rows if x["status"]==200]; good=[x for x in ok if x["ttft_ms"]<=args.ttft_slo_ms]
    summary={"variant":args.variant,"offered":len(rows),"accepted":len(ok),"rejected":len(rows)-len(ok),"wall_s":round(wall,3),
      "accepted_ttft_p95_ms":round(pct([x["ttft_ms"] for x in ok],95),2) if ok else None,"slo_goodput_rps":round(len(good)/wall,3),"accepted_slo_pass_pct":round(100*len(good)/len(ok),2) if ok else 0}
    args.output_dir.mkdir(parents=True,exist_ok=True); (args.output_dir/"requests.jsonl").write_text("".join(json.dumps(x)+"\n" for x in rows)); (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
def main():
    p=argparse.ArgumentParser(); p.add_argument("--variant",required=True); p.add_argument("--base-url",required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--concurrency",type=int,default=32); p.add_argument("--repeats",type=int,default=5); p.add_argument("--ttft-slo-ms",type=float,default=500); asyncio.run(run(p.parse_args()))
if __name__=="__main__": main()
