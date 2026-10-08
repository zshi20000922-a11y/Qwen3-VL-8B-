#!/usr/bin/env python3
"""并发饱和、TPOT 与 SLO Goodput 基准。"""
import argparse, asyncio, base64, csv, json, mimetypes, time, uuid
from datetime import datetime, timezone
from pathlib import Path
import httpx

def percentile(v,p):
    v=sorted(v); pos=(len(v)-1)*p/100; lo=int(pos); hi=min(lo+1,len(v)-1)
    return v[lo]+(v[hi]-v[lo])*(pos-lo)

def image_url(path):
    mime=mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"

async def request(client,args,image,concurrency,repeat,slot):
    nonce=uuid.uuid4().hex
    payload={"model":args.model,"messages":[
        {"role":"system","content":f"请求编号 {nonce}。逐项完成任务，不要提前结束。"},
        {"role":"user","content":[{"type":"image_url","image_url":{"url":image}},
         {"type":"text","text":"描述图片后，继续给出推理部署性能检查清单，保持输出直到长度上限。"}]}],
        "temperature":0,"max_tokens":args.max_tokens,"stream":True,
        "stream_options":{"include_usage":True},"ignore_eos":True}
    started=time.perf_counter(); first=None; usage={}
    async with client.stream("POST",f"{args.base_url}/v1/chat/completions",json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "): continue
            raw=line[6:]
            if raw=="[DONE]": break
            event=json.loads(raw); usage=event.get("usage") or usage; choices=event.get("choices") or []
            content=choices[0].get("delta",{}).get("content") if choices else None
            if content: first=first or time.perf_counter()
    ended=time.perf_counter()
    if first is None: raise RuntimeError("没有收到流式输出 token")
    output_tokens=int(usage.get("completion_tokens") or 0)
    ttft_ms=(first-started)*1000; tpot_ms=(ended-first)*1000/max(output_tokens-1,1)
    return {"concurrency":concurrency,"repeat":repeat,"slot":slot,
        "prompt_tokens":int(usage.get("prompt_tokens") or 0),"output_tokens":output_tokens,
        "ttft_ms":ttft_ms,"tpot_ms":tpot_ms,"e2e_ms":(ended-started)*1000,
        "meets_ttft_slo":ttft_ms<=args.ttft_slo_ms,"meets_tpot_slo":tpot_ms<=args.tpot_slo_ms,
        "started_at":datetime.now(timezone.utc).isoformat()}

def summarize(rows,groups,args):
    result=[]
    for c in args.concurrency:
        values=[x for x in rows if x["concurrency"]==c]; wall=sum(x["wall_s"] for x in groups if x["concurrency"]==c)
        good=[x for x in values if x["meets_ttft_slo"] and x["meets_tpot_slo"]]
        result.append({"concurrency":c,"requests":len(values),
          "ttft_p50_ms":round(percentile([x["ttft_ms"] for x in values],50),2),"ttft_p95_ms":round(percentile([x["ttft_ms"] for x in values],95),2),
          "tpot_p50_ms":round(percentile([x["tpot_ms"] for x in values],50),2),"tpot_p95_ms":round(percentile([x["tpot_ms"] for x in values],95),2),
          "e2e_p95_ms":round(percentile([x["e2e_ms"] for x in values],95),2),
          "aggregate_output_tok_s":round(sum(x["output_tokens"] for x in values)/wall,2),
          "request_throughput_rps":round(len(values)/wall,3),"goodput_rps":round(len(good)/wall,3),
          "slo_pass_rate_pct":round(100*len(good)/len(values),2)})
    return result

async def run(args):
    args.output_dir.mkdir(parents=True,exist_ok=True); image=image_url(args.image); rows=[]; groups=[]
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout)) as client:
        (await client.get(f"{args.base_url}/health")).raise_for_status(); await request(client,args,image,1,-1,0)
        for c in args.concurrency:
            for repeat in range(args.repeats):
                started=time.perf_counter(); batch=await asyncio.gather(*[request(client,args,image,c,repeat,s) for s in range(c)])
                wall=time.perf_counter()-started; rows.extend(batch); groups.append({"concurrency":c,"repeat":repeat,"wall_s":wall})
                print(f"c={c} repeat={repeat+1}/{args.repeats} wall={wall:.2f}s",flush=True)
    summary=summarize(rows,groups,args); meta={"generated_at":datetime.now(timezone.utc).isoformat(),"model":args.model,
      "repeats":args.repeats,"max_tokens":args.max_tokens,"ttft_slo_ms":args.ttft_slo_ms,"tpot_slo_ms":args.tpot_slo_ms,
      "cache_policy":"unique nonce in first text block; image processor cache may hit"}
    (args.output_dir/"requests.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows))
    (args.output_dir/"groups.jsonl").write_text("".join(json.dumps(x)+"\n" for x in groups))
    with (args.output_dir/"summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    (args.output_dir/"summary.json").write_text(json.dumps({"metadata":meta,"scenarios":summary},ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--base-url",default="http://127.0.0.1:8000"); p.add_argument("--model",default="Qwen3-VL-8B-Instruct")
    p.add_argument("--image",type=Path,default=Path("images/test.jpg")); p.add_argument("--concurrency",type=int,nargs="+",default=[1,2,4,8,16,32])
    p.add_argument("--repeats",type=int,default=3); p.add_argument("--max-tokens",type=int,default=128)
    p.add_argument("--ttft-slo-ms",type=float,default=500); p.add_argument("--tpot-slo-ms",type=float,default=30)
    p.add_argument("--timeout",type=float,default=300); p.add_argument("--output-dir",type=Path,default=Path("results/saturation"))
    a=p.parse_args()
    if a.repeats<2 or any(x<1 for x in a.concurrency): p.error("参数必须为正，repeats 至少为 2")
    asyncio.run(run(a))
if __name__=="__main__": main()
