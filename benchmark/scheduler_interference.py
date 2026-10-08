#!/usr/bin/env python3
"""长 Prefill 与短 Decode 混部时的调度干扰实验。"""
import argparse,asyncio,csv,json,statistics,time,uuid
from datetime import datetime,timezone
from pathlib import Path
import httpx
from transformers import AutoTokenizer

def pct(v,p):
    v=sorted(v); x=(len(v)-1)*p/100; lo=int(x); hi=min(lo+1,len(v)-1); return v[lo]+(v[hi]-v[lo])*(x-lo)
def long_text(tok,n):
    seed=tok.encode("长上下文包含系统约束、历史事实、性能数据和需要保留的推理结论。",add_special_tokens=False)
    return tok.decode((seed*(n//len(seed)+1))[:n])
async def request(client,args,role,messages,max_tokens,repeat,slot):
    payload={"model":args.model,"messages":messages,"temperature":0,"max_tokens":max_tokens,"stream":True,"stream_options":{"include_usage":True},"ignore_eos":True}
    start=time.perf_counter(); first=None; usage={}
    async with client.stream("POST",f"{args.base_url}/v1/chat/completions",json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "): continue
            raw=line[6:]
            if raw=="[DONE]": break
            event=json.loads(raw); usage=event.get("usage") or usage; choices=event.get("choices") or []
            content=choices[0].get("delta",{}).get("content") if choices else None
            if content: first=first or time.perf_counter()
    end=time.perf_counter()
    if first is None: raise RuntimeError("no streamed token")
    out=int(usage.get("completion_tokens") or 0)
    return {"variant":args.variant,"role":role,"repeat":repeat,"slot":slot,"prompt_tokens":int(usage.get("prompt_tokens") or 0),"output_tokens":out,
      "ttft_ms":(first-start)*1000,"tpot_ms":(end-first)*1000/max(out-1,1),"e2e_ms":(end-start)*1000,"started_at":datetime.now(timezone.utc).isoformat()}
async def group(client,args,tok,repeat):
    nonce=uuid.uuid4().hex
    long_msg=[{"role":"system","content":f"{nonce} 保留完整上下文。"},{"role":"user","content":long_text(tok,args.long_tokens)+"\n请总结。"}]
    long_task=asyncio.create_task(request(client,args,"long_prefill",long_msg,args.long_output_tokens,repeat,0))
    await asyncio.sleep(args.stagger_ms/1000)
    shorts=[]
    for slot in range(args.short_concurrency):
        short=[{"role":"system","content":f"{uuid.uuid4().hex} 输出完整检查清单。"},{"role":"user","content":"列出大模型推理服务的性能检查项。"}]
        shorts.append(asyncio.create_task(request(client,args,"short_decode",short,args.short_output_tokens,repeat,slot)))
    return [await long_task]+list(await asyncio.gather(*shorts))
async def run(args):
    tok=AutoTokenizer.from_pretrained(args.model_path,local_files_only=True); rows=[]; args.output_dir.mkdir(parents=True,exist_ok=True)
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout)) as client:
        (await client.get(f"{args.base_url}/health")).raise_for_status(); await group(client,args,tok,-1)
        for r in range(args.repeats): rows += await group(client,args,tok,r); print(f"{args.variant} {r+1}/{args.repeats}",flush=True)
    short=[x for x in rows if x["role"]=="short_decode"]; long=[x for x in rows if x["role"]=="long_prefill"]
    summary={"variant":args.variant,"repeats":args.repeats,"short_requests":len(short),"short_ttft_p50_ms":round(pct([x["ttft_ms"] for x in short],50),2),
      "short_ttft_p95_ms":round(pct([x["ttft_ms"] for x in short],95),2),"short_tpot_p95_ms":round(pct([x["tpot_ms"] for x in short],95),2),
      "long_ttft_p50_ms":round(pct([x["ttft_ms"] for x in long],50),2),"long_e2e_p50_ms":round(pct([x["e2e_ms"] for x in long],50),2)}
    (args.output_dir/"requests.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows))
    (args.output_dir/"summary.json").write_text(json.dumps({"metadata":{"long_tokens":args.long_tokens,"short_concurrency":args.short_concurrency,"stagger_ms":args.stagger_ms},"summary":summary},ensure_ascii=False,indent=2))
    with (args.output_dir/"summary.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(summary)); w.writeheader(); w.writerow(summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
def main():
    p=argparse.ArgumentParser(); p.add_argument("--variant",required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--base-url",default="http://127.0.0.1:8000"); p.add_argument("--model",default="Qwen3-VL-8B-Instruct")
    p.add_argument("--model-path",type=Path,default=Path("models/Qwen3-VL-8B-Instruct")); p.add_argument("--long-tokens",type=int,default=12000); p.add_argument("--long-output-tokens",type=int,default=16)
    p.add_argument("--short-concurrency",type=int,default=4); p.add_argument("--short-output-tokens",type=int,default=128); p.add_argument("--stagger-ms",type=float,default=50); p.add_argument("--repeats",type=int,default=5); p.add_argument("--timeout",type=float,default=300)
    asyncio.run(run(p.parse_args()))
if __name__=="__main__": main()
