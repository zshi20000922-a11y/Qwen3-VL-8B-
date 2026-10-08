#!/usr/bin/env python3
"""固定 shape、不同图片内容的视觉 Encoder TTFT A/B workload。"""
import argparse,asyncio,base64,csv,hashlib,io,json,statistics,time
from datetime import datetime,timezone
from pathlib import Path
import httpx
from PIL import Image

def pct(v,p):
    v=sorted(v); x=(len(v)-1)*p/100; lo=int(x); hi=min(lo+1,len(v)-1)
    return v[lo]+(v[hi]-v[lo])*(x-lo)
def variants(path,count):
    base=Image.open(path).convert("RGB"); result=[]
    for i in range(count):
        image=base.copy()
        # 单像素会在 smart_resize 中被抹平，导致预处理张量相同而命中缓存。
        color=((37*(i+1))%256,(83*(i+1))%256,(149*(i+1))%256)
        for y in range(64):
            for x in range(64): image.putpixel((x,y),color)
        buf=io.BytesIO(); image.save(buf,format="JPEG",quality=95); raw=buf.getvalue()
        result.append("data:image/jpeg;base64,"+base64.b64encode(raw).decode())
    return result
async def request(client,args,image,repeat,warmup):
    payload={"model":args.model,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":image}},
      {"type":"text","text":"请用一句中文描述图片的主要场景。"}]}],"temperature":0,"max_tokens":args.max_tokens,"stream":True,"stream_options":{"include_usage":True}}
    start=time.perf_counter(); first=None; usage={}; output=[]
    async with client.stream("POST",f"{args.base_url}/v1/chat/completions",json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "): continue
            raw=line[6:]
            if raw=="[DONE]": break
            event=json.loads(raw); usage=event.get("usage") or usage; choices=event.get("choices") or []
            content=choices[0].get("delta",{}).get("content") if choices else None
            if content: first=first or time.perf_counter(); output.append(content)
    end=time.perf_counter()
    if first is None: raise RuntimeError("no streamed token")
    text="".join(output)
    return {"variant":args.variant,"repeat":repeat,"warmup":warmup,"prompt_tokens":int(usage.get("prompt_tokens") or 0),
      "output_tokens":int(usage.get("completion_tokens") or 0),"ttft_ms":(first-start)*1000,"e2e_ms":(end-start)*1000,
      "output_sha256":hashlib.sha256(text.encode()).hexdigest(),"output_text":text,"started_at":datetime.now(timezone.utc).isoformat()}
async def run(args):
    images=variants(args.image,args.warmups+args.repeats); args.output_dir.mkdir(parents=True,exist_ok=True); rows=[]
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout)) as client:
        (await client.get(f"{args.base_url}/health")).raise_for_status()
        for i in range(args.warmups): await request(client,args,images[i],-1,True)
        for i in range(args.repeats): rows.append(await request(client,args,images[args.warmups+i],i,False))
    summary={"variant":args.variant,"samples":len(rows),"prompt_tokens_mean":statistics.mean(x["prompt_tokens"] for x in rows),
      "ttft_mean_ms":round(statistics.mean(x["ttft_ms"] for x in rows),2),"ttft_p50_ms":round(pct([x["ttft_ms"] for x in rows],50),2),
      "ttft_p95_ms":round(pct([x["ttft_ms"] for x in rows],95),2),"e2e_p50_ms":round(pct([x["e2e_ms"] for x in rows],50),2)}
    (args.output_dir/"requests.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows))
    with (args.output_dir/"summary.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(summary)); w.writeheader(); w.writerow(summary)
    (args.output_dir/"summary.json").write_text(json.dumps({"metadata":{"variant":args.variant,"warmups":args.warmups,"repeats":args.repeats,"image":str(args.image)},"summary":summary},ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
def main():
    p=argparse.ArgumentParser(); p.add_argument("--variant",required=True); p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--base-url",default="http://127.0.0.1:8000"); p.add_argument("--model",default="Qwen3-VL-8B-Instruct"); p.add_argument("--image",type=Path,default=Path("images/test.jpg"))
    p.add_argument("--warmups",type=int,default=2); p.add_argument("--repeats",type=int,default=5); p.add_argument("--max-tokens",type=int,default=16); p.add_argument("--timeout",type=float,default=300)
    asyncio.run(run(p.parse_args()))
if __name__=="__main__": main()
