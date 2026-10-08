#!/usr/bin/env python3
"""真实追加式多轮对话：逐轮测 TTFT 与 Prefix Cache token 命中率。"""
import argparse,asyncio,base64,csv,json,mimetypes,re,time,uuid
from datetime import datetime,timezone
from pathlib import Path
import httpx
from transformers import AutoTokenizer

def image_url(path):
    mime=mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
def material(tok,n):
    seed=tok.encode("历史对话包含图片目标、空间关系、部署约束和已经确认的推理结论。",add_special_tokens=False)
    return tok.decode((seed*(n//len(seed)+1))[:n]) if n else ""
def metric(text,name):
    values=[]
    for line in text.splitlines():
        if line.startswith(name+"{") or line.startswith(name+" "):
            try: values.append(float(line.rsplit(" ",1)[1]))
            except ValueError: pass
    return sum(values)
async def metrics(client,base):
    text=(await client.get(f"{base}/metrics")).text
    return {"queries":metric(text,"vllm:prefix_cache_queries_total"),"hits":metric(text,"vllm:prefix_cache_hits_total")}
async def completion(client,args,messages):
    payload={"model":args.model,"messages":messages,"temperature":0,"max_tokens":args.max_tokens,"stream":True,"stream_options":{"include_usage":True}}
    started=time.perf_counter(); first=None; usage={}; out=[]
    async with client.stream("POST",f"{args.base_url}/v1/chat/completions",json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "): continue
            raw=line[6:]
            if raw=="[DONE]": break
            event=json.loads(raw); usage=event.get("usage") or usage; choices=event.get("choices") or []
            content=choices[0].get("delta",{}).get("content") if choices else None
            if content: first=first or time.perf_counter(); out.append(content)
    if first is None: raise RuntimeError("没有流式文本 token")
    return "".join(out),int(usage.get("prompt_tokens") or 0),(first-started)*1000
async def run(args):
    tok=AutoTokenizer.from_pretrained(args.model_path,local_files_only=True); img=image_url(args.image); rows=[]
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout)) as client:
        (await client.get(f"{args.base_url}/health")).raise_for_status()
        for session in range(args.sessions):
            messages=[{"role":"system","content":f"会话 {uuid.uuid4().hex}。你是严谨的多模态助手。"}]
            previous=0
            for turn,target in enumerate(args.cumulative_history_tokens,1):
                delta=max(target-previous,0); previous=target
                content=[]
                if turn==1: content.append({"type":"image_url","image_url":{"url":img}})
                content.append({"type":"text","text":f"第{turn}轮新增信息：{material(tok,delta)} 请简短确认。"})
                messages.append({"role":"user","content":content}); before=await metrics(client,args.base_url)
                answer,prompt_tokens,ttft=await completion(client,args,messages); after=await metrics(client,args.base_url)
                queries=after["queries"]-before["queries"]; hits=after["hits"]-before["hits"]
                rows.append({"session":session,"turn":turn,"target_cumulative_history_tokens":target,"prompt_tokens":prompt_tokens,
                  "cache_query_tokens":queries,"cache_hit_tokens":hits,"cache_hit_rate_pct":100*hits/queries if queries else 0,"ttft_ms":ttft,
                  "started_at":datetime.now(timezone.utc).isoformat()})
                messages.append({"role":"assistant","content":answer})
    args.output_dir.mkdir(parents=True,exist_ok=True); (args.output_dir/"requests.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows))
    summary=[]
    for turn,target in enumerate(args.cumulative_history_tokens,1):
        v=[x for x in rows if x["turn"]==turn]; summary.append({"turn":turn,"target_cumulative_history_tokens":target,
          "prompt_tokens_mean":round(sum(x["prompt_tokens"] for x in v)/len(v),2),"cache_hit_rate_mean_pct":round(sum(x["cache_hit_rate_pct"] for x in v)/len(v),2),
          "ttft_mean_ms":round(sum(x["ttft_ms"] for x in v)/len(v),2),"samples":len(v)})
    with (args.output_dir/"summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    (args.output_dir/"summary.json").write_text(json.dumps({"metadata":{"sessions":args.sessions,"max_tokens":args.max_tokens,"mode":"append-only real conversation"},"turns":summary},ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
def main():
    p=argparse.ArgumentParser(); p.add_argument("--base-url",default="http://127.0.0.1:8000"); p.add_argument("--model",default="Qwen3-VL-8B-Instruct")
    p.add_argument("--model-path",type=Path,default=Path("models/Qwen3-VL-8B-Instruct")); p.add_argument("--image",type=Path,default=Path("images/test.jpg"))
    p.add_argument("--cumulative-history-tokens",type=int,nargs="+",default=[0,512,1024,2048,4096,8192]); p.add_argument("--sessions",type=int,default=5)
    p.add_argument("--max-tokens",type=int,default=16); p.add_argument("--timeout",type=float,default=300); p.add_argument("--output-dir",type=Path,default=Path("results/conversation_cache"))
    asyncio.run(run(p.parse_args()))
if __name__=="__main__": main()
