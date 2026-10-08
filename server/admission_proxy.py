#!/usr/bin/env python3
"""面向 vLLM 的轻量 Admission Control：限制在途请求并快速拒绝过载。"""
import asyncio,os
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse,StreamingResponse

UPSTREAM=os.getenv("UPSTREAM_URL","http://127.0.0.1:8000")
MAX_INFLIGHT=int(os.getenv("ADMISSION_MAX_INFLIGHT","2")); WAIT_MS=float(os.getenv("ADMISSION_WAIT_MS","50"))
semaphore=asyncio.Semaphore(MAX_INFLIGHT); client=None
@asynccontextmanager
async def lifespan(app):
    global client; client=httpx.AsyncClient(timeout=httpx.Timeout(300)); yield; await client.aclose()
app=FastAPI(lifespan=lifespan)
@app.get("/health")
async def health(): return {"status":"ok","max_inflight":MAX_INFLIGHT,"available":semaphore._value}
@app.post("/v1/chat/completions")
async def completions(request:Request):
    try: await asyncio.wait_for(semaphore.acquire(),timeout=WAIT_MS/1000)
    except asyncio.TimeoutError:
        return JSONResponse(status_code=429,headers={"Retry-After":"1"},content={"error":{"message":"GPU admission queue full","type":"overloaded"}})
    body=await request.body()
    async def stream():
        try:
            async with client.stream("POST",UPSTREAM+"/v1/chat/completions",content=body,headers={"content-type":"application/json"}) as response:
                async for chunk in response.aiter_raw(): yield chunk
        finally: semaphore.release()
    return StreamingResponse(stream(),media_type="text/event-stream")
