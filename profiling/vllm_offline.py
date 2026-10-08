#!/usr/bin/env python3
"""Single-process vLLM workload so Nsight can observe production kernels."""
import argparse, json, os, time
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor

os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING','0')
os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER','0')
from vllm import LLM, SamplingParams

p=argparse.ArgumentParser(); p.add_argument('--variant',choices=['bf16','fp8'],default='bf16')
p.add_argument('--scenario',choices=['cold','steady','prefill','decode'],default='steady'); p.add_argument('--concurrency',type=int,default=1)
p.add_argument('--output-tokens',type=int,default=256); p.add_argument('--output',required=True); a=p.parse_args()
root=Path(__file__).resolve().parents[1]; model=root/'models/Qwen3-VL-8B-Instruct'
processor=AutoProcessor.from_pretrained(model,local_files_only=True)
image_path=root/('images/resolution_test/3840x2160.jpg' if a.scenario=='prefill' else 'images/resolution_test/640x360.jpg' if a.scenario=='decode' else 'images/test.jpg')
messages=[{'role':'user','content':[{'type':'image','image':str(image_path)},{'type':'text','text':'客观描述图片中的主要目标、背景和运动状态。'}]}]
prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
kwargs=dict(model=str(model),dtype='bfloat16',max_model_len=16384,max_num_seqs=8,enable_prefix_caching=True,
            limit_mm_per_prompt={'image':1,'video':0},gpu_memory_utilization=.90)
if a.variant=='fp8': kwargs['quantization']='fp8'
llm=LLM(**kwargs); params=SamplingParams(temperature=0,max_tokens=a.output_tokens)
request={'prompt':prompt,'multi_modal_data':{'image':Image.open(image_path).convert('RGB')}}
if a.scenario!='cold': llm.generate([request],SamplingParams(temperature=0,max_tokens=16))
torch.cuda.synchronize(); torch.cuda.nvtx.range_push(f'{a.variant}:{a.scenario}')
start=time.perf_counter(); outputs=llm.generate([request for _ in range(a.concurrency)],params); torch.cuda.synchronize(); elapsed=time.perf_counter()-start
torch.cuda.nvtx.range_pop(); tokens=len(outputs[0].outputs[0].token_ids)
result={'variant':a.variant,'scenario':a.scenario,'concurrency':a.concurrency,'output_tokens':tokens*a.concurrency,'e2e_ms':elapsed*1000,'output_tok_s':tokens*a.concurrency/elapsed,
        'image':str(image_path.relative_to(root)),'profiling_mode':'vllm_offline_single_process_same_model_kernels'}
Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if torch.distributed.is_available() and torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
