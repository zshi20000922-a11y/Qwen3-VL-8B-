import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "./models/Qwen3-0.6B"

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True,
).eval()

messages = [
    {
        "role": "user",
        "content": "请解释大语言模型推理中的Prefill和Decode阶段。",
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

inputs = tokenizer(
    text,
    return_tensors="pt",
).to("cuda")

input_tokens = inputs.input_ids.shape[1]

# 先预热，避免首次CUDA初始化影响测试
with torch.inference_mode():
    _ = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
        use_cache=True,
    )

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

start = time.perf_counter()

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        use_cache=True,
    )

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

generated_ids = outputs[0, input_tokens:]
output_tokens = generated_ids.numel()

answer = tokenizer.decode(
    generated_ids,
    skip_special_tokens=True,
)

peak_memory = torch.cuda.max_memory_allocated() / 1024**3

print("\n========== 模型输出 ==========")
print(answer)

print("\n========== 性能结果 ==========")
print(f"输入Token数      : {input_tokens}")
print(f"输出Token数      : {output_tokens}")
print(f"总推理时间       : {elapsed:.3f} s")
print(f"平均输出吞吐     : {output_tokens / elapsed:.2f} token/s")
print(f"峰值显存         : {peak_memory:.3f} GiB")
