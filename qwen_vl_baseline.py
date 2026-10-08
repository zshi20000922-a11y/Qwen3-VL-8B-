import argparse
import statistics
import time
from pathlib import Path

import torch
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)
from transformers.generation.streamers import BaseStreamer


class TokenTimingStreamer(BaseStreamer):
    """
    记录generate()过程中每个新Token产生的时间。

    Transformers通常第一次调用put()时传入原始Prompt，
    后续每次put()对应一个新生成Token。
    """

    def __init__(self):
        super().__init__()
        self.prompt_received = False
        self.token_timestamps = []
        self.stream_end_time = None

    def put(self, value):
        if not self.prompt_received:
            # 第一次put通常是完整Prompt，不属于输出Token
            self.prompt_received = True
            return

        self.token_timestamps.append(time.perf_counter())

    def end(self):
        self.stream_end_time = time.perf_counter()


def percentile(values, percent):
    """计算百分位数，使用线性插值。"""

    if not values:
        return 0.0

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * percent / 100.0
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index

    return (
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )


def gib(num_bytes):
    """Byte转换为GiB。"""
    return num_bytes / (1024 ** 3)


def mib(num_bytes):
    """Byte转换为MiB。"""
    return num_bytes / (1024 ** 2)


def get_model_parameter_memory(model):
    """统计模型参数理论占用空间。"""

    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )

    buffer_bytes = sum(
        buffer.numel() * buffer.element_size()
        for buffer in model.buffers()
    )

    return parameter_bytes, buffer_bytes


def count_special_tokens(inputs, model):
    """
    尝试统计视觉和视频占位Token数量。
    不同Transformers版本的配置字段可能不同。
    """

    input_ids = inputs["input_ids"]

    image_token_id = getattr(
        model.config,
        "image_token_id",
        None,
    )

    video_token_id = getattr(
        model.config,
        "video_token_id",
        None,
    )

    image_tokens = 0
    video_tokens = 0

    if image_token_id is not None:
        image_tokens = int(
            (input_ids == image_token_id).sum().item()
        )

    if video_token_id is not None:
        video_tokens = int(
            (input_ids == video_token_id).sum().item()
        )

    return {
        "image_token_id": image_token_id,
        "video_token_id": video_token_id,
        "image_tokens": image_tokens,
        "video_tokens": video_tokens,
    }


def print_gpu_information():
    """打印GPU基础信息。"""

    properties = torch.cuda.get_device_properties(0)

    print("\n========== GPU环境 ==========")
    print(f"PyTorch版本       : {torch.__version__}")
    print(f"PyTorch CUDA      : {torch.version.cuda}")
    print(f"GPU名称           : {torch.cuda.get_device_name(0)}")
    print(
        "Compute Capability:",
        torch.cuda.get_device_capability(0),
    )
    print(
        f"GPU总显存         : "
        f"{gib(properties.total_memory):.2f} GiB"
    )
    print(
        f"BF16支持          : "
        f"{torch.cuda.is_bf16_supported()}"
    )


def build_messages(image_path, prompt):
    """构造多模态Chat消息。"""

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path),
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]


def prepare_inputs(processor, messages, device):
    """
    完成：
    1. 图片读取和预处理
    2. 文本Tokenization
    3. 多模态Chat Template
    4. CPU到GPU传输
    """

    preprocess_start = time.perf_counter()

    cpu_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    preprocess_end = time.perf_counter()

    torch.cuda.synchronize()
    transfer_start = time.perf_counter()

    gpu_inputs = cpu_inputs.to(device)

    torch.cuda.synchronize()
    transfer_end = time.perf_counter()

    return {
        "inputs": gpu_inputs,
        "preprocess_time": preprocess_end - preprocess_start,
        "transfer_time": transfer_end - transfer_start,
    }


def warmup_model(model, inputs, warmup_tokens):
    """
    预热CUDA Context、Kernel和Allocator。

    预热不计入正式测试结果。
    """

    print(
        f"\n开始预热，生成 {warmup_tokens} 个Token……"
    )

    with torch.inference_mode():
        _ = model.generate(
            **inputs,
            max_new_tokens=warmup_tokens,
            do_sample=False,
            use_cache=True,
        )

    torch.cuda.synchronize()

    print("预热完成。")


def run_benchmark(
    model,
    processor,
    inputs,
    max_new_tokens,
):
    """执行一次正式推理并记录完整指标。"""

    prompt_tokens = inputs["input_ids"].shape[-1]

    streamer = TokenTimingStreamer()

    # 清理预热阶段产生的临时缓存
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # 将当前模型常驻显存设置为峰值统计起点
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()

    memory_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()

    benchmark_start = time.perf_counter()

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            streamer=streamer,
        )

    torch.cuda.synchronize()
    benchmark_end = time.perf_counter()

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    memory_after = torch.cuda.memory_allocated()
    reserved_after = torch.cuda.memory_reserved()

    generated_ids_trimmed = generated_ids[:, prompt_tokens:]

    output_tokens = generated_ids_trimmed.shape[-1]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    timestamps = streamer.token_timestamps

    # 某些特殊停止条件可能导致Streamer统计数与输出略有差异
    timing_token_count = len(timestamps)

    e2e_latency = benchmark_end - benchmark_start

    if timing_token_count >= 1:
        ttft = timestamps[0] - benchmark_start
    else:
        ttft = 0.0

    if timing_token_count >= 2:
        inter_token_latencies = [
            timestamps[index] - timestamps[index - 1]
            for index in range(1, timing_token_count)
        ]

        decode_duration = (
            timestamps[-1] - timestamps[0]
        )

        tpot = statistics.mean(inter_token_latencies)

        decode_throughput = (
            (timing_token_count - 1) / decode_duration
            if decode_duration > 0
            else 0.0
        )
    else:
        inter_token_latencies = []
        decode_duration = 0.0
        tpot = 0.0
        decode_throughput = 0.0

    end_to_end_throughput = (
        output_tokens / e2e_latency
        if e2e_latency > 0
        else 0.0
    )

    # 这是包含视觉编码和首次Decode的有效吞吐，
    # 不是严格意义上的纯LLM Prefill吞吐
    effective_input_throughput = (
        prompt_tokens / ttft
        if ttft > 0
        else 0.0
    )

    return {
        "output_text": output_text,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "timing_token_count": timing_token_count,
        "ttft": ttft,
        "tpot": tpot,
        "decode_duration": decode_duration,
        "decode_throughput": decode_throughput,
        "e2e_latency": e2e_latency,
        "e2e_throughput": end_to_end_throughput,
        "effective_input_throughput":
            effective_input_throughput,
        "itl_mean": (
            statistics.mean(inter_token_latencies)
            if inter_token_latencies
            else 0.0
        ),
        "itl_min": (
            min(inter_token_latencies)
            if inter_token_latencies
            else 0.0
        ),
        "itl_max": (
            max(inter_token_latencies)
            if inter_token_latencies
            else 0.0
        ),
        "itl_p50": percentile(
            inter_token_latencies,
            50,
        ),
        "itl_p90": percentile(
            inter_token_latencies,
            90,
        ),
        "itl_p95": percentile(
            inter_token_latencies,
            95,
        ),
        "itl_p99": percentile(
            inter_token_latencies,
            99,
        ),
        "memory_before": memory_before,
        "memory_after": memory_after,
        "reserved_before": reserved_before,
        "reserved_after": reserved_after,
        "peak_allocated": peak_allocated,
        "peak_reserved": peak_reserved,
    }


def print_results(
    results,
    preprocess_time,
    transfer_time,
    special_tokens,
    parameter_bytes,
    buffer_bytes,
):
    """格式化输出基准结果。"""

    print("\n========== 模型输出 ==========")
    print(results["output_text"])

    print("\n========== 输入统计 ==========")
    print(
        f"输入Token总数          : "
        f"{results['prompt_tokens']}"
    )
    print(
        f"图像占位Token数        : "
        f"{special_tokens['image_tokens']}"
    )
    print(
        f"视频占位Token数        : "
        f"{special_tokens['video_tokens']}"
    )
    print(
        f"图像Token ID           : "
        f"{special_tokens['image_token_id']}"
    )
    print(
        f"输出Token数            : "
        f"{results['output_tokens']}"
    )
    print(
        f"Streamer记录Token数    : "
        f"{results['timing_token_count']}"
    )

    print("\n========== 阶段耗时 ==========")
    print(
        f"CPU图片与文本预处理    : "
        f"{preprocess_time * 1000:.2f} ms"
    )
    print(
        f"CPU→GPU输入传输        : "
        f"{transfer_time * 1000:.2f} ms"
    )
    print(
        f"TTFT首Token延迟        : "
        f"{results['ttft'] * 1000:.2f} ms"
    )
    print(
        f"Decode阶段时间         : "
        f"{results['decode_duration']:.3f} s"
    )
    print(
        f"端到端生成时间         : "
        f"{results['e2e_latency']:.3f} s"
    )

    print("\n========== Token性能 ==========")
    print(
        f"TPOT平均单Token时间    : "
        f"{results['tpot'] * 1000:.3f} ms/token"
    )
    print(
        f"Decode吞吐             : "
        f"{results['decode_throughput']:.2f} token/s"
    )
    print(
        f"端到端输出吞吐         : "
        f"{results['e2e_throughput']:.2f} token/s"
    )
    print(
        f"有效输入处理吞吐       : "
        f"{results['effective_input_throughput']:.2f} token/s"
    )

    print("\n========== ITL分布 ==========")
    print(
        f"ITL平均值              : "
        f"{results['itl_mean'] * 1000:.3f} ms"
    )
    print(
        f"ITL最小值              : "
        f"{results['itl_min'] * 1000:.3f} ms"
    )
    print(
        f"ITL最大值              : "
        f"{results['itl_max'] * 1000:.3f} ms"
    )
    print(
        f"ITL P50                : "
        f"{results['itl_p50'] * 1000:.3f} ms"
    )
    print(
        f"ITL P90                : "
        f"{results['itl_p90'] * 1000:.3f} ms"
    )
    print(
        f"ITL P95                : "
        f"{results['itl_p95'] * 1000:.3f} ms"
    )
    print(
        f"ITL P99                : "
        f"{results['itl_p99'] * 1000:.3f} ms"
    )

    print("\n========== 显存统计 ==========")
    print(
        f"模型参数理论显存       : "
        f"{gib(parameter_bytes):.3f} GiB"
    )
    print(
        f"模型Buffer理论显存     : "
        f"{mib(buffer_bytes):.2f} MiB"
    )
    print(
        f"推理前Allocated        : "
        f"{gib(results['memory_before']):.3f} GiB"
    )
    print(
        f"推理后Allocated        : "
        f"{gib(results['memory_after']):.3f} GiB"
    )
    print(
        f"峰值Allocated          : "
        f"{gib(results['peak_allocated']):.3f} GiB"
    )
    print(
        f"推理前Reserved         : "
        f"{gib(results['reserved_before']):.3f} GiB"
    )
    print(
        f"推理后Reserved         : "
        f"{gib(results['reserved_after']):.3f} GiB"
    )
    print(
        f"峰值Reserved           : "
        f"{gib(results['peak_reserved']):.3f} GiB"
    )

    print("\n========== 指标说明 ==========")
    print("TTFT包含：视觉编码 + 多模态Prefill + 第一次Decode")
    print("TPOT反映：后续逐Token Decode的平均耗时")
    print("ITL反映：相邻输出Token之间的延迟分布")
    print("Allocated：PyTorch张量实际占用的GPU显存")
    print("Reserved：PyTorch缓存分配器保留的GPU显存")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL单请求推理性能基准"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="./models/Qwen3-VL-8B-Instruct",
        help="本地模型目录",
    )

    parser.add_argument(
        "--image",
        type=str,
        default="./images/test.jpg",
        help="输入图片路径",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "请详细描述图片中的主要目标、场景、目标状态以及"
            "目标之间的空间关系。如果存在无人机，请指出其"
            "位置、类型和可能的运动状态。"
        ),
        help="文本提示词",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="最大输出Token数量",
    )

    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=8,
        help="预热生成Token数量",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    model_path = Path(args.model).expanduser().resolve()
    image_path = Path(args.image).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            f"模型目录不存在：{model_path}"
        )

    if not image_path.exists():
        raise FileNotFoundError(
            f"图片不存在：{image_path}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")

    print_gpu_information()

    print("\n========== 加载模型 ==========")
    print(f"模型路径：{model_path}")
    print(f"图片路径：{image_path}")

    load_start = time.perf_counter()

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    ).eval()

    torch.cuda.synchronize()
    load_end = time.perf_counter()

    print(
        f"模型加载时间：{load_end - load_start:.3f} s"
    )

    parameter_bytes, buffer_bytes = (
        get_model_parameter_memory(model)
    )

    messages = build_messages(
        image_path=image_path,
        prompt=args.prompt,
    )

    preparation = prepare_inputs(
        processor=processor,
        messages=messages,
        device=model.device,
    )

    inputs = preparation["inputs"]

    special_tokens = count_special_tokens(
        inputs=inputs,
        model=model,
    )

    warmup_model(
        model=model,
        inputs=inputs,
        warmup_tokens=args.warmup_tokens,
    )

    results = run_benchmark(
        model=model,
        processor=processor,
        inputs=inputs,
        max_new_tokens=args.max_new_tokens,
    )

    print_results(
        results=results,
        preprocess_time=preparation["preprocess_time"],
        transfer_time=preparation["transfer_time"],
        special_tokens=special_tokens,
        parameter_bytes=parameter_bytes,
        buffer_bytes=buffer_bytes,
    )


if __name__ == "__main__":
    main()