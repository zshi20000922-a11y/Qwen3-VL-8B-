#!/usr/bin/env python3
"""保存固定多模态样例的完整输出，用于 BF16/FP8 人工质量复核。"""

import argparse
import base64
import json
import mimetypes
from datetime import datetime, timezone
from pathlib import Path

import httpx


def data_url(path):
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prompts = [
        "客观描述图片中的船只、背景和运动状态，不要猜测看不清的信息。",
        "读取图片中清晰可见的文字，并指出文字所在目标；不确定的字符用问号表示。",
    ]
    cases = []
    with httpx.Client(timeout=300) as client:
        for image in [Path("images/test.jpg"), Path("images/resolution_test/640x360.jpg")]:
            for prompt in prompts:
                response = client.post(f"{args.base_url}/v1/chat/completions", json={
                    "model": "Qwen3-VL-8B-Instruct", "temperature": 0, "max_tokens": 256,
                    "messages": [{"role":"user", "content":[
                        {"type":"image_url", "image_url":{"url":data_url(image)}},
                        {"type":"text", "text":prompt},
                    ]}],
                })
                response.raise_for_status()
                body = response.json()
                cases.append({
                    "image": str(image), "prompt": prompt,
                    "answer": body["choices"][0]["message"]["content"],
                    "usage": body.get("usage", {}),
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"variant": args.variant, "generated_at": datetime.now(timezone.utc).isoformat(), "cases": cases}, ensure_ascii=False, indent=2))
    print(f"保存 {len(cases)} 个 {args.variant} 质量样例到 {args.output}")


if __name__ == "__main__":
    main()
