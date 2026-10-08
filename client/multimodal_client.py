#!/usr/bin/env python3
import argparse
import base64
import mimetypes
from pathlib import Path

from openai import OpenAI


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL OpenAI API 图片问答客户端")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--image", type=Path, default=Path("images/test.jpg"))
    parser.add_argument("--prompt", default="请描述这张图片。")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    stream = client.chat.completions.create(
        model="Qwen3-VL-8B-Instruct",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url(args.image)}},
            {"type": "text", "text": args.prompt},
        ]}],
        temperature=0,
        max_tokens=args.max_tokens,
        stream=True,
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content if chunk.choices else None
        if content:
            print(content, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
