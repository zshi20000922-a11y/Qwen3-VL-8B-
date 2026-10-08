import json
import os

import requests
from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__)
UPSTREAM = os.environ.get("LIGHTLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = (5, None)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/models")
def models():
    try:
        upstream = requests.get(f"{UPSTREAM}/v1/models", timeout=5)
        return Response(upstream.content, status=upstream.status_code,
                        content_type=upstream.headers.get("Content-Type", "application/json"))
    except requests.RequestException as exc:
        return jsonify(error=f"无法连接 LightLLM ({UPSTREAM}): {exc}"), 502


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return jsonify(error="messages 必须是非空数组"), 400

    upstream_payload = {
        "model": payload.get("model", "Qwen3-VL-8B-Instruct"),
        "messages": messages,
        "stream": True,
        "temperature": payload.get("temperature", 0.7),
        "max_tokens": payload.get("max_tokens", 1024),
    }
    try:
        upstream = requests.post(
            f"{UPSTREAM}/v1/chat/completions",
            json=upstream_payload,
            headers={"Accept": "text/event-stream"},
            stream=True,
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        return jsonify(error=f"请求 LightLLM 失败: {exc}"), 502

    if upstream.status_code >= 400:
        body = upstream.content.decode("utf-8", errors="replace")
        upstream.close()
        return jsonify(error=f"LightLLM 返回 HTTP {upstream.status_code}: {body}"), 502

    def relay():
        try:
            for chunk in upstream.iter_content(chunk_size=1024):
                if chunk:
                    yield chunk
        finally:
            upstream.close()

    return Response(relay(), content_type="text/event-stream; charset=utf-8",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    app.run(host=os.environ.get("WEBUI_HOST", "127.0.0.1"),
            port=int(os.environ.get("WEBUI_PORT", "7860")), threaded=True)
