# LightLLM Web 聊天界面

适用于已启动 OpenAI 兼容接口的 LightLLM，默认连接 `http://127.0.0.1:8000`，网页监听 `127.0.0.1:7860`。

启动：

```bash
python lightllm_webui/app.py
```

浏览器访问 <http://127.0.0.1:7860>。如果 LightLLM 在其他地址，可用 `LIGHTLLM_BASE_URL=http://主机:端口 python lightllm_webui/app.py` 启动。默认仅本机可访问；需要局域网访问时可设置 `WEBUI_HOST=0.0.0.0`。
