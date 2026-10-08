# Qwen3-VL 项目交互式报告

该成果页使用 `report_snapshot.json` 中的 reviewed 数据，集中展示 FP8、SLO 容量、长上下文 TTFT、调度 A/B 和 Nsight Kernel 证据。

已构建的静态文件位于 `dist/`。本地预览：

```bash
python3 -m http.server 4173 --directory dist
```

然后访问 <http://127.0.0.1:4173/>。

完整技术叙事与证据索引见 `../docs/AI_INFERENCE_PROJECT_REPORT.md`。原始性能数据不复制到页面源码，仍以 `../results/` 和 `../profiles/` 为准。
