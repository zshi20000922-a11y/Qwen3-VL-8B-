# Qwen3-VL-8B-Instruct 模型文件说明

| 文件 | 格式与作用 | 推理时是否关键 |
|---|---|---|
| `config.json` | Hugging Face 主配置：架构名、文本层数/维度、注意力头、视觉编码器、token id、dtype | 是 |
| `configuration.json` | 模型仓库附带的另一份配置/兼容元数据；常规 `from_pretrained` 主要读取 `config.json` | 通常否 |
| `generation_config.json` | 默认生成参数，如 EOS/PAD token、采样默认值 | 是，可被命令/API 覆盖 |
| `model-00001-of-00004.safetensors` 等 | 分片模型权重。Safetensors 是只存张量的安全、可 mmap 格式，不执行任意 Python 代码 | 是，四片缺一不可 |
| `model.safetensors.index.json` | 权重索引：把每个 tensor 名映射到对应分片，同时记录总大小 | 是 |
| `tokenizer.json` | 完整 tokenizer 规则与词表的序列化文件，通常是 Rust fast tokenizer 的主入口 | 是 |
| `tokenizer_config.json` | tokenizer 类型、特殊 token、最大长度等外围配置 | 是 |
| `vocab.json` | token 到 id 的词表 | 是，通常由 tokenizer 加载 |
| `merges.txt` | BPE 合并规则，决定字符/子词如何组合 | 是，通常由 tokenizer 加载 |
| `chat_template.json` | 将 system/user/assistant 及图片、视频内容渲染成模型训练时的对话格式 | 聊天推理关键 |
| `preprocessor_config.json` | 图片 resize、归一化、patch/像素范围等视觉预处理配置 | 图像输入关键 |
| `video_preprocessor_config.json` | 视频抽帧、尺寸及预处理参数 | 视频输入关键 |
| `README.md` | Model Card：模型说明、示例、许可证/限制提示 | 运行不需要，使用前应读 |
| `.gitattributes` | Git LFS 和仓库文本/二进制属性 | 本地推理不需要 |

这里没有 `.bin` 或 `.pt` 并不异常：四个 `.safetensors` 就是模型参数。也没有单独的 `mmproj`，因为 Hugging Face/vLLM 从同一组权重和 `vision_config` 加载视觉编码器与投影层；llama.cpp 的 GGUF 路线才通常把语言模型和多模态 projector 分开。

## 为什么确定不是 MoE

主配置中的架构是 `Qwen3VLForConditionalGeneration`，文本侧 36 层、hidden size 4096、intermediate size 12288，并未出现专家数量、每 token 激活专家数或 router 参数。vLLM 官方也把它与 `Qwen3VLMoeForConditionalGeneration` 分成两个架构条目。因此该 8B 版本每个 token 都走各层的同一套 Dense FFN。

