# MLX INFERENCE

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![English](https://img.shields.io/badge/Docs-English-blue.svg)](README.md)
[![HitCount](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https://github.com/AreChen/mlx_inference)](https://github.com/AreChen/mlx_inference)

### 项目简介

MLX INFERENCE 是基于 MLX-LM 和 MLX-VLM 实现的兼容 OpenAI API 的推理服务，提供以下端点：
- `/v1/chat/completions` - 聊天补全接口
- `/v1/responses` - 响应接口
- `/v1/models` - 获取可用模型列表

### 安装依赖

```bash
pip install -r requirements.txt
# 复制env文件并编辑指定自己需要加载的模型
cp .env.example .env
```

### 启动服务

在项目根目录下执行：

```bash
uvicorn mlx_Inference:app --workers 1 --port 8002
```

参数说明：
- `--workers`: 工作进程数
- `--port`: 服务端口号

### 功能特性

- 兼容 OpenAI API 规范
- 后端推理使用 [MLX-LM](https://github.com/ml-explore/mlx-lm) 和 [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) ,可以使用[mlx-community](https://huggingface.co/mlx-community)模型
- 易于部署和使用