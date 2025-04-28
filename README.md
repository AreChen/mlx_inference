# MLX INFERENCE

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![中文文档](https://img.shields.io/badge/Docs-中文-blue.svg)](README_ZH.md)


### Project Introduction

MLX INFERENCE is an OpenAI API compatible inference service based on MLX-LM and MLX-VLM, providing the following endpoints:
- `/v1/chat/completions` - Chat completion interface
- `/v1/responses` - Response interface
- `/v1/models` - Get available model list

### Installation

```bash
pip install -r requirements.txt
# Copy environment file
cp .env.example .env
```

### Start Service

Execute in project root directory:

```bash
uvicorn mlx_Inference:app --workers 1 --port 8002
```

Parameters:
- `--workers`: Number of worker processes
- `--port`: Service port number

### Features

- Compatible with OpenAI API specifications
- Backend inference uses [MLX-LM](https://github.com/ml-explore/mlx-lm) and [MLX-VLM](https://github.com/Blaizzy/mlx-vlm), supports [mlx-community](https://huggingface.co/mlx-community) models
- Easy to deploy and use