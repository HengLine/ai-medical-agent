# 本地vLLM服务详细启动指南

本指南提供了在本地环境中安装、配置和启动vLLM服务的详细步骤，以及常见问题的排查方法。

## 目录

- [1. 系统要求](#1-系统要求)
- [2. 安装vLLM](#2-安装vllm)
  - [2.1 使用pip安装](#21-使用pip安装)
  - [2.2 从源代码安装](#22-从源代码安装)
- [3. 准备模型](#3-准备模型)
  - [3.1 直接使用在线模型](#31-直接使用在线模型)
  - [3.2 使用本地下载的模型](#32-使用本地下载的模型)
  - [3.3 使用Ollama管理的模型（如您的配置）](#33-使用ollama管理的模型如您的配置)
- [4. 启动vLLM服务](#4-启动vllm服务)
  - [4.1 基本启动命令](#41-基本启动命令)
  - [4.2 常用参数配置](#42-常用参数配置)
  - [4.3 启动示例](#43-启动示例)
- [5. 验证vLLM服务](#5-验证vllm服务)
  - [5.1 使用测试脚本验证](#51-使用测试脚本验证)
  - [5.2 使用HTTP请求验证](#52-使用http请求验证)
  - [5.3 检查服务日志](#53-检查服务日志)
- [6. 集成到依赖问答系统](#6-集成到依赖问答系统)
- [7. 常见问题和解决方案](#7-常见问题和解决方案)
  - [7.1 GPU内存不足](#71-gpu内存不足)
  - [7.2 端口被占用](#72-端口被占用)
  - [7.3 模型加载失败](#73-模型加载失败)
  - [7.4 连接超时](#74-连接超时)
- [8. 高级配置](#8-高级配置)
  - [8.1 量化配置](#81-量化配置)
  - [8.2 多GPU配置](#82-多gpu配置)
  - [8.3 性能优化](#83-性能优化)

## 1. 系统要求

- Python 3.8或更高版本
- 足够的内存（推荐至少16GB）
- 支持的GPU（推荐NVIDIA GPU，显存至少8GB，对于大型模型可能需要更多）
- CUDA支持（如使用GPU）

## 2. 安装vLLM

### 2.1 使用pip安装

```bash
# 基本安装
pip install vllm

# 安装特定版本
pip install vllm==0.4.0

# 安装带额外功能的版本（如CUDA 12.1支持）
pip install vllm[cu121]
```

### 2.2 从源代码安装

```bash
# 克隆仓库
git clone https://github.com/vllm-project/vllm.git
cd vllm

# 安装依赖
pip install -e .

# 或带有特定CUDA版本
pip install -e "[cu121]"
```

## 3. 准备模型


### 3.1 直接使用在线模型

如果您想直接从Hugging Face下载模型，可以使用模型标识符：

```bash
vllm serve meta-llama/Llama-3-8b-instruct --port 8000
```

### 3.2 使用本地下载的模型

如果您已将模型下载到本地，可以指定本地路径：

```bash
pip install -U huggingface_hub
$env:HF_ENDPOINT = "https://hf-mirror.com"
huggingface-cli download --resume-download gpt2 --local-dir gpt2
vllm serve ./gpt2 --port 8000
```


## 4. 启动vLLM服务

### 4.1 基本启动命令

```bash
vllm serve [模型路径或名称] --port 8000
```

根据您的配置，您应该使用：

```bash
vllm serve "E:\AI\models\vllm\qwen3" --port 8000
```

### 4.2 常用参数配置

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--port` | 服务端口号 | 8000 | 8000 |
| `--host` | 服务主机地址 | 0.0.0.0 | 0.0.0.0 |
| `--gpu-memory-utilization` | GPU内存使用率 | 0.9 | 0.8-0.9 |
| `--max-model-len` | 最大模型长度 | 取决于模型 | 4096或更大 |
| `--tensor-parallel-size` | 张量并行大小（多GPU使用） | 1 | 与GPU数量匹配 |
| `--quantization` | 模型量化方法 | 无 | awq, gptq等（内存不足时使用） |
| `--dtype` | 数据类型 | auto | auto |

### 4.3 启动示例

```bash
# 基本启动（使用您的本地Ollama模型）
vllm serve "E:\AI\models\vllm\qwen3" --port 8000

# 配置GPU内存使用率
vllm serve "E:\AI\models\vllm\qwen3" --port 8000 --gpu-memory-utilization 0.8

# 使用量化（内存不足时）
vllm serve "E:\AI\models\vllm\gpt2" --port 8000 --quantization awq

# 在多GPU环境中使用
vllm serve "E:\AI\models\vllm\qwen3" --port 8000 --tensor-parallel-size 2
```

## 5. 验证vLLM服务

### 5.1 使用测试脚本验证

使用我们提供的测试脚本验证服务是否正常运行：

```bash
cd e:\Projects\blogs\ai-medical-agent\langchain\vllm
python test_local_vllm.py
```

### 5.2 使用HTTP请求验证

您也可以使用curl或浏览器直接访问vLLM服务的API：

```bash
# 检查服务状态和可用模型
curl http://localhost:8000/v1/models

# 发送简单的生成请求
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3", "prompt": "什么是依赖管理？", "max_tokens": 100}'
```

### 5.3 检查服务日志

查看vLLM服务启动时的输出日志，确认没有错误信息，并且服务已成功启动。

成功启动的标志包括：
- `INFO:     Started server process`
- `INFO:     Waiting for application startup`
- `INFO:     Application startup complete`
- `INFO:     Uvicorn running on http://0.0.0.0:8000`

## 6. 集成到依赖问答系统

确保`vllm_config.py`文件中的配置与您的vLLM服务设置一致：

```python
MODEL_CONFIG = {
    "model": "E:\\AI\\models\\ollama\\manifests\\registry.ollama.ai\\library\\qwen3",  # 本地模型路径
    "temperature": 0.1,
    "max_tokens": 1024,
    "top_p": 0.95,
    "vllm_kwargs": {
        "base_url": "http://localhost:8000/v1",  # vLLM服务地址
        "gpu_memory_utilization": 0.8,
        "max_model_len": 4096,
        "tensor_parallel_size": 1,
        "trust_remote_code": True,
        "dtype": "auto",
    }
}
```

然后运行依赖问答系统：

```bash
python vllm_dependency_qa.py
```

## 7. 常见问题和解决方案

### 7.1 GPU内存不足

**问题**：启动时出现CUDA out of memory错误

**解决方案**：
- 使用`--gpu-memory-utilization`参数降低GPU内存使用率
- 启用模型量化：`--quantization awq`或`--quantization gptq`
- 使用更小的模型

### 7.2 端口被占用

**问题**：启动时出现`Address already in use`错误

**解决方案**：
- 使用不同的端口：`--port 8001`
- 关闭占用该端口的进程

### 7.3 模型加载失败

**问题**：无法加载指定的模型

**解决方案**：
- 检查模型路径是否正确
- 确认模型文件完整无损坏
- 对于Ollama模型，确保已成功拉取模型

### 7.4 连接超时

**问题**：连接vLLM服务时出现超时错误

**解决方案**：
- 确认vLLM服务正在运行
- 检查防火墙设置，确保端口已开放
- 增加客户端的超时设置

## 8. 高级配置

### 8.1 量化配置

对于内存受限的环境，可以启用模型量化：

```bash
vllm serve "E:\AI\models\vllm\qwen3" --port 8000 --quantization awq
```

### 8.2 多GPU配置

在多GPU环境中，可以使用张量并行来加速推理：

```bash
vllm serve "E:\AI\models\vllm\qwen3" --port 8000 --tensor-parallel-size 2
```

### 8.3 性能优化

```bash
# 启用连续批处理
vllm serve "E:\AI\models\vllm\qwen3" --port 8000 --enable-continuous-batching

# 配置最大批处理大小
vllm serve "E:\AI\models\vllm\qwen3" --port 8000 --max-batch-size 16
```

## 附录：常用命令速查

```bash
# 启动vLLM服务（使用您的Ollama模型）
vllm serve "E:\AI\models\vllm\qwen3" --port 8000

# 验证服务状态
python test_local_vllm.py

# 查看vLLM版本
pip show vllm

# 查看端口占用情况（Windows）
netstat -ano | findstr 8000

# 停止占用端口的进程（Windows）
taskkill /PID [进程ID] /F
```

希望本指南能帮助您成功启动和配置本地vLLM服务！如有任何问题，请参考vLLM的[官方文档](https://docs.vllm.ai/en/latest/)或在GitHub上提交issue。