# 医疗AI智能体

一个基于LangChain和LangGraph的医疗智能体系统，支持四种不同的后端实现方式：本地Ollama、本地VLLM、远程API调用和生成式智能体。

## 功能特性

- 支持四种不同的后端实现：
  - 本地Ollama模型
  - 本地VLLM模型
  - 远程API（如OpenAI API）
  - 生成式智能体（支持多种内容生成模式）
- 基于LangGraph的智能体架构，支持工具调用
- 医疗知识库检索功能
- 症状提取和严重程度评估
- 网络搜索功能（获取最新医疗信息）
- 统一的配置管理
- RESTful API接口，支持健康检查、配置管理、问题查询等功能
- 交互式命令行界面
- 多种医疗内容生成模式（general_info, detailed_explanation, patient_education, medical_case）

## 目录结构

```
├── .gitignore
├── README.md           # 项目说明文档
├── config/             # 配置文件目录
│   └── config.json     # 主配置文件
├── data/               # 知识库数据目录
├── hengline/           # 核心代码目录
│   ├── agent/          # 智能体相关代码
│   │   ├── api/        # 远程API智能体实现
│   │   ├── generative/ # 生成式智能体实现
│   │   ├── ollama/     # Ollama智能体实现
│   │   ├── vllm/       # VLLM智能体实现
│   │   ├── base_agent.py # 智能体基类
│   │   └── medical_agent.py # 统一入口文件
│   ├── api/            # API相关代码
│   │   └── medical_api.py # API入口文件
│   ├── api_app.py      # FastAPI应用入口
│   ├── config.py       # 配置读取器
│   ├── demo/           # 演示脚本
│   │   ├── demo_usage.py # 基本使用演示
│   │   ├── generative_demo.py # 生成式智能体演示
│   │   └── simple_medical_agent.py # 简单医疗智能体演示
│   ├── logger.py       # 日志模块
│   └── tools/          # 医疗相关工具
├── requirements.txt    # 项目依赖
├── run_medical.py      # API服务启动脚本
└── utils/              # 工具函数
    └── console_colors.py # 控制台颜色工具
```

## 安装指南

1. 克隆项目代码

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境：
   - 根据需要修改 `config/config.json` 文件中的配置项
   - 对于Ollama智能体，需要先安装并启动Ollama服务
   - 对于VLLM智能体，需要准备好本地模型文件
   - 对于远程API智能体，需要配置API密钥
   - 医疗知识库文件应放置在 `data/` 目录下

4. 确保知识库目录存在：
```bash
mkdir -p data
```

## 使用方法

### 统一入口使用

通过统一入口文件，可以选择使用哪种类型的智能体：

```bash
# 使用Ollama智能体（默认）
python hengline/agent/medical_agent.py

# 使用VLLM智能体
python hengline/agent/medical_agent.py --type vllm

# 使用远程API智能体
python hengline/agent/medical_agent.py --type api

# 使用生成式智能体
python hengline/agent/medical_agent.py --type generative
```

### 单独使用各智能体

也可以直接运行各智能体的实现文件：

```bash
# 运行Ollama智能体
python hengline/agent/ollama/ollama_medical_agent.py

# 运行VLLM智能体
python hengline/agent/vllm/vllm_medical_agent.py

# 运行远程API智能体
python hengline/agent/api/remote_api_agent.py

# 运行生成式智能体
python hengline/agent/generative/generative_medical_agent.py
```

### 启动API服务

```bash
# 启动默认Ollama类型的API服务
python run_medical.py

# 指定智能体类型启动API服务
python run_medical.py --type ollama
python run_medical.py --type vllm
python run_medical.py --type api
python run_medical.py --type generative

# 自定义主机和端口
python run_medical.py --host 0.0.0.0 --port 8080

# 开发模式下启用自动重载
python run_medical.py --reload
```

API文档地址：http://localhost:8000/docs

可用的API端点：
- GET /api/health - 健康检查（检查API和智能体的运行状态）
- PUT /api/config - 更新LLM配置（修改当前使用的LLM参数）
- POST /api/query - 查询医疗智能体（向医疗智能体发送问题并获取回答）

## 配置说明

配置文件位于 `config/config.json`，主要包含以下配置项：

### 默认LLM设置
```json
"default_llm": "ollama"  # 默认使用的LLM类型，可以是 ollama, vllm, api
```

### LLM配置（所有类型统一组织在llm对象下）
```json
"llm": {
    "ollama": {
        "model_name": "llama3.2",       # Ollama模型名称
        "base_url": "http://localhost:11434",  # Ollama服务地址
        "timeout": 300,                 # 超时时间
        "temperature": 0.1,             # 生成温度
        "keep_alive": 300,              # 保持连接时间
        "top_p": 0.95,                  # 采样参数
        "max_tokens": 1024              # 最大生成令牌数
    },
    "vllm": {
        "model": "E:\\AI\\models\\vllm\\gpt2",  # 本地模型路径
        "temperature": 0.1,             # 生成温度
        "max_tokens": 1024,             # 最大生成令牌数
        "top_p": 0.95,                  # 采样参数
        "vllm_kwargs": {
            "device": "cpu",           # 运行设备
            "max_model_len": 4096,      # 最大模型长度
            "trust_remote_code": true,  # 信任远程代码
            "dtype": "auto",           # 数据类型
            "disable_log_requests": true,  # 禁用请求日志
            "disable_log_stats": false  # 启用统计日志
        },
        "embeddings": {
            "model_name": "E:\\AI\\models\\vllm\\bge-small-zh",  # 嵌入模型路径
            "model_kwargs": {
                "device": "cpu"
            },
            "encode_kwargs": {
                "normalize_embeddings": true
            }
        }
    },
    "api": {
        "api_key": "",                  # API密钥
        "api_url": "https://api.openai.com/v1",  # API地址
        "model": "gpt-4o",             # 模型名称
        "temperature": 0.1,             # 生成温度
        "max_tokens": 2048              # 最大生成令牌数
    }
}

### 共享配置
```json
"retrieval": {                      # 检索链配置
    "chain_type": "stuff",         # 链类型
    "search_kwargs": {              # 搜索参数
        "k": 3                      # 检索文档数量
    },
    "return_source_documents": true # 是否返回源文档
},
"text_splitter": {                  # 文本分割配置
    "chunk_size": 1000,             # 块大小
    "chunk_overlap": 200            # 重叠部分
},
"knowledge_base": {                 # 知识库配置
    "data_dir": "data",            # 数据目录
    "medical_keywords": [           # 医疗相关关键词
      "medical",
      "health",
      "disease",
      "treatment",
      "medicine",
      "symptoms"
    ]
},
"example_questions": [              # 示例问题列表
    "什么是高血压？如何预防？",
    "老年高血压患者有哪些注意事项？",
    "脑血栓的高危因素有哪些？如何预防？",
    "糖尿病的预防措施有哪些？",
    "老年人如何保持健康的生活方式？"
]
```

### 共享配置
```json
"retrieval": {                      # 检索链配置
    "chain_type": "stuff",         # 链类型
    "search_kwargs": {              # 搜索参数
        "k": 3                      # 检索文档数量
    },
    "return_source_documents": true # 是否返回源文档
},
"text_splitter": {                  # 文本分割配置
    "chunk_size": 1000,             # 块大小
    "chunk_overlap": 200            # 重叠部分
},
"knowledge_base": {                 # 知识库配置
    "data_dir": "data",            # 数据目录
    "medical_keywords": [...]       # 医疗相关关键词
},
"example_questions": [              # 示例问题列表
    ...
]
```

## 注意事项

1. 使用前请确保相关服务已正确安装和配置
2. 对于本地模型，需要确保有足够的硬件资源
3. 远程API调用可能会产生费用，请确保配置了有效的API密钥
4. 本项目仅用于演示和学习目的，不应用于实际医疗诊断
5. 所有医疗相关回答仅供参考，不构成医疗建议
6. 使用Ollama智能体前，请确保Ollama服务已启动（默认端口11434）

## 使用示例

### 命令行交互示例

使用演示脚本启动交互式问答：

```bash
python hengline/demo/demo_usage.py
```

运行后，可以直接输入问题进行交互：

```
===== AI医疗问答系统演示 =====
初始化完成！请输入您的医疗问题（输入'退出'结束程序）

示例问题：
1. 什么是高血压？如何预防？
2. 我最近发热、咳嗽、喉咙痛，应该怎么办？
3. 心肺复苏的步骤是什么？
4. 如何保持健康的生活方式？

-----------------------------------
请输入您的问题: 什么是高血压？

正在获取答案，请稍候...

===== 回答 =====
高血压是一种常见的慢性疾病，指血液在血管中流动时对血管壁造成的压力持续高于正常水平...
```

## API使用示例

### 使用curl调用API

```bash
# 健康检查
curl http://localhost:8000/api/health
# 响应示例：{"api_status":"running","agent_status":"initialized"}

# 更新配置
curl -X PUT http://localhost:8000/api/config -H "Content-Type: application/json" -d '{"api_url":"http://localhost:11434","models":"llama3.2","temperature":0.1,"timeout":300}'
# 响应示例：{"llm":{...},"updated_at":"2024-06-15T10:30:00Z","message":"配置获取成功"}

# 查询问题
curl -X POST http://localhost:8000/api/query -H "Content-Type: application/json" -d '{"question": "什么是高血压？", "request_id": "user_123"}'
# 响应示例：{"answer":"高血压是一种常见的慢性疾病...","request_id":"user_123","sources":"可选，来源文档信息（如回答中包含）","timestamp":"2024-06-15T10:31:15Z"}
```

### 使用Python调用API

```python
import requests
import json

# 健康检查
def check_health():
    url = "http://localhost:8000/api/health"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"请求失败: {response.status_code}"}

# 更新配置
def update_llm_config(config_data):
    url = "http://localhost:8000/api/config"
    headers = {"Content-Type": "application/json"}
    response = requests.put(url, headers=headers, data=json.dumps(config_data))
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"请求失败: {response.status_code}"}

# 查询问题
def query_medical_agent(question, request_id=None):
    url = "http://localhost:8000/api/query"
    headers = {"Content-Type": "application/json"}
    data = {"question": question}
    if request_id:
        data["request_id"] = request_id
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"请求失败: {response.status_code}"}

# 使用示例
# 1. 检查健康状态
health_status = check_health()
print("健康状态:", health_status)

# 2. 更新配置
config_data = {
    "api_url": "http://localhost:11434",
    "models": "llama3.2",
    "temperature": 0.1,
    "timeout": 300
}
config_result = update_llm_config(config_data)
print("配置更新结果:", config_result)

# 3. 查询问题
result = query_medical_agent("什么是高血压？", request_id="user_123")
print("回答:", result["answer"])
print("请求ID:", result["request_id"])
print("时间戳:", result["timestamp"])
if "sources" in result and result["sources"]:
    print("来源:", result["sources"])
```

## 知识库管理

### 添加知识文档

将医疗相关的文本文件放置在项目根目录的 `data/` 文件夹中，系统会自动加载这些文档到知识库中：

```bash
# 创建知识文档
echo "高血压是一种常见的慢性疾病..." > data/hypertension_knowledge.txt
```

支持的文件格式：目前仅支持纯文本文件（.txt）

## 常见问题解决

### Ollama服务连接失败

错误信息：`[WinError 10061] 由于目标计算机积极拒绝，无法连接`

解决方案：
1. 确认Ollama服务已安装并启动
2. 检查Ollama服务端口是否为11434（默认端口）
3. 如使用自定义端口，请修改config.json中的base_url配置

### 知识库检索结果为空

解决方案：
1. 确认data目录中包含有效的医疗知识文档
2. 检查文档格式是否为纯文本（.txt）
3. 重启服务以重新加载知识库

## 免责声明

本项目提供的医疗信息仅供参考，不构成医疗建议、诊断或治疗。在进行任何医疗决策之前，请咨询专业医疗人员。项目开发者不对因使用本项目而导致的任何后果负责。