# AI Medical Agent - 智能医疗问答系统

## 项目介绍
AI Medical Agent是一个基于LangChain和RAG技术栈构建的智能医疗问答系统。该系统能够利用预设的医疗知识库和网络搜索功能，为用户提供专业、准确的医疗健康咨询服务。最新版本已升级使用LangGraph替代deprecated的AgentExecutor，提供更稳定、更现代的智能体架构。

## 项目结构
```
ai-medical-agent/
├── langchain/            # LangChain相关代码
│   ├── ollama/           # Ollama模型相关实现
│   │   ├── medical_agent.py            # 原始医疗智能体实现（使用AgentExecutor）
│   │   └── medical_agent_langgraph.py  # 基于LangGraph的医疗智能体实现（推荐使用）
│   └── vllm/             # vLLM模型相关实现
├── data/                 # RAG数据目录
│   ├── chinese_medicine_basic.txt     # 中医基础知识
│   ├── chinese_medicine_classics.txt  # 中医经典
│   ├── chinese_medicine_common_diseases.txt  # 中医常见病
│   ├── chinese_medicine_health.txt    # 中医养生
│   ├── chinese_medicine_treatment.txt # 中医治疗
│   ├── daily_diet_health.txt          # 日常饮食健康
│   ├── elderly_diseases_prevention.txt # 老年疾病预防
│   ├── food_nutrition_guide.txt       # 食物营养指南
│   ├── health_lifestyle.txt           # 健康生活方式
│   ├── medical_knowledge_1.txt        # 基础医疗知识库1
│   └── medical_knowledge_2.txt        # 基础医疗知识库2
├── tools/                # 工具函数目录
│   └── medical_tools.py  # 医疗专用工具函数
├── requirements.txt      # 项目依赖
└── README.md             # 项目说明文档
```

## 功能特点
1. **基于RAG技术**：结合检索增强生成技术，利用预设知识库提高回答准确性
2. **向量存储**：使用Chroma向量数据库高效存储和检索医疗知识
3. **多工具集成**：集成医疗知识库查询和网络搜索功能
4. **专业医疗工具**：提供症状提取、严重程度评估等医疗专用工具
5. **灵活配置**：支持本地LLM模型，可根据需求调整配置
6. **现代智能体架构**：使用LangGraph替代deprecated的AgentExecutor，提供更稳定的智能体实现
7. **模型兼容性**：优化的模型选择逻辑，支持多种Ollama模型
8. **错误处理**：完善的错误处理机制，确保在不同环境下都能提供服务

## 环境要求
- Python 3.8+ 
- 安装项目依赖：`pip install -r requirements.txt`
- 本地Ollama服务运行在 http://localhost:11434
- Ollama中已下载至少一个支持工具调用的模型（如llama3、mistral等）

## 使用方法
1. **安装依赖**
```bash
# 创建虚拟环境（推荐）
python -m venv .venv
# 激活虚拟环境（Windows）
.venv\Scripts\activate
# 激活虚拟环境（macOS/Linux）
# source .venv/bin/activate

# 配置国内源（可选）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

# 安装依赖
pip install -r requirements.txt
```

2. **安装和配置Ollama**
   - 从[Ollama官网](https://ollama.com/)下载并安装Ollama
   - 启动Ollama服务
   - 下载支持工具调用的模型，例如：
     ```bash
     ollama pull llama3
     ollama pull mistral
     ollama pull phi3
     ```

3. **修改配置（可选）**
如果需要调整模型配置，可以在`medical_agent_langgraph.py`文件中修改：
```python
# 在MedicalAgentLangGraph类的__init__方法中
models_to_try = ["llama3", "mistral", "phi3", "gemma:2b"]
```

4. **运行基于LangGraph的智能体（推荐）**
```bash
cd langchain/ollama
python medical_agent_langgraph.py
```

5. **运行原始智能体（不推荐，已弃用）**
```bash
cd langchain/ollama
python medical_agent.py
```

6. **自定义使用**
你可以在自己的代码中导入并使用`MedicalAgentLangGraph`类：
```python
from langchain.ollama.medical_agent_langgraph import MedicalAgentLangGraph

# 创建实例
agent = MedicalAgentLangGraph()

# 提问
question = "什么是高血压？如何预防？"
result = agent.run(question)

print(result)
```

## 添加更多医疗知识
你可以在`data`目录下添加更多的医疗知识文件（.txt格式），系统会自动加载这些文件到知识库中。

## 项目扩展建议
1. **添加更多医疗专业工具**：可以根据需求扩展`medical_tools.py`，添加更多医疗专用功能
2. **优化向量存储**：可以根据数据量和查询需求调整向量存储参数
3. **集成更多数据源**：可以集成医学数据库、学术论文等更多数据源
4. **添加用户界面**：可以开发Web或桌面界面，提高用户体验

## 注意事项
1. 本系统仅供参考，不构成医疗建议。如有健康问题，请咨询专业医生
2. 使用前请确保本地LLM服务已正确配置和运行
3. 根据实际需求调整RAG参数，以获得最佳的回答效果
4. 定期更新知识库，确保医疗信息的准确性和时效性