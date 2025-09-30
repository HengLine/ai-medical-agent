"""
@FileName: api_openai_generative_medical_agent.py
@Description:
@Author: HengLine
@Time: 2025/9/30 14:19
"""
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# 导入日志模块
from hengline.logger import logger

# 从基类导入
from hengline.agent.base_agent import BaseMedicalAgent

# 导入LangChain相关库
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class QwenGenerativeAgent(BaseMedicalAgent):
    """基于生成式AI的医疗智能体，专注于生成丰富、自然的医疗内容"""

    def __init__(self):
        super().__init__()
