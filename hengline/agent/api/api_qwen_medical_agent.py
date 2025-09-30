"""
@FileName: api_qwen_medical_agent.py
@Description:
@Author: HengLine
@Time: 2025/9/30 14:28
"""

# 导入日志模块
from hengline.logger import logger

# 从基类导入
from hengline.agent.base_agent import BaseMedicalAgent

class QwenMedicalAgent(BaseMedicalAgent):
    """基于生成式AI的医疗智能体，专注于生成丰富、自然的医疗内容"""
    def __init__(self):
        # 初始化API调用统计
        self.api_call_count = 0
        self.total_tokens_used = 0

        # 调用基类初始化
        super().__init__()

        # 使用项目的日志模块
        self.logger = logger