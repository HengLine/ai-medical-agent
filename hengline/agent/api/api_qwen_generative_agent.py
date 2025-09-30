"""@FileName: api_qwen_generative_agent.py
@Description: 基于通义千问API的生成式医疗智能体
@Author: HengLine
@Time: 2025/9/30 14:20
"""
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# 导入日志模块
from hengline.logger import logger

# 从基类导入
from hengline.agent.base_agent import BaseMedicalAgent

# 导入Qwen特定的库
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class QwenGenerativeAgent(BaseMedicalAgent):
    """基于通义千问API的生成式医疗智能体"""

    def __init__(self):
        super().__init__()

        # 创建生成链
        self.generative_chains = self._create_generative_chains()

        # 支持的生成类型
        self.supported_generation_types = ["general_info", "detailed_explanation", "patient_education", "medical_case"]

        logger.info("通义千问生成式智能体初始化完成")

    def _initialize_llm(self):
        """初始化通义千问语言模型"""
        try:
            # 从配置中获取Qwen模型参数
            llm_config = self.config_reader.config.get("llm", {})
            qwen_config = llm_config.get("qwen", {})

            # 获取API密钥和模型名称
            api_key = qwen_config.get("api_key", "")
            api_url = qwen_config.get("api_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            model_name = qwen_config.get("model", "qwen-3")

            if not api_key:
                logger.warning("未配置通义千问API密钥，尝试从环境变量获取")
                import os
                api_key = os.environ.get("DASHSCOPE_API_KEY", "")

                if not api_key:
                    logger.warning("未找到通义千问API密钥，将使用空密钥继续")

            # 初始化通义千问模型
            llm = ChatTongyi(
                model=model_name,
                api_key=api_key,
                streaming=True,
                max_retries=3,
                model_kwargs={
                    "base_url": api_url,
                    "temperature": qwen_config.get("temperature", 0.1),
                    "max_tokens": qwen_config.get("max_tokens", 2048)
                },
            )

            self.model_supports_tools = hasattr(self, '_check_tool_support') and self._check_tool_support(model_name) or True
            logger.info(f"成功初始化通义千问模型: {model_name}, 支持工具调用: {self.model_supports_tools}")
            return llm
        except Exception as e:
            logger.error(f"初始化通义千问语言模型时出错: {str(e)}")
            # 返回None，基类会处理这种情况
            return None

    def _create_generative_chains(self):
        """创建不同类型的生成链"""
        if not self.llm:
            return {}

        try:
            # 定义提示模板
            prompt_templates = {
                "general_info": ChatPromptTemplate.from_template(
                    "你是一位经验丰富的医学专家。请提供关于{topic}的一般医学信息，包括定义、常见症状和基本预防措施。"
                    "回答应简洁明了，适合一般读者理解。"
                ),
                "detailed_explanation": ChatPromptTemplate.from_template(
                    "你是一位经验丰富的医学专家。请提供关于{topic}的详细医学解释，包括病理机制、临床表现、诊断标准、治疗方案和预后评估。"
                    "回答应包含专业医学术语，适合医疗专业人员或有医学背景的读者。"
                ),
                "patient_education": ChatPromptTemplate.from_template(
                    "你是一位经验丰富的医学专家。请为患者创建关于{topic}的教育材料，以简单易懂的语言解释该主题。"
                    "包括：什么是该病症、为什么会发生、患者可能有什么感觉、如何治疗、日常生活中如何管理、何时需要就医等内容。"
                    "请使用友好、支持性的语气。"
                ),
                "medical_case": ChatPromptTemplate.from_template(
                    "你是一位经验丰富的医学专家。请创建一个关于{topic}的临床案例，包括患者基本信息、主诉、现病史、既往史、体格检查、辅助检查、诊断过程、治疗方案和随访建议。"
                    "案例应尽可能真实，包含详细的医学信息和临床推理过程。"
                )
            }

            # 创建生成链
            generative_chains = {}
            for gen_type, prompt_template in prompt_templates.items():
                generative_chains[gen_type] = prompt_template | self.llm | StrOutputParser()

            return generative_chains
        except Exception as e:
            logger.error(f"创建生成链时出错: {str(e)}")
            return {}

    def generate_content(self, topic, generation_type="general_info"):
        """生成指定主题的医疗内容
        
        Args:
            topic: 要生成内容的主题
            generation_type: 生成类型，支持general_info、detailed_explanation、patient_education、medical_case
        
        Returns:
            str: 生成的医疗内容
        """
        # 验证生成类型
        if generation_type not in self.supported_generation_types:
            valid_types = ", ".join(self.supported_generation_types)
            return f"不支持的生成类型: {generation_type}。支持的类型: {valid_types}"

        # 验证主题
        if not topic or topic.strip() == "":
            return "主题不能为空"

        try:
            # 获取相应的生成链
            if generation_type not in self.generative_chains:
                return f"生成链 {generation_type} 未初始化"

            # 准备输入
            input_data = {
                "topic": topic.strip()
            }

            # 调用生成链
            generated_content = self.generative_chains[generation_type].invoke(input_data)

            return generated_content
        except Exception as e:
            logger.error(f"生成内容时出错: {str(e)}")
            return f"生成内容时出错: {str(e)}"

    def _determine_generation_type(self, question):
        """根据问题确定生成类型"""
        question_lower = question.lower()

        if any(keyword in question_lower for keyword in ["详细", "机制", "病理", "专业"]):
            return "detailed_explanation"
        elif any(keyword in question_lower for keyword in ["患者", "教育", "解释给", "如何告诉"]):
            return "patient_education"
        elif any(keyword in question_lower for keyword in ["案例", "病例", "实例"]):
            return "medical_case"
        else:
            return "general_info"

    def run(self, question, generate_extra_content=False):
        """运行智能体回答问题，可选择是否生成额外内容"""
        # 调用基类的run方法获取基本回答
        result = super().run(question)

        # 如果需要生成额外内容
        if generate_extra_content:
            try:
                # 确定生成类型
                gen_type = self._determine_generation_type(question)

                # 生成额外内容
                extra_content = self.generate_content(question, gen_type)

                # 组合回答
                result = f"{result}\n\n\n===== 额外生成内容 ({gen_type}) =====\n{extra_content}"
            except Exception as e:
                logger.error(f"生成额外内容时出错: {str(e)}")
                # 不影响基本回答

        return result


if __name__ == "__main__":
    # 创建基于通义千问的生成式医疗智能体实例
    generative_agent = QwenGenerativeAgent()

    # 示例问题
    example_topics = [
        "高血压的预防",
        "糖尿病的病理机制",
        "如何向患者解释心脏病发作",
        "一名65岁男性胸痛患者的诊断流程"
    ]

    # 示例生成类型
    example_types = ["general_info", "detailed_explanation", "patient_education", "medical_case"]

    print("\n===== 通义千问生成式医疗智能体演示 =====")
    print("输入'退出'结束演示。\n")

    try:
        # 运行交互模式
        while True:
            user_input = input("请输入要生成内容的主题 (或输入示例序号 1-4): ")

            if user_input.lower() in ['退出', 'quit', 'exit']:
                print("感谢使用，再见！")
                break

            # 处理示例选择
            if user_input.isdigit() and 1 <= int(user_input) <= len(example_topics):
                idx = int(user_input) - 1
                topic = example_topics[idx]
                gen_type = example_types[idx]
                print(f"\n选择了示例 {user_input}: 主题='{topic}', 生成类型='{gen_type}'")
            else:
                # 用户自定义主题
                topic = user_input

                # 让用户选择生成类型
                print("请选择生成类型:")
                for i, gen_type in enumerate(example_types, 1):
                    print(f"{i}. {gen_type}")

                type_input = input("请输入类型序号 (默认: 1): ")
                if type_input.isdigit() and 1 <= int(type_input) <= len(example_types):
                    gen_type = example_types[int(type_input) - 1]
                else:
                    gen_type = "general_info"

            # 生成内容
            print(f"\n正在生成{gen_type}类型的内容...")
            content = generative_agent.generate_content(topic, gen_type)

            # 显示生成的内容
            print(f"\n===== 生成内容 ({gen_type}) ====\n{content}\n")
            print("=" * 50)
    except KeyboardInterrupt:
        print("\n程序被用户中断，再见！")
