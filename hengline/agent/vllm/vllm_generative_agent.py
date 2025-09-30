"""@FileName: vllm_generative_agent.py
@Description: 基于vLLM的生成式医疗智能体
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
from langchain_community.llms.vllm import VLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class VllmGenerativeAgent(BaseMedicalAgent):
    """基于vLLM的生成式医疗智能体"""

    def __init__(self):
        super().__init__()

        # 初始化生成链字典
        self.generative_chains = {}

        # 初始化日志
        self.logger = logger

        # 创建生成链
        self._create_generative_chains()

        logger.info("vLLM生成式医疗智能体初始化完成")

    def _initialize_llm(self):
        """初始化vLLM语言模型"""
        try:
            # 从配置中获取vLLM模型参数
            llm_config = self.config_reader.config.get("llm", {})
            vllm_config = llm_config.get("vllm", {})

            # 获取vLLM服务URL和模型名称
            model_name = vllm_config.get("model", "")

            # 初始化vLLM模型
            llm = VLLM(
                model=model_name,
                temperature=vllm_config.get("temperature", 0.1),
                max_new_tokens=vllm_config.get("max_tokens", 1024),
                top_p=vllm_config.get("top_p", 0.95),
                **vllm_config.get("vllm_kwargs", {})
            )

            logger.info(f"成功初始化vLLM模型: {model_name}")
            return llm
        except Exception as e:
            logger.error(f"初始化vLLM语言模型时出错: {str(e)}")
            # 返回None，基类会处理这种情况
            return None

    def _create_generative_chains(self):
        """创建各种生成类型的链"""
        # 检查语言模型是否已初始化
        if not self.llm:
            logger.warning("语言模型尚未初始化，无法创建生成链")
            return

        # 通用信息生成提示
        general_info_template = ChatPromptTemplate.from_template(
            "你是一位经验丰富的医学专家。请提供关于{topic}的简明扼要的概述信息。\n\n" \
            "要求：\n" \
            "1. 内容准确、专业\n" \
            "2. 语言通俗易懂\n" \
            "3. 包含最重要的关键点\n" \
            "4. 不包含过于专业的术语解释\n" \
            "5. 长度适中，约200-300字"
        )

        # 详细解释生成提示
        detailed_explanation_template = ChatPromptTemplate.from_template(
            "你是一位经验丰富的医学专家。请提供关于{topic}的详细解释。\n\n" \
            "要求：\n" \
            "1. 内容深入、全面\n" \
            "2. 包含相关的医学原理和机制\n" \
            "3. 适当使用专业术语并给出解释\n" \
            "4. 结构清晰，逻辑连贯\n" \
            "5. 长度约500-800字"
        )

        # 患者教育材料生成提示
        patient_education_template = ChatPromptTemplate.from_template(
            "你是一位经验丰富的医学专家。请创建一份面向患者的关于{topic}的教育材料。\n\n" \
            "要求：\n" \
            "1. 语言简单易懂，避免专业术语\n" \
            "2. 内容实用，关注患者关心的问题\n" \
            "3. 包含常见问题解答\n" \
            "4. 提供明确的建议和注意事项\n" \
            "5. 语气亲切，有帮助性\n" \
            "6. 长度约300-500字"
        )

        # 医疗案例生成提示
        medical_case_template = ChatPromptTemplate.from_template(
            "你是一位经验丰富的医学专家。请创建一个关于{topic}的临床案例。\n\n" \
            "要求：\n" \
            "1. 案例描述详细、真实\n" \
            "2. 包含患者基本信息、症状、诊断过程、治疗方案和预后\n" \
            "3. 反映真实的临床思维过程\n" \
            "4. 包含相关的医学知识要点\n" \
            "5. 长度约800-1000字"
        )

        # 创建各种生成类型的链
        self.generative_chains["general_info"] = general_info_template | self.llm | StrOutputParser()
        self.generative_chains["detailed_explanation"] = detailed_explanation_template | self.llm | StrOutputParser()
        self.generative_chains["patient_education"] = patient_education_template | self.llm | StrOutputParser()
        self.generative_chains["medical_case"] = medical_case_template | self.llm | StrOutputParser()

        logger.info(f"成功创建{len(self.generative_chains)}种生成类型的链")

    def generate_content(self, topic, generation_type="general_info"):
        """生成医疗内容
        
        Args:
            topic: 要生成内容的主题
            generation_type: 生成类型，可选值：general_info（通用信息）、detailed_explanation（详细解释）、
                            patient_education（患者教育）、medical_case（医疗案例）
        
        Returns:
            str: 生成的内容
        """
        try:
            # 验证生成类型
            if generation_type not in self.generative_chains:
                logger.warning(f"不支持的生成类型: {generation_type}，将使用默认类型 general_info")
                generation_type = "general_info"

            # 验证主题
            if not topic or not isinstance(topic, str) or topic.strip() == "":
                raise ValueError("主题必须是非空字符串")

            logger.info(f"正在生成关于'{topic}'的{generation_type}内容")

            # 执行生成链
            content = self.generative_chains[generation_type].invoke({"topic": topic})

            return content
        except Exception as e:
            logger.error(f"生成内容时出错: {str(e)}")
            return f"生成内容时出错: {str(e)}"

    def _determine_generation_type(self, query):
        """根据查询自动确定生成类型
        
        Args:
            query: 用户查询
        
        Returns:
            str: 生成类型
        """
        # 转换为小写以便匹配
        query_lower = query.lower()

        # 定义关键词列表
        general_info_keywords = ["什么是", "概述", "简介", "定义", "基本信息"]
        detailed_explanation_keywords = ["详细", "深入", "机制", "原理", "病理"]
        patient_education_keywords = ["患者", "教育", "注意事项", "自我护理", "日常生活"]
        medical_case_keywords = ["案例", "病例", "实例", "临床案例", "真实案例"]

        # 检查关键词
        if any(keyword in query_lower for keyword in detailed_explanation_keywords):
            return "detailed_explanation"
        elif any(keyword in query_lower for keyword in patient_education_keywords):
            return "patient_education"
        elif any(keyword in query_lower for keyword in medical_case_keywords):
            return "medical_case"
        elif any(keyword in query_lower for keyword in general_info_keywords):
            return "general_info"
        else:
            # 默认返回通用信息
            return "general_info"

    def run(self, question):
        """运行智能体回答问题或生成内容
        
        Args:
            question: 用户的问题或内容生成请求
        
        Returns:
            str: 智能体的回答或生成的内容
        """
        try:
            # 检查是否初始化成功
            if not self.llm:
                return "智能体初始化失败，请检查配置"

            # 如果没有生成链，创建生成链
            if not self.generative_chains:
                self._create_generative_chains()

                if not self.generative_chains:
                    return "无法创建生成链，无法生成内容"

            # 确定生成类型
            generation_type = self._determine_generation_type(question)

            # 提取主题（这里简化处理，直接使用问题作为主题）
            topic = question

            # 生成内容
            content = self.generate_content(topic, generation_type)

            return content
        except Exception as e:
            logger.error(f"运行智能体时出错: {str(e)}")
            return f"运行智能体时出错: {str(e)}"


if __name__ == "__main__":
    # 创建基于vLLM的生成式医疗智能体实例
    generative_agent = VllmGenerativeAgent()

    # 示例主题和生成类型
    example_topics = [
        "高血压",
        "糖尿病",
        "冠心病",
        "哮喘"
    ]

    generation_types = {
        "1": ("general_info", "通用信息"),
        "2": ("detailed_explanation", "详细解释"),
        "3": ("patient_education", "患者教育"),
        "4": ("medical_case", "医疗案例")
    }

    print("\n===== vLLM生成式医疗智能体演示 =====")
    print("输入'退出'结束演示。\n")

    try:
        # 运行交互模式
        while True:
            # 获取用户输入的主题
            user_input = input("请输入您想了解的医疗主题 (或输入示例序号 1-4): ")

            if user_input.lower() in ['退出', 'quit', 'exit']:
                print("感谢使用，再见！")
                break

            # 处理示例主题选择
            if user_input.isdigit() and 1 <= int(user_input) <= len(example_topics):
                idx = int(user_input) - 1
                topic = example_topics[idx]
                print(f"\n选择了示例主题 {user_input}: {topic}")
            else:
                # 用户自定义主题
                topic = user_input

            # 获取用户选择的生成类型
            print("\n请选择生成类型:")
            for key, (_, desc) in generation_types.items():
                print(f"{key}. {desc}")

            type_choice = input("请输入选择 (默认为1): ")

            # 确定生成类型
            if type_choice in generation_types:
                generation_type, type_desc = generation_types[type_choice]
            else:
                generation_type, type_desc = generation_types["1"]
                print(f"\n使用默认生成类型: {type_desc}")

            # 生成内容
            print(f"\n正在生成关于'{topic}'的{type_desc}内容...")
            content = generative_agent.generate_content(topic, generation_type)

            # 显示生成的内容
            print(f"\n===== 生成的{type_desc} =====\n{content}\n")
            print("=" * 50)
    except KeyboardInterrupt:
        print("\n程序被用户中断，再见！")
