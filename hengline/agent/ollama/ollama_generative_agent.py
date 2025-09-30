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


class OllamaGenerativeAgent(BaseMedicalAgent):
    """基于生成式AI的医疗智能体，专注于生成丰富、自然的医疗内容"""
    
    def __init__(self):
        super().__init__()
        
        # 初始化生成式提示模板
        self.prompt_templates = {
            "general_info": ChatPromptTemplate.from_template(
                """请提供关于{topic}的详细医学信息。包括：
                1. 定义和基本概念
                2. 主要症状和临床表现
                3. 常见病因
                4. 诊断方法
                5. 治疗方案
                6. 预防措施
                7. 预后情况
                
                请使用自然、易懂的语言，避免过于技术性的术语。"""
            ),
            "detailed_explanation": ChatPromptTemplate.from_template(
                """请详细解释{topic}。要求：
                1. 深入分析该主题的医学原理
                2. 提供最新的研究进展
                3. 解释临床应用中的关键点
                4. 讨论相关争议或未解决的问题
                
                适合医疗专业人士阅读的详细解释。"""
            ),
            "patient_education": ChatPromptTemplate.from_template(
                """为患者创建关于{topic}的教育材料。要求：
                1. 使用简单易懂的语言
                2. 重点关注患者需要了解的关键信息
                3. 提供实际的生活建议
                4. 解释重要的注意事项
                5. 包含常见问题解答
                
                格式应当友好、易于阅读。"""
            ),
            "medical_case": ChatPromptTemplate.from_template(
                """创建一个关于{topic}的临床案例。要求：
                1. 提供详细的患者病史
                2. 描述症状和体征
                3. 列出诊断过程和检查结果
                4. 解释治疗方案的制定和实施
                5. 讨论治疗效果和随访情况
                6. 提供临床启示和经验教训
                
                请确保案例具有教育意义和现实参考价值。"""
            )
        }
        
        # 初始化输出解析器
        self.output_parser = StrOutputParser()
        
        # 初始化生成链
        self.generative_chains = self._create_generative_chains()
        
        logger.info("生成式医疗智能体初始化完成")
    
    def _initialize_llm(self):
        """初始化生成式语言模型"""
        try:
            # 从配置中获取生成式模型参数
            generative_config = self.config_reader.get_generative_config()
            
            # 尝试使用配置的生成式模型
            try:
                llm = ChatOllama(
                    model=generative_config.get("model_name", "llama3.2"),
                    temperature=generative_config.get("temperature", 0.7),  # 生成式任务使用较高的temperature
                    base_url=generative_config.get("base_url", "http://localhost:11434"),
                    keep_alive=generative_config.get("keep_alive", 300),
                    top_p=generative_config.get("top_p", 0.95),
                    max_tokens=generative_config.get("max_tokens", 2048)  # 生成式任务需要更多token
                )
                logger.info(f"成功初始化生成式模型: {generative_config.get('model_name', 'llama3.2')}")
                return llm
            except Exception as e:
                # 如果首选模型不可用，尝试使用备选模型
                logger.warning(f"尝试使用首选模型失败: {str(e)}")
                
                # 备选模型列表
                fallback_models = ["qwen3", "mistral", "llama2"]
                
                for fallback_model in fallback_models:
                    try:
                        llm = ChatOllama(
                            model=fallback_model,
                            temperature=generative_config.get("temperature", 0.7),
                            base_url=generative_config.get("base_url", "http://localhost:11434"),
                            keep_alive=generative_config.get("keep_alive", 300)
                        )
                        logger.info(f"已切换到备选生成式模型: {fallback_model}")
                        return llm
                    except Exception:
                        continue
                
                # 如果所有备选模型都不可用，抛出异常
                raise Exception("无法初始化生成式模型，请检查Ollama服务是否正常运行")
        except Exception as e:
            logger.error(f"初始化生成式语言模型时出错: {str(e)}")
            # 返回None，基类会处理这种情况
            return None
    
    def _create_generative_chains(self):
        """创建各种生成式任务的链"""
        if not self.llm:
            return {}
        
        generative_chains = {}
        
        for chain_name, prompt_template in self.prompt_templates.items():
            try:
                chain = prompt_template | self.llm | self.output_parser
                generative_chains[chain_name] = chain
            except Exception as e:
                logger.error(f"创建{chain_name}生成链时出错: {str(e)}")
        
        return generative_chains
    
    def generate_content(self, topic, generation_type="general_info", **kwargs):
        """
        生成医疗相关内容
        
        参数:
            topic: 要生成内容的主题
            generation_type: 生成类型，可选值：general_info, detailed_explanation, patient_education, medical_case
            **kwargs: 其他生成参数
        
        返回:
            生成的内容字符串
        """
        try:
            # 验证生成类型
            if generation_type not in self.generative_chains:
                available_types = ", ".join(self.generative_chains.keys())
                return f"不支持的生成类型: {generation_type}。支持的类型: {available_types}"
            
            # 准备输入
            input_data = {"topic": topic}
            input_data.update(kwargs)
            
            # 记录生成请求
            logger.info(f"开始生成关于'{topic}'的{generation_type}内容")
            
            # 执行生成
            result = self.generative_chains[generation_type].invoke(input_data)
            
            # 记录生成完成
            logger.info(f"完成关于'{topic}'的{generation_type}内容生成")
            
            return result
        except Exception as e:
            logger.error(f"生成内容时出错: {str(e)}")
            # 提供更友好的错误信息
            if "积极拒绝" in str(e):
                return f"无法连接到Ollama服务。请确保Ollama服务已启动，然后重新尝试。\n错误详情: {str(e)}"
            return f"生成内容时出错: {str(e)}"
    
    def run(self, question, **kwargs):
        """运行智能体回答问题，同时提供生成式功能"""
        # 首先使用基类的run方法回答问题
        base_answer = super().run(question)
        
        # 检查是否需要生成额外内容
        if kwargs.get("generate_extra_content", False):
            # 根据问题自动确定生成类型
            generation_type = self._determine_generation_type(question)
            
            # 生成额外内容
            extra_content = self.generate_content(question, generation_type)
            
            # 组合回答
            full_answer = f"{base_answer}\n\n===== 详细说明 =====\n{extra_content}"
            return full_answer
        
        return base_answer
    
    def _determine_generation_type(self, question):
        """根据问题自动确定合适的生成类型"""
        question_lower = question.lower()
        
        if any(keyword in question_lower for keyword in ["详细", "深入", "研究进展", "原理"]):
            return "detailed_explanation"
        elif any(keyword in question_lower for keyword in ["患者教育", "科普", "生活建议"]):
            return "patient_education"
        elif any(keyword in question_lower for keyword in ["案例", "病例", "实例"]):
            return "medical_case"
        else:
            return "general_info"


if __name__ == "__main__":
    # 创建生成式医疗智能体实例
    generative_agent = OllamaGenerativeAgent()

    # 从配置中获取示例问题
    example_questions = generative_agent.config_reader.get_example_questions()
    
    # 如果配置中没有示例问题，使用默认问题
    if not example_questions:
        example_questions = [
            "什么是高血压？如何预防？",
            "老年高血压患者有哪些注意事项？",
            "脑血栓的高危因素有哪些？如何预防？",
            "糖尿病的预防措施有哪些？",
            "老年人如何保持健康的生活方式？"
        ]

    # 运行示例
    logger.info("基于生成式AI的医疗智能体已启动。输入'退出'或'quit'结束会话。")
    try:
        # 首先运行预设问题
        for q in example_questions:
            print(f"\n问题: {q}")
            result = generative_agent.run(q)
            print(f"回答: {result}")
            logger.info(f"处理预设问题: {q[:50]}...")

        # 然后进入交互模式
        print("\n预设问题已完成。现在可以输入您自己的问题：")
        logger.info("预设问题已完成，进入用户交互模式")
        while True:
            try:
                user_question = input("\n请输入您的问题: ")
                if user_question.lower() in ['退出', 'quit', 'exit']:
                    logger.info("感谢使用生成式医疗智能体，再见！")
                    break
                
                # 询问用户是否需要生成额外内容
                generate_extra = input("是否需要生成更详细的内容？(y/n): ").lower() == 'y'
                
                result = generative_agent.run(user_question, generate_extra_content=generate_extra)
                print(f"回答: {result}")
                logger.info(f"处理用户问题: {user_question[:50]}...")
            except KeyboardInterrupt:
                logger.info("\n程序被用户中断，再见！")
                break
            except Exception as e:
                logger.error(f"处理问题时出错: {str(e)}")
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")