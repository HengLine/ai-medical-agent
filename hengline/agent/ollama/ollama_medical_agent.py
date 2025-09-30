import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# 导入日志模块
from hengline.logger import logger

# 从基类导入
from hengline.agent.base_agent import BaseMedicalAgent

# 导入Ollama特定的库
from langchain_ollama import ChatOllama


class OllamaMedicalAgent(BaseMedicalAgent):
    """基于Ollama本地模型的医疗智能体"""
    
    def _initialize_llm(self):
        """初始化Ollama语言模型"""
        try:
            # 从配置中获取Ollama模型参数
            ollama_config = self.config_reader.get_ollama_config()
            
            # 尝试使用支持工具调用的模型
            try:
                llm = ChatOllama(
                    model=ollama_config.get("model_name", "llama3.2"),
                    temperature=ollama_config.get("temperature", 0.1),
                    base_url=ollama_config.get("base_url", "http://localhost:11434"),
                    keep_alive=ollama_config.get("keep_alive", 300),
                    top_p=ollama_config.get("top_p", 0.95),
                    max_tokens=ollama_config.get("max_tokens", 1024)
                )
                self.model_supports_tools = True
                logger.info(f"成功初始化Ollama模型: {ollama_config.get('model_name', 'llama3.2')}")
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
                            temperature=self.config_reader.get_llm_value("ollama", "temperature", 0.1),
                            base_url=self.config_reader.get_llm_value("ollama", "base_url", "http://localhost:11434"),
                            keep_alive=self.config_reader.get_llm_value("ollama", "keep_alive", 300)
                        )
                        self.model_supports_tools = self._check_tool_support(fallback_model)
                        logger.info(f"已切换到备选模型: {fallback_model}")
                        if not self.model_supports_tools:
                            logger.warning("当前模型不支持工具调用，将使用简化的回答模式")
                        return llm
                    except Exception:
                        continue
                
                # 如果所有备选模型都不可用，抛出异常
                raise Exception("无法初始化Ollama模型，请检查Ollama服务是否正常运行")
        except Exception as e:
            logger.error(f"初始化Ollama语言模型时出错: {str(e)}")
            # 返回None，基类会处理这种情况
            return None
    
    def _check_tool_support(self, model_name):
        """检查模型是否支持工具调用"""
        # 这里可以根据模型名称或其他方式检查是否支持工具调用
        # 一般来说，较新的模型如llama3.1及以上版本支持工具调用
        supported_models = ["llama3.1", "llama3.2", "gemma2", "qwen3"]
        return any(supported in model_name.lower() for supported in supported_models)
    
    def run(self, question):
        """运行智能体回答问题，针对Ollama模型进行优化"""
        # 调用基类的run方法
        result = super().run(question)
        
        # 可以在这里添加Ollama特定的后处理
        return result


if __name__ == "__main__":
    # 创建基于Ollama的医疗智能体实例
    medical_agent = OllamaMedicalAgent()

    # 从配置中获取示例问题
    example_questions = medical_agent.config_reader.get_example_questions()
    
    # 如果配置中没有示例问题，使用默认问题
    if not example_questions:
        example_questions = [
            "什么是高血压？如何预防？",
            "老年高血压患者有哪些注意事项？",
            "脑血栓的高危因素有哪些？如何预防？",
            "糖尿病的预防措施有哪些？",
            "老年人如何保持健康的生活方式？",
            "高血压、糖尿病和脑血栓之间有什么关系？",
            "高危人群应该多久进行一次体检？"
        ]

    # 运行示例
    logger.info("基于Ollama的医疗智能体已启动。输入'退出'或'quit'结束会话。")
    try:
        # 首先运行预设问题
        for q in example_questions:
            print(f"\n问题: {q}")
            result = medical_agent.run(q)
            print(f"回答: {result}")
            logger.info(f"处理预设问题: {q[:50]}...")

        # 然后进入交互模式
        print("\n预设问题已完成。现在可以输入您自己的问题：")
        logger.info("预设问题已完成，进入用户交互模式")
        while True:
            try:
                user_question = input("\n请输入您的问题: ")
                if user_question.lower() in ['退出', 'quit', 'exit']:
                    logger.info("感谢使用医疗智能体，再见！")
                    break
                result = medical_agent.run(user_question)
                print(f"回答: {result}")
                logger.info(f"处理用户问题: {user_question[:50]}...")
            except KeyboardInterrupt:
                logger.info("\n程序被用户中断，再见！")
                break
            except Exception as e:
                logger.error(f"处理问题时出错: {str(e)}")
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
