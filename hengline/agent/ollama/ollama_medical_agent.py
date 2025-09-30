import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# 导入日志模块
from hengline.logger import logger

# 从基类导入
from hengline.agent.ollama.ollama_base_agent import OllamaBaseAgent


class OllamaMedicalAgent(OllamaBaseAgent):
    """基于Ollama本地模型的医疗智能体"""

    def __init__(self):
        # 调用基类初始化
        super().__init__()
    
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
