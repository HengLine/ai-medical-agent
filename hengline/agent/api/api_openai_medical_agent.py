import os
import sys
from typing import Dict, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# 导入日志模块
from hengline.logger import logger

# 从基类导入
from hengline.agent.api.api_openai_base_agent import OpenAIBaseAgent


class OpenAIMedicalAgent(OpenAIBaseAgent):
    """基于远程OpenAI API的医疗智能体"""

    def __init__(self):
        # 调用基类初始化
        super().__init__()

    def _create_retrieval_chain(self):
        """创建检索链，针对远程API进行优化"""
        try:
            # 检查是否已经初始化了llm和vectorstore
            if not self.llm or not self.vectorstore:
                logger.warning("LLM或VectorStore未初始化，无法创建检索链")
                return None

            # 从配置中获取检索链参数
            retrieval_config = self.config_reader.get_module_config("retrieval")

            from langchain.chains import RetrievalQA
            retrieval_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type=retrieval_config.get("chain_type", "stuff"),
                retriever=self.vectorstore.as_retriever(
                    search_kwargs=retrieval_config.get("search_kwargs", {"k": 3})
                ),
                return_source_documents=retrieval_config.get("return_source_documents", True)
            )

            # 确保检索链不为None
            if retrieval_chain is None:
                logger.error("创建检索链失败，返回值为None")
                return None

            logger.info("远程API检索链创建成功")
            return retrieval_chain
        except Exception as e:
            logger.error(f"创建检索链时出错: {str(e)}")
            # 回退到基类的实现
            return super()._create_retrieval_chain()

    def run(self, question):
        """运行智能体回答问题，针对远程API进行优化"""
        # 增加API调用计数
        self.api_call_count += 1

        # 调用基类的run方法
        result = super().run(question)

        # 如果结果包含token使用信息，可以更新统计
        # 这里可以根据实际的API响应格式进行调整

        return result

    def get_api_stats(self) -> Dict[str, Any]:
        """获取API调用统计信息"""
        return {
            "api_call_count": self.api_call_count,
            "total_tokens_used": self.total_tokens_used
        }

    def clear_cache(self):
        """清除智能体缓存（如果有）"""
        # 如果在基类或子类中实现了缓存，可以在这里添加清除逻辑
        pass


if __name__ == "__main__":
    # 创建基于远程API的医疗智能体实例
    medical_agent = OpenAIMedicalAgent()

    # 从配置中获取示例问题
    example_questions = medical_agent.config_reader.get_example_questions()

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
    logger.info("基于远程API的医疗智能体已启动。输入'退出'或'quit'结束会话。")
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
                    # 显示API调用统计
                    stats = medical_agent.get_api_stats()
                    logger.info(f"\nAPI调用统计: {stats}")
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
