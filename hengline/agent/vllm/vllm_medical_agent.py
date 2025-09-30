import os
import sys
import os
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# 导入基类
from hengline.agent.vllm.vllm_base_agent import VLLMBaseAgent
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from hengline.logger import warning, info, error


class VLLMMedicalAgent(VLLMBaseAgent):
    """基于VLLM本地模型的医疗智能体"""
    
    def __init__(self):
        # 调用基类初始化
        super().__init__()
        return None
    
    
    def _create_retrieval_chain(self):
        """创建检索链，针对VLLM进行优化"""
        try:
            # 检查是否已经初始化了llm和vectorstore
            if not self.llm or not self.vectorstore:
                warning("LLM或VectorStore未初始化，无法创建检索链")
        
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
            
            info("VLLM检索链创建成功")
            return retrieval_chain
        except Exception as e:
            error(f"创建检索链时出错: {str(e)}")
            # 回退到基类的实现
            return super()._create_retrieval_chain()
    
    def run(self, question):
        """运行智能体回答问题，针对VLLM模型进行优化"""
        # 检查缓存
        cache_key = question.strip().lower()
        if cache_key in self.cache:
            debug(f"从缓存中获取问题答案: {cache_key}")
            return self.cache[cache_key]
        
        # 调用基类的run方法
        result = super().run(question)
        
        # 将结果存入缓存
        self.cache[cache_key] = result
        
        # 限制缓存大小
        if len(self.cache) > 100:
            # 移除最早的缓存项
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        
        return result


if __name__ == "__main__":
    # 创建基于VLLM的医疗智能体实例
    medical_agent = VLLMMedicalAgent()

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
    print("基于VLLM的医疗智能体已启动。输入'退出'或'quit'结束会话。")
    try:
        # 首先运行预设问题
        for q in example_questions:
            print(f"\n问题: {q}")
            result = medical_agent.run(q)
            print(f"回答: {result}")

        # 然后进入交互模式
        print("\n预设问题已完成。现在可以输入您自己的问题：")
        while True:
            try:
                user_question = input("\n请输入您的问题: ")
                if user_question.lower() in ['退出', 'quit', 'exit']:
                    print("感谢使用医疗智能体，再见！")
                    break
                result = medical_agent.run(user_question)
                print(f"回答: {result}")
            except KeyboardInterrupt:
                print("\n程序被用户中断，再见！")
                break
            except Exception as e:
                print(f"处理问题时出错: {str(e)}")
    except Exception as e:
        print(f"程序运行出错: {str(e)}")