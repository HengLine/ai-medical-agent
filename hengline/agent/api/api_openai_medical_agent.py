import os
import sys
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# 导入日志模块
from hengline.logger import logger

# 从基类导入
from hengline.agent.base_agent import BaseMedicalAgent

# 导入远程API相关的库
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class OpenAIMedicalAgent(BaseMedicalAgent):
    """基于远程OpenAI API的医疗智能体"""

    def __init__(self):
        # 初始化API调用统计
        self.api_call_count = 0
        self.total_tokens_used = 0

        # 调用基类初始化
        super().__init__()

        # 使用项目的日志模块
        self.logger = logger

    def _initialize_llm(self):
        """初始化远程API语言模型"""
        try:
            # 从配置中获取远程API参数
            openai_config = self.config_reader.get_llm_config("openai")

            if not openai_config:
                raise ValueError("未找到远程API配置")

            # 检查必要的配置项
            api_key = openai_config.get("api_key", "")
            if not api_key:
                logger.warning("未提供API密钥，尝试使用环境变量")
                api_key = os.environ.get("OPENAI_API_KEY", "")

            if not api_key:
                logger.error("未提供API密钥，请在配置文件中设置或设置环境变量OPENAI_API_KEY")
                return None

            # 初始化ChatOpenAI客户端
            llm = ChatOpenAI(
                api_key=api_key,
                model=openai_config.get("model", "gpt-4o"),
                temperature=openai_config.get("temperature", 0.1),
                streaming=openai_config.get("streaming", True),
                max_tokens=openai_config.get("max_tokens", 2048),
                base_url=openai_config.get("api_url", None)  # 如果使用自定义API端点
            )

            logger.info(f"远程API模型初始化成功: {openai_config.get('model', 'gpt-4o')}")
            return llm
        except Exception as e:
            logger.error(f"远程API模型初始化失败: {str(e)}")
            return None

    def load_medical_knowledge(self):
        """加载医疗知识库数据，远程API版本使用OpenAIEmbeddings进行优化"""
        try:
            # 从配置中获取知识库参数
            kb_config = self.config_reader.get_knowledge_base_config()
            data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../', kb_config.get("data_dir", "data"))

            # 获取API配置
            api_config = self.config_reader.get_llm_config("openai")
            api_key = api_config.get("api_key", "") or os.environ.get("OPENAI_API_KEY", "")

            # 初始化嵌入模型
            try:
                if api_key:
                    # 使用OpenAI的嵌入模型
                    embeddings = OpenAIEmbeddings(
                        api_key=api_key,
                        model="text-embedding-3-small"  # 使用轻量级嵌入模型降低成本
                    )
                else:
                    # 如果没有API密钥，使用开源的嵌入模型作为备选
                    logger.warning("未提供API密钥，将使用开源嵌入模型")
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs={"device": "cpu"},
                        encode_kwargs={"normalize_embeddings": True}
                    )
            except Exception as e:
                logger.warning(f"初始化嵌入模型失败，将使用FakeEmbeddings: {str(e)}")
                # 如果失败，回退到FakeEmbeddings
                from langchain_community.embeddings import FakeEmbeddings
                embeddings = FakeEmbeddings(size=768)

            # 使用基类的文档加载和处理逻辑
            from langchain.text_splitter import CharacterTextSplitter
            from langchain_community.document_loaders import TextLoader
            from langchain_core.documents import Document
            from langchain_chroma import Chroma

            # 获取data目录下的所有txt文件
            files = []
            if os.path.exists(data_dir):
                for root, _, filenames in os.walk(data_dir):
                    for filename in filenames:
                        if filename.endswith('.txt'):
                            # 检查文件是否包含医疗相关关键词
                            if not kb_config.get("medical_keywords") or any(keyword.lower() in filename.lower() for keyword in kb_config["medical_keywords"]):
                                files.append(os.path.join(root, filename))

            if not files:
                logger.warning("未找到医疗知识库文件。将创建一个空的向量存储。")
                # 创建空文档列表
                empty_docs = [Document(page_content="这是一个空的医疗知识库文档", metadata={"source": "empty"})]
                return Chroma.from_documents(empty_docs, embeddings)

            # 加载文档
            documents = []
            for file in files:
                try:
                    loader = TextLoader(file, encoding="utf-8")
                    documents.extend(loader.load())
                except Exception as e:
                    logger.error(f"加载文件 {file} 时出错: {str(e)}")

            if not documents:
                logger.warning("未能加载任何文档。将创建一个空的向量存储。")
                empty_docs = [Document(page_content="这是一个空的医疗知识库文档", metadata={"source": "empty"})]
                return Chroma.from_documents(empty_docs, embeddings)

            # 从配置中获取文本分割参数
            text_splitter_config = self.config_reader.get_module_config("text_splitter")

            # 分割文档
            text_splitter = CharacterTextSplitter(
                chunk_size=text_splitter_config.get("chunk_size", 1000),
                chunk_overlap=text_splitter_config.get("chunk_overlap", 200)
            )
            texts = text_splitter.split_documents(documents)

            # 创建向量存储
            vectorstore = Chroma.from_documents(texts, embeddings)
            logger.info(f"成功创建向量存储，包含{len(texts)}个文档块")
            return vectorstore
        except Exception as e:
            logger.error(f"加载医疗知识库时出错: {str(e)}")
            # 回退到基类的实现
            return super().load_medical_knowledge()

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
