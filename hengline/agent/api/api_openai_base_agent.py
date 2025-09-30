"""@FileName: api_openai_base_agent.py
@Description: OpenAI API的基础智能体类，用于抽离问答型和生成式智能体的通用部分
@Author: HengLine
@Time: 2025/10/1 10:15
"""
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# 导入日志模块
from hengline.logger import logger

# 从基类导入
from hengline.agent.base_agent import BaseMedicalAgent

# 导入OpenAI特定的库
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class OpenAIBaseAgent(BaseMedicalAgent):
    """OpenAI API的基础智能体类，提供OpenAI模型的通用初始化和配置功能"""

    def __init__(self):
        # 初始化API调用统计
        self.api_call_count = 0
        self.total_tokens_used = 0
        
        # 使用项目的日志模块
        self.logger = logger
        
        # 调用基类初始化
        super().__init__()

    def _initialize_llm(self):
        """初始化OpenAI语言模型
        
        该方法从配置中读取OpenAI模型的参数，并初始化ChatOpenAI实例。
        
        Returns:
            ChatOpenAI: 初始化后的OpenAI语言模型实例，或None（初始化失败时）
        """
        try:
            # 从配置中获取OpenAI API参数
            openai_config = self.config_reader.get_llm_config("openai")

            if not openai_config:
                raise ValueError("未找到OpenAI API配置")

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

            logger.info(f"OpenAI API模型初始化成功: {openai_config.get('model', 'gpt-4o')}")
            return llm
        except Exception as e:
            logger.error(f"OpenAI API模型初始化失败: {str(e)}")
            return None

    def load_medical_knowledge(self):
        """加载医疗知识库数据，OpenAI版本使用OpenAIEmbeddings进行优化"""
        try:
           
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

            # 加载文档
            documents = self.load_medical_documents()

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

    def update_api_call_stats(self, tokens_used=0):
        """更新API调用统计信息
        
        Args:
            tokens_used: 使用的tokens数量
        """
        self.api_call_count += 1
        self.total_tokens_used += tokens_used
        logger.debug(f"API调用统计 - 总调用次数: {self.api_call_count}, 总tokens使用量: {self.total_tokens_used}")

    def get_api_call_stats(self):
        """获取API调用统计信息
        
        Returns:
            dict: 包含调用次数和tokens使用量的字典
        """
        return {
            "call_count": self.api_call_count,
            "total_tokens": self.total_tokens_used
        }

    def _create_retrieval_chain(self):
        """创建检索链，针对OpenAI API进行优化"""
        try:
            # 检查是否已经初始化了llm和vectorstore
            if not self.llm or not self.vectorstore:
                logger.warning("语言模型或向量存储未初始化，无法创建检索链")
                return None

            # 获取配置
            retrieval_config = self.config_reader.get_retrieval_config()
            
            # 为OpenAI API优化的检索链配置
            optimized_config = retrieval_config.copy()
            
            # 根据OpenAI模型特性调整检索配置
            # 例如，对于GPT-4系列模型，可以使用更复杂的链类型
            if hasattr(self, 'llm') and hasattr(self.llm, 'model'):
                model_name = self.llm.model
                if 'gpt-4' in model_name:
                    optimized_config["chain_type"] = "refine"  # 对于GPT-4，使用refine链可以获得更好的结果
                    optimized_config["chain_type_kwargs"] = {"verbose": False}
                else:
                    optimized_config["chain_type"] = "stuff"  # 对于其他模型，使用stuff链保持简单高效

            # 创建优化后的检索链
            retrieval_chain = super()._create_retrieval_chain()
            
            logger.info(f"为OpenAI API创建了优化的检索链，模型: {self.llm.model if hasattr(self.llm, 'model') else 'unknown'}")
            return retrieval_chain
        except Exception as e:
            logger.error(f"创建OpenAI优化的检索链时出错: {str(e)}")
            # 回退到基类的实现
            return super()._create_retrieval_chain()