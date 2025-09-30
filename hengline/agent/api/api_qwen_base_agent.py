"""@FileName: api_qwen_base_agent.py
@Description: 通义千问API的基础智能体类，用于抽离问答型和生成式智能体的通用部分
@Author: HengLine
@Time: 2025/10/1 10:00
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


class QwenBaseAgent(BaseMedicalAgent):
    """通义千问API的基础智能体类，提供通义千问模型的通用初始化和配置功能"""

    def __init__(self):
        # 初始化API调用统计
        self.api_call_count = 0
        self.total_tokens_used = 0
        
        # 模型工具支持标记
        self.model_supports_tools = False
        
        # 使用项目的日志模块
        self.logger = logger
        
        # 调用基类初始化
        super().__init__()

    def _initialize_llm(self):
        """初始化通义千问语言模型
        
        该方法从配置中读取通义千问模型的参数，并初始化ChatTongyi实例。
        同时检查模型是否支持工具调用功能。
        
        Returns:
            ChatTongyi: 初始化后的通义千问语言模型实例，或None（初始化失败时）
        """
        try:
            # 从配置中获取Qwen模型参数
            model_name = self.config_reader.get_llm_value("qwen", "model", "qwen-plus")
            api_url = self.config_reader.get_llm_value("qwen", "api_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            temperature = self.config_reader.get_llm_value("qwen", "temperature", 0.1)
            max_tokens = self.config_reader.get_llm_value("qwen", "max_tokens", 2048)

            # 获取API密钥
            api_key = self.config_reader.get_qwen_api_key()

            # 初始化通义千问模型
            llm = ChatTongyi(
                model=model_name,
                dashscope_api_key=api_key,
                streaming=True,
                max_retries=3,
                model_kwargs={
                    "base_url": api_url,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
            )

            self.model_supports_tools = self._check_tool_support(model_name)
            logger.info(f"成功初始化通义千问模型: {model_name}, 支持工具调用: {self.model_supports_tools}")
            return llm
        except Exception as e:
            logger.error(f"初始化通义千问语言模型时出错: {str(e)}")
            # 确保model_supports_tools有默认值
            self.model_supports_tools = False
            # 返回None，基类会处理这种情况
            return None

    def _check_tool_support(self, model_name):
        """检查模型是否支持工具调用
        
        Args:
            model_name: 模型名称
        
        Returns:
            bool: 是否支持工具调用
        """
        # 支持工具调用的Qwen模型列表
        tool_supported_models = [
            "qwen-plus",
            "qwen-max",
            "qwen-max-longcontext",
            "qwen-72b-chat",
            "qwen-14b-chat"
        ]

        # 检查模型名称是否为None或空
        if model_name is None or not model_name:
            logger.warning("模型名称为空，默认为不支持工具调用")
            return False

        # 检查模型名称是否在支持列表中
        model_name_lower = model_name.lower()
        for supported_model in tool_supported_models:
            if supported_model.lower() in model_name_lower:
                return True

        logger.warning(f"模型 {model_name} 可能不支持完整的工具调用功能")
        return False

    def load_medical_knowledge(self):
        """加载医疗知识库数据，Qwen版本进行了优化"""
        try:

            # 初始化嵌入模型 - Qwen版本可以使用开源嵌入模型或FakeEmbeddings
            try:
                # 尝试使用HuggingFaceEmbeddings
                from langchain_community.embeddings import HuggingFaceEmbeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )
            except Exception as e:
                logger.warning(f"初始化HuggingFaceEmbeddings失败，将使用FakeEmbeddings: {str(e)}")
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