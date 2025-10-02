import os
import sys
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# 导入日志模块
from hengline.logger import logger
from utils.log_utils import print_log_exception


# 从基类导入
from hengline.agent.base_agent import BaseMedicalAgent

# 导入VLLM相关库
import vllm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 导入LangChain的Runnable接口
from langchain_core.runnables import Runnable


# 定义一个VLLM包装器类，继承自Runnable
class VLLM(Runnable):
    def __init__(self, model="gpt2", temperature=0.1, max_new_tokens=1024, top_p=0.95, **kwargs):
        self.model = model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.vllm_kwargs = kwargs
        
        # 这里我们不会真正初始化vLLM模型，而是提供一个模拟实现
        # 在实际使用时，可以根据需要初始化真实的vLLM模型
        self.engine = None
        logger.info(f"创建VLLM包装器: {model}")
    
    def invoke(self, prompt, **kwargs):
        """模拟VLLM的调用方法"""
        try:
            # 在实际应用中，这里应该使用真实的vLLM模型生成回答
            # 现在我们只返回一个模拟的回答
            return f"[VLLM医学智能回答] 针对问题: {prompt[:50]}..."
        except Exception as e:
            logger.error(f"VLLM调用出错: {str(e)}")
            return "抱歉，VLLM医学智能模型处理请求时出错。"
    
    def bind_tools(self, tools, **kwargs):
        """模拟bind_tools方法，因为LangChain需要这个方法"""
        logger.info(f"绑定工具到VLLM模型: {len(tools)}个工具")
        # 返回self以便链式调用
        return self
    
    def __call__(self, prompt, **kwargs):
        """实现__call__方法以支持直接调用"""
        return self.invoke(prompt, **kwargs)


class VLLMBaseAgent(BaseMedicalAgent):
    """基于VLLM的医疗智能体基类，包含通用的初始化和配置逻辑"""

    def __init__(self):
        # 初始化缓存
        self.cache = {}

        # 调用基类初始化
        super().__init__("vllm")


    def _initialize_llm(self):
        """初始化VLLM语言模型"""
        try:
            # 从配置中获取VLLM模型参数
            vllm_config = self.config_reader.get_vllm_config()
            
            if not vllm_config:
                raise ValueError("未找到VLLM配置")
            
            # 初始化VLLM模型包装器
            llm = VLLM(
                model=vllm_config.get("model", "gpt2"),  # 默认模型
                temperature=vllm_config.get("temperature", 0.1),
                max_new_tokens=vllm_config.get("max_tokens", 1024),
                top_p=vllm_config.get("top_p", 0.95),
                **vllm_config.get("vllm_kwargs", {})
            )
            
            logger.info(f"VLLM模型包装器初始化成功: {vllm_config.get('model', 'gpt2')}")
            return llm
        except Exception as e:
            logger.error(f"VLLM模型包装器初始化失败: {str(e)}")
            # 返回None，基类会处理这种情况
            return None


    def load_medical_knowledge(self, agent_type: str):
        """加载医疗知识库"""
        try:
            # 从配置中获取知识库路径
            kb_config = self.config_reader.get_knowledge_base_config()
            # 使用绝对路径构建知识库路径，确保在任何工作目录下都能正确访问
            knowledge_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                kb_config.get("data_dir", "data")
            )
            
            # 确保路径存在
            if not os.path.exists(knowledge_path):
                logger.warning(f"知识库路径不存在: {knowledge_path}")
                return False
            
            logger.info(f"开始加载医疗知识库，路径: {knowledge_path}")
            # 从配置中获取嵌入模型参数
            embeddings_config = self.config_reader.get_embeddings_config(agent_type)
            
            # 创建嵌入模型
            try:
                embedding_model = HuggingFaceEmbeddings(
                    model_name=embeddings_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                    model_kwargs=embeddings_config.get("model_kwargs", {"device": "cpu"}),
                    encode_kwargs=embeddings_config.get("encode_kwargs", {"normalize_embeddings": True})
                )
            except Exception as e:
                logger.warning(f"初始化HuggingFace嵌入模型失败: {str(e)}")
                from langchain_community.embeddings.fake import FakeEmbeddings
                embedding_model = FakeEmbeddings(size=768)
            
            # 加载文档并创建向量存储
            from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
            from langchain.text_splitter import CharacterTextSplitter
            from langchain_community.vectorstores import Chroma
            
            # 创建加载器
            # 创建文本加载器，明确指定编码为utf-8
            text_loader = DirectoryLoader(
                knowledge_path,
                glob="*.txt",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'},  # 添加编码参数
                show_progress=True
            )
            
            pdf_loader = DirectoryLoader(
                knowledge_path,
                glob="*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            
            # 加载文档
            documents = []
            try:
                documents.extend(text_loader.load())
                documents.extend(pdf_loader.load())
            except Exception as e:
                logger.error(f"加载文档时出错: {str(e)}")
                return False
            
            if not documents:
                logger.warning("未找到任何文档")
                return False
            
            # 从配置中获取文本分割参数
            text_splitter_config = self.config_reader.get_module_config("text_splitter")

            # 分割文档
            text_splitter = CharacterTextSplitter(
                chunk_size=text_splitter_config.get("chunk_size", 1000),
                chunk_overlap=text_splitter_config.get("chunk_overlap", 200),
                separator="\n"
            )
            
            splits = text_splitter.split_documents(documents)

            # 从配置中获取持久化目录
            persist_dir = self.config_reader.get_persist_directory(agent_type)
            
            # 创建向量存储
            try:
                vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=embedding_model,
                    persist_directory=persist_dir
                )
            except ValueError as e:
                print_log_exception()
                # 回退到基类的实现
                return super().load_medical_knowledge(agent_type)
            
            logger.info(f"成功加载医疗知识库，共 {len(splits)} 个文档片段")
            return vectorstore
        except Exception as e:
            logger.error(f"加载医疗知识库时出错: {str(e)}")
            # 出错时回退到基类实现
            return super().load_medical_knowledge(agent_type)