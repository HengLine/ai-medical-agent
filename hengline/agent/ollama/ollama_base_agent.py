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


class OllamaBaseAgent(BaseMedicalAgent):
    """基于Ollama的医疗智能体基类，包含通用的初始化和配置逻辑"""

    def __init__(self):
        # 初始化API调用统计
        self.api_call_count = 0
        self.total_tokens_used = 0

        # 调用基类初始化
        super().__init__("ollama")

        # 使用项目的日志模块
        self.logger = logger

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
                    num_predict=ollama_config.get("max_tokens", 1024)
                )
                self.model_supports_tools = self._check_tool_support(ollama_config.get("model_name", "llama3.2"))
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
                            temperature=ollama_config.get("temperature", 0.1),
                            base_url=ollama_config.get("base_url", "http://localhost:11434"),
                            keep_alive=ollama_config.get("keep_alive", 300)
                        )
                        self.model_supports_tools = self._check_tool_support(fallback_model)
                        logger.info(f"已切换到备选Ollama模型: {fallback_model}")
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
            # Ollama没有内置的嵌入模型，所以使用HuggingFaceEmbeddings
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                embedding_model = HuggingFaceEmbeddings(
                    model_name=embeddings_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                    model_kwargs=embeddings_config.get("model_kwargs", {"device": "cpu"}),
                    encode_kwargs=embeddings_config.get("encode_kwargs", {"normalize_embeddings": True})
                )
            except Exception as e:
                logger.warning(f"初始化HuggingFace嵌入模型失败，使用FakeEmbeddings: {str(e)}")
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
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embedding_model,
                persist_directory=persist_dir
            )
            
            logger.info(f"成功加载医疗知识库，共 {len(splits)} 个文档片段")
            return vectorstore
        except Exception as e:
            logger.error(f"加载医疗知识库时出错: {str(e)}")
            # 出错时回退到基类实现
            return super().load_medical_knowledge(agent_type)