import sys
import os
from abc import ABC, abstractmethod
from typing import List

from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
# LangChain和LangGraph相关导入
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# 导入日志模块
from hengline.logger import logger
from utils.log_utils import print_log_exception

# 导入工具和配置
from hengline.tools.medical_tools import MedicalTools
from hengline.config import config_reader


class MedicalAgentState:
    """定义智能体的状态结构"""
    messages: List[BaseMessage] = add_messages()
    # 可以添加其他状态字段，如思考过程、使用的工具等


class BaseMedicalAgent(ABC):
    """医疗智能体基类，定义通用接口和共享功能"""

    def __init__(self, agent_type: str):
        # 初始化配置读取器
        self.config_reader = config_reader
        self.agent_type = agent_type

        # 初始化医疗工具
        self.medical_tools = MedicalTools()
        
        # 从配置中获取知识库参数
        self.kb_config = self.config_reader.get_knowledge_base_config()
        self.data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                self.kb_config.get("data_dir", "data")
            )

        # 加载RAG数据
        self.vectorstore = self.load_medical_knowledge(self.agent_type)

        # 创建网络搜索工具
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            self.search = DuckDuckGoSearchRun()
        except ImportError:
            self.search = None
            logger.warning("未能导入DuckDuckGoSearchRun，网络搜索功能将不可用")

        # 初始化语言模型
        self.llm = self._initialize_llm()

        # 创建检索链
        self.retrieval_chain = self._create_retrieval_chain()

        # 定义工具列表
        self.tools = self._define_tools()

        # 初始化LangGraph智能体
        self.agent = self._initialize_langgraph_agent()

    @abstractmethod
    def _initialize_llm(self):
        """初始化语言模型，由子类实现"""
        pass

    def _create_retrieval_chain(self):
        """创建检索链"""
        if not self.llm or not self.vectorstore:
            return None

        try:
            # 从配置中获取检索链参数
            retrieval_config = self.config_reader.get_retrieval_config()

            return RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type=retrieval_config.get("chain_type", "stuff"),
                chain_type_kwargs=retrieval_config.get("chain_type_kwargs", {}),
                retriever=self.vectorstore.as_retriever(
                    search_kwargs=retrieval_config.get("search_kwargs", {"k": 3})
                ),
                return_source_documents=retrieval_config.get("return_source_documents", True),
                verbose=retrieval_config.get("verbose", False)
            )
        except Exception as e:
            logger.error(f"创建检索链时出错: {str(e)}")
            return None

    def _define_tools(self):
        """定义智能体可用的工具"""
        tools = []

        # 添加查询医疗知识库的工具
        if self.retrieval_chain:
            tools.append(self.query_medical_knowledge_tool)

        # 添加网络搜索工具
        if self.search:
            tools.append(self.web_search_tool)

        # 添加症状提取工具
        tools.append(self.extract_symptoms_tool)

        # 添加症状评估工具
        tools.append(self.assess_severity_tool)

        return tools

    def _initialize_langgraph_agent(self):
        """初始化LangGraph智能体"""
        if not self.llm or not self.tools:
            return None

        try:
            # 创建React Agent
            agent_executor = create_react_agent(
                self.llm,
                self.tools
            )
            return agent_executor
        except Exception as e:
            logger.error(f"初始化LangGraph智能体时出错: {str(e)}")
            return None


    def get_knowledge_files(self):
        """检查知识库是否存在"""
        files = []

        if not os.path.exists(self.data_dir):
            logger.error(f"知识库目录不存在: {self.data_dir}")
            return files

        for root, _, filenames in os.walk(self.data_dir):
            for filename in filenames:
                if filename.endswith('.txt'):
                    # 检查文件是否包含医疗相关关键词
                    if not self.kb_config.get("medical_keywords") or any(keyword.lower() in filename.lower() for keyword in self.kb_config["medical_keywords"]):
                        files.append(os.path.join(root, filename))
        return files


    def load_medical_documents(self):
        """加载医疗知识库数据"""
        # 获取data目录下的所有txt文件
        files = self.get_knowledge_files()
        logger.info(f"发现 {len(files)} 个医疗知识库文件")

        if not files:
            logger.warning("未找到医疗知识库文件。将创建一个空的向量存储。")
            from langchain_core.documents import Document
            # 创建空文档列表并使用from_documents方法初始化Chroma
            empty_docs = [Document(page_content="这是一个空的医疗知识库文档", metadata={"source": "empty"})]
            return Chroma.from_documents(empty_docs, FakeEmbeddings(size=768))

        # 加载文档
        documents = []
        for file in files:
            try:
                loader = TextLoader(file, encoding="utf-8")
                documents.extend(loader.load())
            except Exception as e:
                logger.error(f"加载文件 {file} 时出错: {str(e)}")

        return documents


    def load_medical_knowledge(self, agent_type: str):
        """加载医疗知识库数据"""
        try:
            # 从配置中获取嵌入模型参数
            # embeddings_config = self.config_reader.get_embeddings_config(agent_type)
            
            # 加载文档
            documents = self.load_medical_documents()

            if not documents:
                logger.warning("未能加载任何文档。将创建一个空的向量存储。")
                from langchain_core.documents import Document
                empty_docs = [Document(page_content="这是一个空的医疗知识库文档", metadata={"source": "empty"})]
                return Chroma.from_documents(empty_docs, FakeEmbeddings(size=768))

            # 从配置中获取文本分割参数
            text_splitter_config = self.config_reader.get_text_splitter_config()

            # 分割文档
            text_splitter = CharacterTextSplitter(
                chunk_size=text_splitter_config.get("chunk_size", 1000),
                chunk_overlap=text_splitter_config.get("chunk_overlap", 200)
            )
            texts = text_splitter.split_documents(documents)

            # 从配置中获取持久化目录
            persist_dir = self.config_reader.get_persist_directory(agent_type)
            
            # 创建向量存储
            if persist_dir:
                try:
                    vectorstore = Chroma.from_documents(
                        texts, 
                        FakeEmbeddings(size=768), 
                        persist_directory=persist_dir
                    )
                    logger.info(f"成功创建持久化向量存储，包含{len(texts)}个文档块，持久化目录: {persist_dir}")
                except ValueError as e:
                    if "dimension" in str(e).lower():
                        return self.recreate_vectorstore(e, persist_dir, texts)
                    else:
                        raise
            else:
                vectorstore = Chroma.from_documents(texts, FakeEmbeddings(size=768))
                logger.info(f"成功创建向量存储，包含{len(texts)}个文档块")
            
            return vectorstore
        except Exception as e:
            logger.error(f"加载医疗知识库时出错: {str(e)}")
            if "dimension" in str(e).lower():
                return self.recreate_vectorstore(e, persist_dir, texts)
            else:
                from langchain_core.documents import Document
                empty_docs = [Document(page_content="这是一个空的医疗知识库文档", metadata={"source": "empty"})]
                return Chroma.from_documents(empty_docs, FakeEmbeddings(size=768))

    def recreate_vectorstore(self, e, persist_dir: str, texts):
        """重新创建向量存储"""
        logger.warning(f"检测到向量维度不匹配错误: {str(e)}，将重新创建向量存储")
        import shutil
        import time
        import os
        import subprocess
        import sys
        
        # 先尝试重命名旧目录，再重新创建（解决Windows文件锁定问题）
        dir_exists = os.path.exists(persist_dir)
        if dir_exists:
            temp_dir = None
            
            # 1. 尝试常规重命名
            try:
                timestamp = int(time.time())
                temp_dir = f"{persist_dir}_old_{timestamp}"
                os.rename(persist_dir, temp_dir)
                logger.info(f"成功将旧向量存储目录重命名为: {temp_dir}")
            except (PermissionError, OSError) as e:
                logger.warning(f"无法重命名目录（文件可能被锁定）: {str(e)}，将尝试强制删除")
                temp_dir = None
              
            if temp_dir is None:
                # 2. 尝试常规删除
                try:
                    shutil.rmtree(persist_dir)
                    logger.info(f"成功删除旧的向量存储目录: {persist_dir}")
                except PermissionError as pe:
                    logger.warning(f"常规删除失败: {str(pe)}，尝试强制删除")
                    
                    # 3. 尝试强制删除（Windows特有方法）
                    try:
                        if sys.platform == 'win32':
                            # 使用Windows命令行强制删除
                            cmd = f'rmdir /s /q "{persist_dir}"' if os.path.isdir(persist_dir) else f'del /f /q "{persist_dir}"'
                            subprocess.run(cmd, shell=True, check=True)
                            logger.info(f"成功使用命令行强制删除目录: {persist_dir}")
                        else:
                            # Linux/Mac下的强制删除
                            shutil.rmtree(persist_dir, ignore_errors=True)
                            logger.info(f"成功强制删除目录: {persist_dir}")
                    except Exception as force_e:
                        logger.error(f"强制删除也失败: {str(force_e)}")
                
                # 检查删除是否成功
                dir_still_exists = os.path.exists(persist_dir)
                if dir_still_exists:
                    logger.warning(f"所有删除方法都失败，将尝试直接创建新的向量存储并强制覆盖现有内容")
                    # 不修改persist_dir，继续执行后续代码，让Chroma尝试直接覆盖
        
        # 创建新的向量存储（即使目录存在也尝试创建，实现强制覆盖）
        try:
            # 直接尝试创建向量存储，即使目录存在
            vectorstore = Chroma.from_documents(
                texts, 
                FakeEmbeddings(size=768), 
                persist_directory=persist_dir
            )
            logger.info(f"成功重新创建持久化向量存储，包含{len(texts)}个文档块，路径: {persist_dir}")
            return vectorstore
        except Exception as create_e:
            print_log_exception()
            logger.error(f"重新创建向量存储时出错: {str(create_e)}")
            # 如果强制覆盖也失败，使用带时间戳的备用路径
            # if dir_exists:
            #     timestamp = int(time.time())
            #     backup_persist_dir = f"{persist_dir}_{timestamp}"
            #     logger.warning(f"无法在现有路径创建向量存储: {str(create_e)}，将使用备用路径: {backup_persist_dir}")
                
            #     vectorstore = Chroma.from_documents(
            #         texts, 
            #         FakeEmbeddings(size=768), 
            #         persist_directory=backup_persist_dir
            #     )
            #     logger.info(f"成功在备用路径创建向量存储，包含{len(texts)}个文档块，路径: {backup_persist_dir}")
            #     return vectorstore
            # else:
            #     raise

    @tool
    def query_medical_knowledge_tool(self, query: str) -> str:
        """适合用来回答医学知识相关的问题，包括疾病、药物、急救和健康生活方式等内容"""
        return self.query_medical_knowledge(query)

    @tool
    def web_search_tool(self, query: str) -> str:
        """适合用来搜索最新的医疗信息、研究进展和新闻等互联网信息"""
        if self.search:
            return self.search.run(query)
        else:
            return "网络搜索功能不可用"

    @tool
    def extract_symptoms_tool(self, text: str) -> List[str]:
        """适合用来从文本中提取症状信息"""
        return self.extract_symptoms(text)

    @tool
    def assess_severity_tool(self, symptoms: List[str]) -> str:
        """适合用来评估症状的严重程度"""
        return self.assess_severity(symptoms)

    def query_medical_knowledge(self, query):
        """查询医疗知识库"""
        if not self.retrieval_chain:
            return "医疗知识库不可用"

        try:
            # 查询知识库
            result = self.retrieval_chain.invoke(query)

            # 提取回答和来源
            answer = result["result"]
            sources = [doc.metadata["source"] for doc in result.get("source_documents", [])]

            # 返回格式化的回答
            if sources:
                return f"{answer}\n\n信息来源: {', '.join(sources)}"
            return answer
        except Exception as e:
            return f"查询知识库时出错: {str(e)}"

    def extract_symptoms(self, text):
        """从文本中提取症状信息"""
        return self.medical_tools.extract_symptoms(text)

    def assess_severity(self, symptoms):
        """评估症状的严重程度"""
        return self.medical_tools.assess_severity(symptoms)

    def run(self, question):
        """运行智能体回答问题"""
        # 验证医疗查询是否合适
        is_valid, error_msg = self.medical_tools.validate_medical_query(question)
        if not is_valid:
            return error_msg

        try:
            # 尝试使用LangGraph智能体处理问题
            if self.agent:
                result = self.agent.invoke({
                    "messages": [HumanMessage(content=question)]
                })

                # 从结果中提取回答
                if isinstance(result, dict) and "messages" in result:
                    for message in reversed(result["messages"]):
                        if isinstance(message, AIMessage):
                            return message.content

                return str(result)
            elif self.llm:
                # 如果没有智能体，直接使用语言模型回答
                response = self.llm.invoke([HumanMessage(content=question)])
                return response.content
            else:
                return "智能体未正确初始化，无法回答问题"
        except Exception as e:
            # 捕获工具调用相关的错误
            logger.error(f"处理问题时出错: {str(e)}")
            # 尝试直接使用语言模型回答
            if self.llm:
                try:
                    response = self.llm.invoke([HumanMessage(content=question)])
                    return response.content
                except Exception:
                    pass
            return f"处理问题时出错: {str(e)}"
