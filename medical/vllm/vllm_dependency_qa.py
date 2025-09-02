# from langchain_community.chains import RetrievalQA
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import VLLM
from langchain.text_splitter import CharacterTextSplitter
import os
import time
import logging

# 导入配置文件
from vllm_config import vllm_config

"""
基于LangChain + vLLM的依赖问答系统
用于回答关于软件依赖关系、版本冲突和依赖管理的问题
"""

class VLLMDependencyQA:
    def __init__(self):
        # 初始化日志
        self._setup_logging()
        
        # 初始化缓存
        self.cache = {}
        
        # 初始化vLLM模型
        self.llm = self._initialize_vllm()
        
        # 加载依赖管理知识库
        self.vectorstore = self._load_knowledge_base()
        
        # 创建检索链
        self.retrieval_chain = self._create_retrieval_chain()
        
        logging.info("vLLM依赖问答系统初始化完成!")
    
    def _setup_logging(self):
        """设置日志系统"""
        log_config = vllm_config.get_logging_config()
        
        # 配置日志级别
        logging.basicConfig(
            level=getattr(logging, log_config['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 如果配置了日志文件，添加文件处理器
        if log_config.get('log_file'):
            file_handler = logging.FileHandler(log_config['log_file'])
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logging.getLogger().addHandler(file_handler)
    
    def _initialize_vllm(self):
        """初始化vLLM模型"""
        try:
            # 从配置文件获取vLLM模型参数
            model_config = vllm_config.get_model_config()
            
            llm = VLLM(
                model=model_config['model'],
                temperature=model_config['temperature'],
                max_tokens=model_config['max_tokens'],
                top_p=model_config['top_p'],
                vllm_kwargs=model_config['vllm_kwargs']
            )
            logging.info(f"vLLM模型初始化成功: {model_config['model']}")
            return llm
        except Exception as e:
            logging.error(f"vLLM模型初始化失败: {str(e)}")
            # 提供一个简单的回退实现
            from langchain_community.llms import FakeListLLM
            return FakeListLLM(
                responses=["抱歉，vLLM模型加载失败，无法提供回答。请检查vLLM服务配置。"]
            )
    
    def _load_knowledge_base(self):
        """加载依赖管理知识库"""
        try:
            # 从配置文件获取知识库配置
            kb_config = vllm_config.get_knowledge_base_config()
            text_split_config = vllm_config.get_text_splitter_config()
            embeddings_config = vllm_config.get_embeddings_config()
            
            # 获取知识库目录
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), kb_config['data_dir'])
            
            # 初始化文档列表
            documents = []
            
            try:
                # 查找相关的依赖管理文档
                dependency_files = []
                data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
                
                if os.path.exists(data_dir):
                    for root, _, files in os.walk(data_dir):
                        for file in files:
                            if any(keyword in file.lower() for keyword in kb_config['dependency_keywords']):
                                dependency_files.append(os.path.join(root, file))
                    
                    # 加载找到的文档
                    for file in dependency_files:
                        try:
                            loader = TextLoader(file, encoding="utf-8")
                            documents.extend(loader.load())
                        except Exception as file_error:
                            logging.warning(f"无法加载文件 {file}: {str(file_error)}")
                else:
                    logging.warning(f"知识库目录不存在: {data_dir}")
                
                # 如果没有找到文档或加载失败，使用内嵌文档
                if not documents:
                    logging.warning("未找到或无法加载专门的依赖管理文档，使用内嵌知识库")
                    # 直接使用内嵌的依赖管理知识文本
                    dependency_text = """软件依赖管理知识：
1. 依赖管理工具：pip（Python）、npm（JavaScript）、Maven（Java）、Gradle（Java/Kotlin）等。
2. 依赖冲突：当两个或多个包依赖于同一包的不同版本时发生。
3. 语义化版本控制：遵循X.Y.Z格式，其中X是主版本，Y是次版本，Z是补丁版本。
4. 虚拟环境：用于隔离不同项目的依赖，避免版本冲突。
5. 依赖锁定：锁定依赖的精确版本，确保构建的可重复性。
6. 传递性依赖：项目直接依赖的包所依赖的其他包。"""
                    
                    # 创建Document对象
                    from langchain_core.documents import Document
                    documents = [Document(
                        page_content=dependency_text,
                        metadata={"source": "内嵌依赖管理知识"}
                    )]
            except Exception as e:
                logging.error(f"加载文档时出错: {str(e)}")
                # 使用内嵌文档作为最后的回退
                from langchain_core.documents import Document
                dependency_text = "基础软件依赖管理知识"
                documents = [Document(
                    page_content=dependency_text,
                    metadata={"source": "内嵌基础知识"}
                )]
            
            # 分割文档
            text_splitter = CharacterTextSplitter(
                chunk_size=text_split_config['chunk_size'],
                chunk_overlap=text_split_config['chunk_overlap']
            )
            texts = text_splitter.split_documents(documents)
            
            # 创建向量存储
            embeddings = HuggingFaceEmbeddings(
                model_name=embeddings_config['model_name'],
                model_kwargs=embeddings_config['model_kwargs'],
                encode_kwargs=embeddings_config.get('encode_kwargs', {})
            )
            vectorstore = FAISS.from_documents(texts, embeddings)
            
            logging.info(f"成功加载知识库，共{len(texts)}个文档块")
            return vectorstore
        except Exception as e:
            logging.error(f"加载知识库失败: {str(e)}")
            # 提供一个简单的回退实现
            from langchain_community.embeddings import FakeEmbeddings
            return FAISS.from_texts(["依赖管理知识"], FakeEmbeddings(size=384))
    
    def _create_retrieval_chain(self):
        """创建检索链"""
        try:
            # 从配置文件获取检索链配置
            retrieval_config = vllm_config.get_retrieval_config()
            
            retrieval_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type=retrieval_config['chain_type'],
                retriever=self.vectorstore.as_retriever(
                    search_kwargs=retrieval_config['search_kwargs']
                ),
                return_source_documents=retrieval_config['return_source_documents']
            )
            logging.info("检索链创建成功")
            return retrieval_chain
        except Exception as e:
            logging.error(f"创建检索链失败: {str(e)}")
            # 返回一个简单的回退函数
            def fallback_chain(query):
                return {"result": "抱歉，无法处理您的查询。系统组件初始化失败。", "source_documents": []}
            return fallback_chain
    
    def ask_question(self, question):
        """回答用户的依赖管理问题"""
        try:
            logging.info(f"用户问题: {question}")
            start_time = time.time()
            
            # 检查缓存
            cache_config = vllm_config.get_cache_config()
            if cache_config['enable_cache'] and question in self.cache:
                cached_response, timestamp = self.cache[question]
                # 检查缓存是否过期
                if time.time() - timestamp < cache_config['cache_ttl']:
                    logging.info(f"从缓存返回回答: {question}")
                    return cached_response
                
            # 检查是否是依赖管理相关问题
            is_relevant = self._check_relevance(question)
            if not is_relevant:
                response = "抱歉，我专注于回答软件依赖管理相关的问题。请提出相关问题。"
                return response
            
            # 执行查询
            result = self.retrieval_chain.invoke(question)
            
            end_time = time.time()
            logging.info(f"处理时间: {end_time - start_time:.2f}秒")
            
            # 格式化回答
            answer = result["result"]
            
            # 提取来源信息
            sources = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    if "source" in doc.metadata:
                        sources.append(os.path.basename(doc.metadata["source"]))
            
            if sources:
                formatted_sources = "\n\n来源: " + ", ".join(set(sources))
                response = f"{answer}{formatted_sources}"
            else:
                response = answer
            
            # 更新缓存
            if cache_config['enable_cache']:
                self.cache[question] = (response, time.time())
                # 限制缓存大小
                if len(self.cache) > 100:
                    # 移除最早的缓存项
                    self.cache.pop(next(iter(self.cache)))
            
            return response
        except Exception as e:
            logging.error(f"回答问题时出错: {str(e)}")
            return f"抱歉，处理您的问题时发生错误: {str(e)}\n错误详情: {str(e)}"
    
    def _check_relevance(self, question):
        """检查问题是否与依赖管理相关"""
        # 从配置文件获取相关性检查关键词
        relevance_config = vllm_config.get_relevance_config()
        keywords = relevance_config['keywords']
        
        question_lower = question.lower()
        is_relevant = any(keyword in question_lower for keyword in keywords)
        logging.debug(f"问题相关性检查: '{question}' -> {is_relevant}")
        return is_relevant
    
    def run_demo(self):
        """运行演示问答"""
        print("\n===== vLLM依赖问答系统演示 =====")
        print("输入'退出'结束对话")
        
        demo_questions = [
            "什么是依赖冲突？如何解决？",
            "pip和conda有什么区别？",
            "如何创建和使用Python虚拟环境？",
            "什么是语义化版本控制？",
            "如何解决npm依赖冲突？"
        ]
        
        print("\n示例问题:")
        for i, q in enumerate(demo_questions, 1):
            print(f"{i}. {q}")
        
        while True:
            user_input = input("\n请输入您的问题: ")
            if user_input.lower() in ["退出", "quit", "exit"]:
                print("感谢使用vLLM依赖问答系统，再见！")
                break
            
            response = self.ask_question(user_input)
            print(f"\n回答: {response}")

if __name__ == "__main__":
    # 创建依赖问答系统实例
    dependency_qa = VLLMDependencyQA()
    
    # 运行演示
    dependency_qa.run_demo()