import os
import sys
import logging
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 从基类导入
from hengline.agent.base_agent import BaseMedicalAgent

# 导入VLLM特定的库
from langchain_community.llms.vllm import VLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class VLLMMedicalAgent(BaseMedicalAgent):
    """基于VLLM本地模型的医疗智能体"""
    
    def __init__(self):
        # 设置日志
        self._setup_logging()
        
        # 初始化缓存
        self.cache = {}
        
        # 调用基类初始化
        super().__init__()
    
    def _setup_logging(self):
        """设置日志系统"""
        try:
            # 配置日志
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        except Exception as e:
            print(f"设置日志时出错: {str(e)}")
    
    def _initialize_llm(self):
        """初始化VLLM语言模型"""
        try:
            # 从配置中获取VLLM模型参数
            vllm_config = self.config_reader.get_vllm_config()
            
            if not vllm_config:
                raise ValueError("未找到VLLM配置")
            
            # 初始化VLLM模型
            llm = VLLM(
                model=vllm_config.get("model", "gpt2"),  # 默认模型
                temperature=vllm_config.get("temperature", 0.1),
                max_new_tokens=vllm_config.get("max_tokens", 1024),
                top_p=vllm_config.get("top_p", 0.95),
                vllm_kwargs=vllm_config.get("vllm_kwargs", {})
            )
            
            logging.info(f"VLLM模型初始化成功: {vllm_config.get('model', 'gpt2')}")
            return llm
        except Exception as e:
            logging.error(f"VLLM模型初始化失败: {str(e)}")
            
            # 提供一个简单的回退实现
            try:
                from langchain_community.llms import FakeListLLM
                fallback_llm = FakeListLLM(
                    responses=["抱歉，VLLM模型加载失败，无法提供回答。请检查VLLM服务配置。"]
                )
                return fallback_llm
            except Exception:
                # 如果回退也失败，返回None
                print(f"初始化回退模型也失败: {str(e)}")
                return None
    
    def load_medical_knowledge(self):
        """加载医疗知识库数据，VLLM版本使用FAISS进行优化"""
        try:
            # 从配置中获取知识库参数
            kb_config = self.config_reader.get_module_config("knowledge_base")
            data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../', kb_config.get("data_dir", "data"))
            
            # 获取VLLM嵌入配置
            embeddings_config = self.config_reader.get_embeddings_config("vllm")
            
            # 初始化嵌入模型
            try:
                # 尝试使用HuggingFaceEmbeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name=embeddings_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                    model_kwargs=embeddings_config.get("model_kwargs", {"device": "cpu"}),
                    encode_kwargs=embeddings_config.get("encode_kwargs", {"normalize_embeddings": True})
                )
            except Exception as e:
                logging.warning(f"初始化HuggingFaceEmbeddings失败，将使用FakeEmbeddings: {str(e)}")
                # 如果失败，回退到FakeEmbeddings
                from langchain_community.embeddings import FakeEmbeddings
                embeddings = FakeEmbeddings(size=768)
            
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
                logging.warning("未找到医疗知识库文件。将创建一个空的向量存储。")
                from langchain_core.documents import Document
                # 创建空文档列表并使用from_documents方法初始化FAISS
                empty_docs = [Document(page_content="这是一个空的医疗知识库文档", metadata={"source": "empty"})]
                return FAISS.from_documents(empty_docs, embeddings)
            
            # 加载文档
            documents = []
            for file in files:
                try:
                    from langchain_community.document_loaders import TextLoader
                    loader = TextLoader(file, encoding="utf-8")
                    documents.extend(loader.load())
                except Exception as e:
                    logging.error(f"加载文件 {file} 时出错: {str(e)}")
            
            if not documents:
                logging.warning("未能加载任何文档。将创建一个空的向量存储。")
                from langchain_core.documents import Document
                empty_docs = [Document(page_content="这是一个空的医疗知识库文档", metadata={"source": "empty"})]
                return FAISS.from_documents(empty_docs, embeddings)
            
            # 从配置中获取文本分割参数
            text_splitter_config = self.config_reader.get_module_config("text_splitter")
            
            # 分割文档
            from langchain.text_splitter import CharacterTextSplitter
            text_splitter = CharacterTextSplitter(
                chunk_size=text_splitter_config.get("chunk_size", 1000),
                chunk_overlap=text_splitter_config.get("chunk_overlap", 200)
            )
            texts = text_splitter.split_documents(documents)
            
            # 创建向量存储 - 使用FAISS进行优化
            vectorstore = FAISS.from_documents(texts, embeddings)
            logging.info(f"成功创建FAISS向量存储，包含{len(texts)}个文档块")
            return vectorstore
        except Exception as e:
            logging.error(f"加载医疗知识库时出错: {str(e)}")
            # 回退到基类的实现
            return super().load_medical_knowledge()
    
    def _create_retrieval_chain(self):
        """创建检索链，针对VLLM进行优化"""
        try:
            # 检查是否已经初始化了llm和vectorstore
            if not self.llm or not self.vectorstore:
                logging.warning("LLM或VectorStore未初始化，无法创建检索链")
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
            
            logging.info("VLLM检索链创建成功")
            return retrieval_chain
        except Exception as e:
            logging.error(f"创建检索链时出错: {str(e)}")
            # 回退到基类的实现
            return super()._create_retrieval_chain()
    
    def run(self, question):
        """运行智能体回答问题，针对VLLM模型进行优化"""
        # 检查缓存
        cache_key = question.strip().lower()
        if cache_key in self.cache:
            logging.debug(f"从缓存中获取问题答案: {cache_key}")
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