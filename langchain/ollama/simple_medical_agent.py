# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import os
import glob
import time

"""
简化版医疗代理应用程序
移除不必要的复杂性，专注于核心功能
"""

class SimpleMedicalAgent:
    def __init__(self):
        # 初始化语言模型
        self.llm = ChatOllama(
            model="qwen3",
            temperature=0,
            base_url="http://localhost:11434",
            timeout=300
        )
        
        # 加载向量存储
        self.vectorstore = self._load_vectorstore()
        
        # 创建检索链
        self.retrieval_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 1}),  # 只检索1个最相关的文档
            return_source_documents=True
        )
        
        print("简化版医疗代理初始化完成")
    
    def _load_vectorstore(self):
        """加载向量存储，使用最小配置"""
        # 获取data目录
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../data")
        
        # 只加载elderly_diseases_prevention.txt文件用于测试
        files = glob.glob(os.path.join(data_dir, "elderly_diseases_prevention.txt"))
        
        if not files:
            print("警告: 未找到知识库文件。创建测试文档用于初始化。")
            # 创建一个测试文档来初始化Chroma
            from langchain_core.documents import Document
            test_docs = [Document(page_content="这是一个测试文档，用于初始化向量存储。", metadata={"source": "test_doc.txt"})]
            return Chroma.from_documents(test_docs, FakeEmbeddings(size=768))
        
        # 加载文档
        documents = []
        for file in files:
            loader = TextLoader(file, encoding="utf-8")
            documents.extend(loader.load())
        
        # 分割文档
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        # 创建向量存储
        return Chroma.from_documents(texts, FakeEmbeddings(size=768))
    
    def query(self, question):
        """简单查询方法，直接使用检索链"""
        print(f"\n查询: {question}")
        start_time = time.time()
        
        try:
            # 直接使用检索链查询
            result = self.retrieval_chain.invoke(question)
            
            end_time = time.time()
            print(f"响应时间: {end_time - start_time:.2f}秒")
            
            # 返回回答和来源
            answer = result["result"]
            sources = []
            for doc in result["source_documents"]:
                if "source" in doc.metadata:
                    sources.append(os.path.basename(doc.metadata["source"]))
            
            return answer, sources
            
        except Exception as e:
            end_time = time.time()
            print(f"查询失败，耗时: {end_time - start_time:.2f}秒，错误: {str(e)}")
            return f"查询失败: {str(e)}", []

# 测试简化版医疗代理
if __name__ == "__main__":
    print("开始测试简化版医疗代理...")
    
    # 创建代理实例
    agent = SimpleMedicalAgent()
    
    # 测试问题
    test_questions = [
        "什么是高血压？",
        "高血压的预防措施有哪些？",
        "糖尿病的症状是什么？"
    ]
    
    # 运行测试
    for q in test_questions:
        answer, sources = agent.query(q)
        print(f"回答: {answer}")
        if sources:
            print(f"来源: {', '.join(sources)}")
    
    print("\n测试完成！")