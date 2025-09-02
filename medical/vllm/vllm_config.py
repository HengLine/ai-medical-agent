"""
vLLM依赖问答系统配置文件
包含模型参数、检索配置和系统设置
"""

class VLLMConfig:
    """vLLM模型和检索链配置"""
    
    # vLLM模型配置 - 直接加载本地模型
    MODEL_CONFIG = {
        # 模型名称 - 本地模型路径或名称
        "model": "E:\\AI\\models\\vllm\\gpt2",  # 本地模型路径
        # 生成参数
        "temperature": 0.1,  # 低温度使输出更确定
        "max_tokens": 1024,  # 最大生成令牌数
        "top_p": 0.95,  # 采样参数
        # 本地模型特定参数 - CPU模式
        "vllm_kwargs": {
            "device": "cpu",  # 使用CPU模式
            # "tensor_parallel_size": 1,  #GPU 张量并行大小
            "max_model_len": 4096,  # 最大模型长度
            "trust_remote_code": True,  # 信任远程代码（本地模型可能需要）
            "dtype": "auto",  # 自动选择数据类型
            "disable_log_requests": True,  # 禁用请求日志
            "disable_log_stats": False,  # 启用统计日志
        }
    }
    
    # 检索链配置
    RETRIEVAL_CONFIG = {
        "chain_type": "stuff",  # 使用stuff链类型
        "search_kwargs": {
            "k": 3  # 检索3个最相关的文档
        },
        "return_source_documents": True  # 返回源文档信息
    }
    
    # 嵌入模型配置
    EMBEDDINGS_CONFIG = {
        "model_name": "E:\\AI\\models\\vllm\\bge-small-zh",  # 轻量级嵌入模型
        "model_kwargs": {
            "device": "cpu"  # 在CPU上运行
        },
        "encode_kwargs": {
            "normalize_embeddings": True  # 归一化嵌入向量
        }
    }
    
    # 文本分割配置
    TEXT_SPLITTER_CONFIG = {
        "chunk_size": 1000,  # 每个文档块的大小
        "chunk_overlap": 200  # 文档块之间的重叠部分
    }
    
    # 知识库配置
    KNOWLEDGE_BASE_CONFIG = {
        "data_dir": "../../data",  # 知识库数据目录
        "dependency_keywords": ["dependency", "依赖", "package", "包管理", "依赖管理"]  # 依赖相关文档关键词
    }
    
    # 相关性检查配置
    RELEVANCE_CONFIG = {
        "keywords": [
            "依赖", "dependency", "包", "package", "版本", "version", "冲突", "conflict",
            "pip", "npm", "maven", "gradle", "虚拟环境", "virtualenv", "conda",
            "requirements.txt", "package.json", "pom.xml", "依赖树", "传递依赖"
        ]
    }
    
    # 日志和性能监控配置
    LOGGING_CONFIG = {
        "enable_detailed_logs": False,  # 是否启用详细日志
        "log_file": "../../logs/vllm_dependency_qa.log",  # 日志文件路径
        "log_level": "INFO"  # 日志级别
    }
    
    # 缓存配置
    CACHE_CONFIG = {
        "enable_cache": True,  # 是否启用缓存
        "cache_dir": ".vllm_cache",  # 缓存目录
        "cache_ttl": 3600  # 缓存过期时间（秒）
    }
    
    @classmethod
    def get_model_config(cls):
        """获取模型配置"""
        return cls.MODEL_CONFIG
    
    @classmethod
    def get_retrieval_config(cls):
        """获取检索链配置"""
        return cls.RETRIEVAL_CONFIG
    
    @classmethod
    def get_embeddings_config(cls):
        """获取嵌入模型配置"""
        return cls.EMBEDDINGS_CONFIG
    
    @classmethod
    def get_text_splitter_config(cls):
        """获取文本分割配置"""
        return cls.TEXT_SPLITTER_CONFIG
    
    @classmethod
    def get_knowledge_base_config(cls):
        """获取知识库配置"""
        return cls.KNOWLEDGE_BASE_CONFIG
    
    @classmethod
    def get_relevance_config(cls):
        """获取相关性检查配置"""
        return cls.RELEVANCE_CONFIG
    
    @classmethod
    def get_logging_config(cls):
        """获取日志配置"""
        return cls.LOGGING_CONFIG
    
    @classmethod
    def get_cache_config(cls):
        """获取缓存配置"""
        return cls.CACHE_CONFIG
    
    @classmethod
    def update_config(cls, config_updates):
        """更新配置参数"""
        for config_name, updates in config_updates.items():
            if hasattr(cls, config_name) and isinstance(getattr(cls, config_name), dict):
                getattr(cls, config_name).update(updates)
                print(f"已更新{config_name}配置")
            else:
                print(f"警告：配置项{config_name}不存在或不是字典类型")

# 默认配置实例
vllm_config = VLLMConfig()

# 示例：如何使用配置
if __name__ == "__main__":
    # 获取模型配置
    model_config = VLLMConfig.get_model_config()
    print("模型配置:")
    print(f"  模型名称: {model_config['model']}")
    print(f"  温度参数: {model_config['temperature']}")
    
    # 更新配置示例
    print("\n更新配置示例:")
    VLLMConfig.update_config({
        "MODEL_CONFIG": {
            "temperature": 0.2,
            "max_tokens": 2048
        }
    })
    
    # 查看更新后的配置
    updated_config = VLLMConfig.get_model_config()
    print("\n更新后的模型配置:")
    print(f"  温度参数: {updated_config['temperature']}")
    print(f"  最大令牌数: {updated_config['max_tokens']}")