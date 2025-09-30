"""
@FileName: config.json.py
@Description:
@Author: HengLine
@Time: 2025/9/29 15:23
"""
# -*- coding: utf-8 -*-
"""
配置读取工具类 - 支持从优化后的配置结构中读取配置
"""

import json
import os
from typing import Dict, Any, List

# 加载dotenv以支持.env文件
from dotenv import load_dotenv


class ConfigReader:
    """
    从 config.json 中读取配置的工具类，支持配置优先级：JSON配置 > 环境变量
    """

    def __init__(self, config_path: str = None):
        """
        初始化配置读取器
        :param config_path: 配置文件路径，默认为项目根目录下的 config/config.json
        """
        # 加载.env文件
        load_dotenv()
        
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            "config.json"
        )
        self._config = self._load_config()
    
    def get_config_with_env_fallback(self, module_name: str, key: str, default: Any = None, env_var_name: str = None) -> Any:
        """
        先从JSON配置中获取值，如果为空则从环境变量中获取
        
        :param module_name: 模块名称
        :param key: 配置项键
        :param default: 默认值
        :param env_var_name: 环境变量名称，如果未指定则自动生成 (MODULE_KEY)
        :return: 配置值
        """
        # 1. 首先从JSON配置中获取值
        config_value = self.get_value(module_name, key, default)
        
        # 2. 如果JSON配置中有值且不为空，直接返回
        if config_value not in (None, ""):
            return config_value
        
        # 3. 如果JSON配置中没有值或为空，尝试从环境变量中获取
        # 生成环境变量名（默认为模块名_键名，大写）
        if env_var_name is None:
            env_var_name = f"{module_name.upper()}_{key.upper()}"
        
        # 从环境变量中获取值
        env_value = os.environ.get(env_var_name, default)
        
        return env_value

    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        :return: 配置字典
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件 {self.config_path} 未找到")
        except json.JSONDecodeError:
            raise ValueError(f"配置文件 {self.config_path} 格式错误")

    def get_all_config(self) -> Dict[str, Any]:
        """
        获取全部配置
        :return: 完整的配置字典
        """
        return self._config

    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """
        按模块获取配置
        :param module_name: 模块名称（如 'retrieval', 'text_splitter'）
        :return: 模块配置字典
        """
        return self._config.get(module_name, {})

    def get_value(self, module_name: str, key: str, default: Any = None) -> Any:
        """
        获取指定模块的配置项值
        :param module_name: 模块名称
        :param key: 配置项键
        :param default: 默认值（如果未找到）
        :return: 配置项值
        """
        module_config = self.get_module_config(module_name)
        return module_config.get(key, default)

    def get_example_questions(self) -> List[str]:
        """
        获取示例问题列表
        :return: 示例问题列表
        """
        return self._config.get("example_questions", [])

    def get_llm_config(self, llm_type: str) -> Dict[str, Any]:
        """
        获取特定类型LLM的配置
        :param llm_type: LLM类型（'ollama', 'vllm', 'qwen', 'openai'）
        :return: LLM配置字典
        """
        llm_configs = self._config.get("llm", {})
        return llm_configs.get(llm_type, {})

    def get_ollama_config(self) -> Dict[str, Any]:
        """
        获取Ollama配置
        :return: Ollama配置字典
        """
        return self.get_llm_config("ollama")

    def get_vllm_config(self) -> Dict[str, Any]:
        """
        获取VLLM配置
        :return: VLLM配置字典
        """
        return self.get_llm_config("vllm")


    def get_retrieval_config(self) -> Dict[str, Any]:
        """
        获取检索配置
        :return: 检索配置字典
        """
        return self.get_module_config("retrieval")

    def get_text_splitter_config(self) -> Dict[str, Any]:
        """
        获取文本分割器配置
        :return: 文本分割器配置字典
        """
        return self.get_module_config("text_splitter")

    def get_knowledge_base_config(self) -> Dict[str, Any]:
        """
        获取知识库配置
        :return: 知识库配置字典
        """
        return self.get_module_config("knowledge_base")

    def get_llm_value(self, llm_type: str, key: str, default: Any = None) -> Any:
        """
        获取特定类型LLM的配置项值
        :param llm_type: LLM类型
        :param key: 配置项键
        :param default: 默认值
        :return: 配置项值
        """
        # 检查是否有嵌套的配置路径（如 'qwen.model'）
        if '.' in key:
            parts = key.split('.')
            parent_key = parts[0]
            child_key = '.'.join(parts[1:])
            
            # 先尝试从llm.llm_type.parent_key中获取
            llm_config = self.get_llm_config(llm_type)
            if parent_key in llm_config and isinstance(llm_config[parent_key], dict):
                return llm_config[parent_key].get(child_key, default)
            
            # 如果没找到，再尝试直接从llm.parent_key中获取（兼容之前的配置结构）
            parent_config = self.get_llm_config(parent_key)
            return parent_config.get(child_key, default)
        
        # 如果是简单键，直接从llm.llm_type中获取
        llm_config = self.get_llm_config(llm_type)
        return llm_config.get(key, default)

    def get_embeddings_config(self, llm_type: str) -> Dict[str, Any]:
        """
        获取特定类型LLM的嵌入模型配置
        :param llm_type: LLM类型
        :return: 嵌入模型配置字典
        """
        llm_config = self.get_llm_config(llm_type)
        return llm_config.get("embeddings", {})

    def get_generative_config(self) -> Dict[str, Any]:
        """
        获取生成式智能体配置
        :return: 生成式智能体配置字典
        """
        # 默认使用Ollama配置，但允许在llm中单独配置generative
        generative_config = self.get_llm_config("generative")
        if not generative_config:
            # 如果没有单独配置generative，则使用Ollama配置的副本
            ollama_config = self.get_ollama_config()
            # 调整温度参数以适合生成式任务
            generative_config = ollama_config.copy()
            generative_config["temperature"] = generative_config.get("temperature", 0.7)
            generative_config["max_tokens"] = generative_config.get("max_tokens", 2048)
        return generative_config
    
    def get_openai_api_key(self, default: str = "") -> str:
        """
        获取OpenAI API密钥，优先从JSON配置中获取，为空则从环境变量获取
        
        :param default: 默认值
        :return: OpenAI API密钥
        """
        # 1. 先从llm.openai.api_key配置中获取
        llm_config = self.get_module_config("llm")
        openai_config = llm_config.get("openai", {})
        api_key = openai_config.get("api_key", "")
        
        # 2. 如果JSON配置中有值且不为空，直接返回
        if api_key not in (None, ""):
            return api_key
        
        # 3. 如果JSON配置中没有值或为空，从环境变量OPENAI_API_KEY获取
        return os.environ.get("OPENAI_API_KEY", default)
    
    def get_qwen_api_key(self, default: str = "") -> str:
        """
        获取通义千问API密钥，优先从JSON配置中获取，为空则从环境变量获取
        
        :param default: 默认值
        :return: 通义千问API密钥
        """
        # 1. 先从llm.qwen.api_key配置中获取
        llm_config = self.get_module_config("llm")
        qwen_config = llm_config.get("qwen", {})
        api_key = qwen_config.get("api_key", "")
        
        # 2. 如果JSON配置中有值且不为空，直接返回
        if api_key not in (None, ""):
            return api_key
        
        # 3. 如果JSON配置中没有值或为空，从环境变量DASHSCOPE_API_KEY获取
        return os.environ.get("DASHSCOPE_API_KEY", default)
    
    def get_api_key_for_llm(self, llm_type: str, default: str = "") -> str:
        """
        根据LLM类型获取对应的API密钥
        
        :param llm_type: LLM类型（'openai', 'qwen'等）
        :param default: 默认值
        :return: API密钥
        """
        if llm_type == "openai":
            return self.get_openai_api_key(default)
        elif llm_type == "qwen":
            return self.get_qwen_api_key(default)
        else:
            # 其他LLM类型可能不需要API密钥
            return default
