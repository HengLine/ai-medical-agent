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


class ConfigReader:
    """
    从 config.json 中读取配置的工具类
    """

    def __init__(self, config_path: str = None):
        """
        初始化配置读取器
        :param config_path: 配置文件路径，默认为项目根目录下的 config/config.json
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            "config.json"
        )
        self._config = self._load_config()

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
        :param llm_type: LLM类型（'ollama', 'vllm', 'api'）
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

    def get_api_config(self) -> Dict[str, Any]:
        """
        获取远程API配置（原remote_api）
        :return: 远程API配置字典
        """
        return self.get_llm_config("api")

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
