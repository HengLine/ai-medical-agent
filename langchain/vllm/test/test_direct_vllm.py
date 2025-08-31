#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试vLLM直接加载本地模型功能
此脚本用于验证vLLM是否能够直接加载本地模型而不需要通过服务连接
"""

import sys
import logging
import time
import os

# 配置详细日志
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('direct_vllm_test')

# 打印Python环境信息
logger.info(f"当前Python解释器路径: {sys.executable}")
logger.info(f"当前Python版本: {sys.version}")

# 尝试导入vllm包进行验证
try:
    # 先直接导入vllm
    import vllm
    logger.info(f"直接导入vllm包成功，版本: {vllm.__version__}")
    
    # 再导入langchain_community.llms中的VLLM类
    from langchain_community.llms import VLLM
    logger.info("从langchain_community.llms导入VLLM类成功")
    
    # 导入配置
    from vllm_config import VLLMConfig
    logger.info("导入vllm_config成功")
    
except ImportError as e:
    logger.error(f"导入失败: {str(e)}")
    print(f"错误: 导入vllm相关包失败 - {str(e)}")
    print("请检查vllm是否已正确安装，当前Python环境: {sys.executable}")
    sys.exit(1)

logger.info("所有导入成功，继续执行测试")

# 日志配置已在文件顶部完成


def test_direct_vllm_model():
    """
    测试直接加载本地vLLM模型
    """
    try:
        # 获取配置
        config = VLLMConfig()
        model_config = config.MODEL_CONFIG
        
        logger.info(f"开始测试直接加载本地模型: {model_config['model']}")
        logger.debug(f"模型配置: {model_config}")
        
        start_time = time.time()
        
        # 尝试修改vllm_kwargs，移除可能导致问题的参数
        vllm_kwargs = model_config['vllm_kwargs'].copy()
        
        # 尝试使用CPU模式作为备选方案
        try:
            logger.info("尝试使用原始配置加载模型...")
            llm = VLLM(
                model=model_config['model'],
                temperature=model_config['temperature'],
                max_tokens=model_config['max_tokens'],
                top_p=model_config['top_p'],
                vllm_kwargs=vllm_kwargs
            )
        except Exception as original_error:
            logger.warning(f"原始配置加载失败: {str(original_error)}")
            logger.info("尝试使用CPU模式加载模型...")
            # 尝试使用CPU模式
            vllm_kwargs_cpu = vllm_kwargs.copy()
            vllm_kwargs_cpu['device'] = 'cpu'
            del vllm_kwargs_cpu['gpu_memory_utilization']  # CPU模式不需要GPU内存配置
            
            llm = VLLM(
                model=model_config['model'],
                temperature=model_config['temperature'],
                max_tokens=model_config['max_tokens'],
                top_p=model_config['top_p'],
                vllm_kwargs=vllm_kwargs_cpu
            )
        
        load_time = time.time() - start_time
        logger.info(f"模型加载成功，耗时: {load_time:.2f}秒")
        
        # 测试推理
        test_prompt = "什么是依赖管理？请简要解释。"
        logger.info(f"测试推理，提示词: {test_prompt}")
        
        inference_start_time = time.time()
        response = llm.invoke(test_prompt)
        inference_time = time.time() - inference_start_time
        
        logger.info(f"推理完成，耗时: {inference_time:.2f}秒")
        logger.info(f"推理结果: {response}")
        
        # 格式化输出结果
        print("\n" + "="*50)
        print("vLLM直接加载本地模型测试结果")
        print("="*50)
        print(f"模型路径: {model_config['model']}")
        print(f"模型加载时间: {load_time:.2f}秒")
        print(f"推理时间: {inference_time:.2f}秒")
        print("\n推理结果:")
        print(response)
        print("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {str(e)}", exc_info=True)
        print(f"\n测试失败: {str(e)}")
        
        # 提供更详细的故障排查建议
        print("\n故障排查建议:")
        print("1. 确保模型路径正确且模型文件完整")
        print(f"   当前模型路径: {model_config['model']}")
        print(f"   路径存在: {os.path.exists(model_config['model'])}")
        print("2. 检查vllm版本是否兼容")
        try:
            import vllm
            print(f"   当前vllm版本: {vllm.__version__}")
        except:
            print("   无法获取vllm版本")
        print("3. 检查CUDA环境是否正确配置")
        print(f"   CUDA_PATH环境变量: {os.environ.get('CUDA_PATH', '未设置')}")
        print("4. 对于大型模型，确保有足够的内存/GPU内存")
        print("5. 尝试使用简单模型(如gpt2)进行测试")
        print("\n提示: 您可以尝试直接在Python中使用以下代码测试vllm:")
        print("import vllm")
        print("from vllm import LLM, SamplingParams")
        print("sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)")
        print("llm = LLM(model='gpt2')  # 使用小模型进行测试")
        print("outputs = llm.generate(['什么是依赖管理？'], sampling_params)")
        print("print(outputs[0].outputs[0].text)")
        return False


def check_model_path():
    """
    检查模型路径是否存在
    """
    import os
    config = VLLMConfig()
    model_path = config.MODEL_CONFIG['model']
    
    if os.path.exists(model_path):
        logger.info(f"模型路径存在: {model_path}")
        # 检查是否有必要的模型文件
        required_files = ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json']
        found_files = []
        missing_files = []
        
        for root, _, files in os.walk(model_path):
            for file in files:
                if file in required_files:
                    found_files.append(file)
                    required_files.remove(file)
                    
        for file in required_files:
            missing_files.append(file)
            
        if missing_files:
            logger.warning(f"缺少必要的模型文件: {missing_files}")
            print(f"警告: 缺少必要的模型文件: {missing_files}")
        else:
            logger.info("所有必要的模型文件都已找到")
        
        return True
    else:
        logger.error(f"模型路径不存在: {model_path}")
        print(f"错误: 模型路径不存在: {model_path}")
        return False


if __name__ == "__main__":
    print("\n启动vLLM直接加载本地模型测试...")
    
    # 首先检查模型路径
    if check_model_path():
        # 然后测试模型加载和推理
        test_direct_vllm_model()
    else:
        print("请先确保模型路径正确且包含完整的模型文件。")