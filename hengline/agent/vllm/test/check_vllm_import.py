# -*- coding: utf-8 -*-
"""
用于检查vllm包是否能正常导入的测试脚本
"""

import sys

print("当前Python解释器路径:", sys.executable)
print("当前Python版本:", sys.version)
print("\n尝试导入vllm包...")

try:
    # 尝试直接导入vllm
    import vllm
    print("✓ 成功导入vllm包")
    print(f"vllm版本: {vllm.__version__}")
    
    # 尝试导入langchain_community.llms中的VLLM类
    from langchain_community.llms import VLLM
    print("✓ 成功从langchain_community.llms导入VLLM类")
    
    # 检查环境变量中的CUDA相关配置
    import os
    print("\nCUDA相关环境变量:")
    cuda_vars = [var for var in os.environ if "CUDA" in var.upper()]
    if cuda_vars:
        for var in cuda_vars:
            print(f"{var}: {os.environ.get(var)}")
    else:
        print("没有找到CUDA相关的环境变量")
    
    # 检查Python包路径
    print("\nPython包搜索路径:")
    for path in sys.path:
        print(f"- {path}")
    
    # 检查是否有安装vllm的证据
    import pkg_resources
    try:
        dist = pkg_resources.get_distribution("vllm")
        print(f"\nvllm包安装信息:")
        print(f"- 版本: {dist.version}")
        print(f"- 位置: {dist.location}")
    except pkg_resources.DistributionNotFound:
        print("vllm包未在pkg_resources中找到")
    
except ImportError as e:
    print(f"✗ 导入vllm包失败: {str(e)}")
    print("\n可能的原因:")
    print("1. vllm包未正确安装")
    print("2. Python解释器无法在包搜索路径中找到vllm")
    print("3. CUDA环境配置问题")
    print("4. Python版本与vllm不兼容")
    print("\n建议解决方案:")
    print(f"- 在当前Python环境中重新安装: {sys.executable} -m pip install vllm")
    print("- 检查CUDA环境是否正确配置")
    print("- 确认Python版本与vllm兼容")
except Exception as e:
    print(f"✗ 发生未知错误: {str(e)}")