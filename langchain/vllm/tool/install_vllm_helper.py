import sys
import os
import subprocess
import platform

"""
VLLM安装助手脚本

这个脚本可以帮助您诊断和解决vllm安装问题，特别是针对"No module named 'vllm._C'"错误。
"""


def print_system_info():
    """打印系统信息"""
    print("="*60)
    print("系统环境信息")
    print("="*60)
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python解释器: {sys.executable}")
    print(f"Python版本: {sys.version}")
    
    # 检查CUDA相关环境变量
    print("\nCUDA环境变量:")
    cuda_path = os.environ.get('CUDA_PATH', '未设置')
    print(f"CUDA_PATH: {cuda_path}")
    
    # 检查NVIDIA驱动和GPU信息（如果可用）
    try:
        nvidia_smi = subprocess.run(
            ["nvidia-smi"], 
            capture_output=True, 
            text=True
        )
        if nvidia_smi.returncode == 0:
            print("\nNVIDIA GPU信息:")
            print(nvidia_smi.stdout[:500] + "...")
        else:
            print("\n未检测到NVIDIA GPU或nvidia-smi不可用")
    except FileNotFoundError:
        print("\nnvidia-smi工具不可用")
    
    print("="*60)


def check_python_packages():
    """检查相关Python包的安装情况"""
    print("\nPython包安装检查")
    print("="*60)
    
    packages = ["vllm", "torch", "transformers", "langchain", "langchain_community"]
    
    for pkg in packages:
        try:
            # 使用pip show命令获取包信息
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", pkg],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"\n{pkg} 已安装:")
                # 提取版本和位置信息
                for line in result.stdout.splitlines():
                    if line.startswith("Version:") or line.startswith("Location:"):
                        print(f"  {line}")
            else:
                print(f"{pkg} 未安装")
        except Exception as e:
            print(f"检查{pkg}时出错: {str(e)}")
    
    print("="*60)


def suggest_solutions():
    """提供解决方案建议"""
    print("\n解决方案建议")
    print("="*60)
    
    print("\n问题分析:")
    print("您遇到的'No module named 'vllm._C''错误通常表示vllm的C/C++扩展模块没有正确安装。")
    print("这可能是由于安装过程中出现问题，或者您的系统环境不满足vllm的要求。")
    
    print("\n解决方案1: 重新安装vllm（推荐）")
    print("执行以下命令完全重新安装vllm:")
    print("\n# 卸载当前vllm")
    print(f"{sys.executable} -m pip uninstall -y vllm")
    print("\n# 安装vllm的最新稳定版本")
    print(f"{sys.executable} -m pip install vllm --upgrade --no-cache-dir")
    
    print("\n解决方案2: 安装特定版本的vllm")
    print("如果最新版本有问题，可以尝试安装已知稳定的版本:")
    print(f"{sys.executable} -m pip install vllm==0.9.0 --upgrade --no-cache-dir")
    
    print("\n解决方案3: 针对CUDA版本的特定安装")
    print("如果您知道您的CUDA版本，可以尝试安装对应版本的vllm:")
    print("# CUDA 12.x")
    print(f"{sys.executable} -m pip install vllm --extra-index-url=https://download.pytorch.org/whl/cu121")
    print("# CUDA 11.8")
    print(f"{sys.executable} -m pip install vllm --extra-index-url=https://download.pytorch.org/whl/cu118")
    
    print("\n解决方案4: 使用CPU版本的vllm（如果没有GPU）")
    print(f"{sys.executable} -m pip install vllm[cpu] --upgrade --no-cache-dir")
    
    print("\n解决方案5: 创建新的虚拟环境")
    print("如果以上方法都失败，可以创建一个全新的虚拟环境:")
    print("# 使用conda创建新环境")
    print("conda create -n new_ai_env python=3.10 -y")
    print("conda activate new_ai_env")
    print("pip install vllm")
    
    print("\n安装后验证")
    print("安装完成后，可以使用以下简单脚本验证vllm是否正常工作:")
    print("import vllm\nfrom vllm import LLM, SamplingParams\nprint('vllm版本:', vllm.__version__)\nprint('vllm已成功导入!')")
    
    print("\n更多帮助")
    print("如果问题仍然存在，请访问vllm的GitHub仓库查看最新的安装指南和常见问题解答:")
    print("https://github.com/vllm-project/vllm")
    print("="*60)


def main():
    """主函数"""
    print("\n" + "="*60)
    print("VLLM安装助手")
    print("="*60)
    print("此工具可以帮助您诊断和解决vllm安装问题，特别是'No module named 'vllm._C''错误。")
    
    # 打印系统信息
    print_system_info()
    
    # 检查Python包
    check_python_packages()
    
    # 提供解决方案建议
    suggest_solutions()
    
    print("\n请根据上述建议尝试解决vllm安装问题。")
    print("如果您在安装过程中需要更多帮助，可以参考vllm的官方文档或在GitHub上提交issue。")
    print("="*60)


if __name__ == "__main__":
    main()