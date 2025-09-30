import os
import subprocess
import sys

"""
vLLM重新安装工具

这个脚本将帮助您完全卸载并重新安装vllm，解决'No module named 'vllm._C''错误
"""


def run_command(cmd, shell=False):
    """运行命令并显示输出"""
    print(f"\n执行命令: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(
        cmd, 
        shell=shell,
        capture_output=True,
        text=True
    )
    
    # 打印输出
    if result.stdout:
        print(f"输出:\n{result.stdout[:500]}{'...' if len(result.stdout) > 500 else ''}")
    if result.stderr:
        print(f"错误:\n{result.stderr[:500]}{'...' if len(result.stderr) > 500 else ''}")
    
    return result


def main():
    print("="*60)
    print("vLLM重新安装工具")
    print("="*60)
    print("此工具将帮助您解决'No module named 'vllm._C''错误")
    print("="*60)
    
    # 步骤1: 完全卸载当前的vllm
    print("\n1. 完全卸载当前的vllm...")
    run_command([sys.executable, "-m", "pip", "uninstall", "-y", "vllm"])
    
    # 步骤2: 清理pip缓存
    print("\n2. 清理pip缓存...")
    run_command([sys.executable, "-m", "pip", "cache", "purge"])
    
    # 步骤3: 重新安装vllm（使用--no-cache-dir选项）
    print("\n3. 重新安装vllm（使用--no-cache-dir选项）...")
    print("注意：这可能需要一些时间，特别是如果需要编译C扩展模块")
    
    # 尝试安装特定版本的vllm（已知更稳定的版本）
    result = run_command([sys.executable, "-m", "pip", "install", "vllm==0.9.0", "--upgrade", "--no-cache-dir"])
    
    if result and result.returncode == 0:
        print("\nvllm重新安装成功！")
        
        # 步骤4: 验证安装
        print("\n4. 验证安装...")
        verify_result = run_command([sys.executable, "-c", 
            "import vllm; print('vllm版本:', vllm.__version__); "
            "try: import vllm._C; print('vllm._C模块已成功导入！'); "
            "except ImportError: print('警告: vllm._C模块导入失败');"
        ])
        
        print("\n" + "="*60)
        print("安装总结")
        print("="*60)
        
        if verify_result and verify_result.returncode == 0:
            print("✅ vllm重新安装成功！")
            print("\n您现在可以尝试运行vllm相关的程序了。")
        else:
            print("⚠️ vllm重新安装完成，但验证时出现问题。")
            print("\n建议尝试以下解决方案：")
            print("1. 尝试使用不同版本的vllm")
            print("2. 确保您的Python版本兼容（推荐3.8-3.10）")
            print("3. 考虑创建一个新的虚拟环境")
            
    else:
        print("\n❌ vllm重新安装失败。")
        
        print("\n" + "="*60)
        print("安装失败，建议尝试以下方法：")
        print("="*60)
        
        # 提供替代安装方法
        print("\n方法1: 尝试安装不同版本的vllm")
        print(f"{sys.executable} -m pip install vllm==0.8.0 --no-cache-dir")
        
        print("\n方法2: 使用预编译的whl文件")
        print("访问 https://github.com/vllm-project/vllm/releases 下载适合您系统的whl文件")
        print("然后使用 pip install <下载的文件名>.whl 进行安装")
        
        print("\n方法3: 创建新的虚拟环境")
        print("# 使用conda创建新环境")
        print("conda create -n vllm_env python=3.10 -y")
        print("conda activate vllm_env")
        print("pip install vllm")
        
        print("\n方法4: 使用Docker容器")
        print("docker pull vllm/vllm-openai")
        print("docker run --gpus all -p 8000:8000 vllm/vllm-openai")
        
        print("\n方法5: 如果您不需要GPU加速，可以考虑使用其他轻量级模型")
        print("例如: pip install transformers torch")
        
    print("="*60)


if __name__ == "__main__":
    main()