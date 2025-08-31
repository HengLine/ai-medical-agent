import os
import subprocess
import sys

"""
安装vllm[cpu]并测试配置修改后的系统

这个脚本将帮助您：
1. 安装vllm[cpu]版本
2. 测试修改后的配置是否工作正常
3. 提供故障排查信息
"""


def run_command(cmd, shell=False):
    """运行命令并返回输出"""
    try:
        print(f"\n执行命令: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        result = subprocess.run(
            cmd, 
            shell=shell,
            capture_output=True,
            text=True
        )
        
        # 打印输出和错误（如果有）
        if result.stdout:
            print(f"输出:\n{result.stdout[:500]}{'...' if len(result.stdout) > 500 else ''}")
        if result.stderr:
            print(f"错误:\n{result.stderr[:500]}{'...' if len(result.stderr) > 500 else ''}")
        
        return result
    except Exception as e:
        print(f"执行命令时出错: {str(e)}")
        return None


def install_vllm_cpu():
    """安装vllm[cpu]版本"""
    print("="*60)
    print("开始安装vllm[cpu]版本")
    print("="*60)
    
    # 卸载当前vllm（如果存在）
    print("\n1. 卸载当前vllm（如果存在）...")
    run_command([sys.executable, "-m", "pip", "uninstall", "-y", "vllm"])
    
    # 安装vllm[cpu]版本
    print("\n2. 安装vllm[cpu]版本...")
    result = run_command([sys.executable, "-m", "pip", "install", "vllm[cpu]", "--upgrade", "--no-cache-dir"])
    
    if result and result.returncode == 0:
        print("\nvllm[cpu]安装成功！")
        return True
    else:
        print("\nvllm[cpu]安装失败，请检查错误信息。")
        return False


def fix_retrieval_qa_import():
    """修复vllm_dependency_qa.py中缺少RetrievalQA导入的问题"""
    print("\n3. 检查并修复vllm_dependency_qa.py文件...")
    
    file_path = os.path.join(os.path.dirname(__file__), "vllm_dependency_qa.py")
    
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已经导入了RetrievalQA
        if "from langchain.chains import RetrievalQA" not in content:
            # 在导入部分添加RetrievalQA导入
            import_lines = content.find("from langchain_community.llms import VLLM")
            if import_lines != -1:
                # 在VLLM导入后添加RetrievalQA导入
                updated_content = content[:import_lines] + \
                                 "from langchain.chains import RetrievalQA\n" + \
                                 content[import_lines:]
                
                # 写回文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                
                print("已添加RetrievalQA导入语句。")
            else:
                print("警告：无法找到VLLM导入语句，可能需要手动修复文件。")
        else:
            print("RetrievalQA导入语句已存在，无需修改。")
        
        return True
    except Exception as e:
        print(f"修复文件时出错: {str(e)}")
        return False


def test_vllm_cpu():
    """测试vllm[cpu]配置是否工作正常"""
    print("\n4. 测试vllm[cpu]配置...")
    
    # 创建一个简单的测试脚本
    test_script = """
import sys
import os
# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.llms import VLLM
from vllm_config import vllm_config

print("测试vllm[cpu]模式...")

# 获取配置
model_config = vllm_config.get_model_config()
print(f"模型配置: {model_config}")
print(f"vllm_kwargs: {model_config['vllm_kwargs']}")

# 尝试初始化VLLM
print("\n尝试初始化VLLM模型...")
try:
    llm = VLLM(
        model=model_config['model'],
        temperature=model_config['temperature'],
        max_tokens=model_config['max_tokens'],
        top_p=model_config['top_p'],
        vllm_kwargs=model_config['vllm_kwargs']
    )
    print("VLLM模型初始化成功！")
    
    # 测试简单推理
    print("\n测试简单推理...")
    response = llm.invoke("什么是依赖管理？请简要解释。")
    print(f"推理结果: {response[:200]}{'...' if len(response) > 200 else ''}")
    print("\n测试成功！vllm[cpu]模式工作正常。")
except Exception as e:
    print(f"测试失败: {str(e)}")
    import traceback
    traceback.print_exc()
    print("\n请检查错误信息并根据提示进行故障排查。")
"""
    
    # 写入测试脚本
    test_script_path = os.path.join(os.path.dirname(__file__), "test_vllm_cpu_config.py")
    with open(test_script_path, 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    # 运行测试脚本
    print(f"\n运行测试脚本: {test_script_path}")
    result = run_command([sys.executable, test_script_path])
    
    return result and result.returncode == 0


def main():
    """主函数"""
    print("\n" + "="*60)
    print("vLLM CPU模式安装和测试工具")
    print("="*60)
    
    # 步骤1: 安装vllm[cpu]
    if not install_vllm_cpu():
        print("\n安装失败，无法继续测试。")
        print("请尝试手动安装vllm[cpu]:")
        print(f"{sys.executable} -m pip install vllm[cpu] --upgrade --no-cache-dir")
        return
    
    # 步骤2: 修复RetrievalQA导入
    if not fix_retrieval_qa_import():
        print("\n修复文件失败，建议手动检查vllm_dependency_qa.py文件。")
    
    # 步骤3: 测试vllm[cpu]配置
    test_result = test_vllm_cpu()
    
    # 总结
    print("\n" + "="*60)
    print("安装和测试总结")
    print("="*60)
    
    if test_result:
        print("✅ vllm[cpu]安装和配置成功！")
        print("\n您现在可以运行主程序了:")
        print(f"{sys.executable} {os.path.join(os.path.dirname(__file__), 'vllm_dependency_qa.py')}")
    else:
        print("❌ vllm[cpu]测试失败。")
        print("\n故障排查建议:")
        print("1. 确保您的系统满足vllm[cpu]的要求")
        print("2. 检查Python版本是否兼容（推荐3.8-3.10）")
        print("3. 尝试使用更小的模型（如'gpt2'）进行测试")
        print("4. 查看完整的错误日志获取更多信息")
        print("\n如果问题仍然存在，可以尝试修改vllm_config.py中的模型路径为'huggingface.co/gpt2'等公共模型")
    
    print("="*60)


if __name__ == "__main__":
    main()