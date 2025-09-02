#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Conda环境下的vLLM安装和测试工具
专为解决conda虚拟环境中的vllm._C模块缺失问题而设计
支持CPU模式和特定版本安装
"""

import os
import sys
import subprocess
import logging
from typing import Optional, Dict, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('conda_vllm_installer')

class CondaVLLMInstaller:
    def __init__(self):
        """初始化Conda VLLM安装器"""
        self.python_exe = sys.executable
        self.conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
        self.is_conda_env = 'CONDA_PREFIX' in os.environ
        logger.info(f"当前Python解释器: {self.python_exe}")
        logger.info(f"当前conda环境: {self.conda_env}")
        logger.info(f"是否在conda环境中: {self.is_conda_env}")
        
        # 检查系统架构
        self.system_info = self._get_system_info()
        logger.info(f"系统信息: {self.system_info}")
    
    def _get_system_info(self) -> Dict:
        """获取系统信息"""
        import platform
        return {
            'os': platform.system(),
            'version': platform.version(),
            'architecture': platform.architecture(),
            'python_version': platform.python_version(),
            'processor': platform.processor()
        }
    
    def run_command(self, cmd: List[str], cwd: Optional[str] = None) -> bool:
        """运行命令并返回执行状态"""
        try:
            logger.info(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=cwd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info(f"命令执行成功: {result.stdout[:100]}...")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"命令执行失败: {e.stderr}")
            print(f"错误: {e.stderr}")
            return False
    
    def install_dependencies(self) -> bool:
        """安装必要的依赖项"""
        logger.info("开始安装必要的依赖项...")
        
        # 安装依赖包
        success = self.run_command([
            self.python_exe, '-m', 'pip', 'install', 
            'numpy', 'torch', 'transformers', 'sentencepiece', 'accelerate'
        ])
        
        return success
    
    def uninstall_vllm(self) -> bool:
        """卸载现有的vllm"""
        logger.info("开始卸载现有的vllm...")
        
        # 先尝试卸载
        self.run_command([self.python_exe, '-m', 'pip', 'uninstall', '-y', 'vllm'])
        
        # 清理可能的残留文件
        try:
            import site
            site_packages = site.getsitepackages()
            for sp in site_packages:
                vllm_dir = os.path.join(sp, 'vllm')
                if os.path.exists(vllm_dir):
                    logger.info(f"发现残留的vllm目录: {vllm_dir}")
                    # 在实际运行中，用户可能需要手动删除这些目录
                    print(f"警告: 发现残留的vllm目录 {vllm_dir}，建议手动删除")
        except Exception as e:
            logger.warning(f"清理残留文件时出错: {e}")
        
        return True
    
    def install_vllm(self, version: str = '0.9.0', cpu_only: bool = True) -> bool:
        """安装指定版本的vllm"""
        logger.info(f"开始安装vllm版本 {version} {'(CPU模式)' if cpu_only else ''}...")
        
        # 根据CPU模式选择安装命令
        install_cmd = [self.python_exe, '-m', 'pip', 'install']
        
        if cpu_only:
            # CPU模式下的特定安装参数
            install_cmd.extend([f'vllm=={version}'])  # 注意：vllm没有[cpu]扩展，直接安装即可
        else:
            # GPU模式
            install_cmd.extend([f'vllm=={version}'])  # 可以添加 --no-cache-dir 来避免缓存问题
        
        return self.run_command(install_cmd)
    
    def test_vllm_installation(self) -> bool:
        """测试vllm安装是否成功"""
        logger.info("开始测试vllm安装...")
        
        # 创建临时测试脚本
        test_script = """import sys
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('vllm_test')

try:
    # 尝试导入vllm的核心模块
    import vllm
    logger.info(f"成功导入vllm，版本: {vllm.__version__}")
    
    # 尝试初始化一个简单的模型
    from vllm import LLM, SamplingParams
    
    # 使用轻量级模型进行测试
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
llm = LLM(
    model="gpt2",  # 使用HuggingFace的gpt2模型
    device="cpu",
    max_model_len=2048,
    trust_remote_code=True,
    dtype="float32",  # CPU模式下使用float32
    disable_log_requests=True
)
logger.info("成功初始化vllm模型")
    
    # 生成文本
    prompts = ["什么是依赖管理？请简要解释。"]
    outputs = llm.generate(prompts, sampling_params)
    
    # 打印结果
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"提示: {prompt}")
        print(f"生成: {generated_text}")
        
    logger.info("vllm测试成功!")
    sys.exit(0)
except ImportError as e:
    logger.error(f"导入vllm失败: {e}")
    print(f"错误: 导入vllm失败 - {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"vllm测试过程中出错: {e}")
    print(f"错误: vllm测试失败 - {e}")
    sys.exit(1)
"""
        
        test_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_conda_vllm.py')
        
        with open(test_script_path, 'w', encoding='utf-8') as f:
            f.write(test_script)
        
        logger.info(f"创建测试脚本: {test_script_path}")
        
        # 运行测试脚本
        result = subprocess.run(
            [self.python_exe, test_script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 打印测试结果
        print("\n===== vLLM测试结果 =====")
        print(result.stdout)
        if result.stderr:
            print("\n===== 错误输出 =====")
            print(result.stderr)
        
        # 检查测试是否成功
        success = result.returncode == 0
        if success:
            logger.info("vllm安装测试成功!")
        else:
            logger.error(f"vllm安装测试失败，退出码: {result.returncode}")
        
        return success
    
    def update_project_config(self) -> bool:
        """更新项目配置以使用conda环境中的vllm"""
        logger.info("更新项目配置...")
        
        # 这里可以根据需要更新项目配置文件
        # 例如，更新requirements.txt或其他配置文件
        
        # 检查并更新requirements.txt
        requirements_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'requirements.txt')
        if os.path.exists(requirements_path):
            with open(requirements_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 如果requirements.txt中没有vllm，则添加
            if 'vllm' not in content:
                with open(requirements_path, 'a', encoding='utf-8') as f:
                    f.write('vllm==0.9.0\n')
                logger.info(f"已添加vllm==0.9.0到{requirements_path}")
            else:
                logger.info(f"{requirements_path}中已包含vllm依赖")
        
        return True
    
    def run_full_installation(self) -> bool:
        """运行完整的安装流程"""
        logger.info("开始完整的vllm安装流程...")
        
        # 步骤1: 安装必要的依赖项
        if not self.install_dependencies():
            logger.error("安装依赖项失败，终止安装流程")
            return False
        
        # 步骤2: 卸载现有的vllm
        self.uninstall_vllm()  # 即使失败也继续，因为可能之前没有安装
        
        # 步骤3: 安装指定版本的vllm
        if not self.install_vllm(version='0.9.0', cpu_only=True):
            logger.error("安装vllm失败，终止安装流程")
            return False
        
        # 步骤4: 测试vllm安装
        if not self.test_vllm_installation():
            logger.error("vllm安装测试失败")
            # 提供故障排查建议
            self._provide_troubleshooting()
            return False
        
        # 步骤5: 更新项目配置
        self.update_project_config()
        
        logger.info("vllm安装和配置流程已完成!")
        return True
    
    def _provide_troubleshooting(self):
        """提供故障排查建议"""
        print("\n===== 故障排查建议 =====")
        print("1. 确保您的conda环境已激活: conda activate ai-agent")
        print("2. 检查Python版本是否兼容(推荐3.8-3.10)")
        print("3. 尝试升级pip: python -m pip install --upgrade pip")
        print("4. 清理pip缓存: python -m pip cache purge")
        print("5. 检查系统是否安装了C++编译工具")
        print("   - Windows: 安装Visual Studio Build Tools")
        print("   - Linux: 安装gcc和g++")
        print("6. 尝试使用特定版本的vllm: python -m pip install vllm==0.2.0")
        print("7. 查看完整的错误日志以获取更详细的信息")
        print("8. 如果使用CPU模式，确保有足够的内存(至少8GB)")

if __name__ == "__main__":
    print("===== Conda环境vLLM安装工具 =====")
    print("此工具将帮助您在conda虚拟环境中安装和配置vLLM")
    print(f"当前环境: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")
    print(f"Python解释器: {sys.executable}")
    print()
    
    installer = CondaVLLMInstaller()
    
    # 检查是否在conda环境中
    if not installer.is_conda_env:
        print("警告: 您似乎不在conda环境中运行此脚本")
        print("请先激活您的conda环境: conda activate ai-agent")
        print("是否继续在当前环境中安装vLLM? (y/n)")
        choice = input().strip().lower()
        if choice != 'y':
            print("安装已取消")
            sys.exit(0)
    
    # 运行完整的安装流程
    success = installer.run_full_installation()
    
    if success:
        print("\n恭喜! vLLM已成功安装和配置。")
        print("您可以通过运行项目中的vllm_dependency_qa.py来使用它。")
    else:
        print("\n安装过程中遇到问题，请查看上面的错误信息和故障排查建议。")