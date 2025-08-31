
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
print("
尝试初始化VLLM模型...")
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
    print("
测试简单推理...")
    response = llm.invoke("什么是依赖管理？请简要解释。")
    print(f"推理结果: {response[:200]}{'...' if len(response) > 200 else ''}")
    print("
测试成功！vllm[cpu]模式工作正常。")
except Exception as e:
    print(f"测试失败: {str(e)}")
    import traceback
    traceback.print_exc()
    print("
请检查错误信息并根据提示进行故障排查。")
