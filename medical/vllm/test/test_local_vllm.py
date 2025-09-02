import time
from langchain_community.llms import VLLM


def test_local_vllm_connection():
    """测试连接本地vLLM服务"""
    print("开始测试本地vLLM连接...")
    
    try:
        # 配置本地vLLM模型
        llm = VLLM(
            model="E:\\AI\\models\\vllm\\gpt2",  # 本地模型名称
            temperature=0.1,
            max_tokens=1024,
            top_p=0.95,
            vllm_kwargs={
                "base_url": "http://localhost:8000/v1",  # 本地vLLM服务地址
                "gpu_memory_utilization": 0.8,
                "max_model_len": 4096,
                "tensor_parallel_size": 1,
                "trust_remote_code": True,
                "dtype": "auto"
            }
        )
        
        print("成功初始化vLLM模型连接")
        
        # 测试简单问题
        test_question = "什么是依赖管理？"
        print(f"发送测试问题: {test_question}")
        
        start_time = time.time()
        response = llm.invoke(test_question)
        end_time = time.time()
        
        print(f"响应时间: {end_time - start_time:.2f}秒")
        print(f"回答:")
        print(response)
        
        return True
    except Exception as e:
        print(f"连接本地vLLM服务失败: {str(e)}")
        return False


def check_vllm_service_status():
    """检查vLLM服务状态"""
    import requests
    
    try:
        response = requests.get("http://localhost:8000/v1/models")
        if response.status_code == 200:
            models = response.json()
            print("本地vLLM服务正在运行!")
            print(f"可用模型: {[model['id'] for model in models.get('data', [])]}")
            return True
        else:
            print(f"vLLM服务返回非成功状态码: {response.status_code}")
            return False
    except requests.ConnectionError:
        print("无法连接到vLLM服务，请确保服务已启动并且端口正确")
        return False
    except Exception as e:
        print(f"检查vLLM服务状态时出错: {str(e)}")
        return False


if __name__ == "__main__":
    print("===== 本地vLLM服务测试 ======")
    
    # 检查服务状态
    service_status = check_vllm_service_status()
    
    if service_status:
        print("\n尝试连接到本地vLLM模型...")
        test_result = test_local_vllm_connection()
        
        if test_result:
            print("\n测试成功! 本地vLLM模型可以正常使用。")
        else:
            print("\n测试失败，请检查vLLM服务和模型配置。")
    else:
        print("\n请先启动vLLM服务，然后再运行此测试脚本。")
        print("启动vLLM服务的命令示例:")
        print("vllm serve qwen3 --port 8000")