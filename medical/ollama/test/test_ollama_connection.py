from langchain_community.chat_models import ChatOllama
import time

"""
简单的Ollama连接测试脚本
用于诊断ChatOllama连接问题
"""

print("开始测试Ollama连接...")
print(f"当前时间: {time.strftime('%H:%M:%S')}")

try:
    # 初始化ChatOllama客户端
    llm = ChatOllama(
        model="qwen3",  # 使用的模型名称
        temperature=0,
        base_url="http://localhost:11434",  # Ollama默认API地址
        timeout=300  # 增加超时时间
    )
    
    print("ChatOllama客户端初始化成功")
    
    # 测试简单的查询
    test_query = "你好，你能正常工作吗？"
    print(f"发送测试查询: {test_query}")
    
    start_time = time.time()
    response = llm.invoke(test_query)
    end_time = time.time()
    
    print(f"收到响应，耗时: {end_time - start_time:.2f}秒")
    print(f"响应内容: {response.content}")
    
    print("\n测试完成，连接正常!")
    
except Exception as e:
    print(f"\n测试失败，错误信息: {str(e)}")
    import traceback
    traceback.print_exc()