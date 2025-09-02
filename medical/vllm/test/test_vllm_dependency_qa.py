import time
from vllm_dependency_qa import VLLMDependencyQA


def run_tests():
    """测试vLLM依赖问答系统的功能和性能"""
    print("开始测试vLLM依赖问答系统...")
    
    # 创建问答系统实例
    try:
        qa_system = VLLMDependencyQA()
        print("成功初始化VLLMDependencyQA实例")
    except Exception as e:
        print(f"初始化失败: {str(e)}")
        return
    
    # 准备测试问题
    test_questions = [
        "什么是依赖管理？",
        "pip和conda有什么区别？",
        "如何解决依赖冲突问题？",
        "什么是虚拟环境，为什么需要它？",
        "如何在CI/CD流程中管理依赖？"
    ]
    
    # 运行测试
    total_time = 0
    
    for i, question in enumerate(test_questions):
        print(f"\n测试问题 {i+1}/{len(test_questions)}: {question}")
        start_time = time.time()
        
        try:
            response = qa_system.ask_question(question)
            end_time = time.time()
            response_time = end_time - start_time
            total_time += response_time
            
            print(f"响应时间: {response_time:.2f}秒")
            print(f"回答: {response[:100]}...")  # 只显示前100个字符
        except Exception as e:
            print(f"处理问题时出错: {str(e)}")
    
    # 打印总体统计信息
    if len(test_questions) > 0:
        avg_time = total_time / len(test_questions)
        print(f"\n测试完成！")
        print(f"总测试问题数: {len(test_questions)}")
        print(f"平均响应时间: {avg_time:.2f}秒")
    
    # 测试不相关问题
    print("\n测试不相关问题:")
    unrelated_question = "如何制作巧克力蛋糕？"
    print(f"问题: {unrelated_question}")
    
    try:
        response = qa_system.ask_question(unrelated_question)
        print(f"回答: {response}")
    except Exception as e:
        print(f"处理不相关问题时出错: {str(e)}")


if __name__ == "__main__":
    run_tests()