import os
import sys
import time
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入医疗代理类
from medical_agent import MedicalAgent

if __name__ == "__main__":
    print("正在初始化医疗代理...")
    start_init = time.time()
    
    try:
        # 创建医疗智能体实例
        medical_agent = MedicalAgent()
        
        init_time = time.time() - start_init
        print(f"初始化完成，耗时: {init_time:.2f}秒\n")
        
        # 测试问题
        test_questions = [
            "什么是高血压？",
            "糖尿病的主要症状有哪些？",
            "老年人如何预防脑血栓？"
        ]
        
        # 测试每个问题的响应时间
        total_time = 0
        valid_responses = 0
        
        for i, question in enumerate(test_questions):
            print(f"测试问题 {i+1}/{len(test_questions)}: {question}")
            start_time = time.time()
            
            try:
                result = medical_agent.run(question)
                response_time = time.time() - start_time
                total_time += response_time
                valid_responses += 1
                
                print(f"响应时间: {response_time:.2f}秒")
                print(f"答案长度: {len(result)}字符")
                print("-" * 50)
            except Exception as e:
                print(f"发生错误: {e}")
                print("-" * 50)
        
        # 计算平均响应时间
        avg_time = total_time / valid_responses if valid_responses > 0 else 0
        print(f"\n测试完成！")
        print(f"成功响应次数: {valid_responses}/{len(test_questions)}")
        print(f"平均响应时间: {avg_time:.2f}秒")
        print(f"总耗时: {total_time:.2f}秒")
        
    except Exception as e:
        print(f"初始化失败: {e}")