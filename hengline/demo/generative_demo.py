import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# 导入日志模块
from hengline.logger import logger

# 导入生成式智能体
from hengline.agent.generative.generative_medical_agent import GenerativeMedicalAgent


def run_generative_demo():
    """运行生成式智能体演示"""
    logger.info("===== 生成式医疗智能体演示 =====")
    
    try:
        # 初始化生成式智能体
        logger.info("正在初始化生成式医疗智能体...")
        agent = GenerativeMedicalAgent()
        logger.info("生成式医疗智能体初始化成功！")
        
        print("\n欢迎使用生成式医疗智能体演示程序！")
        print("此智能体可以生成多种类型的医疗内容，包括基本信息、详细解释、患者教育材料和临床案例。")
        print("\n请选择要演示的内容生成类型：")
        print("1. 基本医疗信息 (general_info)")
        print("2. 详细医学解释 (detailed_explanation)")
        print("3. 患者教育材料 (patient_education)")
        print("4. 临床案例 (medical_case)")
        
        # 预设的医疗主题列表
        topics = [
            "高血压",
            "糖尿病",
            "冠心病",
            "脑卒中",
            "阿尔茨海默病"
        ]
        
        while True:
            # 用户选择生成类型
            type_choice = input("\n请输入生成类型的编号 (1-4，输入'退出'结束): ")
            
            if type_choice.lower() in ['退出', 'quit', 'exit']:
                break
            
            # 验证用户输入
            if not type_choice.isdigit() or int(type_choice) not in range(1, 5):
                print("无效的选择，请输入1-4之间的数字。")
                continue
            
            # 映射用户选择到生成类型
            generation_types = [
                "general_info",
                "detailed_explanation",
                "patient_education",
                "medical_case"
            ]
            generation_type = generation_types[int(type_choice) - 1]
            
            # 显示预设主题
            print("\n请选择一个医疗主题：")
            for i, topic in enumerate(topics, 1):
                print(f"{i}. {topic}")
            print(f"{len(topics)+1}. 自定义主题")
            
            # 用户选择主题
            topic_choice = input("请输入主题的编号 (1-" + str(len(topics)+1) + "): ")
            
            # 验证用户输入
            if not topic_choice.isdigit() or int(topic_choice) not in range(1, len(topics)+2):
                print("无效的选择，请输入有效的数字。")
                continue
            
            # 确定主题
            if int(topic_choice) == len(topics)+1:
                topic = input("请输入自定义医疗主题: ")
            else:
                topic = topics[int(topic_choice) - 1]
            
            # 生成内容
            print(f"\n正在生成关于'{topic}'的{generation_type}内容...")
            logger.info(f"开始生成关于'{topic}'的{generation_type}内容")
            
            # 执行生成
            result = agent.generate_content(topic, generation_type)
            
            # 显示结果
            print(f"\n===== {generation_type} 结果 =====")
            print(result)
            print("================================")
            
            logger.info(f"完成关于'{topic}'的{generation_type}内容生成")
            
            # 询问是否继续
            again = input("\n是否继续生成其他内容？(y/n): ")
            if again.lower() != 'y':
                break
        
        logger.info("生成式医疗智能体演示结束")
        print("\n感谢使用生成式医疗智能体演示程序！")
        
    except Exception as e:
        logger.error(f"演示程序运行出错: {str(e)}")
        print(f"\n演示程序运行出错: {str(e)}")


def run_combined_demo():
    """运行组合问答和生成式功能的演示"""
    logger.info("===== 组合问答和生成式功能演示 =====")
    
    try:
        # 初始化生成式智能体
        logger.info("正在初始化生成式医疗智能体...")
        agent = GenerativeMedicalAgent()
        logger.info("生成式医疗智能体初始化成功！")
        
        print("\n欢迎使用组合问答和生成式功能演示程序！")
        print("此程序允许您先提问，然后选择是否生成更详细的相关内容。")
        
        # 预设问题
        questions = [
            "什么是高血压？",
            "糖尿病有哪些类型？",
            "如何预防冠心病？",
            "脑卒中的急救措施是什么？",
            "阿尔茨海默病的早期症状有哪些？"
        ]
        
        while True:
            # 显示预设问题
            print("\n请选择一个预设问题或输入自己的问题：")
            for i, question in enumerate(questions, 1):
                print(f"{i}. {question}")
            print(f"{len(questions)+1}. 自定义问题")
            
            # 用户选择问题
            q_choice = input("请输入问题的编号 (1-" + str(len(questions)+1) + "): ")
            
            # 验证用户输入
            if not q_choice.isdigit() or int(q_choice) not in range(1, len(questions)+2):
                print("无效的选择，请输入有效的数字。")
                continue
            
            # 确定问题
            if int(q_choice) == len(questions)+1:
                question = input("请输入自定义问题: ")
            else:
                question = questions[int(q_choice) - 1]
            
            # 回答问题
            print(f"\n正在回答问题: '{question}'")
            logger.info(f"处理问题: {question[:50]}...")
            
            # 执行问答
            answer = agent.run(question)
            
            # 显示回答
            print(f"\n===== 回答结果 =====")
            print(answer)
            print("====================")
            
            # 询问是否生成额外内容
            generate_extra = input("\n是否需要生成更详细的相关内容？(y/n): ").lower() == 'y'
            
            if generate_extra:
                # 获取更详细的回答
                full_answer = agent.run(question, generate_extra_content=True)
                
                # 显示完整回答
                print(f"\n===== 详细回答结果 =====")
                print(full_answer)
                print("========================")
            
            # 询问是否继续
            again = input("\n是否继续提问？(y/n): ")
            if again.lower() != 'y':
                break
        
        logger.info("组合问答和生成式功能演示结束")
        print("\n感谢使用组合问答和生成式功能演示程序！")
        
    except Exception as e:
        logger.error(f"演示程序运行出错: {str(e)}")
        print(f"\n演示程序运行出错: {str(e)}")


def main():
    """主函数"""
    print("===== 医疗智能体生成式功能演示程序 =====")
    print("1. 生成式功能演示")
    print("2. 组合问答和生成式功能演示")
    
    choice = input("请选择演示类型 (1-2，输入'退出'结束): ")
    
    if choice.lower() in ['退出', 'quit', 'exit']:
        print("感谢使用演示程序！")
        return
    
    if choice == '1':
        run_generative_demo()
    elif choice == '2':
        run_combined_demo()
    else:
        print("无效的选择，请输入1或2。")


if __name__ == "__main__":
    main()