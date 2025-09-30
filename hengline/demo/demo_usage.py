#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
医疗智能体使用示例
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 验证路径是否正确添加
# print(f"Python路径: {sys.path}")

# 导入日志模块
from hengline.logger import logger

# 导入医疗智能体
from hengline.agent.ollama.ollama_medical_agent import OllamaMedicalAgent


def run_demo():
    """运行医疗智能体演示"""
    logger.info("===== AI医疗问答系统演示 =====")
    
    try:
        # 创建医疗智能体实例
        medical_agent = OllamaMedicalAgent()
        
        logger.info("初始化完成！请输入您的医疗问题（输入'退出'结束程序）")
        print("初始化完成！请输入您的医疗问题（输入'退出'结束程序）")
        logger.info("\n示例问题：")
        print("\n示例问题：")
        print("1. 什么是高血压？如何预防？")
        print("2. 我最近发热、咳嗽、喉咙痛，应该怎么办？")
        print("3. 心肺复苏的步骤是什么？")
        print("4. 如何保持健康的生活方式？")
        
        while True:
            print("\n-----------------------------------")
            question = input("请输入您的问题: ")
            
            if question.lower() in ['退出', 'exit', 'quit', 'bye']:
                logger.info("感谢使用AI医疗问答系统，再见！")
                print("感谢使用AI医疗问答系统，再见！")
                break
            
            logger.info("\n正在获取答案，请稍候...")
            print("\n正在获取答案，请稍候...")
            try:
                # 调用智能体回答问题
                result = medical_agent.run(question)
                print("\n===== 回答 =====")
                print(result)
                logger.info(f"处理问题: {question[:50]}...")
            except Exception as e:
                logger.error(f"处理问题时出错: {str(e)}")
                print(f"处理问题时出错: {str(e)}")
                print("请检查您的问题或稍后再试。")
    
    except Exception as e:
        logger.error(f"初始化医疗智能体失败: {str(e)}")
        print(f"初始化医疗智能体失败: {str(e)}")
        print("请确保已安装所有依赖，并且本地LLM服务已正确配置。")
        print("可以参考README.md文件中的说明进行配置。")


if __name__ == "__main__":
    run_demo()