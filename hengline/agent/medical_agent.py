import argparse
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 导入日志模块
from hengline.logger import info, error

# 导入不同类型的医疗智能体
from hengline.agent.ollama.ollama_medical_agent import OllamaMedicalAgent
from hengline.agent.ollama.ollama_generative_agent import OllamaGenerativeAgent
from hengline.agent.vllm.vllm_medical_agent import VLLMMedicalAgent
from hengline.agent.vllm.vllm_generative_agent import VllmGenerativeAgent
from hengline.agent.api.api_openai_medical_agent import OpenAIMedicalAgent
from hengline.agent.api.api_openai_generative_agent import OpenAIGenerativeAgent
from hengline.agent.api.api_qwen_medical_agent import QwenMedicalAgent
from hengline.agent.api.api_qwen_generative_agent import QwenGenerativeAgent


class MedicalAgentFactory:
    """医疗智能体工厂类，用于创建不同类型的医疗智能体"""

    @staticmethod
    def create_agent(agent_type: str) -> object:
        """创建指定类型的医疗智能体"""
        agent_type = agent_type.lower()

        if agent_type == "ollama":
            return OllamaMedicalAgent(), OllamaGenerativeAgent()
        elif agent_type == "vllm":
            return VLLMMedicalAgent(), VllmGenerativeAgent()
        else:
            return MedicalAgentFactory.create_api_agent(agent_type)

    """创建基于API的医疗智能体"""

    @staticmethod
    def create_api_agent(agent_type: str) -> object:
        if agent_type == "openai":
            return OpenAIMedicalAgent(), OpenAIGenerativeAgent()
        elif agent_type == "qwen":
            return QwenMedicalAgent(), QwenGenerativeAgent()
        else:
            raise ValueError(f"不支持的智能体类型: {agent_type}。支持的类型: api/openai, api/qwen")


def main():
    """主函数，提供命令行接口让用户选择智能体类型"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="医疗智能体 - 支持三种不同的后端实现")
    parser.add_argument(
        "--type",
        choices=["ollama", "vllm", "openai", "qwen"],
        default="ollama",
        help="选择智能体的后端类型 (默认: ollama)"
    )

    # 解析命令行参数
    args = parser.parse_args()

    try:
        # 创建指定类型的智能体
        info(f"正在初始化 {args.type} 类型的医疗智能体...")
        agent, generative_agent = MedicalAgentFactory.create_agent(args.type)
        info(f"{args.type} 医疗智能体初始化成功！")

        # 运行智能体的交互模式
        run_interactive_mode(agent, args.type)

    except Exception as e:
        error(f"初始化智能体时出错: {str(e)}")
        sys.exit(1)


def run_interactive_mode(agent: object, agent_type: str):
    """运行智能体的交互模式"""
    info(f"基于{agent_type}的医疗智能体已启动。输入'退出'或'quit'结束会话。")

    try:
        # 进入交互模式
        while True:
            try:
                user_question = input("\n请输入您的问题: ")
                if user_question.lower() in ['退出', 'quit', 'exit']:
                    # 对于远程API智能体，显示API调用统计
                    if agent_type == "openai" and hasattr(agent, "get_api_stats"):
                        stats = agent.get_api_stats()
                        info(f"\nAPI调用统计: {stats}")
                    info("感谢使用医疗智能体，再见！")
                    break

                # 处理用户问题
                result = agent.run(user_question)
                print(f"回答: {result}")  # 保留控制台输出以便用户查看
                info(f"处理问题: {user_question[:50]}...")

            except KeyboardInterrupt:
                info("\n程序被用户中断，再见！")
                break
            except Exception as e:
                error(f"处理问题时出错: {str(e)}")
    except Exception as e:
        error(f"交互模式运行出错: {str(e)}")


if __name__ == "__main__":
    main()
