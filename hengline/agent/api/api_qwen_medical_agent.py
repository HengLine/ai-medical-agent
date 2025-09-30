"""@FileName: api_qwen_medical_agent.py
@Description: 基于通义千问API的医疗智能体
@Author: HengLine
@Time: 2025/9/30 14:21
"""
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# 导入日志模块
from hengline.logger import info, logger

# 从基类导入
from hengline.agent.base_agent import BaseMedicalAgent, MedicalAgentState

# 导入LangChain相关库
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END


class QwenMedicalAgent(BaseMedicalAgent):
    """基于通义千问API的医疗智能体"""

    def __init__(self):
        # 初始化API调用统计
        self.api_call_count = 0
        self.total_tokens_used = 0

        # 调用基类初始化
        super().__init__()

        # 使用项目的日志模块
        self.logger = logger

        logger.info("通义千问医疗智能体初始化完成")

    def _initialize_llm(self):
        """初始化通义千问语言模型"""
        try:
            # 从配置中获取Qwen模型参数
            model_name = self.config_reader.get_llm_value("qwen", "model", "qwen-plus")
            api_url = self.config_reader.get_llm_value("qwen", "api_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            temperature = self.config_reader.get_llm_value("qwen", "temperature", 0.1)
            max_tokens = self.config_reader.get_llm_value("qwen", "max_tokens", 2048)

            # 获取API密钥
            api_key = self.config_reader.get_qwen_api_key()

            # 初始化通义千问模型
            llm = ChatTongyi(
                model=model_name,
                dashscope_api_key=api_key,
                streaming=True,
                max_retries=3,
                model_kwargs={
                    "base_url": api_url,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
            )

            self.model_supports_tools = self._check_tool_support(model_name)
            logger.info(f"成功初始化通义千问模型: {model_name}, 支持工具调用: {self.model_supports_tools}")
            return llm
        except Exception as e:
            logger.error(f"初始化通义千问语言模型时出错: {str(e)}")
            # 确保model_supports_tools有默认值
            self.model_supports_tools = False
            # 返回None，基类会处理这种情况
            return None

    def _check_tool_support(self, model_name):
        """检查模型是否支持工具调用
        
        Args:
            model_name: 模型名称
        
        Returns:
            bool: 是否支持工具调用
        """
        # 支持工具调用的Qwen模型列表
        tool_supported_models = [
            "qwen-plus",
            "qwen-max",
            "qwen-max-longcontext",
            "qwen-72b-chat",
            "qwen-14b-chat"
        ]

        # 检查模型名称是否为None或空
        if model_name is None or not model_name:
            logger.warning("模型名称为空，默认为不支持工具调用")
            return False

        # 检查模型名称是否在支持列表中
        model_name_lower = model_name.lower()
        for supported_model in tool_supported_models:
            if supported_model.lower() in model_name_lower:
                return True

        logger.warning(f"模型 {model_name} 可能不支持完整的工具调用功能")
        return False

    def _define_tools(self):
        """定义智能体可以使用的工具"""
        # 如果模型不支持工具调用，返回空列表
        if not self.model_supports_tools:
            logger.warning("当前模型不支持工具调用，将跳过工具定义")
            return []

        # 定义工具列表
        tools = [
            self.query_medical_knowledge_tool,
            self.web_search_tool,
            self.extract_symptoms_tool,
            self.assess_severity_tool
        ]

        return tools

    def _initialize_langgraph_agent(self):
        """初始化LangGraph智能体"""
        # 定义状态图
        workflow = StateGraph(MedicalAgentState)

        # 添加节点
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", ToolNode(self._define_tools()))

        # 设置边
        workflow.set_entry_point("agent")
        workflow.add_edge("tools", "agent")

        # 如果模型支持工具调用，设置条件边
        if self.model_supports_tools:
            workflow.add_conditional_edges(
                "agent",
                self._should_continue,
                {
                    "continue": "tools",
                    "end": END
                }
            )
        else:
            # 否则直接结束
            workflow.add_edge("agent", END)

        # 编译工作流
        self.agent = workflow.compile()

    def query_medical_knowledge(self, query, top_k=3):
        """查询医疗知识库
        
        Args:
            query: 查询文本
            top_k: 返回的相关文档数量
        
        Returns:
            str: 知识库中检索到的相关信息
        """
        try:
            # 检查是否有可用的检索链
            if not self.retrieval_chain:
                logger.warning("检索链未初始化，正在创建...")
                self.retrieval_chain = self._create_retrieval_chain()

                if not self.retrieval_chain:
                    return "无法初始化检索链，无法查询知识库"

            # 执行检索
            result = self.retrieval_chain.invoke({
                "query": query,
                "k": top_k
            })

            # 格式化结果
            if isinstance(result, dict) and "result" in result:
                return result["result"]
            else:
                return str(result)
        except Exception as e:
            logger.error(f"查询知识库时出错: {str(e)}")
            return f"查询知识库时出错: {str(e)}"

    def extract_symptoms(self, text):
        """从文本中提取症状
        
        Args:
            text: 包含症状的文本
        
        Returns:
            str: 提取出的症状列表
        """
        try:
            # 创建症状提取提示
            extract_prompt = ChatPromptTemplate.from_template(
                "你是一位经验丰富的医学专家。请从以下文本中提取出所有症状，并以列表形式返回。\n\n"
                "文本: {text}\n\n"
                "请只返回提取出的症状列表，不要添加任何额外的解释或说明。"
            )

            # 创建症状提取链
            extract_chain = extract_prompt | self.llm | StrOutputParser()

            # 执行症状提取
            symptoms = extract_chain.invoke({"text": text})

            return symptoms
        except Exception as e:
            logger.error(f"提取症状时出错: {str(e)}")
            return f"提取症状时出错: {str(e)}"

    def assess_severity(self, symptoms):
        """评估症状严重程度
        
        Args:
            symptoms: 症状列表
        
        Returns:
            str: 严重程度评估结果
        """
        try:
            # 创建严重程度评估提示
            assess_prompt = ChatPromptTemplate.from_template(
                "你是一位经验丰富的急诊医学专家。请根据以下症状评估患者的病情严重程度，并提供相应的建议。\n\n"
                "症状: {symptoms}\n\n"
                "评估应包括以下几个方面:\n"
                "1. 总体严重程度评级（轻度、中度、重度、紧急）\n"
                "2. 主要风险点\n"
                "3. 建议的行动（如休息、观察、就医等）\n"
                "4. 就医时机建议（如立即、24小时内、非紧急等）"
            )

            # 创建严重程度评估链
            assess_chain = assess_prompt | self.llm | StrOutputParser()

            # 执行严重程度评估
            assessment = assess_chain.invoke({"symptoms": symptoms})

            return assessment
        except Exception as e:
            logger.error(f"评估症状严重程度时出错: {str(e)}")
            return f"评估症状严重程度时出错: {str(e)}"

    def _agent_node(self, state: MedicalAgentState):
        """代理节点，用于处理输入并决定下一步行动"""
        # 创建代理提示
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一位经验丰富的医学专家助手。你的任务是回答用户的医疗问题，提供准确、专业的医学建议。\n" \
                       "请基于你所掌握的医学知识和可用的工具来回答用户的问题。\n" \
                       "如果需要更多信息来回答问题，请使用提供的工具。\n" \
                       "请记住，你的回答仅供参考，不能替代专业医生的诊断和治疗建议。"),
            MessagesPlaceholder(variable_name="messages")
        ])

        # 创建代理链
        agent_chain = agent_prompt | self.llm

        # 执行代理链
        result = agent_chain.invoke({
            "messages": state["messages"]
        })

        # 更新API调用统计
        self.api_call_count += 1

        # 返回结果
        return {"messages": [result]}

    def _should_continue(self, state: MedicalAgentState):
        """决定是否继续执行（使用工具）或结束对话"""
        # 获取最后的消息
        last_message = state["messages"][-1]

        # 检查是否有工具调用请求
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        else:
            return "end"

    def run(self, question):
        """运行智能体回答问题
        
        Args:
            question: 用户的问题
        
        Returns:
            str: 智能体的回答
        """
        try:
            # 检查是否初始化成功
            if not self.llm:
                return "智能体初始化失败，请检查配置"

            # 如果模型支持工具调用，使用LangGraph智能体
            if self.model_supports_tools and hasattr(self, "agent"):
                logger.info(f"使用支持工具调用的LangGraph智能体回答问题: {question}")

                # 执行智能体
                result = self.agent.invoke({
                    "messages": [{"role": "user", "content": question}]
                })

                # 提取回答
                if "messages" in result and len(result["messages"]) > 0:
                    return result["messages"][-1].content
                else:
                    return "无法获取智能体的回答"
            else:
                # 否则使用简化的问答链
                logger.info(f"使用简化的问答链回答问题: {question}")

                # 创建问答提示
                qa_prompt = ChatPromptTemplate.from_template(
                    "你是一位经验丰富的医学专家。请回答以下问题，并提供准确、专业的医学建议。\n\n"
                    "问题: {question}\n\n"
                    "请记住，你的回答仅供参考，不能替代专业医生的诊断和治疗建议。"
                )

                # 创建问答链
                qa_chain = qa_prompt | self.llm | StrOutputParser()

                # 执行问答链
                answer = qa_chain.invoke({"question": question})

                return answer
        except Exception as e:
            logger.error(f"运行智能体时出错: {str(e)}")
            return f"运行智能体时出错: {str(e)}"


if __name__ == "__main__":
    # 创建基于通义千问的医疗智能体实例
    medical_agent = QwenMedicalAgent()

    # 示例问题
    example_questions = [
        "什么是高血压？有哪些症状？",
        "糖尿病患者应该注意哪些饮食问题？",
        "如何判断感冒和流感的区别？",
        "心悸可能是什么原因引起的？"
    ]

    print("\n===== 通义千问医疗智能体演示 =====")
    print("输入'退出'结束演示。\n")

    try:
        # 运行交互模式
        while True:
            user_input = input("请输入您的医疗问题 (或输入示例序号 1-4): ")

            if user_input.lower() in ['退出', 'quit', 'exit']:
                print("感谢使用，再见！")
                break

            # 处理示例选择
            if user_input.isdigit() and 1 <= int(user_input) <= len(example_questions):
                idx = int(user_input) - 1
                question = example_questions[idx]
                print(f"\n选择了示例 {user_input}: {question}")
            else:
                # 用户自定义问题
                question = user_input

            # 运行智能体回答问题
            print("\n正在思考...")
            answer = medical_agent.run(question)

            # 显示回答
            print(f"\n===== 智能体回答 =====\n{answer}\n")
            print("=" * 50)
    except KeyboardInterrupt:
        print("\n程序被用户中断，再见！")
