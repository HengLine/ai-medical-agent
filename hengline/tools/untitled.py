from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

# 定义工具
tools = [...]  # 你的工具列表

# 创建 StateGraph
workflow = StateGraph()

# 添加工具节点
tool_node = ToolNode(tools)
workflow.add_node("tool_node", tool_node)

# 设置入口和出口
workflow.set_entry_point("tool_node")
workflow.set_finish_point("tool_node")

# 编译并运行
app = workflow.compile()
result = app.invoke({"input": "你的查询"})
