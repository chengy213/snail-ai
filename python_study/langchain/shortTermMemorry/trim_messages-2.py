import os
from typing import Literal
from langchain_core.messages import RemoveMessage, AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.chat_models import ChatTongyi

# =======================
# 1. 基础组件初始化
# =======================

# 初始化模型
model = ChatTongyi(
    model="qwen-max",
    temperature=0,
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)


# 定义工具
@tool
def get_weather(location: str) -> str:
    """获取指定城市天气"""
    return f"{location} 今天多云，气温 22℃。"


tools = [get_weather]
# 将工具绑定到模型（如果需要）
model_with_tools = model.bind_tools(tools)


# =======================
# 2. 定义清理节点 (核心逻辑)
# 由于涉及到langgraph 这里只展示了如何使用RemoveMessage
# =======================

def cleanup_node(state: MessagesState):
    """
    自定义清理规则：
    1. 保留最近的 6 条消息（保证短期上下文连贯）。
    2. 其余消息使用 RemoveMessage 标记删除。
    """
    messages = state["messages"]

    # 规则：只保留最后 6 条
    keep_count = 6

    if len(messages) > keep_count:
        # 获取需要删除的消息 ID（除了最后 keep_count 条之外的所有消息）
        ids_to_remove = [msg.id for msg in messages[:-keep_count]]

        print(f"🧹 [清理节点] 当前消息数: {len(messages)}, 准备删除: {len(ids_to_remove)} 条老旧消息")

        # 返回 RemoveMessage 命令
        # LangGraph 执行到这个返回值时，会自动从 Checkpoint 中物理删除这些消息
        return {"messages": [RemoveMessage(id=id) for id in ids_to_remove]}

    print(f"✅ [清理节点] 当前消息数: {len(messages)}, 无需清理")
    return {}


# =======================
# 3. 定义对话节点
# =======================

def agent_node(state: MessagesState):
    """调用模型生成回复"""
    print(f"🤖 [对话节点] 正在处理消息...")
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# =======================
# 4. 构建图
# =======================

workflow = StateGraph(MessagesState)

# 添加节点
workflow.add_node("agent", agent_node)
workflow.add_node("cleanup", cleanup_node)

# 设置入口
workflow.set_entry_point("agent")

# 定义边：对话 -> 清理 -> 结束
workflow.add_edge("agent", "cleanup")
workflow.add_edge("cleanup", END)

# 编译图（带记忆）
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# =======================
# 5. 运行测试
# =======================

config = {"configurable": {"thread_id": "thread-cleanup-01"}}

print("\n" + "=" * 60)
print("【第 1 轮对话】")
app.invoke({"messages": [HumanMessage(content="你好，我是 Alice，编号 001。")]}, config)

print("\n" + "=" * 60)
print("【第 2 轮对话】")
app.invoke({"messages": [HumanMessage(content="北京天气怎么样？")]}, config)

print("\n" + "=" * 60)
print("【第 3 轮对话】(模拟长对话，触发清理)")
# 假设这次对话导致消息总数超过 6 条，从而触发 cleanup_node 中的删除逻辑
for i in range(5):
    app.invoke({"messages": [HumanMessage(content=f"测试消息 {i + 1}")]}, config)

print("\n" + "=" * 60)
print("【查看当前状态】")
final_state = app.get_state(config)
print(f"最终历史消息数: {len(final_state.values['messages'])}")
print("最后一条消息内容:", final_state.values['messages'][-1].content[:20] + "...")
