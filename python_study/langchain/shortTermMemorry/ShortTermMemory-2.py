from langchain_community.chat_models import ChatTongyi
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import os
# 1. 初始化通义千问模型
model = ChatTongyi(
    model="qwen-max",
    temperature=0,
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")  # 替换为你的API Key
)
# 2. 定义工具
@tool
def get_weather(location: str) -> str:
    """获取指定城市天气"""
    if "nyc" in location.lower():
        return "纽约多云，22℃"
    return f"暂无{location}天气数据"

tools = [get_weather]

# 3. 使用 with 语句创建 SQLite 检查点
with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    # 4. 创建带持久化记忆的智能体
    agent = create_react_agent(model=model, tools=tools, checkpointer=checkpointer)
    # 绑定对话线程ID
    config = {"configurable": {"thread_id": "thread-001"}}
    # 第一轮对话
    res1 = agent.invoke({"messages": [("user", "你好，我叫Bob！")]}, config=config)
    print("第一轮回复：", res1["messages"][-1].content)
    # 第二轮对话
    res2 = agent.invoke({"messages": [("user", "我叫什么名字？")]}, config=config)
    print("第二轮回复：", res2["messages"][-1].content)