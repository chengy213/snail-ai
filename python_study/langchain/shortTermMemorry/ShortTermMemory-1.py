import os

from langchain_community.chat_models import ChatTongyi
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# 1. 初始化通义千问模型
model = ChatTongyi(
    model="qwen-max",  # 可选: qwen-max, qwen-plus, qwen-turbo
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
# 3. 初始化内存检查点（短期记忆核心）
memory = MemorySaver()
# 4. 创建带记忆智能体
agent = create_react_agent(model=model, tools=tools, checkpointer=memory)
# 绑定对话线程ID，维持上下文记忆
config = {"configurable": {"thread_id": "thread-001"}}
# 第一轮对话，告知AI姓名
res1 = agent.invoke({"messages": [("user", "你好，我叫Bob！")]}, config=config)
print("第一轮回复：", res1["messages"][-1].content)
# 第二轮对话，验证记忆是否生效
res2 = agent.invoke({"messages": [("user", "我叫什么名字？")]}, config=config)
print("第二轮回复：", res2["messages"][-1].content)