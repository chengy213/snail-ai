from langchain_community.chat_models import ChatTongyi
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import trim_messages, RemoveMessage
import os

# 1. 初始化通义千问模型
model = ChatTongyi(
    model="qwen-max",
    temperature=0,
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)


# 2. 定义工具
@tool
def get_weather(location: str) -> str:
    """获取指定城市天气"""
    if "纽约" in location.lower() or "new york" in location.lower():
        return "纽约今天多云，气温22℃，湿度65%，适合户外活动。"
    elif "beijing" in location.lower() or "北京" in location.lower():
        return "北京今天晴朗，气温28℃，空气质量良好。"
    else:
        return f"暂无{location}的天气数据，请查询其他城市。"


tools = [get_weather]

# 3. 初始化内存检查点
memory = MemorySaver()

# 4. 创建带记忆的智能体 (使用 create_react_agent)
agent = create_react_agent(
    model=model,
    tools=tools,
    checkpointer=memory
)

# 绑定对话线程ID
config = {"configurable": {"thread_id": "thread-001"}}

print("=" * 60)
print("开始测试带记忆功能的对话智能体")
print("=" * 60)

# 第一轮对话：自我介绍
print("\n【第1轮对话】")
res1 = agent.invoke({"messages": [("user", "你好，我叫Bob！")]}, config=config)
print(f"用户: 你好，我叫Bob！")
print(f"AI: {res1['messages'][-1].content}")
print(f"当前存储消息数: {len(res1['messages'])}")

# 第二轮对话：测试记忆
print("\n【第2轮对话】")
res2 = agent.invoke({"messages": [("user", "我叫什么名字？")]}, config=config)
print(f"用户: 我叫什么名字？")
print(f"AI: {res2['messages'][-1].content}")
print(f"当前存储消息数: {len(res2['messages'])}")

# 第三轮对话：测试工具调用
print("\n【第3轮对话】")
res3 = agent.invoke({"messages": [("user", "纽约天气怎么样？")]}, config=config)
print(f"用户: 纽约天气怎么样？")
print(f"AI: {res3['messages'][-1].content}")
print(f"当前存储消息数: {len(res3['messages'])}")

# 第四轮对话：测试综合记忆
print("\n【第4轮对话】")
res4 = agent.invoke({"messages": [("user", "我叫什么名字？我刚才问了哪里的天气？")]}, config=config)
print(f"用户: 我叫什么名字？我刚才问了哪里的天气？")
print(f"AI: {res4['messages'][-1].content}")
print(f"当前存储消息数: {len(res4['messages'])}")

# 第五轮对话：测试消息修剪
print("\n【第5轮对话 - 测试消息修剪】")


# 定义一个简单的 Token 计数器 (避免 NameError)
def simple_token_counter(messages):
    """简单的 token 计数器，按字符数/2 估算"""
    total_tokens = 0
    for msg in messages:
        if hasattr(msg, 'content'):
            total_tokens += len(str(msg.content)) // 2
    return max(total_tokens, 1)


# 获取当前所有消息
current_state = agent.get_state(config)
all_messages = current_state.values.get("messages", [])

print(f"修剪前消息数: {len(all_messages)}")

# 应用消息修剪
# 注意：这里为了演示效果，将 max_tokens 设得很小，强行触发修剪
# 实际使用时请设置为合理的值，例如 2000
try:
    trimmed = trim_messages(
        all_messages,
        max_tokens=10,  # 设置极小值以测试修剪逻辑
        strategy="last",
        token_counter=simple_token_counter,
        allow_partial=False,
        start_on="human"
    )

    print(f"修剪后消息数: {len(trimmed)}")

    # ✅ 关键步骤：把被修剪掉的消息用 RemoveMessage 从 Agent 状态中删除
    # 注意：直接 update_state(config, {"messages": trimmed}) 不会生效——
    # MessagesState 的 add_messages 归约器按消息 id 合并，只会新增/替换，不会删除
    keep_ids = {msg.id for msg in trimmed}
    ids_to_remove = [msg.id for msg in all_messages if msg.id not in keep_ids]
    agent.update_state(config, {"messages": [RemoveMessage(id=id_) for id_ in ids_to_remove]})
    print(f"✅ 已从 Agent 状态删除 {len(ids_to_remove)} 条被修剪的消息")

except Exception as e:
    print(f"修剪过程出错: {e}")

# 继续对话验证记忆
print("\n【第6轮对话 - 验证修剪后的记忆】")
# 因为第5步可能把最早的消息（比如"我叫Bob"）剪掉了，所以Agent可能记不住了
res5 = agent.invoke({"messages": [("user", "我刚才问过天气吗？")]}, config=config)
print(f"用户: 我刚才问过天气吗？")
print(f"AI: {res5['messages'][-1].content}")
print(f"当前存储消息数: {len(res5['messages'])}")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
