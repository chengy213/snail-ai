import os
from typing import List
from langchain_core.messages import (
    BaseMessage, SystemMessage, HumanMessage, AIMessage
)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatTongyi

# =======================
# 1. 模型初始化
# =======================

# 主对话模型
llm = ChatTongyi(
    model="qwen-max",
    temperature=0,
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

# 摘要模型（轻量级，节省成本）
summary_llm = ChatTongyi(
    model="qwen-turbo",
    temperature=0,
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)


# =======================
# 2. 自定义历史记录类（核心逻辑）
# =======================

class SummarizedChatHistory(BaseChatMessageHistory):
    """
    带自动摘要功能的内存历史记录。
    当消息数量超过阈值时，自动将早期消息压缩成摘要。
    """

    def __init__(self, session_id: str, max_messages: int = 6):
        self.session_id = session_id
        self.messages: List[BaseMessage] = []
        self.max_messages = max_messages

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """添加消息，并在必要时执行摘要清理"""
        self.messages.extend(messages)

        # 检查是否需要清理
        if len(self.messages) > self.max_messages:
            self._summarize_old_messages()

    def _summarize_old_messages(self):
        """将旧消息压缩成摘要并替换"""
        # 1. 分割：保留最新的 max_messages 条，其余的需要摘要
        messages_to_summarize = self.messages[:-self.max_messages]
        recent_messages = self.messages[-self.max_messages:]

        if not messages_to_summarize:
            return

        print(f"📝 [History] 触发摘要: 压缩 {len(messages_to_summarize)} 条旧消息...")

        # 2. 调用轻量模型生成摘要
        summarization_prompt = ChatPromptTemplate.from_messages([
            ("system", "请将以下对话历史浓缩成一段简短的摘要，保留关键信息（如人名、地点、事件）。"),
            MessagesPlaceholder(variable_name="messages"),
        ])

        chain = summarization_prompt | summary_llm | StrOutputParser()
        summary_text = chain.invoke({"messages": messages_to_summarize})

        print(f"✂️ [History] 生成摘要: {summary_text[:40]}...")

        # 3. 重写历史：摘要 + 最近的消息
        summary_msg = SystemMessage(content=f"历史对话摘要:\n{summary_text}")

        # 覆盖当前消息列表
        self.messages = [summary_msg] + recent_messages

    def clear(self) -> None:
        self.messages = []


# =======================
# 3. 全局存储（确保会话持久化）
# =======================

_history_store = {}


def get_history(session_id):
    if session_id not in _history_store:
        _history_store[session_id] = SummarizedChatHistory(
            session_id=session_id,
            max_messages=4
        )
    return _history_store[session_id]


# =======================
# 4. 构建对话链
# =======================

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的助手。"),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])

# 简单的对话链（不使用工具）
chain = prompt | llm

# =======================
# 5. 包装历史记录能力
# =======================

runnable_with_history = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

# =======================
# 6. 迄行测试
# =======================

session_id = "user-123-session"

print("\n" + "=" * 60)
print("【第 1 轮】建立身份")
config = {"configurable": {"session_id": session_id}}
res1 = runnable_with_history.invoke({"input": "你好，我是张三，正在计划去北京旅游。"}, config)
print(f"AI: {res1.content}")

print("\n" + "=" * 60)
print("【第 2-4 轮】积累上下文 (将填满缓存)")
runnable_with_history.invoke({"input": "北京有哪些好玩的地方？"}, config)
runnable_with_history.invoke({"input": "我想去故宫。"}, config)
runnable_with_history.invoke({"input": "帮我查一下故宫门票。"}, config)

print("\n" + "=" * 60)
print("【第 5 轮】触发摘要 (总消息数 > 4)")
# 这条新消息进入时，SummarizedChatHistory.add_messages 会检测到超限
res5 = runnable_with_history.invoke({"input": "门票贵吗？"}, config)
print(f"AI: {res5.content}")

print("\n" + "=" * 60)
print("【验证内部状态】")
current_history = get_history(session_id)
print(f"当前消息总数: {len(current_history.messages)}")
print("-" * 40)
for i, msg in enumerate(current_history.messages):
    content = msg.content.replace("\n", " ")
    print(f"{i + 1}. [{type(msg).__name__}]: {content[:60]}...")

print("\n" + "=" * 60)
print("测试完成！")
