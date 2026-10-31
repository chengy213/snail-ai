from langchain_community.chat_models import ChatTongyi
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import os
import psycopg2

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
    if "纽约" in location.lower():
        return "纽约多云，22℃"
    return f"暂无{location}天气数据"



tools = [get_weather]

# 3. PostgreSQL 连接配置（匹配你的容器）
DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"


# 4. 测试连接（可选）
def test_connection():
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="postgres",
            user="postgres",
            password="postgres"
        )
        print("✅ PostgreSQL 连接成功！")
        conn.close()
        return True
    except Exception as e:
        print(f"❌ PostgreSQL 连接失败: {e}")
        return False


# 5. 使用上下文管理器连接数据库
if test_connection():
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        # 自动创建数据表（首次运行必须执行）
        checkpointer.setup()
        print("✅ 检查点表初始化成功")

        # 创建带持久化记忆的智能体
        agent = create_react_agent(model=model, tools=tools, checkpointer=checkpointer)

        # 绑定对话线程ID
        config = {"configurable": {"thread_id": "thread-001"}}

        # 第一轮对话
        res1 = agent.invoke({"messages": [("user", "你好，我叫Bob！")]}, config=config)
        print("第一轮回复：", res1["messages"][-1].content)

        # 第二轮对话（验证记忆）
        res2 = agent.invoke({"messages": [("user", "我叫什么名字？")]}, config=config)
        print("第二轮回复：", res2["messages"][-1].content)

        # 第三轮对话（测试工具）
        res3 = agent.invoke({"messages": [("user", "纽约天气怎么样？")]}, config=config)
        print("第三轮回复：", res3["messages"][-1].content)