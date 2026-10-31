from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from base.state_models import AgentState
from graph.agent import create_data_stats_agent, create_classification_agent, create_chit_chat_agent ,\
    create_business_qa_agent, create_system_issue_agent, create_business_operation_agent
from model.factory import chat_model

checkpointer = InMemorySaver()


def build_graph():
    """构建LangGraph工作流"""

    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("save_input", save_user_message)
    workflow.add_node("classify", create_classification_agent())
    workflow.add_node("chit_chat", create_chit_chat_agent())
    workflow.add_node("business_qa", create_business_qa_agent())
    workflow.add_node("system_issue", create_system_issue_agent())
    workflow.add_node("business_operation", create_business_operation_agent())
    workflow.add_node("data_stats", create_data_stats_agent())
    workflow.add_node("save_output", save_ai_response)

    # ---- 设置流程 ----
    workflow.add_edge(START, "save_input")  # 1. 保存用户输入
    workflow.add_edge("save_input", "classify")  # 2. 分类

    # 路由函数
    def route_by_category(state: AgentState) -> str:
        """根据分类结果路由到对应的处理Agent"""
        category_map = {
            "闲聊": "chit_chat",
            "业务问题": "business_qa",
            "系统问题": "system_issue",
            "业务操作": "business_operation",
            "数据统计": "data_stats",
        }
        category = state["classification"].category
        return category_map.get(category, "chit_chat")  # 默认走闲聊

    # 添加条件边
    workflow.add_conditional_edges(
        "classify",
        route_by_category,
        {
            "chit_chat": "chit_chat",
            "business_qa": "business_qa",
            "system_issue": "system_issue",
            "business_operation": "business_operation",
            "data_stats": "data_stats",
        }
    )

    # 所有处理节点结束后退出
    workflow.add_edge("chit_chat", "save_output")
    workflow.add_edge("business_qa", "save_output")
    workflow.add_edge("system_issue", "save_output")
    workflow.add_edge("business_operation", "save_output")
    workflow.add_edge("data_stats", "save_output")
    workflow.add_edge("save_output", END)

    return workflow.compile(checkpointer=MemorySaver())

# ---- 记忆管理节点 ----
def save_user_message(state: AgentState) -> AgentState:
    """保存用户消息到共享记忆"""
    # add_messages会自动处理追加
    return {"messages": [HumanMessage(content=state["user_input"])]}

def save_ai_response(state: AgentState) -> AgentState:
    """保存AI回复到共享记忆"""
    if state.get("final_response"):
        return {"messages": [AIMessage(content=state["final_response"])]}
    return {}



graph = build_graph()

