from langchain_core.messages import HumanMessage, SystemMessage

from base.state_models import AgentState, ClassificationResult
from model.factory import chat_model
from utils.prompt_loader import load_question_classification_prompt, load_chit_chat_prompt, load_business_qa_prompt, \
    load_system_issue_prompt, load_data_statistics_prompt, load_business_operation_prompt


# ==================== 分类Agent ====================
def create_classification_agent():
    """问题分类处理Agent"""
    def classify(state: AgentState) -> AgentState:
        prompt = load_question_classification_prompt()

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=state["user_input"])
        ]

        # 如果有图片，可以加入图片描述逻辑
        if state.get("image_paths"):
            messages.append(HumanMessage(content=f"用户还上传了{len(state['image_paths'])}张图片"))

        # 使用结构化输出
        structured_llm = chat_model.with_structured_output(ClassificationResult)
        result = structured_llm.invoke(messages)

        state["classification"] = result
        return state

    return classify


# ==================== 闲聊Agent ====================
def create_chit_chat_agent():
    """闲聊处理Agent"""

    def handle(state: AgentState) -> AgentState:
        prompt = load_chit_chat_prompt()
        messages = [SystemMessage(content=prompt)]
        messages.extend(state["messages"])  # 全部历史
        messages.append(HumanMessage(content=state["user_input"]))
        # 使用结构化输出
        response = chat_model.invoke(messages)

        state["final_response"] = response.content
        return state

    return handle


# ==================== 业务问题Agent ====================
def create_business_qa_agent():
    """业务问题处理Agent - 可接RAG检索知识库"""

    def handle(state: AgentState) -> AgentState:
        # 这里可以接入你的业务知识库(RAG)
        prompt = load_business_qa_prompt()

        response = chat_model.invoke([HumanMessage(content=prompt)])
        state["final_response"] = response.content
        return state

    return handle


# ==================== 系统问题Agent ====================
def create_system_issue_agent():
    """系统问题处理Agent - 诊断和排查"""

    def handle(state: AgentState) -> AgentState:
        prompt = load_system_issue_prompt()

        response = chat_model.invoke([HumanMessage(content=prompt)])
        state["final_response"] = response.content
        return state

    return handle


# ==================== 业务操作Agent ====================
def create_business_operation_agent():
    """业务操作Agent - 执行具体操作(可接API/工具调用)"""

    # 定义业务操作工具(示例)
    tools = [
        {
            "name": "create_order",
            "description": "创建订单",
            "parameters": {"product": "str", "quantity": "int"}
        },
        {
            "name": "update_info",
            "description": "修改用户信息",
            "parameters": {"field": "str", "value": "str"}
        }
    ]

    def handle(state: AgentState) -> AgentState:
        prompt = load_business_operation_prompt()

        response = chat_model.invoke([HumanMessage(content=prompt)])
        state["final_response"] = response.content
        return state

    return handle


# ==================== 数据统计Agent ====================
def create_data_stats_agent():
    """数据统计Agent - 接数据库或数据平台查询"""

    def handle(state: AgentState) -> AgentState:
        prompt = load_data_statistics_prompt()

        response = chat_model.invoke([HumanMessage(content=prompt)])
        state["final_response"] = response.content
        return state

    return handle
