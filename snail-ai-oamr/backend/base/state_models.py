# 导入类型提示相关的类和工具
# TypedDict: 用于定义具有固定键和特定值类型的字典结构
# Literal: 限制变量只能取特定的字面量值
# Optional: 表示一个值可以是指定类型或None
# List: 列表类型注解
# Union: 联合类型，表示可以是多种类型中的一种
from typing import TypedDict, Literal, Optional, List, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
# 从pydantic库导入数据验证和序列化的基类和字段定义工具
# BaseModel: pydantic的基础模型类，用于定义数据结构和自动验证
# Field: 为模型字段提供额外的元数据和验证规则
from pydantic import BaseModel, Field

# 从langgraph库导入构建状态图所需的核心组件
# StateGraph: 状态图类，用于定义和管理多步骤的工作流程
# END: 特殊节点标记，表示工作流程的终止点
# from langgraph.graph import StateGraph, END


# ---------- Pydantic 模型：分类结果 ----------
class ClassificationResult(BaseModel):
    """
    分类Agent的输出结构
    
    使用Pydantic模型定义分类结果的数据结构，提供自动验证和序列化功能
    """
    
    # category字段：问题分类的类别，只能是以下5个选项之一
    # Literal类型确保了类型安全，如果赋值不在列表中会报错
    category: Literal["闲聊", "业务问题", "系统问题", "业务操作", "数据统计"] = Field(
        description="问题分类"  # 字段描述，用于文档生成和提示信息
    )
    
    # confidence字段：分类的置信度，浮点数类型
    # ge=0.0: 大于等于0.0（greater than or equal to）
    # le=1.0: 小于等于1.0（less than or equal to）
    confidence: float = Field(
        ge=0.0,           # 最小值约束
        le=1.0,           # 最大值约束
        description="置信度"  # 字段说明
    )
    
    # reason字段：分类的理由说明，字符串类型
    reason: str = Field(
        description="分类理由"  # 字段说明
    )


# ---------- LangGraph 状态 ----------
class AgentState(TypedDict):
    """
    全局状态定义
    
    继承TypedDict，定义了整个Agent系统在不同节点间传递的共享状态
    所有字段都是可选的（通过Optional标记），允许在流程中逐步填充
    """
    
    # user_input: 存储用户的原始输入文本
    # 必填字段，用于记录用户发送的完整消息内容
    user_input: str
    
    # image_paths: 存储用户上传的图片文件路径列表
    # Optional[List[str]]: 可以是None（没有图片）或字符串列表（有图片）
    # 支持多图片上传场景
    image_paths: Optional[List[str]]
    
    # classification: 存储分类Agent的分类结果
    # Optional[ClassificationResult]: 可以是None（尚未分类）或分类结果对象
    # 在分类节点执行后被填充
    classification: Optional[ClassificationResult]
    
    # final_response: 存储系统最终返回给用户的回复内容
    # Optional[str]: 可以是None（未生成回复）或最终的回复文本
    # 在回复生成节点执行后被填充
    final_response: Optional[str]

    # ---- 对话记忆（自动追加，所有Agent共享）----
    messages: Annotated[List[BaseMessage], add_messages]
    
    # error: 存储处理过程中发生的错误信息
    # Optional[str]: 可以是None（无错误）或错误描述字符串
    # 用于异常处理和错误提示
    error: Optional[str]