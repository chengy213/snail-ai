"""
统一响应格式模块
定义标准的API响应结构，保持接口返回一致性
"""
from typing import Any, Optional, TypeVar, Generic
from pydantic import BaseModel

T = TypeVar('T')  # 泛型类型变量，用于数据类型


class ResponseModel(BaseModel, Generic[T]):
    """
    统一响应模型
    使用泛型支持不同数据类型的返回

    响应结构：
    {
        "code": 200,      # 状态码
        "message": "成功",  # 提示信息
        "data": {...}     # 返回数据
    }
    """
    code: int = 200  # 响应状态码，默认200
    message: str = "操作成功"  # 响应提示信息
    data: Optional[T] = None  # 响应数据，支持任意类型


class PaginatedResponse(BaseModel, Generic[T]):
    """
    分页响应模型
    用于列表查询接口的分页数据返回

    响应结构：
    {
        "code": 200,
        "message": "成功",
        "data": {
            "items": [...],    # 数据列表
            "total": 100,      # 总条数
            "page": 1,         # 当前页码
            "page_size": 10    # 每页条数
        }
    }
    """
    code: int = 200  # 响应状态码
    message: str = "操作成功"  # 响应提示信息
    data: Optional[dict] = None  # 分页数据结构


def success_response(data: Any = None, message: str = "操作成功") -> dict:
    """
    成功响应快捷函数
    直接返回符合统一格式的字典

    Args:
        data: 返回的数据
        message: 提示信息

    Returns:
        dict: 统一格式的响应数据
    """
    return {"code": 200, "message": message, "data": data}


def error_response(code: int = 400, message: str = "操作失败", data: Any = None) -> dict:
    """
    错误响应快捷函数

    Args:
        code: 错误状态码
        message: 错误提示信息
        data: 错误详情数据

    Returns:
        dict: 统一格式的错误响应
    """
    return {"code": code, "message": message, "data": data}