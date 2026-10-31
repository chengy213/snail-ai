"""
用户API路由模块
定义用户相关的RESTful API接口
"""
from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy.orm import Session
from typing import List

from python_study.fastapi.app.core.database import get_db
from python_study.fastapi.app.schemas.user import (
    UserCreateRequest,
    UserUpdateRequest,
    UserQueryRequest,
    UserResponse,
    UserListResponse
)
from python_study.fastapi.app.services.user import UserService
from python_study.fastapi.app.utils.response import success_response, error_response

# 创建路由实例，添加前缀和标签
router = APIRouter(prefix="/users", tags=["用户管理"])


@router.post("/", response_model=dict, summary="创建用户")
async def create_user(
        user_data: UserCreateRequest,  # 请求体，自动验证
        db: Session = Depends(get_db)  # 依赖注入数据库会话
):
    """
    创建新用户接口

    请求示例：
    {
        "username": "zhangsan",
        "password": "123456",
        "real_name": "张三",
        "gender": "男",
        "email": "zhangsan@example.com",
        "phone": "13800138000"
    }
    """
    # 调用业务逻辑层创建用户
    user = UserService.create_user(db, user_data)
    return success_response(
        data=user.model_dump(),  # 将Pydantic模型转为字典
        message="用户创建成功"
    )


@router.get("/{user_id}", response_model=dict, summary="获取用户详情")
async def get_user(
        user_id: int = Path(..., ge=1, description="用户ID"),  # 路径参数，必须大于0
        db: Session = Depends(get_db)
):
    """
    根据ID获取用户详细信息

    请求示例：
    GET /api/v1/users/1
    """
    user = UserService.get_user_by_id(db, user_id)
    return success_response(data=user.model_dump())


@router.get("/", response_model=dict, summary="获取用户列表")
async def get_users(
        page: int = Query(default=1, ge=1, description="页码"),
        page_size: int = Query(default=10, ge=1, le=100, description="每页条数"),
        username: str = Query(default=None, description="登录账号（模糊查询）"),
        real_name: str = Query(default=None, description="真实姓名（模糊查询）"),
        email: str = Query(default=None, description="电子邮箱（模糊查询）"),
        phone: str = Query(default=None, description="电话号码（模糊查询）"),
        is_active: bool = Query(default=None, description="是否激活"),
        gender: str = Query(default=None, description="性别"),
        db: Session = Depends(get_db)
):
    """
    获取用户列表，支持分页和条件查询

    请求示例：
    GET /api/v1/users?page=1&page_size=10&username=zhang
    """
    # 构建查询参数对象
    query_params = UserQueryRequest(
        page=page,
        page_size=page_size,
        username=username,
        real_name=real_name,
        email=email,
        phone=phone,
        is_active=is_active,
        gender=gender
    )

    # 调用服务层获取列表
    result = UserService.get_user_list(db, query_params)

    return success_response(data=result.model_dump())


@router.put("/{user_id}", response_model=dict, summary="更新用户信息")
async def update_user(
        user_data: UserUpdateRequest,  # 请求体，必填；缺失时由 FastAPI 返回 422 校验错误
        user_id: int = Path(..., ge=1, description="用户ID"),
        db: Session = Depends(get_db)
):
    """
    更新用户信息，只更新提供的字段

    请求示例：
    PUT /api/v1/users/1
    {
        "real_name": "张三丰",
        "email": "zhangsanfeng@example.com"
    }
    """
    user = UserService.update_user(db, user_id, user_data)
    return success_response(
        data=user.model_dump(),
        message="用户更新成功"
    )


@router.delete("/{user_id}", response_model=dict, summary="删除用户")
async def delete_user(
        user_id: int = Path(..., ge=1, description="用户ID"),
        db: Session = Depends(get_db)
):
    """
    删除指定用户

    请求示例：
    DELETE /api/v1/users/1
    """
    UserService.delete_user(db, user_id)
    return success_response(message="用户删除成功")


@router.put("/{user_id}/activate", response_model=dict, summary="激活/停用用户")
async def toggle_user_active(
        user_id: int = Path(..., ge=1, description="用户ID"),
        db: Session = Depends(get_db)
):
    """
    切换用户激活状态

    请求示例：
    PUT /api/v1/users/1/activate
    """
    # 先获取用户当前状态
    user = UserService.get_user_by_id(db, user_id)

    # 切换状态
    from python_study.fastapi.app.schemas.user import UserUpdateRequest
    update_data = UserUpdateRequest(is_active=not user.is_active)

    user = UserService.update_user(db, user_id, update_data)
    status = "激活" if user.is_active else "停用"

    return success_response(
        data=user.model_dump(),
        message=f"用户已{status}"
    )