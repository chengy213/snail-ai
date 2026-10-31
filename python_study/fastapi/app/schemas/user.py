"""
用户 Pydantic 模式模块
定义请求和响应的数据结构验证
"""
from pydantic import BaseModel, Field, EmailStr, field_validator, ConfigDict
from typing import Optional
from datetime import datetime
import re


# ==================== 请求相关模式 ====================

class UserCreateRequest(BaseModel):
    """
    创建用户请求模式
    定义创建用户时需要提交的字段
    """
    username: str = Field(..., min_length=3, max_length=50, description="登录账号")
    password: str = Field(..., min_length=6, max_length=50, description="登录密码")
    real_name: str = Field(..., min_length=1, max_length=50, description="真实姓名")
    gender: Optional[str] = Field(default="未知", description="性别：男/女/未知")
    email: Optional[EmailStr] = Field(default=None, description="电子邮箱")
    phone: Optional[str] = Field(default=None, description="电话号码")
    is_active: Optional[bool] = Field(default=True, description="是否激活")
    remark: Optional[str] = Field(default=None, description="备注")

    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        """验证登录账号格式：只允许字母、数字、下划线"""
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('登录账号只能包含字母、数字和下划线')
        return v.strip()  # 去除首尾空格

    @field_validator('phone')
    @classmethod
    def validate_phone(cls, v: Optional[str]) -> Optional[str]:
        """验证电话号码格式"""
        if v is not None:
            # 中国大陆手机号：1开头，第二位3-9，后面9位数字
            if not re.match(r'^1[3-9]\d{9}$', v):
                raise ValueError('电话号码格式不正确，请输入11位手机号')
        return v

    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v: str) -> str:
        """验证性别字段"""
        if v not in ["男", "女", "未知"]:
            raise ValueError('性别只能为：男、女、未知')
        return v


class UserUpdateRequest(BaseModel):
    """
    更新用户请求模式
    所有字段都是可选的，只更新提供的字段
    """
    real_name: Optional[str] = Field(default=None, min_length=1, max_length=50, description="真实姓名")
    gender: Optional[str] = Field(default=None, description="性别")
    email: Optional[EmailStr] = Field(default=None, description="电子邮箱")
    phone: Optional[str] = Field(default=None, description="电话号码")
    is_active: Optional[bool] = Field(default=None, description="是否激活")
    remark: Optional[str] = Field(default=None, description="备注")

    @field_validator('phone')
    @classmethod
    def validate_phone(cls, v: Optional[str]) -> Optional[str]:
        """验证电话号码格式"""
        if v is not None and not re.match(r'^1[3-9]\d{9}$', v):
            raise ValueError('电话号码格式不正确')
        return v

    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v: Optional[str]) -> Optional[str]:
        """验证性别字段"""
        if v is not None and v not in ["男", "女", "未知"]:
            raise ValueError('性别只能为：男、女、未知')
        return v


class UserQueryRequest(BaseModel):
    """
    用户查询请求模式
    支持分页和条件查询
    """
    page: int = Field(default=1, ge=1, description="页码，从1开始")
    page_size: int = Field(default=10, ge=1, le=100, description="每页条数，最大100")
    username: Optional[str] = Field(default=None, description="登录账号（模糊查询）")
    real_name: Optional[str] = Field(default=None, description="真实姓名（模糊查询）")
    email: Optional[str] = Field(default=None, description="电子邮箱（模糊查询）")
    phone: Optional[str] = Field(default=None, description="电话号码（模糊查询）")
    is_active: Optional[bool] = Field(default=None, description="是否激活")
    gender: Optional[str] = Field(default=None, description="性别")


# ==================== 响应相关模式 ====================

class UserResponse(BaseModel):
    """
    用户响应模式
    定义返回给客户端的用户数据结构
    注意：不包含密码等敏感信息
    """
    id: int
    username: str
    real_name: str
    gender: str
    email: Optional[str] = None
    phone: Optional[str] = None
    is_active: bool
    remark: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime] = None

    # 启用从ORM对象自动转换
    model_config = ConfigDict(from_attributes=True)


class UserListResponse(BaseModel):
    """用户列表响应模式"""
    items: list[UserResponse]  # 用户数据列表
    total: int  # 总记录数
    page: int  # 当前页码
    page_size: int  # 每页条数