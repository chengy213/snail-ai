"""
自定义异常体系模块
定义业务级别的异常，用于统一异常处理
"""
from typing import Any, Optional, Dict


class AppBaseException(Exception):
    """
    应用基础异常类
    所有自定义异常都应该继承此类
    """
    def __init__(self, message: str, status_code: int = 400, data: Any = None):
        self.message = message  # 异常消息
        self.status_code = status_code  # HTTP状态码
        self.data = data  # 附加数据
        super().__init__(self.message)


class ResourceNotFoundException(AppBaseException):
    """
    资源未找到异常
    当请求的资源（如用户）不存在时抛出
    """
    def __init__(self, resource: str = "资源", resource_id: Any = None):
        message = f"{resource}未找到"
        if resource_id:
            message = f"{resource} (ID: {resource_id}) 未找到"
        super().__init__(message=message, status_code=404)


class DuplicateResourceException(AppBaseException):
    """
    资源重复异常
    当尝试创建已存在的资源时抛出
    """
    def __init__(self, field: str, value: Any):
        message = f"{field} '{value}' 已存在"
        super().__init__(message=message, status_code=409)


class ValidationException(AppBaseException):
    """
    数据验证异常
    当输入数据不符合要求时抛出
    """
    def __init__(self, message: str = "数据验证失败"):
        super().__init__(message=message, status_code=422)


class DatabaseException(AppBaseException):
    """
    数据库操作异常
    当数据库操作失败时抛出
    """
    def __init__(self, message: str = "数据库操作失败"):
        super().__init__(message=message, status_code=500)


class AuthenticationException(AppBaseException):
    """
    认证异常
    当用户认证失败时抛出
    """
    def __init__(self, message: str = "认证失败"):
        super().__init__(message=message, status_code=401)