"""
用户业务逻辑层
封装所有用户相关的业务操作
"""
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, func
from typing import Optional, Tuple, List
import hashlib
import secrets
from datetime import datetime

from python_study.fastapi.app.models.user import SysUser, GenderEnum
from python_study.fastapi.app.schemas.user import (
    UserCreateRequest,
    UserUpdateRequest,
    UserQueryRequest,
    UserResponse,
    UserListResponse
)
from python_study.fastapi.app.core.exceptions import (
    ResourceNotFoundException,
    DuplicateResourceException,
    ValidationException,
    DatabaseException
)

# 移除 passlib 和 bcrypt 的导入，使用内置库


class UserService:
    """
    用户服务类
    封装所有用户相关的业务逻辑
    """

    @staticmethod
    def _hash_password(password: str) -> str:
        """
        使用 SHA-256 + 随机盐值进行密码哈希
        这种方法：
        1. 没有长度限制
        2. 不需要额外安装包
        3. 足够安全
        """
        # 生成随机盐值（16字节 = 32个十六进制字符）
        salt = secrets.token_hex(16)
        # 组合密码和盐值
        salted_password = password + salt
        # 计算哈希
        hash_obj = hashlib.sha256(salted_password.encode('utf-8'))
        # 返回格式: salt$hash (salt是32字符，hash是64字符)
        return f"{salt}${hash_obj.hexdigest()}"

    @staticmethod
    def _verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        验证密码是否正确
        """
        try:
            # 分割存储的哈希值
            salt, stored_hash = hashed_password.split('$')
            # 使用相同的盐值计算哈希
            salted_password = plain_password + salt
            hash_obj = hashlib.sha256(salted_password.encode('utf-8'))
            computed_hash = hash_obj.hexdigest()
            # 使用时间恒定的比较方法防止时序攻击
            return secrets.compare_digest(computed_hash, stored_hash)
        except Exception:
            return False

    @staticmethod
    def _check_unique_fields(db: Session, username: str, email: Optional[str] = None,
                             phone: Optional[str] = None, exclude_id: Optional[int] = None):
        """
        检查唯一字段是否已存在

        Args:
            db: 数据库会话
            username: 登录账号
            email: 电子邮箱
            phone: 电话号码
            exclude_id: 排除的用户ID（用于更新时的自身检查）

        Raises:
            DuplicateResourceException: 当字段值已存在时抛出
        """
        # 构建查询条件
        conditions = []

        # 检查用户名
        user_query = db.query(SysUser).filter(SysUser.username == username)
        if exclude_id:
            user_query = user_query.filter(SysUser.id != exclude_id)
        if user_query.first():
            raise DuplicateResourceException(field="登录账号", value=username)

        # 检查邮箱
        if email:
            email_query = db.query(SysUser).filter(SysUser.email == email)
            if exclude_id:
                email_query = email_query.filter(SysUser.id != exclude_id)
            if email_query.first():
                raise DuplicateResourceException(field="电子邮箱", value=email)

        # 检查电话号码
        if phone:
            phone_query = db.query(SysUser).filter(SysUser.phone == phone)
            if exclude_id:
                phone_query = phone_query.filter(SysUser.id != exclude_id)
            if phone_query.first():
                raise DuplicateResourceException(field="电话号码", value=phone)

    @staticmethod
    def create_user(db: Session, user_data: UserCreateRequest) -> UserResponse:
        """
        创建新用户

        Args:
            db: 数据库会话
            user_data: 用户创建数据

        Returns:
            UserResponse: 创建的用户信息

        Raises:
            DuplicateResourceException: 账号/邮箱/电话已存在
            DatabaseException: 数据库操作失败
        """
        try:
            # 1. 检查唯一性约束
            UserService._check_unique_fields(
                db,
                username=user_data.username,
                email=user_data.email,
                phone=user_data.phone
            )

            # 2. 创建用户对象
            db_user = SysUser(
                username=user_data.username,
                password=UserService._hash_password(user_data.password),  # 使用新的哈希方法
                real_name=user_data.real_name,
                gender=user_data.gender,
                email=user_data.email,
                phone=user_data.phone,
                is_active=user_data.is_active if user_data.is_active is not None else True,
                remark=user_data.remark
            )

            # 3. 保存到数据库
            db.add(db_user)  # 添加到会话
            db.commit()  # 提交事务
            db.refresh(db_user)  # 刷新对象，获取数据库生成的默认值

            # 4. 返回用户信息
            return UserResponse.model_validate(db_user)

        except DuplicateResourceException:
            db.rollback()  # 回滚事务
            raise  # 重新抛出异常
        except Exception as e:
            db.rollback()
            raise DatabaseException(f"创建用户失败: {str(e)}")

    @staticmethod
    def get_user_by_id(db: Session, user_id: int) -> UserResponse:
        """
        根据ID获取用户

        Args:
            db: 数据库会话
            user_id: 用户ID

        Returns:
            UserResponse: 用户信息

        Raises:
            ResourceNotFoundException: 用户不存在
        """
        user = db.query(SysUser).filter(SysUser.id == user_id).first()
        if not user:
            raise ResourceNotFoundException(resource="用户", resource_id=user_id)
        return UserResponse.model_validate(user)

    @staticmethod
    def get_user_list(db: Session, query_params: UserQueryRequest) -> UserListResponse:
        """
        获取用户列表（支持分页和条件查询）

        Args:
            db: 数据库会话
            query_params: 查询参数

        Returns:
            UserListResponse: 分页的用户列表
        """
        # 1. 构建基础查询
        query = db.query(SysUser)

        # 2. 添加过滤条件
        filters = []

        if query_params.username:
            # 模糊查询：LIKE %keyword%
            filters.append(SysUser.username.like(f"%{query_params.username}%"))

        if query_params.real_name:
            filters.append(SysUser.real_name.like(f"%{query_params.real_name}%"))

        if query_params.email:
            filters.append(SysUser.email.like(f"%{query_params.email}%"))

        if query_params.phone:
            filters.append(SysUser.phone.like(f"%{query_params.phone}%"))

        if query_params.is_active is not None:
            filters.append(SysUser.is_active == query_params.is_active)

        if query_params.gender:
            filters.append(SysUser.gender == query_params.gender)

        # 应用所有过滤条件
        if filters:
            query = query.filter(and_(*filters))

        # 3. 获取总记录数
        total = query.count()

        # 4. 分页查询
        offset = (query_params.page - 1) * query_params.page_size
        users = query.order_by(SysUser.created_at.desc()).offset(offset).limit(query_params.page_size).all()

        # 5. 构建响应
        return UserListResponse(
            items=[UserResponse.model_validate(user) for user in users],
            total=total,
            page=query_params.page,
            page_size=query_params.page_size
        )

    @staticmethod
    def update_user(db: Session, user_id: int, user_data: UserUpdateRequest) -> UserResponse:
        """
        更新用户信息

        Args:
            db: 数据库会话
            user_id: 用户ID
            user_data: 更新数据

        Returns:
            UserResponse: 更新后的用户信息

        Raises:
            ResourceNotFoundException: 用户不存在
            DuplicateResourceException: 唯一字段冲突
        """
        try:
            # 1. 查找用户
            user = db.query(SysUser).filter(SysUser.id == user_id).first()
            if not user:
                raise ResourceNotFoundException(resource="用户", resource_id=user_id)

            # 2. 检查唯一性约束（排除自身）
            if user_data.email or user_data.phone:
                UserService._check_unique_fields(
                    db,
                    username=user.username,  # 用户名不可修改
                    email=user_data.email,
                    phone=user_data.phone,
                    exclude_id=user_id  # 排除当前用户
                )

            # 3. 更新字段（只更新提供的字段）
            update_data = user_data.model_dump(exclude_unset=True)  # exclude_unset=True 只包含明确设置的字段

            for field, value in update_data.items():
                if value is not None:  # 只更新非None的字段
                    setattr(user, field, value)

            # 4. 保存到数据库
            db.commit()
            db.refresh(user)

            return UserResponse.model_validate(user)

        except (ResourceNotFoundException, DuplicateResourceException):
            db.rollback()
            raise
        except Exception as e:
            db.rollback()
            raise DatabaseException(f"更新用户失败: {str(e)}")

    @staticmethod
    def delete_user(db: Session, user_id: int) -> bool:
        """
        删除用户

        Args:
            db: 数据库会话
            user_id: 用户ID

        Returns:
            bool: 是否删除成功

        Raises:
            ResourceNotFoundException: 用户不存在
        """
        try:
            # 1. 查找用户
            user = db.query(SysUser).filter(SysUser.id == user_id).first()
            if not user:
                raise ResourceNotFoundException(resource="用户", resource_id=user_id)

            # 2. 删除用户
            db.delete(user)
            db.commit()

            return True

        except ResourceNotFoundException:
            db.rollback()
            raise
        except Exception as e:
            db.rollback()
            raise DatabaseException(f"删除用户失败: {str(e)}")

    @staticmethod
    def update_last_login(db: Session, user_id: int):
        """
        更新用户最后登录时间

        Args:
            db: 数据库会话
            user_id: 用户ID
        """
        user = db.query(SysUser).filter(SysUser.id == user_id).first()
        if user:
            user.last_login_at = datetime.now()
            db.commit()