"""
用户数据模型模块
使用 SQLAlchemy ORM 定义数据库表结构
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Enum as SQLEnum
from sqlalchemy.sql import func
from python_study.fastapi.app.core.database import Base

import enum


class GenderEnum(str, enum.Enum):
    """性别枚举类"""
    MALE = "男"  # 男性
    FEMALE = "女"  # 女性
    UNKNOWN = "未知"  # 未知


class SysUser(Base):
    """
    系统用户表模型
    映射到数据库 sys_user 表
    """
    __tablename__ = "sys_user"  # 指定表名

    # 主键ID，自增整数类型
    id = Column(Integer, primary_key=True, autoincrement=True, comment="主键ID")

    # 登录账号，唯一索引，用于登录
    username = Column(String(50), unique=True, nullable=False, index=True, comment="登录账号")

    # 登录密码，存储哈希后的密码
    password = Column(String(255), nullable=False, comment="登录密码(哈希存储)")

    # 真实姓名
    real_name = Column(String(50), nullable=False, comment="真实姓名")

    # 性别，使用枚举类型
    gender = Column(
        SQLEnum(GenderEnum, values_callable=lambda x: [e.value for e in x]),
        default=GenderEnum.UNKNOWN,
        comment="性别"
    )

    # 电子邮箱，唯一索引
    email = Column(String(100), unique=True, nullable=True, index=True, comment="电子邮箱")

    # 电话号码，唯一索引
    phone = Column(String(20), unique=True, nullable=True, index=True, comment="电话号码")

    # 是否激活，布尔类型，默认True
    is_active = Column(Boolean, default=True, comment="是否激活")

    # 备注，长文本类型
    remark = Column(Text, nullable=True, comment="备注")

    # 创建时间，自动填充当前时间
    created_at = Column(
        DateTime,
        server_default=func.now(),  # 数据库层面的默认值
        comment="创建时间"
    )

    # 更新时间，自动更新
    updated_at = Column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),  # 每次更新时自动更新为当前时间
        comment="更新时间"
    )

    # 最后登录时间
    last_login_at = Column(DateTime, nullable=True, comment="最后登录时间")

    def __repr__(self):
        """对象的字符串表示，方便调试"""
        return f"<SysUser(id={self.id}, username='{self.username}', real_name='{self.real_name}')>"