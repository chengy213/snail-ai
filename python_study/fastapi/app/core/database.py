"""
数据库连接管理模块
使用 SQLAlchemy 2.0 风格管理数据库连接和会话
"""
from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from typing import Generator
from .config import settings


# 创建数据库引擎
# echo=True 会在控制台输出SQL语句，生产环境应设为False
# pool_size 设置连接池大小
# max_overflow 设置超出pool_size后可创建的最大连接数
# pool_pre_ping 每次从连接池获取连接前检查连接是否有效
engine = create_engine(
    settings.database_url,
    echo=settings.DEBUG,  # 调试模式下打印SQL
    poolclass=QueuePool,
    pool_size=10,  # 连接池保持的连接数
    max_overflow=20,  # 超出pool_size后最多创建的连接数
    pool_pre_ping=True,  # 连接前检测连接是否有效
    pool_recycle=3600  # 连接回收时间（秒）
)

# 创建会话工厂
# autocommit=False 关闭自动提交，需要手动commit
# autoflush=False 关闭自动刷新，手动控制刷新时机
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# 创建声明式基类，所有模型需要继承此类
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    获取数据库会话的依赖函数
    使用生成器确保会话在使用后正确关闭

    使用方式：
    @app.get("/")
    def read_root(db: Session = Depends(get_db)):
        ...
    """
    db = SessionLocal()  # 创建会话实例
    try:
        yield db  # 将会话传递给调用方
    finally:
        db.close()  # 确保会话在请求结束后关闭

