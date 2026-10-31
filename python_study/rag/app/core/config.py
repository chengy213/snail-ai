"""
应用配置模块

负责从 .env 文件和系统环境变量中读取配置参数。
使用 pydantic-settings 进行配置管理和验证。
"""
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    应用配置类

    从 .env 文件和环境变量中读取配置参数。
    配置优先级：系统环境变量 > .env 文件 > 默认值
    """

    # ===== 大模型 API 配置 =====

    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY")  # 阿里云百炼API密钥
    DASHSCOPE_BASE_URL: str = ""  # API基础URL（可选）
    DASHSCOPE_MODEL: str  # 使用的模型名称
    EMBEDDINGS_MODEL_NAME: str  # 向量化模型名称


    # ===== Milvus 向量数据库配置 =====

    MILVUS_HOST: str  # Milvus服务地址
    MILVUS_PORT: int  # Milvus服务端口
    COLLECTION_NAME: str  # 集合名称


    # ===== 文档处理配置 =====

    UPLOAD_DIR: str = "uploads"  # 上传文件存储目录
    CHUNK_SIZE: int = 500  # 文档分块大小（字符数）
    CHUNK_OVERLAP: int = 50  # 分块重叠大小
    TOP_K: int = 3  # 检索返回的最相关结果数
    SEPARATORS: list[str] = ["\n\n", "\n", "。", ".", "?", "？", "!", "", "  "]  # 文本分割符


    # ===== Pydantic Settings 配置 =====

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent.parent / ".env",  # .env文件路径
        case_sensitive=True,  # 环境变量名区分大小写
        env_file_encoding="utf-8"  # 文件编码
    )


# 创建配置单例实例
settings = Settings()