"""
应用配置管理模块
使用 Pydantic Settings 管理应用配置，支持从环境变量和 .env 文件读取

Q：为什么 .env 和 config.py 各有一份数据库配置？
它们不是重复，而是“配置的两个层次”，这和 Java 里 Spring Boot 的模式完全对应：

层次	                        Python	                                                对应 Java (Spring Boot)
配置的结构定义 + 兜底默认值	    config.py 的字段声明(DB_HOST: str = "localhost")	        @ConfigurationProperties 类
当前环境的实际值	            .env 文件	                                            application.yml / 环境变量

注意：config.py 只声明有哪些配置项、什么类型、没给值时默认是什么代码里统一通过 settings.DB_HOST 访问。而 .env 提供这台机器/这个环境的真实值，启动时覆盖默认值。
优先级(pydantic-settings 的规则，从高到低)：真实环境变量  >  .env 文件  >  config.py 中的默认值

这样设计的价值：

1.换环境只改 .env(或注入环境变量)，代码零改动 —— 开发/测试/生产各自一份 .env
2.生产部署(Docker/K8s)时密码通过环境变量注入(-e DB_PASSWORD=xxx),优先级最高，连 .env 都不用带
3.对你这位 Java 开发者来说，这就是熟悉的 profile 思路
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """应用配置类，继承 BaseSettings 自动读取环境变量"""

    # 应用基本信息
    APP_NAME: str = "User Manage"  # 应用名称
    APP_VERSION: str = "1.0.0"  # 应用版本
    DEBUG: bool = True  # 调试模式开关

    # 数据库配置
    DB_HOST: str = "localhost"  # 数据库主机地址
    DB_PORT: int = 3306  # 数据库端口
    DB_USER: str = "root"  # 数据库用户名
    DB_PASSWORD: str = "password"  # 数据库密码
    DB_NAME: str = "demo"  # 数据库名称

    @property
    def database_url(self) -> str:
        """
        构建数据库连接URL
        使用 @property 装饰器使其可以像属性一样访问
        """
        return f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?charset=utf8mb4"

    # 指定 .env 文件路径
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent.parent / ".env",  # 向上两级到根目录寻找 .env 文件
        case_sensitive=True,  # 区分大小写，环境变量名必须与字段名完全一致
        env_file_encoding="utf-8"  # 使用 UTF-8 编码读取 .env 文件
    )


# 创建全局配置实例，供其他模块使用
settings = Settings()
