"""
日志配置模块
使用 loguru 库实现统一的日志管理
"""
import sys
from pathlib import Path
from loguru import logger

# 移除 loguru 默认的控制台输出配置
# 因为我们要自定义输出格式和级别
logger.remove()

# ==========================================
# 控制台输出配置（开发环境）
# ==========================================
logger.add(
    sys.stderr,  # 输出到标准错误流（控制台）
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="DEBUG",      # 输出 DEBUG 及以上级别的日志
    colorize=True       # 启用彩色输出，便于区分日志级别
)
# 格式说明：
# - <green>...</green> : 绿色显示时间
# - {time:HH:mm:ss}    : 时间格式为 时:分:秒
# - {level: <8}        : 日志级别，左对齐，宽度8个字符
# - <level>...</level> : 根据日志级别显示不同颜色
# - {message}          : 日志消息内容

# ==========================================
# 常规日志文件配置（生产环境）
# ==========================================
logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",  # 日志文件路径，按日期命名
    rotation="10 MB",      # 文件大小达到 10MB 时自动切割，创建新文件
    retention="30 days",   # 日志文件保留 30 天，过期自动删除
    compression="zip",     # 切割后的旧日志压缩为 zip 格式，节省磁盘空间
    level="INFO",          # 只记录 INFO 及以上级别的日志（不记录 DEBUG）
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
)
# 格式说明：
# - {time:YYYY-MM-DD HH:mm:ss} : 完整时间格式（年-月-日 时:分:秒）
# - {level}                     : 日志级别
# - {name}:{function}:{line}    : 记录哪个文件的哪个函数的第几行
# - {message}                   : 日志消息

# ==========================================
# 错误日志单独文件配置
# ==========================================
logger.add(
    "logs/error_{time:YYYY-MM-DD}.log",  # 错误日志专用文件，按日期命名
    rotation="10 MB",      # 文件大小达到 10MB 时切割
    retention="90 days",   # 错误日志保留更长时间（90天）
    level="ERROR",         # 只记录 ERROR 及以上级别的日志
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message} | {exception}"
)
# 格式说明：
# - {exception} : 记录完整的异常堆栈信息（如果存在）