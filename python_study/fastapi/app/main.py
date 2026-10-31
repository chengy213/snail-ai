"""
FastAPI 应用主入口
包含应用初始化、全局异常处理、路由注册等
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# 以仓库根目录为导入起点，支持 `python app/main.py` 和 `uvicorn python_study.fastapi.app.main:app` 两种方式
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from loguru import logger
import os
import traceback

from python_study.fastapi.app.core.config import settings
from python_study.fastapi.app.core.database import Base, engine
from python_study.fastapi.app.core.exceptions import AppBaseException
from python_study.fastapi.app.utils.response import error_response

# 导入路由
from python_study.fastapi.app.api.v1.user import router as user_router
from python_study.fastapi.app.models.user import SysUser # 必须引入，不然自动创建表不成功

# ==================== 启动事件 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # === 启动逻辑 ===
    logger.debug("正在初始化数据库...")
    init_db()
    logger.debug("数据库初始化成功")
    yield
    # === 关闭逻辑 ===
    logger.debug("应用关闭")


def init_db():
    """
    初始化数据库表结构
    根据模型定义创建所有表（如果不存在）
    注意：不会修改已存在的表结构
    """
    try:
        logger.debug("正在连接数据库...")
        logger.debug(f"Base 中注册的表: {list(Base.metadata.tables.keys())}")

        with engine.begin() as conn:
            logger.debug("连接成功，开始创建表...")
            Base.metadata.create_all(conn)
            logger.debug("✅ 数据库表创建完成")

            # 验证表是否创建成功
            logger.debug("\n验证创建的表...")
            result = conn.execute(text("SHOW TABLES;"))
            tables = result.fetchall()
            logger.debug("已创建的表:", [t[0] for t in tables])

    except Exception as e:
        logger.error(f"❌ 数据库初始化失败: {type(e).__name__}: {e}")
        logger.opt(exception=True).error("数据库异常详情")  # 记录完整堆栈
        raise  # 启动阶段失败应直接终止应用，避免"假启动"后请求持续报错
    finally:
        engine.dispose()
        logger.debug("数据库连接已关闭")

# 创建 FastAPI 应用实例
app = FastAPI(
    title=settings.APP_NAME,  # API文档标题
    version=settings.APP_VERSION,  # API版本
    description="FastAPI 用户管理 CRUD 示例项目",  # API描述
    docs_url="/docs",  # Swagger文档路径
    redoc_url="/redoc",  # ReDoc文档路径  # ← 这里需要添加逗号
    lifespan=lifespan
)

# 配置CORS中间件（跨域资源共享）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境应限制
    allow_credentials=True,  # 允许携带凭证
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)


# ==================== 全局异常处理器 ====================

@app.exception_handler(AppBaseException)
async def app_exception_handler(request: Request, exc: AppBaseException):
    """
    处理自定义业务异常

    当业务逻辑层抛出 AppBaseException 及其子类异常时，
    会由此处理器统一处理，返回标准格式的错误响应
    """
    logger.warning(f"业务异常: {exc.message}")

    return JSONResponse(
        status_code=exc.status_code,  # HTTP状态码
        content=error_response(
            code=exc.status_code,
            message=exc.message,
            data=exc.data
        )
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    处理请求参数验证异常（Pydantic验证失败）

    当请求参数不符合 Pydantic 模型定义时，
    FastAPI 会抛出 RequestValidationError
    """
    # 提取第一个验证错误信息
    errors = exc.errors()
    if errors:
        # 构建友好的错误信息
        error_msg = errors[0].get("msg", "参数验证失败")
        field = errors[0].get("loc", ["unknown"])[-1]  # 获取字段名
        message = f"参数验证失败: [{field}] {error_msg}"
    else:
        message = "请求参数验证失败"

    logger.warning(f"参数验证失败: {message}")

    return JSONResponse(
        status_code=422,
        content=error_response(
            code=422,
            message=message,
            data={"errors": errors}  # 返回详细错误信息
        )
    )


@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    """
    处理数据库异常

    捕获 SQLAlchemy 抛出的数据库相关异常
    """
    logger.error(f"数据库异常: {str(exc)}")
    logger.error(traceback.format_exc())

    return JSONResponse(
        status_code=500,
        content=error_response(
            code=500,
            message="数据库操作失败，请稍后重试",
            data=None
        )
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    全局异常处理器（兜底）

    捕获所有未被其他处理器处理的异常
    """
    logger.error(f"未处理的异常: {str(exc)}")
    logger.error(traceback.format_exc())

    return JSONResponse(
        status_code=500,
        content=error_response(
            code=500,
            message="服务器内部错误",
            data=None
        )
    )


# ==================== 路由注册 ====================

# 注册用户路由，添加API版本前缀
app.include_router(user_router, prefix="/api/v1")

# 挂载静态文件目录（前端页面）
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")


# ==================== 健康检查 ====================

@app.get("/", tags=["系统"])
async def root():
    """根路径，返回应用基本信息"""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "frontend": "/static/index.html"
    }


@app.get("/health", tags=["系统"])
async def health_check():
    """
    健康检查接口
    用于监控系统状态
    """
    return {"status": "healthy", "message": "服务运行正常"}


# 应用入口
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "python_study.fastapi.app.main:app",
        host="0.0.0.0",  # 监听所有网络接口
        port=8000,  # 端口号
        reload=settings.DEBUG  # 调试模式下启用热重载
    )