"""
FastAPI 应用入口

RAG 问答系统的主入口文件，负责：
- 应用生命周期管理
- FastAPI 应用初始化和配置
- 路由注册
- 服务启动
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Windows 控制台默认使用 GBK 编码，无法输出 emoji，
# 将 stdout 重配置为 UTF-8 避免 UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8")

# 以仓库根目录为导入起点，支持 `python app/main.py` 和 `uvicorn python_study.rag.app.main:app` 两种方式
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from python_study.rag.app.api.routes import router
from python_study.rag.app.core.dependencies import init_vector_store


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    应用生命周期管理器

    使用 FastAPI 的生命周期功能，在应用启动和关闭时执行特定操作。

    执行顺序：
    1. 应用启动时：初始化向量数据库连接
    2. 应用运行时：处理正常的 API 请求
    3. 应用关闭时：执行清理操作

    参数:
        _app: FastAPI 应用实例

    示例:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # 启动时执行
            await startup()
            yield
            # 关闭时执行
            await shutdown()

    注意:
        - 使用 async 支持异步操作
        - yield 之前的代码在启动时执行
        - yield 之后的代码在关闭时执行
        - yield 本身表示应用运行期间
    """
    # ===== 应用启动阶段 =====
    print("🚀 正在初始化向量数据库...")
    # 初始化 Milvus 向量数据库连接和集合
    # 确保数据库可用后再处理请求
    await init_vector_store()
    print("✅ 千问 RAG 服务已启动")

    # yield 表示应用运行中
    yield

    # ===== 应用关闭阶段 =====
    print("🛑 应用关闭")


# 创建 FastAPI 应用实例
app = FastAPI(
    title="RAG 问答系统（千问版）",
    description="基于通义千问 + Milvus 的文档问答系统",
    version="1.0.0",
    lifespan=lifespan
)

# ============================================================
# 注册 API 路由
# ============================================================
# 将 routes.py 中定义的所有接口注册到应用上，
# prefix="/api/v1" 表示所有接口地址前都加上 /api/v1 前缀
# 例如：原本定义的 /upload → 实际访问 /api/v1/upload
#
# 最终可用的 API 端点：
#   POST /api/v1/upload  - 上传文档并入库
#   POST /api/v1/ask     - 基于文档的智能问答
#   GET  /api/v1/docs    - 自动生成的 Swagger UI 文档
#   GET  /api/v1/redoc   - 自动生成的 ReDoc 文档
#   GET  /               - 静态前端页面
app.include_router(router, prefix="/api/v1")

# 挂载静态文件目录（前端页面），访问根路径 http://host:port/ 即可看到前端界面
# html=True 表示如果请求的是目录，自动寻找 index.html
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)  # 目录不存在则创建
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


# ============================================================
# 直接运行入口
# ============================================================
# 当执行 `python main.py` 时启动服务（而非通过 uvicorn 命令行）
# host="0.0.0.0" 表示监听所有网络接口，允许外部访问
# port=8100      服务端口号
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)