"""
FastAPI 后端入口 —— 提供 /chat 流式接口。

启动方式:
    python main.py
    或
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

接口:
    POST /chat
        Body: { "message": "你好" }
        Response: SSE 流式返回文本片段
"""

import asyncio
import os
import sys
from pathlib import Path

# 以 backend 目录为导入根：保证 utils / model / graph 等扁平导入在任何工作目录下
# 都解析到本目录，避免被 site-packages 中的同名 utils 包（utils 1.0.2）遮蔽
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from service import ChatService

app = FastAPI(title="Snail AI Chat")

# 允许跨域（H5 前端可能不在同一端口）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_service = ChatService()


class ChatRequest(BaseModel):
    message: str


async def sse_generator(message: str):
    """将 ChatService 的 yield 片段包装为 SSE 格式。"""
    last_chunk = ""
    async for chunk in chat_service.chat_stream(message):
        last_chunk = chunk
        # SSE 格式: data: <内容>\n\n
        yield f"data: {chunk}\n\n"
        await asyncio.sleep(0)  # 交出控制权，确保即时发送
    # 发送结束标记
    yield "data: [DONE]\n\n"


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    流式聊天接口（POST），返回 SSE 流。

    前端连接方式见 frontend/index.html。
    """
    return StreamingResponse(
        sse_generator(req.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        },
    )


# ---- 托管前端页面 ----
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


@app.get("/")
async def root():
    """直接访问 http://localhost:8100 即可看到聊天页面"""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
