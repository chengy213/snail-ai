"""
API 路由模块

基于 FastAPI 搭建，封装了项目所有对外接口。
通过 Pydantic 规范统一请求和响应数据格式，保证接口严谨、自动生成 API 文档。

提供的接口：
    POST /api/v1/upload  — 上传文档（PDF/TXT/MD/DOCX），自动切分并写入向量库
    POST /api/v1/ask     — 智能问答，混合检索 + 重排序后由千问生成答案

依赖的服务（通过依赖注入获取）：
    - DocumentService  : 文档解析与切分
    - VectorStoreManager: Milvus 向量数据库操作
    - RetrievalService : 检索与上下文构建
    - LLMService       : 千问大模型生成
"""
import hashlib
import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel, Field

from python_study.rag.app.core.config import settings
from python_study.rag.app.core.dependencies import (
    get_document_service,
    get_vector_store,
    get_retrieval_service,
    get_llm_service
)

# 创建路由器实例，后续所有接口通过 @router 装饰器注册
router = APIRouter()


# ===== 请求/响应模型 =====

class QuestionRequest(BaseModel):
    """
    问答请求模型

    字段说明：
        question: 用户输入的自然语言问题
        top_k   : 期望从向量库中检索多少条相关文档片段（1-10）
                  值越大召回越多，但可能引入噪音；越小越精准，但可能遗漏
    """
    question: str = Field(..., description="用户问题")
    top_k: int = Field(default=3, ge=1, le=10, description="检索数量")


class ReferenceItem(BaseModel):
    """
    引用来源条目

    每个条目对应检索到的一段文档片段，用于前端展示"答案来自哪里"。
    字段说明：
        source     : 来源文件名（如 "技术手册.pdf"）
        score      : 相似度/相关度分数，分数越高越相关
        chunk_index: 该片段在原文档中的位置序号（第几段）
        doc_id     : 文档内容的 SHA256 哈希，用于去重和溯源
    """
    source: str = Field(..., description="来源文件名")
    score: float = Field(..., description="相似度分数")
    chunk_index: int = Field(default=0, description="文档分块索引")
    doc_id: str = Field(default="", description="文档哈希ID")


class QuestionResponse(BaseModel):
    """
    问答响应模型

    字段说明：
        answer    : 千问大模型基于检索到的文档生成的答案
        references: 答案引用了哪些文档片段，方便用户溯源验证
    """
    answer: str = Field(..., description="AI生成的答案")
    references: list[ReferenceItem] = Field(default_factory=list, description="引用来源列表")


class UploadResponse(BaseModel):
    """
    上传响应模型

    字段说明：
        message : 处理结果描述（成功/失败/已存在等）
        filename: 上传的原始文件名
        chunks  : 文档被切分成多少段（0 表示未处理，如文档已存在）
        file_size: 文件大小（字节）
    """
    message: str = Field(..., description="处理结果消息")
    filename: str = Field(..., description="文件名")
    chunks: int = Field(default=0, description="文档切分块数")
    file_size: int = Field(default=0, description="文件大小(字节)")


# ===== 接口 =====

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档并写入向量数据库

    完整的文档入库流水线（安全校验 → 去重 → 切分 → 向量化 → 入库）：
        Step 1: 安全保存文件到本地（清洗文件名，防范路径穿越攻击）
        Step 2: 计算 SHA256 内容哈希，检查是否已入库（避免重复向量化）
        Step 3: 删除同名旧版本（支持文档更新覆盖）
        Step 4: 解析文档（支持 PDF/TXT/MD/DOCX），智能切分成文本块
        Step 5: 将文本块批量向量化并写入 Milvus 向量数据库

    支持的文件格式：.pdf / .txt / .md / .docx
    去重逻辑：相同内容（SHA256一致）只存一份，content 为空时自动跳过
    """
    # =================================================================
    # Step 1: 安全保存文件到本地磁盘
    # =================================================================
    # Path(...).name 只取文件名部分，丢弃路径信息，防止路径穿越攻击
    # 例如：../../etc/passwd → passwd
    safe_filename = Path(file.filename).name
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(settings.UPLOAD_DIR, safe_filename)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # =================================================================
    # Step 2: 计算内容哈希，检查是否已入库（去重）
    # =================================================================
    # SHA256 对文件内容做哈希，相同内容产生相同的哈希值
    # 用于实现"相同文档只存一份"的去重逻辑
    doc_id = hashlib.sha256(content).hexdigest()

    vector_store = get_vector_store()

    # 如果已存在相同内容的文档，直接返回，避免重复向量化
    if vector_store.check_doc_exists(doc_id):
        return UploadResponse(
            message="文档已存在，你可以直接提问",
            filename=file.filename,
            chunks=0,
            file_size=0
        )

    # =================================================================
    # Step 3: 删除同名旧版本（支持文档更新覆盖）
    # =================================================================
    # 即使内容不同，同名文件也应清理旧数据，避免检索时出现多版本混淆
    vector_store.delete_by_file_name(file.filename)

    # =================================================================
    # Step 4: 解析文档内容，智能切分成文本块
    # =================================================================
    # DocumentService 根据文件扩展名自动选择加载器（PDF/TXT/MD/DOCX），
    # 然后按配置的 chunk_size 和 chunk_overlap 切分成多个文本片段
    doc_service = get_document_service()
    texts, metadata_list = await doc_service.process_file(
        file_path,
        file.filename,
        doc_id  # 将内容哈希注入每个文本块的元数据，方便后续溯源
    )

    # =================================================================
    # Step 5: 批量向量化并写入 Milvus 向量数据库
    # =================================================================
    # add_documents 内部会调用 Embedding 服务将文本转成向量，
    # 然后批量插入 Milvus 集合，并等待索引构建完成
    vector_store.add_documents(texts, metadata_list)

    return UploadResponse(
        message="文档上传成功",
        filename=file.filename,
        chunks=len(texts),
        file_size=len(content)
    )


@router.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    智能问答接口

    完整的 RAG 问答流水线（检索 → 增强 → 生成）：
        Step 1 — 检索（Retrieve）：从向量库中查找与问题最相关的文档片段
                使用"混合检索 + 重排序"策略获得最佳效果
        Step 2 — 增强（Augment）：将检索到的文档片段拼接成结构化提示上下文
        Step 3 — 生成（Generate）：将上下文交给千问大模型，生成基于文档的答案

    检索策略可选方案（当前使用方式3）：
        方式1: 纯向量检索              — 最简单，只有语义匹配
        方式2: 混合检索（向量+BM25）    — 语义 + 关键词，覆盖面更广
        方式3: 混合检索 + 重排序        — 最优方案，二次精排提高精准度 ← 当前使用
    """
    # =================================================================
    # Step 1: 检索相关文档片段
    # =================================================================
    retrieval_service = get_retrieval_service()

    #  方式3：混合检索 + 重排序（当前效果最好的组合）
    #  - search_type="hybrid": 向量语义(权重0.7) + BM25关键词(权重0.3) 加权融合
    #  - use_rerank=True     : 召回后由千问重排序模型二次精排，提高Top-K精准度
    retrieved = retrieval_service.retrieve(
        request.question,
        request.top_k,
        search_type="hybrid",
        use_rerank=True
    )
    # 备选方案（可按需切换）：
    # # 方式1：纯向量检索 — 简单场景
    # retrieved = retrieval_service.retrieve(request.question, request.top_k)
    # # 方式2：混合检索无重排序 — 速度快但精度略低
    # retrieved = retrieval_service.retrieve(
    #     request.question, request.top_k, search_type="hybrid"
    # )

    # =================================================================
    # Step 2: 构建上下文（将检索结果格式化为 LLM 可理解的提示文本）
    # =================================================================
    context = retrieval_service.build_context(
        request.question,
        request.top_k,
        results=retrieved
    )

    # =================================================================
    # Step 3: 调用千问大模型生成答案
    # =================================================================
    # LLM 会严格基于 context 中的文档内容回答，不会编造信息
    # 如果文档中没有相关信息，会明确告知用户"无法回答"
    llm_service = get_llm_service()
    answer = llm_service.generate(request.question, context)

    # =================================================================
    # Step 4: 构建引用来源列表，供前端展示"答案依据"
    # =================================================================
    references = [
        ReferenceItem(
            source=item["source"],
            score=item["score"],
            chunk_index=item.get("chunk_index", 0),  # 文档片段序号，取代废弃的 page 字段
            doc_id=item.get("doc_id", ""),           # 文档内容哈希，用于溯源去重
        )
        for item in retrieved
    ]

    return QuestionResponse(answer=answer, references=references)