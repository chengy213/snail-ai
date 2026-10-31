"""
依赖注入模块

管理应用中所有服务的单例实例，提供统一的依赖获取接口。
使用 FastAPI 的依赖注入机制，确保服务实例在整个应用生命周期中唯一。
"""
from python_study.rag.app.repositories.vector_store import VectorStoreManager
from python_study.rag.app.services.context_manager import ContextManager
from python_study.rag.app.services.document_service import DocumentService
from python_study.rag.app.services.embedding_service import EmbeddingService
from python_study.rag.app.services.llm_service import LLMService
from python_study.rag.app.services.retrieval_service import RetrievalService

# ===== 单例实例 =====

# 向量存储管理器单例（管理 Milvus 数据库连接和操作）
_vector_store: VectorStoreManager = None

# 向量化服务单例（负责将文本转换为向量嵌入）
_embedding_service: EmbeddingService = None

# 检索服务单例（负责语义检索和上下文构建）
_retrieval_service: RetrievalService = None


# ===== 服务获取函数 =====

def get_vector_store() -> VectorStoreManager:
    """
    获取向量数据库管理器实例（单例模式）

    负责 Milvus 数据库的连接、集合初始化和向量检索操作。

    返回:
        VectorStoreManager: 向量数据库管理器实例
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreManager()
    return _vector_store


def get_embedding_service() -> EmbeddingService:
    """
    获取 Embedding 服务实例（单例模式）

    负责调用 DashScope 嵌入模型将文本转换为向量。

    返回:
        EmbeddingService: 向量化服务实例
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def get_retrieval_service() -> RetrievalService:
    """
    获取检索服务实例（单例模式）

    负责语义检索和上下文构建。

    返回:
        RetrievalService: 检索服务实例
    """
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService(get_vector_store())
    return _retrieval_service


def get_llm_service() -> LLMService:
    """
    获取大模型服务实例

    负责调用通义千问 API 生成答案。

    返回:
        LLMService: 大模型服务实例
    """
    return LLMService()


def get_document_service() -> DocumentService:
    """
    获取文档处理服务实例

    负责文档的解析、清洗和智能切分。

    返回:
        DocumentService: 文档处理服务实例
    """
    return DocumentService()


def get_context_manager() -> ContextManager:
    return ContextManager()

# ===== 应用生命周期函数 =====

async def init_vector_store():
    """
    应用启动时初始化向量数据库

    执行流程：
    1. 获取向量数据库管理器实例
    2. 获取 Embedding 服务实例
    3. 使用嵌入函数初始化 Milvus 集合

    注意：
        embedding_service.embedding 必须是 DashScopeEmbeddings 实例，
        即具有 embed_documents 和 embed_query 方法的对象。
    """
    vector_store = get_vector_store()
    embedding_service = get_embedding_service()

    # 使用嵌入函数初始化集合
    # embedding_service.embedding 是 DashScopeEmbeddings 实例
    vector_store.init_collection(embedding_service.embedding)

    print("✅ 向量数据库初始化完成")