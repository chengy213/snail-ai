"""
检索服务模块

封装检索和上下文构建逻辑，是 RAG 流水线中"Retrieve + Augment"环节的服务层。

职责：
    - retrieve()     : 调用向量库执行检索（支持多种检索策略 + 重排序）
    - build_context(): 将检索结果格式化为 LLM 可消费的提示文本（Prompt上下文）

重排序逻辑已内置在 VectorStoreManager.search() 中，
本层专注结果格式化和上下文拼接，保持职责清晰。
"""
from python_study.rag.app.core.config import settings
from python_study.rag.app.repositories.vector_store import VectorStoreManager


class RetrievalService:
    """
    检索服务

    作为 VectorStoreManager 的上层封装，提供：
        1. 统一的检索接口（屏蔽底层检索策略的复杂性）
        2. 结果格式化（将 Milvus 原始结果转为标准化的字典列表）
        3. 上下文构建（将检索结果拼接为 LLM 的提示文本）

    使用方式:
        service = RetrievalService(vector_store)
        results = service.retrieve("问题", top_k=3, search_type="hybrid", use_rerank=True)
        context = service.build_context("问题", results=results)
    """

    def __init__(self, vector_store: VectorStoreManager):
        """
        初始化检索服务

        参数:
            vector_store: 向量数据库管理器实例（由依赖注入提供，确保单例）
        """
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        search_type: str = "hybrid",
        use_rerank: bool = False
    ) -> list:
        """
        检索与查询最相关的文档片段

        通用的检索入口，封装了检索策略选择和结果格式化两个步骤。

        参数:
            query      : 用户问题文本
            top_k      : 期望返回的文档片段数量（默认从配置读取）
            search_type: 检索策略
                - "vector" : 纯向量语义检索（理解意思，适合中文）
                - "bm25"   : BM25 关键词检索（精确匹配，适合英文和技术术语）
                - "hybrid" : 混合检索（向量0.7 + BM250.3），推荐使用
            use_rerank : 是否启用千问重排序模型进行二次精排
                - True  : 先多召回（3x top_k），再精筛回 top_k，质量最高
                - False : 直接返回向量库排序结果，速度更快

        返回:
            list[dict]: 标准化格式的检索结果列表，每项包含：
                - content    : 文档片段文本内容
                - score      : 相关度分数（已处理为4位小数）
                - source     : 来源文件名
                - chunk_index: 在文档中的片段序号
                - doc_id     : 文档内容哈希
        """
        if top_k is None:
            top_k = settings.TOP_K

        # 调用 VectorStoreManager 执行实际的向量检索
        # search() 内部已根据 search_type 选择策略，并根据 use_rerank 决定是否重排序
        results = self.vector_store.search(
            query=query,
            top_k=top_k,
            search_type=search_type,
            use_rerank=use_rerank
        )

        # ================================================================
        # 格式化输出：将 Milvus 原始结果转为标准化的字典列表
        # ================================================================
        # 分数优先级：
        #   1. rerank_score  — 重排序分数（最精确）
        #   2. hybrid_score  — 混合检索分数（向量+BM25加权）
        #   3. score         — 原始相似度分数（纯向量或纯BM25）
        formatted = []
        for result in results:
            score = result.get("rerank_score") or result.get("hybrid_score") or result.get("score", 0)
            formatted.append({
                "content": result.get("text", ""),
                "score": round(score, 4),                   # 保留4位小数，前端展示友好
                "source": result.get("file_name", "未知"),
                "chunk_index": result.get("chunk_index", 0),
                "doc_id": result.get("doc_id", ""),
            })

        return formatted

    def build_context(
        self,
        query: str,
        top_k: int = None,
        results: list = None
    ) -> str:
        """
        将检索结果构建为 LLM 的提示上下文（Prompt Context）

        这是 RAG 中"Augment"（增强）环节的实现：
        将检索到的多个文档片段按编号、来源、相关度结构化拼接，
        形成一段 LLM 可以直接理解和引用的提示文本。

        参数:
            query  : 用户问题（预留参数，当前未使用，方便后续扩展）
            top_k  : 检索数量（预留参数，当前未使用）
            results: 检索结果列表（由 retrieve() 返回的标准化格式）

        返回:
            str: 拼接后的上下文字符串，格式如下：
                [1] (来源: xxx.pdf, 相关度: 0.9523)
                文档片段内容...

                [2] (来源: yyy.md, 相关度: 0.8731)
                文档片段内容...

            如果没有检索到任何结果，返回 "未找到相关文档。"
        """
        if not results:
            return "未找到相关文档。"

        # 逐条拼接：编号 + 来源 + 相关度 + 内容
        parts = []
        for i, item in enumerate(results):
            parts.append(
                f"[{i + 1}] (来源: {item['source']}, "
                f"相关度: {item['score']})\n"
                f"{item['content']}"
            )

        # 用空行分隔每条结果
        return "\n\n".join(parts)