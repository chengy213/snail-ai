"""
重排序服务模块 - 基于 DashScope

使用重排序模型（qwen3-rerank）对检索结果进行精排序。

模型说明：
- qwen3-rerank：官方重排序模型，中英文效果好
- 通过 DashScope API 调用，无需本地GPU
- 每次调用最多支持 50 个文档

参考文档：
https://help.aliyun.com/zh/model-studio/getting-started/models
"""
import logging
from typing import List, Dict, Any
from http import HTTPStatus
from dashscope import TextReRank

from python_study.rag.app.core.config import settings

logger = logging.getLogger(__name__)


class RerankerService:
    """
    重排序服务
    
    使用 DashScope TextReRank API 对检索结果进行精排序。
    模型：qwen3-rerank（重排序模型）
    
    优势：
    - API 调用，无需本地GPU
    - 中英文效果优秀
    - 与 DashScope 嵌入模型配套使用
    
    使用示例：
        reranker = RerankerService()
        results = vector_store.search("查询", top_k=20)
        reranked = reranker.rerank("查询", results, top_k=3)
    """

    def __init__(self):
        """
        初始化重排序服务
        
        参数:
            api_key: DashScope API密钥，默认使用配置中的值
            model: 重排序模型名称
                - qwen3-rerank（推荐，官方重排序模型）
        """
        self.api_key = settings.DASHSCOPE_API_KEY
        self.model = "qwen3-rerank"
        
        if not self.api_key:
            logger.warning("⚠️  DashScope API Key 未配置，重排序功能不可用")
            self._available = False
        else:
            self._available = True
            logger.info(f"✅ 重排序服务就绪，模型: {self.model}")

    @property
    def available(self) -> bool:
        """重排序服务是否可用"""
        return self._available

    def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_k: int = None,
        return_documents: bool = True
    ) -> List[Dict[str, Any]]:
        """
        对检索结果进行重排序
        
        参数:
            query: 查询文本
            documents: 检索结果列表
                每项需包含 "text" 或 "content" 字段
            top_k: 返回数量，默认使用配置值
            return_documents: 是否返回完整文档信息
            
        返回:
            重排序后的结果列表（按相关性降序）
            
        注意：
            - API 限制单次最多处理 50 个文档
            - 超过50个会自动截断
        """
        if top_k is None:
            top_k = settings.TOP_K
        
        # 服务不可用或无文档时，直接返回
        if not self._available or not documents:
            return documents[:top_k]
        
        try:
            # 提取文档文本（兼容不同字段名）
            doc_texts = []
            for doc in documents:
                text = doc.get("text") or doc.get("content", "")
                doc_texts.append(text[:2000])  # 限制单篇长度，避免超token
            
            # API 限制：最多 50 个文档
            max_docs = 50
            if len(doc_texts) > max_docs:
                logger.warning(f"文档数量 {len(doc_texts)} 超过限制 {max_docs}，仅处理前 {max_docs} 个")
                doc_texts = doc_texts[:max_docs]
                documents = documents[:max_docs]
            
            # 调用重排序 API
            response = TextReRank.call(
                model=self.model,
                api_key=self.api_key,
                query=query,
                documents=doc_texts,
                top_n=min(top_k, len(doc_texts)),  # 返回数量
                return_documents=return_documents
            )
            
            # 检查响应状态
            if response.status_code != HTTPStatus.OK:
                logger.error(f"重排序 API 调用失败: {response.message}")
                return documents[:top_k]
            
            # 解析重排序结果
            rerank_results = response.output.results
            
            if not rerank_results:
                logger.warning("重排序无结果，回退到原始排序")
                return documents[:top_k]
            
            # 重新组装结果
            reranked = []
            for item in rerank_results:
                index = item.index  # 原始文档索引
                score = item.relevance_score  # 重排序分数
                
                original = documents[index].copy()
                original["rerank_score"] = round(score, 4)
                
                # 如果有重排序返回的文本，使用它（可能被截断）
                if hasattr(item, 'document') and item.document:
                    original["text"] = item.document
                
                reranked.append(original)
            
            logger.debug(f"重排序完成: {len(reranked)} 个结果")
            return reranked
            
        except Exception as e:
            logger.error(f"重排序失败: {e}，回退到原始排序")
            return documents[:top_k]

