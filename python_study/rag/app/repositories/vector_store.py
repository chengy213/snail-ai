"""
向量存储管理模块

负责管理 Milvus 向量数据库的连接、集合初始化和向量检索操作。
提供文档向量的存储和相似度搜索功能。

主要功能：
- 向量相似度搜索（主要推荐，中文语义理解好）
- BM25 关键词搜索（英文效果更好）
- 混合搜索（向量 + BM25 加权融合）
- 混合检索 + 重排序（最佳效果）
"""
from typing import List, Dict, Any
from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import MilvusClient, DataType, Function, FunctionType
import time
import logging

from python_study.rag.app.core.config import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """向量存储管理器 - 负责处理所有数据库操作"""

    def __init__(self):
        """初始化管理器，创建 Milvus 数据库连接"""
        # 集合名称（相当于MySQL中的表名）
        self.collection_name = settings.COLLECTION_NAME

        # 连接 Milvus 数据库
        self.client = MilvusClient(
            uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
            token="root:Milvus"  # 默认用户名密码
        )

        self.embedding = None          # 向量化工具（将文字转成数字向量）
        self._collection_ready = False # 集合是否准备好（类似表是否可用）
        self._reranker = None          # 重排序器（让搜索结果更精准）

    @property
    def reranker(self):
        """延迟加载重排序器 - 需要时才创建，避免启动太慢"""
        if self._reranker is None:
            try:
                from python_study.rag.app.services.reranker_service import RerankerService
                self._reranker = RerankerService()
            except ImportError:
                logger.warning("RerankerService not available")
        return self._reranker

    def init_collection(self, embedding_function: DashScopeEmbeddings) -> bool:
        """
        初始化集合（相当于创建数据库表）

        参数:
            embedding_function: 向量化工具，负责把文字转成数字向量

        返回:
            bool: 是否成功
        """
        self.embedding = embedding_function

        # 如果集合已经存在，直接加载并返回
        if self.client.has_collection(self.collection_name):
            logger.info(f"集合 '{self.collection_name}' 已存在")
            self.client.load_collection(self.collection_name)
            self._collection_ready = True
            return True

        # ========== 第1步：定义表结构（Schema） ==========
        schema = MilvusClient.create_schema(
            auto_id=True,           # 自动生成ID
            enable_dynamic_field=True  # 允许动态添加字段
        )

        # 添加各个字段
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)  # 主键ID
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=1024)  # 向量字段（1024维）
        schema.add_field(
            "text", DataType.VARCHAR, max_length=65535,
            enable_analyzer=True, analyzer_params={"tokenizer": "jieba"}  # 中文分词器
        )  # 原始文本内容
        schema.add_field("file_name", DataType.VARCHAR, max_length=512)  # 文件名
        schema.add_field("doc_id", DataType.VARCHAR, max_length=128)     # 文档ID（用于去重）
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)  # 稀疏向量（BM25用）
        schema.add_field("chunk_index", DataType.INT64)  # 分块索引（第几段）

        # ========== 第2步：配置BM25函数 ==========
        # BM25是一种关键词匹配算法，根据词语出现频率计算相关性
        schema.add_function(Function(
            name="text_bm25_emb",
            input_field_names=["text"],   # 输入：原始文本
            output_field_names=["sparse"], # 输出：稀疏向量
            function_type=FunctionType.BM25,
        ))

        # ========== 第3步：建立索引（加快查询速度） ==========
        index_params = self.client.prepare_index_params()

        # 向量索引：IVF_FLAT是聚类算法，COSINE是余弦相似度
        index_params.add_index("vector", index_type="IVF_FLAT", metric_type="COSINE", params={"nlist": 128})

        # 普通字段索引：TRIE是前缀树，适合精确匹配
        index_params.add_index("file_name", index_type="TRIE")
        index_params.add_index("doc_id", index_type="TRIE")
        index_params.add_index("id", index_type="STL_SORT")      # 排序索引
        index_params.add_index("chunk_index", index_type="STL_SORT")

        # BM25稀疏向量索引
        index_params.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25",
                              params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75})

        # ========== 第4步：创建并加载集合 ==========
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
        self.client.load_collection(self.collection_name)  # 加载到内存
        self._collection_ready = True
        logger.info(f"集合 '{self.collection_name}' 创建并加载成功")
        return True

    def add_documents(self, texts: List[str], metadata_list: List[Dict[str, Any]]) -> bool:
        """
        批量添加文档到向量数据库

        参数:
            texts: 文本列表（每个元素是一段文本）
            metadata_list: 元数据列表（文件名、文档ID等）

        返回:
            bool: 是否全部添加成功
        """
        if not self._collection_ready:
            raise RuntimeError("集合未初始化，请先调用 init_collection")

        batch_size = 200  # 每批200条，避免一次性处理太多
        total = len(texts)
        logger.info(f"正在添加 {total} 条文档")

        for i in range(0, total, batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadata = metadata_list[i:i+batch_size]

            # 把文字转换成向量（调用阿里云百炼的embedding接口）
            vectors = self.embedding.embed_documents(batch_texts)

            # 组装要插入的数据
            data = []
            for j, (txt, meta, vec) in enumerate(zip(batch_texts, batch_metadata, vectors)):
                data.append({
                    "vector": vec,
                    "text": txt,
                    "file_name": meta.get("file_name", ""),
                    "doc_id": meta.get("doc_id", ""),
                    "chunk_index": meta.get("chunk_index", i + j),
                })

            # 批量插入数据库
            self.client.insert(self.collection_name, data=data)

        # 刷新：强制把数据写入磁盘
        self.client.flush(self.collection_name)
        self._wait_for_index()  # 等待索引构建完成
        logger.info(f"成功添加 {total} 条文档")
        return True

    def _wait_for_index(self, timeout: int = 30) -> None:
        """
        等待索引构建完成

        索引是异步构建的，刚插入数据后可能还没建好索引
        这个方法会尝试搜索来确认索引是否可用
        """
        start = time.time()
        while time.time() - start < timeout:
            try:
                # 尝试执行一次搜索，如果成功说明索引已就绪
                self.client.search(self.collection_name, data=["test"], anns_field='sparse', limit=1)
                return
            except Exception:
                time.sleep(2)  # 等2秒再试
        logger.warning("索引构建超时，但集合仍可用")

    def search(self, query: str, top_k: int = None, search_type: str = "vector", use_rerank: bool = False) -> List[Dict[str, Any]]:
        """
        统一搜索接口 - 根据需求选择不同的搜索方式

        参数:
            query: 用户输入的问题
            top_k: 返回多少个结果（默认从配置读取）
            search_type: 搜索类型
                - "vector": 纯向量搜索（语义理解强，适合中文）
                - "bm25": 关键词搜索（精确匹配强，适合英文）
                - "hybrid": 混合搜索（两者结合）
            use_rerank: 是否启用重排序（让结果更精准，但会慢一点）

        返回:
            搜索结果列表，每个结果包含文本内容、文件名、相似度分数等
        """
        if not self._collection_ready:
            raise RuntimeError("集合未初始化，请先调用 init_collection")

        top_k = top_k or settings.TOP_K

        # 如果要用重排序，先多召回一些候选（3倍），给重排序留出筛选空间
        recall_k = top_k * 3 if (use_rerank and self.reranker and self.reranker.available) else top_k

        # 第一步：根据选择的类型进行检索
        if search_type == "vector":
            results = self._vector_search(query, recall_k)
        elif search_type == "bm25":
            results = self._bm25_search(query, recall_k)
        elif search_type == "hybrid":
            results = self._hybrid_search(query, recall_k)
        else:
            raise ValueError(f"未知的搜索类型: {search_type}")

        # 第二步：如果启用重排序，用千问模型精排
        if use_rerank and self.reranker and self.reranker.available and len(results) > top_k:
            results = self.reranker.rerank(query, results, top_k)
        else:
            results = results[:top_k]  # 直接取前top_k个

        return results

    def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        向量语义搜索

        原理：把用户问题也转成向量，然后找最相似的文档向量
        优点：理解语义，比如搜"苹果"能找到"iPhone"
        """
        # 把问题转成向量
        query_vector = self.embedding.embed_query(query)

        # 在数据库中搜索最相似的top_k个向量
        results = self.client.search(
            collection_name=self.collection_name,
            anns_field="vector",
            data=[query_vector],
            limit=top_k,
            search_params={"metric_type": "COSINE"},  # 余弦相似度
            output_fields=['id', 'text', 'file_name', 'doc_id', 'chunk_index']
        )
        return self._format_results(results)

    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        BM25关键词搜索

        原理：统计问题中的关键词，找包含这些词的文档
        优点：精确匹配，比如搜"Python 3.9"能准确定位到相关文档
        """
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query],                    # 直接传原始文本，Milvus会自动用BM25处理
            anns_field='sparse',
            limit=top_k,
            search_params={"metric_type": "BM25", "params": {"drop_ratio_search": 0.0}},
            output_fields=['id', 'text', 'file_name', 'doc_id', 'chunk_index']
        )
        return self._format_results(results)

    def _hybrid_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        混合搜索：向量 + BM25 加权融合

        原理：
        - 同时用向量搜索和BM25搜索
        - 向量结果权重70%（主要靠语义理解）
        - BM25结果权重30%（辅助精确匹配）
        - 把两份结果合并，按加权分数排序

        优点：语义理解好，精确匹配也不落下
        """
        # 同时跑两种搜索
        vector_results = self._vector_search(query, top_k)
        bm25_results = self._bm25_search(query, top_k)

        # 如果某一路没结果，就回退到另一路
        if not bm25_results:
            return vector_results
        if not vector_results:
            return bm25_results

        # 合并结果，加权计算总分
        merged = {}  # 用字典存，key是文档ID，方便合并

        # 向量结果：权重0.7
        for r in vector_results:
            merged[r['id']] = {
                **r,
                'hybrid_score': r.get('score', 0) * 0.7,   # 混合分数
                'vector_score': r.get('score', 0),          # 向量分数（保留）
                'bm25_score': 0
            }

        # BM25结果：权重0.3
        for r in bm25_results:
            if r['id'] in merged:
                # 已存在：累加分数
                merged[r['id']]['hybrid_score'] += r.get('score', 0) * 0.3
                merged[r['id']]['bm25_score'] = r.get('score', 0)
            else:
                # 新文档：直接加入
                merged[r['id']] = {
                    **r,
                    'hybrid_score': r.get('score', 0) * 0.3,
                    'vector_score': 0,
                    'bm25_score': r.get('score', 0)
                }

        # 按混合分数从高到低排序
        return sorted(merged.values(), key=lambda x: x['hybrid_score'], reverse=True)

    def _format_results(self, results: List) -> List[Dict[str, Any]]:
        """
        格式化Milvus搜索结果

        Milvus返回的结果格式有点复杂，这个函数把它转成简单好用的字典列表
        """
        if not results or not results[0]:
            return []

        formatted = []
        for hit in results[0]:
            entity = hit.get('entity', {})  # 原始数据存在entity里
            formatted.append({
                'id': hit.get('id'),
                'text': entity.get('text', ''),
                'file_name': entity.get('file_name', ''),
                'doc_id': entity.get('doc_id', ''),
                'chunk_index': entity.get('chunk_index', 0),
                'score': hit.get('distance', 0),  # 相似度分数
            })
        return formatted

    def check_doc_exists(self, doc_id: str) -> bool:
        """检查文档是否已存在（用于上传去重，避免重复添加）"""
        results = self.client.query(
            self.collection_name,
            filter=f'doc_id == "{doc_id}"',
            output_fields=["doc_id"],
            limit=1
        )
        return len(results) > 0

    def delete_by_file_name(self, file_name: str) -> int:
        """按文件名删除文档（重新上传相同文件时会先删除旧的）"""
        result = self.client.delete(self.collection_name, filter=f'file_name == "{file_name}"')
        delete_count = result.get('delete_count', 0)
        logger.info(f"删除了 {delete_count} 条文档（文件: {file_name}）")
        return delete_count

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息（总文档数、占用空间等）"""
        return self.client.get_collection_stats(self.collection_name)

    def flush(self) -> None:
        """强制刷新数据到磁盘（确保数据持久化）"""
        self.client.flush(self.collection_name)