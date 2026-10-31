"""
向量存储 - 混合检索版本（向量 + BM25）
支持纯向量检索和混合检索（向量语义匹配 + BM25关键词匹配）
"""
from typing import List, Optional
from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import Field
from rank_bm25 import BM25Okapi
import jieba
from model.factory import embedding_model
from utils.config_handler import vector_store_config


class BM25Retriever:
    """
    BM25 关键词检索器

    基于经典的 BM25 算法实现关键词匹配检索
    通过中文分词后建立倒排索引，擅长精确术语匹配
    弥补向量检索在专业术语、缩写等精确匹配场景的不足

    使用示例:
        retriever = BM25Retriever(documents)
        results = retriever.get_relevant_documents("限流怎么实现", k=5)
    """

    def __init__(self, documents: List[Document]):
        """
        初始化 BM25 检索器

        Args:
            documents: 需要建立索引的文档列表
                      需要预先加载向量库中的所有文档
        """
        # 保存原始文档引用
        self.documents = documents

        # 对所有文档进行中文分词，构建 BM25 索引
        # BM25Okapi 需要输入分词后的词语列表，而非原始文本
        tokenized_docs = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def _tokenize(self, text: str) -> List[str]:
        """
        中文分词方法

        使用结巴分词(jieba)将中文文本切分为词语列表
        中文不像英文有天然的空格分隔，必须使用分词工具

        Args:
            text: 待分词的中文文本

        Returns:
            分词后的词语列表
            例如: "限流怎么实现" -> ["限流", "怎么", "实现"]
        """
        return list(jieba.cut(text))

    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        根据查询检索最相关的 k 个文档

        先对查询进行分词，然后计算每个文档的 BM25 分数，
        返回分数最高的 k 个文档

        Args:
            query: 查询文本
            k: 需要返回的文档数量

        Returns:
            相关文档列表，按 BM25 分数从高到低排序
        """
        # 对查询文本进行分词
        tokenized_query = self._tokenize(query)

        # 计算每个文档相对于查询的 BM25 分数
        # BM25 会考虑词频(TF)和逆文档频率(IDF)
        scores = self.bm25.get_scores(tokenized_query)

        # 获取分数最高的 k 个文档的索引
        # sorted 返回排序后的索引列表
        top_k_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],  # 按分数排序
            reverse=True              # 降序，分数高的在前
        )[:k]

        # 根据索引返回对应的文档对象
        return [self.documents[i] for i in top_k_indices]


class HybridRetriever(BaseRetriever):
    """
    混合检索器

    将向量检索（语义匹配）和 BM25 检索（关键词匹配）的结果进行加权合并
    利用两种检索方式互补的优势，提高整体检索准确率

    工作流程:
    1. 分别执行向量检索和 BM25 检索
    2. 对各自结果按排名加权打分
    3. 合并去重（同时被两种方式命中的文档得分累加）
    4. 按总分排序返回 top-k

    权重说明:
    - vector_weight=0.5, bm25_weight=0.5: 平衡模式（推荐默认值）
    - vector_weight=0.7: 偏重语义理解，适合长文本自然语言查询
    - bm25_weight=0.7: 偏重关键词匹配，适合精确术语查询
    """

    # 使用 Field 声明 Pydantic 字段
    # BaseRetriever 继承自 Pydantic BaseModel，需要用 Field 声明属性
    vector_retriever: BaseRetriever = Field(..., description="向量检索器实例")
    bm25_retriever: BM25Retriever = Field(..., description="BM25关键词检索器实例")
    vector_weight: float = Field(default=0.5, description="向量检索权重，范围[0,1]")
    bm25_weight: float = Field(default=0.5, description="BM25检索权重，范围[0,1]")
    k: int = Field(default=5, description="最终返回的文档数量")

    class Config:
        """Pydantic 配置，允许使用自定义类型（非标准 Pydantic 字段）"""
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        执行混合检索

        这是 BaseRetriever 要求实现的抽象方法，
        LangChain 框架会调用此方法进行检索

        检索流程:
        1. 向量检索：基于语义相似度
        2. BM25检索：基于关键词匹配（扩大召回，取2*k个）
        3. 排名加权：排名越靠前，得分越高
        4. 合并去重：同一文档被两种方式命中时分数累加
        5. 排序输出：按总分降序返回前k个

        Args:
            query: 查询文本

        Returns:
            合并排序后的文档列表
        """
        # 第一步：向量检索
        # 利用嵌入模型将查询转为向量，在向量空间中搜索最相似的文档
        vector_docs = self.vector_retriever.get_relevant_documents(query)

        # 第二步：BM25 关键词检索
        # 扩大召回量（2*k），后续通过混合排序筛选出最好的k个
        bm25_docs = self.bm25_retriever.get_relevant_documents(query, k=self.k * 2)

        # 第三步：合并结果并去重
        # 使用字典存储，key为文档标识，value为文档和分数
        doc_scores = {}

        # 处理向量检索结果
        # 使用排名归一化：第1名得满分，最后一名接近0分
        max_vector_rank = len(vector_docs)
        for rank, doc in enumerate(vector_docs):
            # 取文档前100字符作为唯一标识（用于去重）
            # 注：更好的做法是使用doc.metadata中的唯一ID
            doc_id = doc.page_content[:100]

            # 排名分数计算公式：weight * (1 - rank/max_rank)
            # 排名0（第一名）得满分，排名越靠后分数越低
            score = self.vector_weight * (1 - rank / max(max_vector_rank, 1))
            doc_scores[doc_id] = {'doc': doc, 'score': score}

        # 处理 BM25 检索结果
        max_bm25_rank = len(bm25_docs)
        for rank, doc in enumerate(bm25_docs):
            doc_id = doc.page_content[:100]

            # 同样使用排名归一化
            score = self.bm25_weight * (1 - rank / max(max_bm25_rank, 1))

            if doc_id in doc_scores:
                # 文档同时被向量和BM25命中，累加分数
                # 这表示该文档在语义和关键词两个维度都与查询高度相关
                doc_scores[doc_id]['score'] += score
            else:
                # 仅被BM25命中
                doc_scores[doc_id] = {'doc': doc, 'score': score}

        # 第四步：按总分降序排序，返回前 k 个结果
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:self.k]

        # 只返回文档对象，不返回分数
        return [item['doc'] for item in sorted_docs]


class VectorStoreService:
    """
    向量存储服务

    封装了 Milvus 向量数据库的操作，提供:
    1. 纯向量检索
    2. 混合检索（向量 + BM25）
    3. 文档加载和索引管理

    使用示例:
        service = VectorStoreService()

        # 纯向量检索
        results = service.vector_search("查询文本", k=5)

        # 混合检索（推荐）
        results = service.hybrid_search("查询文本", k=5, alpha=0.5)

        # 获取 LangChain 标准检索器
        retriever = service.get_hybrid_retriever(k=5)
        results = retriever.invoke("查询文本")
    """

    def __init__(self):
        """
        初始化向量存储服务

        1. 连接 Milvus 向量数据库
        2. 配置文本分割器
        3. 初始化缓存变量（延迟加载）
        """
        # 初始化 Milvus 向量数据库连接
        # embedding_function: 将文本转换为向量的嵌入模型
        # collection_name: Milvus 中的集合名称（类似数据库的表）
        # connection_args: Milvus 服务的连接参数
        # vector_field: 集合中存储向量的字段名（需与已有集合结构匹配）
        # text_field: 集合中存储文本的字段名
        # drop_old: 是否删除已有集合（False=保留已有数据）
        self.vector_store = Milvus(
            embedding_function=embedding_model,
            collection_name=vector_store_config["collection_name"],
            connection_args={
                "host": vector_store_config["host"],
                "port": vector_store_config["port"],
            },
            vector_field="embedding",
            text_field="content",
            drop_old=False,
        )

        # 初始化文本分割器
        # 用于将长文档切分为适合检索的文本块
        # chunk_size: 每块的最大字符数
        # chunk_overlap: 相邻块之间的重叠字符数（保持上下文连贯）
        # separators: 分隔符优先级列表（优先在段落、句子边界切分）
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=vector_store_config["chunk_size"],
            chunk_overlap=vector_store_config["chunk_overlap"],
            separators=vector_store_config["separators"],
            length_function=len
        )

        # 缓存变量，使用延迟加载模式
        # _all_documents: 缓存所有文档，用于构建 BM25 索引
        # _bm25_retriever: 缓存 BM25 检索器实例
        self._all_documents = None
        self._bm25_retriever = None

    @property
    def all_documents(self) -> List[Document]:
        """
        获取向量库中的所有文档（带缓存）

        首次调用时从 Milvus 加载所有文档，
        后续调用直接返回缓存，避免重复查询

        Returns:
            向量库中所有文档的列表
        """
        if self._all_documents is None:
            # 使用空字符串查询获取所有文档
            # k 值需设置足够大以确保获取全部文档
            self._all_documents = self.vector_store.similarity_search("", k=10000)
        return self._all_documents

    @property
    def bm25_retriever(self) -> BM25Retriever:
        """
        获取 BM25 检索器实例（带缓存）

        首次调用时基于 all_documents 创建 BM25 索引，
        后续调用直接返回缓存的检索器

        Returns:
            BM25 检索器实例
        """
        if self._bm25_retriever is None:
            # 基于所有文档构建 BM25 索引
            self._bm25_retriever = BM25Retriever(self.all_documents)
        return self._bm25_retriever

    def get_retriever(
        self,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ):
        """
        获取纯向量检索器（LangChain Retriever 接口）

        Args:
            k: 返回的文档数量，默认使用配置中的值
            score_threshold: 相似度阈值，低于此分数的文档会被过滤
                           设为0表示不过滤，设高值可提高精度但会减少召回

        Returns:
            LangChain BaseRetriever 对象，可直接用于 RAG 链
        """
        # 使用配置中的默认值或参数指定的值
        k = k or vector_store_config.get("k", 5)
        score_threshold = score_threshold or vector_store_config.get("score_threshold", 0.0)

        # as_retriever 将向量存储包装为标准的 LangChain 检索器
        return self.vector_store.as_retriever(
            search_kwargs={
                "k": k,
                "score_threshold": score_threshold
            }
        )

    def get_hybrid_retriever(
        self,
        k: int = 5,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5
    ) -> HybridRetriever:
        """
        获取混合检索器（向量 + BM25）

        返回实现了 LangChain BaseRetriever 接口的混合检索器，
        可以直接在 RAG 链中替换纯向量检索器

        Args:
            k: 最终返回的文档数量
            vector_weight: 向量检索权重 [0-1]，越大越偏向语义匹配
            bm25_weight: BM25检索权重 [0-1]，越大越偏向关键词匹配

        Returns:
            HybridRetriever 实例，可直接调用 invoke() 方法

        使用示例:
            retriever = service.get_hybrid_retriever(k=5, vector_weight=0.7)
            docs = retriever.invoke("查询文本")
        """
        # 向量检索器先取 2*k 个结果，增加召回率
        # 最终由混合检索器重新排序选出最好的 k 个
        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": k * 2}
        )

        # 创建混合检索器
        return HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=self.bm25_retriever,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            k=k
        )

    def vector_search(self, query: str, k: int = 5) -> List[Document]:
        """
        纯向量检索

        直接基于语义相似度在向量空间中搜索最相关的文档

        Args:
            query: 查询文本
            k: 返回的文档数量

        Returns:
            相关文档列表，按相似度从高到低排序
        """
        # similarity_search_with_score 返回 (文档, 分数) 的元组列表
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)

        # 只返回文档，丢弃分数
        return [doc for doc, _ in docs_with_scores]

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Document]:
        """
        混合检索：向量语义匹配 + BM25关键词匹配

        同时执行向量检索和 BM25 检索，加权合并后返回最佳结果
        相比纯向量检索，能更好地处理专业术语和精确匹配需求

        检索流程:
        1. 向量检索：获取语义相似的文档
        2. BM25检索：获取关键词匹配的文档
        3. 分数归一化：将两种分数的量纲统一
        4. 加权合并：alpha * 向量分数 + (1-alpha) * BM25分数
        5. 去重排序：同一文档被双命中时分数累加

        Args:
            query: 查询文本
            k: 返回的文档数量
            alpha: 向量检索权重，取值范围 [0, 1]
                  0.0 = 纯 BM25 关键词检索
                  0.5 = 平衡模式（推荐默认值）
                  1.0 = 纯向量语义检索

        Returns:
            合并排序后的文档列表

        使用示例:
            # 偏重语义理解
            results = service.hybrid_search("如何实现限流", k=5, alpha=0.7)

            # 偏重关键词匹配
            results = service.hybrid_search("Sentinel 限流", k=5, alpha=0.3)
        """
        # 第一步：向量检索，取 2*k 个结果以增加召回率
        vector_results = self.vector_store.similarity_search_with_score(query, k=k*2)

        # 第二步：BM25 关键词检索
        bm25_docs = self.bm25_retriever.get_relevant_documents(query, k=k*2)

        # 第三步：合并两种检索结果
        # combined 字典用于去重和分数累加
        # key: 文档前100字符作为唯一标识
        # value: {'doc': 文档对象, 'score': 加权后的最终分数}
        combined = {}

        # 处理向量检索结果 - 使用实际相似度分数
        # 取最大的相似度分数用于归一化
        max_vector_score = max([score for _, score in vector_results]) if vector_results else 1

        for doc, score in vector_results:
            doc_id = doc.page_content[:100]

            # 归一化：将分数映射到 [0, 1] 区间
            # 避免绝对分数差异过大影响加权效果
            normalized_score = score / max_vector_score if max_vector_score > 0 else 0

            # 加权：向量分数 * 向量权重
            combined[doc_id] = {
                'doc': doc,
                'score': alpha * normalized_score
            }

        # 处理 BM25 检索结果 - 使用排名归一化
        # BM25 不返回归一化分数，使用排名位置作为分数
        for rank, doc in enumerate(bm25_docs):
            doc_id = doc.page_content[:100]

            # 排名分数：rank=0（第一名）得1.0，逐次递减
            # 公式: 1.0 - rank/总文档数
            rank_score = 1.0 - rank / max(len(bm25_docs), 1)

            if doc_id in combined:
                # 双命中：文档同时被向量和BM25检索到
                # 累加分数，增强该文档的排名
                combined[doc_id]['score'] += (1 - alpha) * rank_score
            else:
                # 仅被BM25命中
                combined[doc_id] = {
                    'doc': doc,
                    'score': (1 - alpha) * rank_score
                }

        # 第四步：按最终分数降序排序
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:k]

        # 只返回文档列表
        return [item['doc'] for item in sorted_results]

    def load_document(self):
        """
        加载新文档到向量库

        加载新文档后需要清空缓存，确保:
        1. BM25 索引包含最新文档
        2. all_documents 缓存被更新

        注意: 目前是占位方法，需要实现具体的文档加载逻辑

        TODO:
        - 从文件系统读取文档
        - 使用 spliter 进行文本分割
        - 通过 embedding_model 生成向量
        - 存入 Milvus 向量库
        - 清空缓存
        """
        # 清空缓存，下次访问时会自动重新加载
        self._all_documents = None
        self._bm25_retriever = None
        return None