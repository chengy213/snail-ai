"""
文本向量化服务模块

提供将文本转换为向量嵌入（Embedding）的功能。
向量化是 RAG 系统的核心环节：将非结构化的自然语言文本映射为高维向量空间中的点，
语义相似的文本在向量空间中距离更近，从而支撑后续的语义检索。

当前使用 DashScope（阿里云百炼）的嵌入模型，通过 API 调用，无需本地 GPU。

备选方案（注释中保留，可按需切换）：
    - BGE-small-zh : 512维轻量中文模型，CPU 即可运行
    - BGE-large-zh : 1024维高精度中文模型，需 GPU
    - M3E-base     : 768维通用中文模型，CPU 可运行
"""
from langchain_community.embeddings import DashScopeEmbeddings

from python_study.rag.app.core.config import settings


class EmbeddingService:
    """
    文本向量化服务

    封装 LangChain 的 DashScopeEmbeddings，提供两种向量化方式：
        - get_embedding       : 文档批量向量化（入库时使用）
        - get_query_embedding : 查询向量化（检索时使用）

    为什么有两种方法？
        虽然底层用的是同一个模型，但 DashScopeEmbeddings 在 LangChain 中
        区分了 embed_documents（文档）和 embed_query（查询），
        部分模型对查询和文档有不同的编码策略，所以分开调用。
    """

    def __init__(self):
        """
        初始化向量化服务

        创建 DashScopeEmbeddings 实例，连接到阿里云百炼的嵌入模型 API。
        embedding 对象是 LangChain 的 Embeddings 接口实现，
        提供 embed_documents（批量）和 embed_query（单条）两个方法。
        """
        # 千问 DashScope 嵌入模型（通过 API 调用，无需本地 GPU）
        # model 名称从配置文件读取（如 text-embedding-v2 等）
        self.embedding = DashScopeEmbeddings(
            model=settings.EMBEDDINGS_MODEL_NAME,
            dashscope_api_key=settings.DASHSCOPE_API_KEY,
        )

        # ================================================================
        # 备选嵌入模型方案（按需切换）
        # ================================================================
        # 方案1：BGE 中文轻量模型（推荐起步方案）
        #   - 512维向量，体积小、速度快
        #   - CPU 即可运行，适合开发和小规模部署
        #   - 使用 normalize_embeddings=True 归一化，相似度计算更准确
        # self.embeddings_bge_small = HuggingFaceEmbeddings(
        #     model_name="BAAI/bge-small-zh-v1.5",
        #     model_kwargs={"device": "cpu"},
        #     encode_kwargs={"normalize_embeddings": True}
        # )
        #
        # 方案2：BGE 中文大模型（高精度场景）
        #   - 1024维向量，精度更高
        #   - 需要 GPU 加速，否则速度很慢
        #   - 适合大规模生产环境
        # self.embeddings_bge_large = HuggingFaceEmbeddings(
        #     model_name="BAAI/bge-large-zh-v1.5",
        #     model_kwargs={"device": "cuda"},
        #     encode_kwargs={"normalize_embeddings": True}
        # )
        #
        # 方案3：M3E 通用中文模型（平衡选择）
        #   - 768维向量，精度和速度的折中
        #   - CPU 可运行，适合中等规模
        # self.embeddings_m3e = HuggingFaceEmbeddings(
        #     model_name="moka-ai/m3e-base",
        #     model_kwargs={"device": "cpu"},
        #     encode_kwargs={"normalize_embeddings": True}
        # )

    def get_embedding(self, text: list) -> list:
        """
        批量获取文档文本的向量嵌入

        用于文档入库场景：将一批文本块一次性转为向量，减少 API 调用次数。

        参数:
            text: 文本列表，每个元素为一个文本块（字符串）

        返回:
            list: 向量列表，每个元素为对应文本块的向量表示（浮点数列表）
                  维度取决于使用的嵌入模型（如 1024 维）

        示例:
            service = EmbeddingService()
            vectors = service.get_embedding(["文档1内容", "文档2内容"])
            # 返回: [[0.123, 0.456, ...], [0.789, 0.012, ...]]
        """
        return self.embedding.embed_documents(text)

    def get_query_embedding(self, query: str) -> list:
        """
        获取查询文本的向量嵌入

        用于检索场景：将用户问题转为向量，然后在向量数据库中搜索最相似的文档向量。

        参数:
            query: 用户查询字符串（一个问题）

        返回:
            list: 查询文本的向量表示（浮点数列表）

        注意:
            查询向量化和文档向量化使用同一模型，但调用不同的方法：
            - 文档: embed_documents(texts)  → 批量处理
            - 查询: embed_query(query)      → 单条处理，部分模型有特殊编码策略

        示例:
            service = EmbeddingService()
            query_vector = service.get_query_embedding("什么是向量数据库？")
            # 返回: [0.123, 0.456, ...]
        """
        return self.embedding.embed_query(query)

