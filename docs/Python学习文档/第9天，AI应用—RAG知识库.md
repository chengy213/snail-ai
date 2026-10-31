> 项目源码：python_study/rag/
>
> 前端界面：http://localhost:8100 （单页聊天应用）
>
> API 文档：http://localhost:8100/docs (FastAPI 自动生成 Swagger)

## 第一章 什么是 RAG？

### 1.1 核心概念

**RAG（Retrieval-Augmented Generation，检索增强生成）** 是一种将信息检索与大语言模型相结合的 AI 应用架构。它的核心思想是：**先检索相关知识，再让模型基于检索结果生成答案**，而不是让模型仅凭训练记忆"编造"答案。

```
用户提问 → 向量检索(找相关文档) → 拼装上下文 → LLM 基于上下文生成 → 返回答案
```

### 1.2 为什么需要 RAG？

| 问题 | 纯 LLM | RAG 方案 |
|------|--------|----------|
| 私有知识 | ❌ 不知道企业内部文档 | ✅ 检索私有文档后回答 |
| 幻觉问题 | ❌ 可能编造不存在的事实 | ✅ 严格基于检索到的文档 |
| 知识时效 | ❌ 训练截止日期后无法更新 | ✅ 上传新文档即可 |
| 可追溯性 | ❌ 无法验证信息来源 | ✅ 每个答案附带引用 |

---

## 第二章 项目架构总览

### 2.1 技术栈

```
┌────────────────────────────────────────────────────┐
│                   前端 (纯 HTML/CSS/JS)              │
│              聊天式 UI / 拖拽上传 / fetch() 调用      │
└──────────────────────┬─────────────────────────────┘
                       │ POST /api/v1/upload
                       │ POST /api/v1/ask
┌──────────────────────▼─────────────────────────────┐
│              FastAPI 后端 (uvicorn :8100)            │
│  ┌──────────┐ ┌──────────┐ ┌────────────────────┐  │
│  │ routes   │ │ core     │ │ services           │  │
│  │ 上传/问答 │ │ 配置/DI  │ │ 文档/嵌入/检索/LLM  │  │
│  └──────────┘ └──────────┘ └─────────┬──────────┘  │
│                                      │              │
│  ┌───────────────────────────────────▼───────────┐  │
│  │          repositories/vector_store.py         │  │
│  │               Milvus 连接 & 操作               │  │
│  └───────────────────────┬───────────────────────┘  │
└──────────────────────────┼──────────────────────────┘
           ┌───────────────┴───────────────┐
    ┌──────▼──────┐                  ┌─────▼──────┐
    │   Milvus    │                  │  DashScope │
    │  向量数据库  │                  │  阿里云百炼  │
    │ :19530      │                  │ 嵌入+LLM    │
    └─────────────┘                  └────────────┘
```

### 2.2 目录结构

```
python_study/rag/
├── .env                          # 环境变量配置
├── requirements.txt              # Python 依赖
└── app/
    ├── main.py                   # 应用入口，FastAPI 实例 + 生命周期
    ├── static/
    │   └── index.html            # 前端单页面
    ├── uploads/                  # 上传文件存储目录
    ├── api/
    │   └── routes.py             # API 路由 + 请求/响应 Pydantic 模型
    ├── core/
    │   ├── config.py             # pydantic-settings 配置管理
    │   └── dependencies.py       # 依赖注入（单例 + 工厂）
    ├── repositories/
    │   └── vector_store.py       # Milvus 向量数据库 CRUD 封装
    └── services/
        ├── document_service.py   # 文档解析与分块
        ├── embedding_service.py  # 文本向量化
        ├── retrieval_service.py  # 语义检索 + 上下文组装
        └── llm_service.py        # 大模型答案生成
```

### 2.3 分层架构

项目采用经典的三层架构：

| 层级 | 目录 | 职责 |
|------|------|------|
| **API 路由层** | `api/routes.py` | 接收 HTTP 请求，参数校验，调用服务，返回响应 |
| **服务层** | `services/` | 业务逻辑：文档处理、向量化、检索、LLM 调用 |
| **仓储层** | `repositories/` | 数据持久化：Milvus 的 CRUD 封装 |
| **核心层** | `core/` | 配置管理和依赖注入（相当于 Spring 的 Bean 容器） |

---

## 第三章 核心组件详解

### 3.1 配置管理 — `core/config.py`

使用 **pydantic-settings** 实现类型安全的配置读取，从 `.env` 文件和系统环境变量中自动加载。

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # DashScope 大模型配置
    DASHSCOPE_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    DASHSCOPE_MODEL: str = "qwen3-max"

    # 嵌入模型配置
    EMBEDDINGS_MODEL_NAME: str = "text-embedding-v4"   # 输出 1024 维向量

    # Milvus 向量数据库配置
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    COLLECTION_NAME: str = "snail_store"

    # 文档分割配置
    TOP_K: int = 3          # 默认返回 3 个相关文档片段
    CHUNK_SIZE: int = 500   # 每块 500 字符
    CHUNK_OVERLAP: int = 50 # 块间重叠 50 字符
    SEPARATORS: list = ["\n\n", "\n", "。", ".", "?", "？", "!", "！", "", " "]

    # API Key（从系统环境变量读取，不在 .env 中明文存储）
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY")

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()  # 模块级单例
```

**关键设计**：API Key 不在 `.env` 文件中存储，而是通过系统环境变量注入，避免密钥泄露。

### 3.2 依赖注入 — `core/dependencies.py`

手动实现轻量级 DI 容器，混合使用单例和工厂模式：

```python
# === 单例（整个应用生命周期内复用）===
_vector_store: VectorStoreManager = None      # Milvus 连接
_embedding_service: EmbeddingService = None   # 嵌入模型客户端

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreManager()
    return _vector_store

# === 工厂（每次请求创建新实例）===
def get_document_service():
    return DocumentService()      # 轻量级，无状态

def get_llm_service():
    return LLMService()           # 每次新建 ChatOpenAI 实例

def get_retrieval_service():
    return RetrievalService(get_vector_store())  # 注入单例依赖
```

**对比 Spring**：Python 生态中，FastAPI 原生的 `Depends()` 提供了更优雅的方式，但手动 DI 更直观、学习成本更低。

### 3.3 文档解析服务 — `services/document_service.py`

支持 4 种文档格式的解析与智能分块：

```python
class DocumentService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,        # 每块 500 字符
            chunk_overlap=50,      # 重叠 50 字符（保持语义连贯）
            separators=["\n\n","\n","。",".","?","？","!","！",""," "]
            # 优先按段落切，其次换行，再按句号/问号/感叹号，最后按字符
        )

    async def process_file(self, file_path, file_name):
        # 1. 按扩展名选择加载器
        if file_name.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_name.endswith('.txt') or file_name.endswith('.md'):
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_name.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_name}")

        # 2. 加载文档
        documents = loader.load()

        # 3. 分割成块
        chunks = self.text_splitter.split_documents(documents)

        # 4. 提取文本和元数据
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [{
            "source": file_path,
            "file_name": file_name,
            "chunk_index": i,
            "page": chunk.metadata.get("page", -1)
        } for i, chunk in enumerate(chunks)]

        return texts, metadatas
```

**分块策略解析**：

```
原始文本: "第一章 概述\n\n1.1 背景。这是内容。\n1.2 目标。更多内容。"
          ↓ 按分隔符优先级递归切割
块1: "第一章 概述\n\n1.1 背景。这是内容。"  (500字符以内)
块2: "这是内容。\n1.2 目标。更多内容。"    (50字符重叠，保持语义)
```

### 3.4 嵌入服务 — `services/embedding_service.py`

将文本转换为高维向量，供语义搜索使用：

```python
from langchain_community.embeddings import DashScopeEmbeddings

class EmbeddingService:
    def __init__(self):
        self.embedding = DashScopeEmbeddings(
            model="text-embedding-v4",       # 阿里云嵌入模型
            dashscope_api_key=settings.DASHSCOPE_API_KEY
        )

    def get_embedding(self, text: list) -> list:
        """批量向量化（文档入库时使用）"""
        return self.embedding.embed_documents(text)   # 每段 → 1024维向量

    def get_query_embedding(self, query: str) -> list:
        """单条向量化（用户提问时使用）"""
        return self.embedding.embed_query(query)       # 问题 → 1024维向量
```

**关键概念**：`embed_documents()` 和 `embed_query()` 即使调用同一个模型，内部处理策略也可能不同（查询可能会加前缀指令），因此 API 上是分开的。

### 3.5 向量数据库 — `repositories/vector_store.py`

管理 Milvus 的完整生命周期：建表 → 入库 → 搜索。

```python
class VectorStoreManager:
    def init_collection(self, embedding_function):
        # 1. 连接 Milvus
        client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")

        # 2. 定义 Schema
        schema = client.create_schema(
            auto_id=True,          # 主键自增
            enable_dynamic_field=True  # 允许动态元数据字段
        )
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=1024)  # 对齐 text-embedding-v4
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("file_name", DataType.VARCHAR, max_length=512)

        # 3. 创建索引（IVF_FLAT + COSINE 相似度）
        index_params = client.prepare_index_params()
        index_params.add_index("vector", IndexType.IVF_FLAT,
                               metric_type="COSINE", params={"nlist": 128})

        # 4. 创建 LangChain 封装层
        self.store = Milvus(
            collection_name="snail_store",
            embedding_function=embedding_function,  # 内部自动调用向量化
            connection_args={"uri": "...", "token": "..."},
            auto_id=True,
            text_field="text"
        )

    def add_documents(self, texts, metadatas):
        """入库文档块（内部自动向量化并写入 Milvus）"""
        return self.store.add_texts(texts, metadatas)

    def search(self, query, top_k=3):
        """语义搜索：问题向量化 → Milvus ANN 搜索 → 返回相似文档"""
        return self.store.similarity_search_with_score(query=query, k=top_k)
```

**Schema 设计要点**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INT64 (PK) | 自增主键 |
| `vector` | FLOAT_VECTOR(1024) | 文本的向量表示 |
| `text` | VARCHAR(65535) | 原始文本内容 |
| `file_name` | VARCHAR(512) | 来源文件名 |
| 动态字段 | — | `enable_dynamic_field=True` 允许任意附加元数据 |

### 3.6 检索服务 — `services/retrieval_service.py`

负责语义搜索 + 将结果组装成 LLM 可理解的上下文：

```python
class RetrievalService:
    def retrieve(self, query, top_k=None):
        """执行语义搜索"""
        docs_with_scores = self.vector_store.search(query, top_k)
        return [{
            "content": doc.page_content,
            "score": round(score, 4),          # 相似度分数，保留4位
            "source": doc.metadata.get("source", ""),
            "page": doc.metadata.get("page", -1)
        } for doc, score in docs_with_scores]

    def build_context(self, query, top_k=None, retrieved_results=None):
        """将检索结果格式化为 LLM 输入"""
        results = retrieved_results or self.retrieve(query, top_k)
        parts = []
        for i, item in enumerate(results, 1):
            parts.append(
                f"[文档{i} 来源:{item['source']} 第{item['page']}页]\n"
                f"{item['content']}"
            )
        return "\n\n".join(parts)
```

**组装后的上下文示例**：

```
[文档1 来源:产品手册.pdf 第3页]
本产品支持最高 100W 快充，兼容 PD3.0 协议...

[文档2 来源:FAQ.md 第-1页]
Q: 充电器是否支持 iPhone？ A: 支持，需使用 C-to-Lightning 线...

[文档3 来源:产品手册.pdf 第8页]
安全警告：请勿在潮湿环境中使用充电器...
```

### 3.7 LLM 服务 — `services/llm_service.py`

基于检索到的上下文，调用大模型生成答案：

```python
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

class LLMService:
    def __init__(self):
        self.model = ChatOpenAI(
            model="qwen3-max",
            temperature=0.3,                             # 低温度 = 更确定性
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=settings.DASHSCOPE_API_KEY
        )

    def generate(self, query: str, context: str) -> str:
        system_prompt = """你是一个基于内部文档的问答助手。
- 严格基于参考资料回答问题
- 如果资料中没有相关信息，请如实说"未找到相关信息"
- 回答时请注明信息来源
- 不要编造任何资料中没有的内容"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"[参考资料]\n{context}\n\n[用户问题]\n{query}")
        ]
        response = self.model.invoke(messages)
        return response.content
```

**系统提示词设计理念**：通过强约束让模型"戴着镣铐跳舞"——既能发挥语言能力，又不会越界编造。

---

## 第四章 API 接口设计

### 4.1 接口总览

| 方法 | 路径 | 功能 | 请求 | 响应 |
|------|------|------|------|------|
| POST | `/api/v1/upload` | 上传文档 | `multipart/form-data` file | `{message, filename, chunks, file_size}` |
| POST | `/api/v1/ask` | 提问 | `{question, top_k}` | `{answer, references[{source, page, score}]}` |

### 4.2 上传接口

```python
@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    # 1. 防路径穿越：只取文件名，丢弃路径部分
    safe_name = Path(file.filename).name
    file_path = os.path.join(settings.UPLOAD_DIR, safe_name)

    # 2. 保存文件
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # 3. 解析 + 分块
    doc_service = get_document_service()
    texts, metadatas = await doc_service.process_file(file_path, safe_name)

    # 4. 向量化 + 入库
    vector_store = get_vector_store()
    vector_store.add_documents(texts, metadatas)

    return UploadResponse(
        message="文件上传并处理成功",
        filename=safe_name,
        chunks=len(texts),
        file_size=len(content)
    )
```

**安全注意**：`Path(file.filename).name` 是关键的安全措施——即使用户上传 `../../../etc/passwd`，也只会提取 `passwd` 作为文件名，防止路径穿越攻击。

### 4.3 问答接口

```python
@router.post("/ask", response_model=QuestionResponse)
async def ask_question(req: QuestionRequest):
    # 1. 语义检索
    retrieval_svc = get_retrieval_service()
    results = retrieval_svc.retrieve(req.question, req.top_k)

    # 2. 构建上下文
    context = retrieval_svc.build_context(req.question, req.top_k, results)

    # 3. 大模型生成
    llm_svc = get_llm_service()
    answer = llm_svc.generate(req.question, context)

    # 4. 组装引用
    refs = [ReferenceItem(source=r["source"], page=r["page"], score=r["score"])
            for r in results]

    return QuestionResponse(answer=answer, references=refs)
```

---

## 第五章 完整业务流程图

### 5.1 文档上传流程

```
用户浏览器                     FastAPI 后端                      外部服务
   │                              │                               │
   │  POST /upload (file)         │                               │
   │─────────────────────────────>│                               │
   │                              │  保存到 ./uploads/             │
   │                              │  DocumentService.process_file()│
   │                              │  ├─ 按扩展名选 Loader          │
   │                              │  ├─ loader.load()              │
   │                              │  └─ splitter.split_documents() │
   │                              │     (500字/块, 50字重叠)       │
   │                              │                               │
   │                              │  VectorStore.add_documents()   │
   │                              │──────────────────────────────>│
   │                              │    内部调用 EmbeddingService   │
   │                              │    ──────────────────────────>│ DashScope
   │                              │    <─ 1024维向量 ─────────────│ text-embedding-v4
   │                              │    写入 Milvus snail_store     │
   │                              │──────────────────────────────>│ Milvus
   │                              │                               │
   │  {message, filename,         │                               │
   │   chunks, file_size}         │                               │
   │<─────────────────────────────│                               │
```

### 5.2 问答流程

```
用户浏览器                     FastAPI 后端                      外部服务
   │                              │                               │
   │  POST /ask {question,top_k}  │                               │
   │─────────────────────────────>│                               │
   │                              │  RetrievalService.retrieve()  │
   │                              │  ├─ EmbeddingService           │
   │                              │  │  .get_query_embedding()     │
   │                              │  │────────────────────────────>│ DashScope
   │                              │  │  <── 1024维查询向量 ───────│ text-embedding-v4
   │                              │  │                             │
   │                              │  └─ Milvus.similarity_search   │
   │                              │──────────────────────────────>│ Milvus
   │                              │  <── [(doc,score),...] ──────│ IVF_FLAT COSINE
   │                              │                               │
   │                              │  RetrievalService              │
   │                              │  .build_context()              │
   │                              │  组装: [文档N 来源:...第X页]   │
   │                              │                               │
   │                              │  LLMService.generate()         │
   │                              │  SystemMessage: 严格基于文档   │
   │                              │  HumanMessage: 上下文+问题      │
   │                              │──────────────────────────────>│ DashScope
   │                              │  <── 生成的答案 ──────────────│ qwen3-max
   │                              │                               │
   │  {answer, references}        │                               │
   │<─────────────────────────────│                               │
```

---

## 第六章 关键技术决策

### 6.1 为什么用 DashScope 而不是直接调用 OpenAI？

| 维度 | OpenAI | DashScope (阿里云百炼) |
|------|--------|----------------------|
| 国内访问 | 需代理 | 直接访问 |
| 数据合规 | 数据出境 | 国内处理 |
| 中文能力 | 强 | 更强（千问系列专门优化） |
| API 兼容 | 原生 | OpenAI 兼容模式，代码改动最小 |
| 成本 | 按 token | 按 token，国内结算更方便 |

项目通过 `base_url` 指向 DashScope 的 OpenAI 兼容端点，使得 `langchain_openai.ChatOpenAI` 可以直接使用，无需额外适配。

### 6.2 为什么分块大小是 500？

```
太小 (100字符):         太大 (2000字符):
 语义碎片化 ❌            检索精度下降 ❌
 上下文不足 ❌            混入无关信息 ❌
 检索召回率高 ✅          单块覆盖完整 ✅

500 字符是经验平衡点：足够承载一个完整的语义单元，又不会因过长而混入噪声。
50 字符重叠则确保跨块边界的信息不会丢失。
```

### 6.3 为什么用 COSINE 而不是 L2 距离？

- **COSINE（余弦相似度）**：只关心方向，不关心长度。对文本向量来说，语义由方向决定——两段话的向量方向越接近，语义越相似。
- **L2（欧氏距离）**：同时受方向和长度影响。文本长度差异会导致向量长度不同，干扰语义判断。

业界实践：文本嵌入模型普遍使用余弦相似度。

---

## 第七章 系统启动过程

```python
# main.py — FastAPI 生命周期
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时：初始化向量数据库连接
    await init_vector_store()    # → 连接 Milvus → 创建 Schema/索引
    yield                         # → 对外提供服务
    # 关闭时：清理资源
    print("应用关闭")

# 初始化顺序
init_vector_store():
    1. 创建 VectorStoreManager 单例
    2. 创建 EmbeddingService 单例（含 DashScope 客户端）
    3. vector_store.init_collection(embedding_service.embedding)
       ├─ pymilvus 连接 Milvus
       ├─ 检查/创建 Collection snail_store
       ├─ 创建 Schema (id, vector, text, file_name)
       ├─ 创建索引 (IVF_FLAT, COSINE)
       └─ 实例化 LangChain Milvus 封装
```

---

## 第八章 不足与改进方向

### 8.1 当前存在的问题

| 问题 | 影响 | 改进方向 |
|------|------|----------|
| `process_file()` 标记 async 但同步阻塞 | 大文件上传时阻塞事件循环 | 使用 `run_in_executor` 放入线程池 |
| 无检索结果缓存 | 相同问题重复检索 | 引入 Redis 缓存检索结果 |
| 上传文件永久存储 | 磁盘空间持续增长 | 增加文件过期清理机制 |
| 明文 token 硬编码 | 安全风险 | 全部走环境变量 |
| 每次请求新建 `ChatOpenAI` | 连接开销 | 改为单例 + 连接池复用 |

### 8.2 可扩展的功能

1. **多轮对话**：加入 `ChatMessageHistory`，支持追问
2. **混合检索**：BM25 关键词 + 向量语义，提升召回率
3. **重排序（Rerank）**：初检 → 精排，进一步提升精度
4. **文档管理**：列表查看/删除已上传文档
5. **用户认证**：多租户隔离，每人独立知识库

---

## 第九章 总结

这个 RAG 知识库项目演示了一套**完整可用的企业级文档问答方案**，核心链路为：

```
文档上传 → 分块 → 向量化 → 存入 Milvus
用户提问 → 向量化 → Milvus ANN 搜索 → 组装上下文 → LLM 生成 → 带引用的答案
```

项目使用了 **Python 主流 AI 技术栈**：FastAPI + LangChain + Milvus + DashScope(千问)，以不到 500 行代码实现了一个完整的 RAG 系统，是学习 AI 应用开发的优秀范例。

**核心收获**：
- RAG 的本质是"检索 + 生成"，检索决定上限，生成决定下限
- 分块策略直接影响检索质量
- 系统提示词的设计决定了 LLM 是否"听话"
- LangChain 虽重但统一了 API，降低切换成本