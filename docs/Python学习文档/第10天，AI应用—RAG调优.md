# RAG 调优：从问题到优化的完整逻辑

## 概述

RAG（Retrieval-Augmented Generation）系统的质量由三个环节共同决定：**文档处理 → 检索召回 → 答案生成**。任何一个环节的短板都会成为整个系统的瓶颈。

本文档围绕实际项目 `python_study/rag` 的代码实现，按"**问题是什么 → 为什么需要优化 → 怎么优化 → 优化后提升在哪里**"的逻辑，系统梳理每一处调优的来龙去脉。

---

## 1、文档处理调优

文档处理是 RAG 的入口环节，决定了知识以什么形态存入向量库。如果这一环节处理不当，后续的检索和生成都无法弥补。

### 1.1、选择合适的文本分割器 —— 解决"切块尺寸与语义完整性的矛盾"

#### 优化什么问题？

文本切块（Chunk）是 RAG 检索精准度的第一道关口。切块策略面临一个根本性矛盾：

| 问题方向 | 具体表现 | 导致的后果 |
|:---------|:---------|:-----------|
| **块过大** | 单块混杂多段独立语义，向量被无关内容稀释 | 检索时匹配到主题偏离的内容，产生**检索噪音**；无法精准定位用户真正关心的那一段信息 |
| **块过小** | 连贯的语义/逻辑/代码被强行割裂 | 向量只能表征碎片内容，丢失完整语境，检索到的内容**残缺不全**，无法支撑完整回答 |

> **核心矛盾**：大块引入噪音，小块丢失上下文 —— 两者此消彼长，不存在"一刀切"的最优解。

#### 怎么优化？

针对不同文档类型的内容特征，差异化设定切块参数：

**方案一：通用文档 → 500 字符 + 50 重叠（均衡基准）**

适用 PDF、TXT、Word 等常规文档。这类文档以叙事/说明/陈述为主，段落长短适中。

- **为什么是 500 字符**：恰好容纳一个完整段落或一个独立语义单元，不会把多个主题混在一起，也不会把一句话切成两半。
- **为什么 50 字符重叠**：弥补相邻切块边界的语义断层，防止跨块关联信息丢失。
- **分隔符优先级** `["\n\n", "\n", "。", ".", " "]`：优先在自然语义边界（段落、句子）处切分，而非生硬截断字符。

```python
# python_study/rag/app/services/document_service.py
self.general_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", ".", " "],
    length_function=len,
)
```

**方案二：技术文档 → 800 字符 + 100 重叠（保留代码完整上下文）**

适用技术文档、API 手册、代码注释等。特点是包含代码块、长逻辑流程、连续参数说明，语义高度绑定、不可拆分。

- **为什么放大到 800 字符**：代码块 + 其说明文字往往远超 500 字符，小切块会把一个函数的代码和注释拆到两个块里，检索时拿到残缺代码完全失去实用价值。
- **为什么提升到 100 字符重叠**：技术文档跨块的逻辑关联更强，更大的重叠确保技术逻辑串联。
- **分隔符额外增加 Markdown 标题层级** `["\n## ", "\n### ", ...]`：技术文档大量使用 `##`/`###` 组织章节，优先在章节边界切分。

```python
# python_study/rag/app/services/document_service.py
self.tech_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n## ", "\n### ", "\n\n", "\n", "。", " "],
)
```

**方案三：对话/短文本 → 300 字符 + 30 重叠（提升检索精细度）**

适用聊天记录、客服对话、问答记录等。特点是单轮对话语义独立、内容简短、用户查询粒度细。

- **为什么缩小到 300 字符**：对话类内容每轮发言本身就是独立语义单元，大切块会把多轮不同主题的对话合并，检索时定位偏差严重。小切块精准定位到单轮对话。
- **为什么降低到 30 字符重叠**：对话块本身语义独立，过大的重叠反而引入相邻轮次的噪音。

```python
# python_study/rag/app/services/document_service.py
self.chat_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
    separators=["\n\n", "\n", "。", ".", " "],
)
```

**方案四：Markdown 专属标题分割 → 保留文档结构化语义**

Markdown 依靠 `#`/`##`/`###` 组织章节结构，标题本身就是天然的语义分界。普通字符分割会破坏这种结构。

- **先按标题做大层级拆分**，再复用通用字符分割器做二次细切：既保住文档的原生章节骨架，又控制单块大小不过长。
- **`strip_headers=False`**：保留标题在文本块中，检索返回结果时自带章节信息，LLM 能更好地理解内容归属。

```python
# python_study/rag/app/services/document_service.py
self.md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ],
    strip_headers=False,  # 保留标题在内容中
)
```

#### 优化后提升在哪里？

| 维度 | 优化前（一刀切） | 优化后（场景适配） |
|:-----|:----------------|:-------------------|
| 通用文档检索 | 块大小不当导致噪音或语义断裂 | 500 字符精准命中段落级语义 |
| 技术文档代码检索 | 代码被拆碎，检索返回残缺片段 | 800 字符完整保留函数级代码+注释 |
| 对话内容检索 | 多轮无关对话混在一起，定位不准 | 300 字符精确匹配单轮问答 |
| Markdown 结构 | 章节结构被破坏，检索结果归属不明 | 标题层级保留，结果自带章节上下文 |

---

### 1.2、分批写入向量库 —— 解决"全量写入导致的内存与可靠性问题"

#### 优化什么问题？

原始做法是将全部文本块一次性传入向量库 `add_texts`：

```python
# ❌ 原始写法（不推荐）
self.store.add_texts(texts, metadata_list)
```

这带来四个严重隐患：

| 问题 | 原因 | 后果 |
|:-----|:-----|:-----|
| **内存溢出** | 大文档切块数成千上万，全量列表同时加载到内存 | 服务 OOM 崩溃，大文件/批量上传场景风险极高 |
| **接口超时/失败** | 向量数据库（Milvus 等）对单次请求体量有限制 | 请求包过大触发限流、超时、连接中断，整批写入失败 |
| **无断点续传** | 全量写入中途报错，无法区分哪些已入库 | 全部数据作废，只能重头再来 |
| **阻塞主线程** | 大同步写入耗时长 | 影响后续请求响应，并发能力下降 |

> **本质问题**：全量写入把"一批数据"和"一次调用"强绑定，任何一个环节出问题都导致全局失败。

#### 怎么优化？

**核心思路**：将大任务拆分为可控的小批次，解耦"数据总量"和"单次调用"。

```python
# ✅ 优化后：分批写入
# python_study/rag/app/repositories/vector_store.py - add_documents()
batch_size = 200
total = len(texts)

for i in range(0, total, batch_size):
    batch_texts = texts[i:i + batch_size]
    batch_metadata = metadata_list[i:i + batch_size]

    vectors = self.embedding.embed_documents(batch_texts)
    data = []
    for j, (text, metadata, vector) in enumerate(
        zip(batch_texts, batch_metadata, vectors)
    ):
        data.append({
            "vector": vector,
            "text": text,
            "file_name": metadata.get("file_name", ""),
            "doc_id": metadata.get("doc_id", ""),
            "chunk_index": metadata.get("chunk_index", i + j),
        })

    self.client.insert(collection_name=self.collection_name, data=data)
    print(f"已写入 {min(i + batch_size, total)}/{total} 条")
```

#### 优化后提升在哪里？

| 维度 | 优化前 | 优化后 |
|:-----|:------|:------|
| **内存** | 全量加载，大文档易 OOM | 每次仅 200 条常驻，内存峰值固定可控 |
| **请求成功率** | 大报文易超时/被拒绝 | 小批次适配向量库接口限制，成功率大幅提升 |
| **故障影响** | 全量失败，需重跑全部 | 仅当前批次受影响，前面批次数据保留有效 |
| **容错重试** | 无法定位失败位置 | 可针对失败批次单独重试 |
| **进度可见** | 黑盒，不知写到哪 | 实时输出进度，排查卡顿/丢失有据可查 |

---

### 1.3、文档去重与更新 —— 解决"重复上传浪费资源，内容更新无法处理"

#### 优化什么问题？

RAG 系统在实际使用中面临两个看似矛盾的需求：

| 场景 | 问题 |
|:-----|:-----|
| **重复上传** | 用户反复上传内容完全相同的文件 → 向量库出现大量重复向量 → 浪费存储空间 + Embedding API 调用费用 + 检索时返回多条相同内容挤占 LLM 上下文窗口 |
| **内容更新** | 用户上传了内容已变化的同名文件 → 这是合理更新需求，不应被当作重复拒绝，但旧版本数据必须清理 |

> **核心难点**：如何区分"完全重复"和"内容变更"？拒绝重复但不能拒绝更新。

#### 怎么优化？

引入 **SHA-256 内容哈希**作为文档唯一指纹，配合**文件名**实现双维度管理：

| 维度 | 用途 |
|:-----|:-----|
| **doc_id（内容哈希）** | 判断内容是否完全重复 —— 哈希相同 = 内容相同 = 拒绝入库 |
| **file_name（文件名）** | 定位旧版本数据并删除 —— 内容变化后哈希也变，无法用新哈希删除旧数据，必须依赖文件名 |

**完整流程**：

```
文件上传
   ↓
计算 SHA-256 内容哈希
   ↓
用哈希查向量库是否已存在
   ├── 存在 → 拒绝（重复上传，直接返回"文档已存在"）
   └── 不存在 → 按文件名删除旧数据 → 写入新数据
```

```python
# python_study/rag/app/api/routes.py - upload_document()
# 1. 计算内容哈希
doc_id = hashlib.sha256(content).hexdigest()

# 2. 查重：哈希存在 = 完全相同的文档已入库
if vector_store.check_doc_exists(doc_id):
    return UploadResponse(
        message="文档已存在，你可以直接提问",
        filename=file.filename,
        chunks=0,
        file_size=0
    )

# 3. 更新：先按文件名清理旧版本，再写入新版本
vector_store.delete_by_file_name(file.filename)
```

向量库 Schema 中新增 `doc_id` 字段并建立索引，支持快速查重：

```python
# python_study/rag/app/repositories/vector_store.py - init_collection()
schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=128)
index_params.add_index(field_name="doc_id", index_type="TRIE")
```

#### 优化后提升在哪里？

| 维度 | 优化前 | 优化后 |
|:-----|:------|:------|
| **存储效率** | 重复向量堆积浪费存储 | 同内容仅存一份 |
| **API 成本** | 重复调用 Embedding 浪费费用 | 哈希查重在前端拦截，零成本拒绝 |
| **检索质量** | 检索返回多条重复内容污染上下文 | 每条内容唯一，上下文窗口利用率最大化 |
| **更新支持** | 旧数据残留，新旧版本共存 | 先删后写，知识库始终保持最新版本 |
| **响应速度** | 重复文件走完整处理流程 | 哈希命中直接返回，毫秒级响应 |

---

## 2、Embedding 模型选择 —— 解决"不同场景对精度、速度、成本的需求冲突"

#### 优化什么问题？

Embedding 模型将文本转为向量，是语义检索的基石。但不存在"万能模型"——不同模型在精度、速度、部署成本三个维度上存在根本性 trade-off：

- **精度高的模型**（如 BGE-large，1024 维）需要 GPU，部署成本高
- **速度快的模型**（如 BGE-small，512 维）精度有限，复杂语义检索效果差
- **API 模型**（如 DashScope）免部署但按量计费，大规模使用成本不可控

> **核心矛盾**：没有一种模型在所有场景下都是最优解，需要根据实际条件做权衡。

#### 四种可选方案对比

| 维度 | BGE-small | BGE-large | M3E-base | DashScope（当前项目使用） |
|:-----|:----------|:----------|:---------|:--------------------------|
| **部署方式** | 本地部署 | 本地部署 | 本地部署 | API 调用，无需部署 |
| **向量维度** | 512 | 1024 | 768 | 1024+（取决于模型版本） |
| **硬件需求** | CPU 即可 | 需 GPU（~1.3GB 显存） | CPU 可跑（较慢） | 零本地资源 |
| **调用成本** | 免费（仅电费） | 免费（仅电费） | 免费（仅电费） | 按 Token 计费 |
| **检索精度** | ★★★ | ★★★★★ | ★★★★ | ★★★★★ |
| **推理速度** | 快 | 慢（无 GPU 不可用） | 中等 | 取决于网络 |
| **适用场景** | 原型验证、小规模 | 高精度生产、大规模知识库 | 不想用 API 但需一定精度 | 快速上线、免运维、弹性伸缩 |

```python
# python_study/rag/app/services/embedding_service.py
# 当前项目采用 DashScope（阿里云 API），免去本地模型部署和 GPU 依赖
self.embedding = DashScopeEmbeddings(
    model=settings.EMBEDDINGS_MODEL_NAME,
    dashscope_api_key=settings.DASHSCOPE_API_KEY,
)

# 其他可选方案（代码中已注释保留，按需切换）：
# BGE-small: HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5", ...)
# BGE-large: HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5", ...)
# M3E-base:  HuggingFaceEmbeddings(model_name="moka-ai/m3e-base", ...)
```

#### 优化后提升在哪里？

- **免运维**：不需要下载和管理 GB 级别的模型文件，不需要配置 GPU 环境
- **弹性伸缩**：阿里云 API 自带并发能力，无需自行搭建模型服务集群
- **精度保障**：云端模型持续迭代，始终使用最新版本，精度通常优于开源小模型
- **快速切换**：代码中保留多种方案的注释代码，切换模型只需改一行配置

---

## 3、检索优化 —— 解决"纯向量检索的关键词盲区"

### 3.1、为什么纯向量检索不够？

纯向量检索（Dense Retrieval）基于 Embedding 的语义相似度进行匹配，存在一个**致命盲区**：

```
用户查询："Python 3.12 的新特性有哪些？"

纯向量检索可能返回：
✅ "Python 3.11 版本更新说明"        ← 语义高度相似，但版本错了！
❌ 漏掉精确包含 "Python 3.12" 的文档  ← 关键词完全匹配但语义向量不够突出

问题根源：Embedding 将 "3.12" 和 "3.11" 映射到向量空间中非常接近的位置，
         模型认为它们"语义相似"，但用户需要的是精确匹配。
```

> **本质问题**：Embedding 做的是**语义泛化**，把相似的词映射到相近的向量；但检索有时需要**精确匹配** —— 数字、日期、代码、专有名词、产品型号等不容模糊。

### 3.2、优化方案：BM25 稀疏向量 + 混合检索

**核心思路**：用 BM25（基于词频的关键词匹配）弥补向量检索的精确匹配盲区。

#### Step 1：在 Milvus 中建立 BM25 检索能力

在集合 Schema 中新增 `sparse` 稀疏向量字段，利用 Milvus 内置的 BM25 Function 自动从 `text` 字段生成词频向量：

```python
# python_study/rag/app/repositories/vector_store.py - init_collection()
# 1. text 字段配置中文分词器（jieba）
schema.add_field(
    field_name="text",
    datatype=DataType.VARCHAR,
    enable_analyzer=True,
    analyzer_params={"tokenizer": "jieba"},  # 中文分词，解决原生分词器对中文不友好
    max_length=65535
)

# 2. 稀疏向量字段（存储 BM25 词频向量）
schema.add_field(
    field_name="sparse",
    datatype=DataType.SPARSE_FLOAT_VECTOR
)

# 3. BM25 Function：自动从 text → sparse
bm25_function = Function(
    name="text_bm25_emb",
    input_field_names=["text"],
    output_field_names=["sparse"],
    function_type=FunctionType.BM25,
)
schema.add_function(bm25_function)

# 4. 稀疏向量索引（BM25 专用）
index_params.add_index(
    field_name="sparse",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="BM25",
    params={
        "inverted_index_algo": "DAAT_MAXSCORE",  # 高效检索算法
        "bm25_k1": 1.2,   # 词频饱和参数：控制词频对评分的影响程度
        "bm25_b": 0.75     # 文档长度归一化：削弱长文档天然的词频优势
    }
)
```

#### Step 2：混合检索 —— 向量 + BM25 加权融合

```python
# python_study/rag/app/repositories/vector_store.py - _hybrid_search()
def _hybrid_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
    """
    混合搜索：向量语义搜索 + BM25 关键词搜索，加权融合

    权重设计逻辑：
    - 向量搜索 0.7：语义理解为主，保证召回广度和同义表达覆盖
    - BM25 0.3：关键词匹配为辅，保证专有名词/数字/代码等精确匹配不遗漏
    - BM25 无结果时自动回退纯向量搜索（兜底）
    """
    vector_results = self._vector_search(query, top_k)
    bm25_results = self._bm25_search(query, top_k)

    # 任一路无结果时自动回退
    if not bm25_results:
        return vector_results
    if not vector_results:
        return bm25_results

    # 加权融合
    merged = {}
    for result in vector_results:
        merged[result['id']] = {**result, 'hybrid_score': result['score'] * 0.7}

    for result in bm25_results:
        if result['id'] in merged:
            merged[result['id']]['hybrid_score'] += result['score'] * 0.3
        else:
            merged[result['id']] = {**result, 'hybrid_score': result['score'] * 0.3}

    # 按融合分数降序排列
    return sorted(merged.values(), key=lambda x: x['hybrid_score'], reverse=True)
```

#### 优化后提升在哪里？

| 维度 | 纯向量检索 | 混合检索 |
|:-----|:----------|:---------|
| **精确关键词** | "Python 3.12"可能返回"Python 3.11" | BM25 精准命中包含 "3.12" 的文档 |
| **语义理解** | 强（同义词、近义词可召回） | 更强（向量 0.7 权重保持语义覆盖） |
| **专有名词** | 容易被泛化模糊 | BM25 精确匹配，不受 Embedding 泛化影响 |
| **召回率** | 单一信号，存在盲区 | 双路召回互补，盲区大幅缩小 |
| **鲁棒性** | BM25 无结果时没有 fallback | 任一路失败自动回退另一路 |

---

## 4、重排序优化 —— 解决"语义相似 ≠ 信息相关"的根本矛盾

### 4.1、为什么检索结果需要重排序？

这是整个 RAG 调优中最关键的一环，因为它解决的是一个**根本性架构矛盾**：

#### 矛盾一：语义相似 ≠ 信息相关

```
用户查询："如何申请年假？"

向量检索按 cosine 相似度排序返回：
  Top 1: "年假管理制度的历史沿革"（相似度 0.95）← 语义高度相似，但与"申请流程"无关！
  Top 2: "年假申请流程说明"（相似度 0.88）      ← 这才是用户真正需要的

问题：向量检索把"语义相似"当作"信息相关"，但两者不是等价关系。
```

#### 矛盾二：Embedding 的信息压缩损失

将非结构化文本压缩为固定维度向量（如 1024 维），必然丢失信息：

- **精确信息模糊化**：数字、日期、金额等在向量中被稀释
- **逻辑关系断裂**：因果关系、条件关系难以在向量空间保留
- **专有名词退化**：特定术语、产品型号被泛化到相近概念

#### 矛盾三：双塔架构的先天局限

主流 Embedding 模型（BERT 系列）采用**双塔架构**：

```
Query  → Encoder → Query Vector    ↘
                                      → cosine(Query Vector, Document Vector) = 相似度
Document → Encoder → Document Vector ↗

核心局限：Query 和 Document 在编码阶段完全独立，没有任何交互。
         模型无法知道"查询在问什么"和"文档在答什么"之间的细粒度匹配关系。
```

> **一句话总结**：初检阶段为了效率使用 ANN 近似检索 + 单一相似度指标，必然导致**排序不准**——真正高质量的文档可能排在后面，与查询"看起来像"但无关的文档占据前排。

### 4.2、优化方案：Cross-encoder 重排序

#### Cross-encoder vs 双塔架构

```
双塔（Bi-encoder，初检用）：
  Query → Encoder → Vector
  Doc   → Encoder → Vector
  相似度 = cosine(V_q, V_d)
  特点：快（向量可预计算），但 Query 和 Doc 零交互

Cross-encoder（重排序用）：
  [Query, Doc] → 完整 Transformer → Relevance Score
  特点：Query 和 Doc 在每一层都有 Full Attention 交互，
        能捕捉词语级的精确匹配、逻辑推理、意图对齐
```

#### 级联检索架构：粗筛 → 精排

```
完整文档集（百万级）
    ↓ 多路召回（向量 + BM25）—— 追求召回率，允许噪音
候选文档（~30 个）  ← recall_k = top_k × 3
    ↓ Cross-encoder 重排序 —— 追求精准度，剔除噪音
精选文档（3~5 个）  ← 送入 LLM 生成答案
```

#### 代码实现

```python
# python_study/rag/app/repositories/vector_store.py - search()
def search(self, query, top_k=None, search_type="vector", use_rerank=False):
    # 第一阶段：多路召回（放大到 3 倍，给重排序留空间）
    recall_k = top_k * 3 if (use_rerank and self.reranker and self.reranker.available) else top_k

    if search_type == "hybrid":
        results = self._hybrid_search(query, recall_k)  # 混合检索粗筛
    # ...

    # 第二阶段：Cross-encoder 精排
    if use_rerank and self.reranker and self.reranker.available and len(results) > top_k:
        results = self.reranker.rerank(query, results, top_k)

    return results
```

```python
# python_study/rag/app/services/reranker_service.py
class RerankerService:
    """使用 DashScope TextReRank API（qwen3-rerank 模型）进行 Cross-encoder 重排序"""

    def rerank(self, query, documents, top_k):
        # 提取文档文本（截断到 2000 字符，避免超 token 限制）
        doc_texts = [doc.get("text", "")[:2000] for doc in documents]

        # 调用 Cross-encoder 重排序 API
        response = TextReRank.call(
            model="qwen3-rerank",
            api_key=self.api_key,
            query=query,
            documents=doc_texts,
            top_n=min(top_k, len(doc_texts)),
            return_documents=True
        )

        # 按 relevance_score 重新排序
        for item in response.output.results:
            original = documents[item.index].copy()
            original["rerank_score"] = round(item.relevance_score, 4)
            reranked.append(original)

        return reranked
```

```python
# python_study/rag/app/api/routes.py - ask_question()
# 实际调用：混合检索 + 重排序 = 最佳效果
retrieved = retrieval_service.retrieve(
    request.question,
    request.top_k,
    search_type="hybrid",
    use_rerank=True   # 开启 Cross-encoder 精排
)
```

#### 重排序的容错设计

```python
# 多层 fallback 保证可用性：
# 1. 服务不可用（无 API Key）→ 跳过重排序，直接返回初检结果
# 2. API 调用失败 → logger.error + 回退初检结果
# 3. 重排序无结果 → 回退初检结果
# 4. 文档数超过 API 限制（50 个）→ 自动截断前 50 个
```

### 4.3、优化后提升在哪里？

| 维度 | 无重排序 | 有重排序 |
|:-----|:--------|:---------|
| **排序准确性** | ANN 近似排序，噪音文档混入前排 | Cross-encoder 精确判断相关性，真实答案排第一 |
| **抗干扰能力** | "年假制度"和"年假申请"区分不开 | Cross-encoder 理解查询意图，精准区分 |
| **长文档处理** | 关键信息被大量无关文本稀释，排名靠后 | Query-Doc 交互能穿透噪音定位关键段落 |
| **上下文质量** | Top-3 中可能仅 1~2 条真正相关 | Top-3 全部精准命中，LLM 生成质量质的提升 |
| **幻觉率** | LLM 基于无关文档编造答案 | 文档精准相关，LLM 严格基于事实回答 |

> **量化效果**（业界基准）：重排序能将 Precision@K 提升 30-50%，答案准确率提升 20-40%，幻觉率降低 50% 以上。

---

## 5、完整优化链路总览

将以上所有优化串联起来，形成一条完整的 RAG 优化链路：

```
┌─────────────────────────────────────────────────────────────────┐
│                        文档处理层                                │
│  场景化切块 ──── 分批写入 ──── 内容哈希去重/更新                   │
│  (精准语义单元)   (稳定可靠入库)   (零冗余 + 支持更新)              │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                        Embedding 层                             │
│  DashScope API ──── 免部署/免GPU/弹性伸缩/持续迭代                 │
│  (可选切换 BGE/M3E 本地模型)                                      │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                         检索层                                   │
│  向量语义搜索 ──── + BM25 关键词搜索 ──── 加权融合(7:3)            │
│  (理解语义)         (精确匹配)            (取长补短)               │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                        重排序层                                  │
│  多路召回(30个) ──── Cross-encoder精排 ──── Top 3~5 精选          │
│  (追求召回率)         (追求精准度)           (送入LLM)             │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                        生成层                                    │
│  精选上下文 + System Prompt约束 ──── LLM 生成基于事实的答案         │
│  (严格基于文档，不编造)                                            │
└─────────────────────────────────────────────────────────────────┘
```

每一层解决一个特定问题，层层递进，最终实现从"用户上传文档"到"AI 精准回答"的完整闭环。

---

## 附录：项目文件结构

```
python_study/rag/app/
├── api/routes.py              # API 路由：上传/问答接口，含去重逻辑
├── core/config.py             # 全局配置：模型/数据库/切块参数
├── repositories/vector_store.py  # 向量库管理：Schema/索引/混合检索/重排序
├── services/
│   ├── document_service.py    # 文档处理：加载/场景化切块
│   ├── embedding_service.py   # Embedding：多种模型可选
│   ├── retrieval_service.py   # 检索服务：统一检索入口/格式化输出
│   ├── reranker_service.py    # 重排序服务：Cross-encoder 精排
│   └── llm_service.py         # LLM 服务：RAG 问答生成
└── main.py                    # FastAPI 应用入口
```
