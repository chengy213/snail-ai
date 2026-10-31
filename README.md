# Snail AI 🐌

**智能蜗牛** — 一个 Java 转 Python 学习实践 + AI 应用研究项目。

> 从「基础概念映射」到「RAG 智能问答」，见证一条完整的 Python + AI 成长曲线。

---

## 📦 项目构成

仓库根目录下的 `python_study/` 汇总了三个 Python 学习子项目（`fastapi` / `rag` / `langchain`），
再加上一个完整的 AI 应用工程 `snail-ai-oamr`：

### 1. `python_study/fastapi` — Python 学习实践（FastAPI 用户管理 CRUD）

从 Java 开发者的视角出发，系统性学习 Python、FastAPI、SQLAlchemy 等 Web 后端技术栈。

- **Web 框架**: FastAPI + Uvicorn
- **ORM**: SQLAlchemy 2.0 + PyMySQL（MySQL）
- **数据验证**: Pydantic / Pydantic-Settings
- **密码安全**: 标准库 `hashlib` + `secrets`（SHA-256 + 随机盐，恒定时间比较）
- **日志**: Loguru（控制台 + 按日切分的文件日志）
- **前端**: 纯 HTML/CSS/JS 单页应用（用户管理界面）
- **端口**: 8000

**配套学习文档**（`docs/Python学习文档/`）：

| 天次 | 内容 |
|------|------|
| 第1天 | 基础概念映射（Java → Python） |
| 第2天 | FastAPI 框架入门 |
| 第3天 | SQLAlchemy ORM |
| 第4天 | 异步编程 |
| 第5天 | 异常与日志 |
| 第6天 | 项目实战 — 用户管理（CRUD + 前端联调） |
| 第7天 | AI 应用 — 基础概念 |
| 第8天 | AI 应用 — Milvus 向量数据库 |
| 第9天 | AI 应用 — RAG 知识库 |
| 第10天 | AI 应用 — RAG 调优 |
| 第11天 | AI 应用 — LangChain（上）— 框架介绍与核心组件 |
| 第12天 | AI 应用 — LangChain（短期记忆）— Checkpointer + State + 上下文管理 |

---

### 2. `python_study/rag` — RAG 智能问答系统

基于通义千问 + Milvus + LangChain 的文档问答系统，历经多轮调优，形成完整的 RAG 优化链路。

- **LLM 模型**: 通义千问（DashScope OpenAI 兼容接口）— qwen3-max
- **嵌入模型**: DashScope text-embedding-v4，1024 维（支持 BGE / M3E 本地模型切换）
- **重排序模型**: qwen3-rerank
- **向量数据库**: Milvus（port 19530，集合 `snail_store`）
- **文档处理**: LangChain（支持 PDF / TXT / Markdown / DOCX）
- **文本分割**: 场景化分割器（通用/技术文档/对话，差异化 chunk_size & overlap）
- **Web 框架**: FastAPI + Uvicorn
- **前后端通信**: RESTful API（JSON 响应）
- **前端**: 静态文件挂载（`/` 即前端页面）
- **端口**: 8100

**RAG 优化链路**：

```
┌─────────────── 文档处理层 ───────────────┐
│ 场景化切块 → 分批写入 → 内容哈希去重/更新    │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────── Embedding 层 ──────────────┐
│ DashScope API（免部署/免GPU/弹性伸缩）       │
└──────────────────┬──────────────────────┘
                   ↓
┌──────────────── 检索层 ──────────────────┐
│ 向量语义搜索 + BM25 关键词搜索 → 加权融合    │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────── 重排序层 ─────────────────┐
│ 多路召回(30个) → Cross-encoder 精排 → Top3 │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────── 生成层 ───────────────────┐
│ 精选上下文 + System Prompt → 千问生成答案   │
└──────────────────────────────────────────┘
```

**调优要点**：

| 优化项 | 解决问题 | 方案 |
|:-------|:---------|:-----|
| 场景化切块 | 通用尺寸无法适配不同文档类型 | 通用 500/技术 800/对话 300 字符 + Markdown 标题分割 |
| 分批写入 | 全量写入内存溢出、超时、无容错 | batch_size=200，小批次循环写入，进度可见 |
| 内容哈希去重 | 重复上传浪费存储与 API 费用 | SHA-256 哈希判重 + 文件名定位旧版实现更新 |
| 混合检索 | 纯向量检索无法精确匹配关键词 | 向量(0.7) + BM25(0.3) 加权融合，补齐精确匹配盲区 |
| Cross-encoder 重排序 | "语义相似 ≠ 信息相关"的架构矛盾 | 多路召回 → qwen3-rerank 精排，Precision@K 提升 30-50% |

**API 接口一览**：

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/upload` | 上传文档并入库（含哈希去重 + 旧版清理） |
| POST | `/api/v1/ask` | 智能问答（混合检索 + 重排序 + 返回引用来源） |
| GET | `/docs` | Swagger UI 接口文档 |
| GET | `/redoc` | ReDoc 接口文档 |

---

### 3. `python_study/langchain` — LangChain 学习实践

从零开始学习 LangChain / LangGraph 的短期记忆，覆盖三种检查点后端与三种上下文压缩方式。

- **LLM 模型**: 通义千问（DashScope）— qwen-max / qwen-turbo
- **智能体**: `create_react_agent`（工具调用 + 检查点）与自建 `StateGraph`
- **检查点后端**: `MemorySaver`（内存）/ `SqliteSaver`（SQLite 文件）/ `PostgresSaver`（PostgreSQL）
- **上下文压缩**: `trim_messages` 裁剪、`RemoveMessage` 删除、历史自动摘要

**示例代码**（`python_study/langchain/shortTermMemorry/`）：

| 文件 | 说明 |
|:-----|:-----|
| `ShortTermMemory-1.py` | 基础短期记忆 — `MemorySaver` 内存检查点 + Agent |
| `ShortTermMemory-2.py` | 持久化记忆 — `SqliteSaver`，检查点写入 `checkpoints.db` |
| `ShortTermMemory-3.py` | 生产级记忆 — `PostgresSaver` + `checkpointer.setup()` 建表 |
| `trim_messages-1.py` | 消息修剪 — `trim_messages` 裁剪后用 `RemoveMessage` 回写状态 |
| `trim_messages-2.py` | 自定义清理节点 — `StateGraph` 中保留最近 6 条，其余删除 |
| `trim_messages-3.py` | 历史自动摘要 — `RunnableWithMessageHistory` 超阈值压缩成摘要 |

> `ShortTermMemory-2.py` 需额外安装 `langgraph-checkpoint-sqlite`，
> `ShortTermMemory-3.py` 需额外安装 `psycopg2-binary` 并准备 PostgreSQL 实例。

---

### 4. `snail-ai-oamr` — AI 智能聊天助手

基于 LangGraph 工作流的多 Agent 对话系统，支持问题分类、RAG 知识检索、流式 SSE 输出。

- **LLM 模型**: 通义千问（DashScope）— qwen3-max
- **嵌入模型**: text-embedding-v4
- **工作流引擎**: LangGraph（状态图 + 条件路由）
- **向量数据库**: Milvus（port 19530）
- **检索策略**: 混合检索（向量语义 + BM25 关键词）
- **中文分词**: jieba
- **前后端通信**: SSE（Server-Sent Events）流式输出
- **前端**: 深色主题 AI 聊天界面
- **端口**: 8100

**Agent 工作流**：

```
用户输入 → 保存记忆 → 问题分类
                          ├── 闲聊 → 闲聊Agent
                          ├── 业务问题 → 业务QA Agent（可接 RAG）
                          ├── 业务操作 → 操作Agent
                          ├── 系统问题 → 系统诊断Agent
                          └── 数据统计 → 数据统计Agent
                                    ↓
                              保存回复 → 输出
```

---

## 🏗️ 项目结构

```
snail-ai/
├── README.md                        # ← 本文档
├── LICENSE                          # MIT
│
├── docs/                            # 学习文档
│   └── Python学习文档/               # Python 学习日课（第1~12天）+ 短期记忆原始素材（新文件2.txt）
│
├── python_study/                    # Python 学习代码，导入以仓库根为起点（python_study.<子项目>.app.*）
│   ├── fastapi/                     # 🅰️ Python 学习 — FastAPI CRUD
│   │   ├── requirements.txt
│   │   ├── .env                     # 数据库配置（DB_HOST / DB_PORT / DB_USER / DB_PASSWORD / DB_NAME）
│   │   ├── app/
│   │   │   ├── main.py              # 应用入口 + 全局异常处理 + CORS
│   │   │   ├── core/
│   │   │   │   ├── config.py        # Pydantic Settings 配置管理
│   │   │   │   ├── database.py      # SQLAlchemy 引擎 & Session
│   │   │   │   ├── exceptions.py    # 自定义业务异常
│   │   │   │   └── logger.py        # 日志配置
│   │   │   ├── models/
│   │   │   │   └── user.py          # ORM 模型（SysUser）
│   │   │   ├── schemas/
│   │   │   │   └── user.py          # Pydantic 请求/响应模型
│   │   │   ├── services/
│   │   │   │   └── user.py          # 业务逻辑层
│   │   │   ├── api/v1/
│   │   │   │   └── user.py          # RESTful 路由
│   │   │   └── utils/
│   │   │       └── response.py      # 统一响应格式
│   │   └── static/
│   │       └── index.html           # 用户管理前端页面
│   │
│   ├── rag/                         # 🅱️ RAG 智能问答系统
│   │   ├── requirements.txt
│   │   ├── .env                     # DashScope + Milvus 配置
│   │   └── app/
│   │       ├── main.py              # FastAPI 入口 + 生命周期管理
│   │       ├── core/
│   │       │   ├── config.py        # Pydantic Settings 配置管理
│   │       │   └── dependencies.py  # 依赖注入（单例服务管理）
│   │       ├── api/
│   │       │   └── routes.py        # 上传 & 问答接口（含去重逻辑）
│   │       ├── services/
│   │       │   ├── document_service.py  # 文档解析 & 场景化切分
│   │       │   ├── embedding_service.py # 文本向量化（DashScope/BGE/M3E）
│   │       │   ├── llm_service.py       # 千问大模型调用
│   │       │   ├── retrieval_service.py # 检索 & 上下文构建
│   │       │   ├── context_manager.py   # 上下文窗口限长与截断
│   │       │   └── reranker_service.py  # Cross-encoder 重排序精排
│   │       ├── repositories/
│   │       │   └── vector_store.py  # Milvus 向量存储（混合检索+重排序）
│   │       └── static/
│   │           └── index.html       # RAG 问答前端页面
│   │
│   └── langchain/                   # 🅲️ LangChain 短期记忆示例
│       └── shortTermMemorry/        # 内存 / SQLite / PostgreSQL 三种检查点 + 裁剪 / 删除 / 摘要
│           ├── ShortTermMemory-1.py # MemorySaver 内存检查点
│           ├── ShortTermMemory-2.py # SqliteSaver 持久化（运行时生成 checkpoints.db）
│           ├── ShortTermMemory-3.py # PostgresSaver 生产级检查点
│           ├── trim_messages-1.py   # trim_messages + RemoveMessage 回写状态
│           ├── trim_messages-2.py   # StateGraph + cleanup 节点删除旧消息
│           └── trim_messages-3.py   # RunnableWithMessageHistory 自动摘要
│
└── snail-ai-oamr/                   # 🅳️ AI 智能聊天助手
    ├── frontend/
    │   └── index.html               # AI 聊天前端（深色主题）
    └── backend/
        ├── requirements.txt
        ├── main.py                  # FastAPI 入口（SSE 流式接口）
        ├── service.py               # ChatService 业务编排
        ├── config/
        │   ├── oamr.yml             # 模型配置（chat/embeddings）
        │   ├── prompts.yml          # 提示词文件路径
        │   └── vector_store.yml     # Milvus 向量库配置
        ├── prompts/                 # Agent 提示词（.txt）
        ├── base/
        │   └── state_models.py      # 状态定义（AgentState / ClassificationResult）
        ├── graph/
        │   ├── workflow.py          # LangGraph 工作流编排
        │   └── agent.py             # 各 Agent 实现（分类/闲聊/业务等）
        ├── model/
        │   └── factory.py           # 模型工厂（ChatTongyi / DashScopeEmbeddings）
        ├── rag/
        │   ├── vector_store.py      # Milvus 向量存储（纯向量检索）
        │   ├── hybrid_retriever.py  # 混合检索（向量 + BM25）
        │   └── RAG检索问题.md        # RAG 优化分析
        └── utils/
            ├── config_handler.py    # YAML 配置加载
            ├── path_tool.py         # 路径工具
            ├── prompt_loader.py     # 提示词加载
            └── logger_handler.py    # 日志处理
```

---

## 🚀 快速开始

### 前提

- Python 3.9+
- MySQL（`python_study/fastapi` 需要）
- Milvus 向量数据库（`python_study/rag` 和 `snail-ai-oamr` 需要）
- DashScope（通义千问）API Key（`python_study/rag`、`python_study/langchain` 和 `snail-ai-oamr` 需要）

> **导入约定**：`python_study/` 下三个子项目的源码统一使用 `python_study.<子项目>.app.*` 绝对导入，
> 因此命令都要在**仓库根目录**执行（`main.py` 内也做了 `sys.path` 兜底，直接 `python .../main.py` 同样可用）。

---

### 1️⃣ 启动用户管理 CRUD（python_study/fastapi）

```bash
# 安装依赖
# 方式一(推荐)：用 uv 来装，不需要给 venv 装 pip:
uv pip install -r python_study/fastapi/requirements.txt
# 方式二：把 pip 引导回 venv,之后就能按 README 用 pip install:
python3 -m ensurepip --upgrade
pip install -r python_study/fastapi/requirements.txt

# 配置数据库（编辑 python_study/fastapi/.env 或设置同名环境变量）
# DB_HOST=localhost  DB_PORT=3306  DB_USER=root  DB_PASSWORD=...  DB_NAME=demo

# 在仓库根目录启动服务
# 推进使用uv命令(推荐)
# 启动方式1(uv 等价)
uv run --no-sync python python_study/fastapi/app/main.py

# 启动方式2(uv 等价)
uv run --no-sync python -m uvicorn python_study.fastapi.app.main:app --reload

# 启动方式3(uv 等价)
uv run --no-sync uvicorn python_study.fastapi.app.main:app --reload

# 原理： uv run 会自动找到项目根目录的 pyproject.toml 和 .venv,用虚拟环境里的解释器/工具执行命令 
# —— 所以不需要手动 source .venv/bin/activate,它等价于“激活 venv 后再执行”。

#另外注意：加 --no-sync 更稳妥:uv run 默认会先按 pyproject.toml/uv.lock 同步一下环境，而这个项目的 lock 目前是空的(之前讨论过)。实测它不会删掉 uv pip install 装的包，但如果你想绝对保险，可以写：
# uv run --no-sync uvicorn python_study.fastapi.app.main:app --reload

#启动方式2 (不推荐)
python python_study/fastapi/app/main.py
#启动方式2
python3 -m uvicorn python_study.fastapi.app.main:app --reload
#启动方式3
uvicorn python_study.fastapi.app.main:app --reload
带参数版本：uvicorn python_study.fastapi.app.main:app --reload --host 0.0.0.0 --port 8000
```

访问：
- API 文档：http://localhost:8000/docs
- 前端页面：http://localhost:8000/static/index.html
- 健康检查：http://localhost:8000/health

**API 接口一览**：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/users/` | 用户列表（分页 + 多条件筛选） |
| POST | `/api/v1/users/` | 创建用户 |
| GET | `/api/v1/users/{id}` | 用户详情 |
| PUT | `/api/v1/users/{id}` | 更新用户 |
| DELETE | `/api/v1/users/{id}` | 删除用户 |
| PUT | `/api/v1/users/{id}/activate` | 启用/停用用户 |

---

### 2️⃣ 启动 RAG 问答系统（python_study/rag）

```bash
# 安装依赖
pip install -r python_study/rag/requirements.txt

# 配置环境变量（编辑 python_study/rag/.env）
# DASHSCOPE_API_KEY=your_api_key          # 也可放系统环境变量
# DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# DASHSCOPE_MODEL=qwen3-max
# EMBEDDINGS_MODEL_NAME=text-embedding-v4
# MILVUS_HOST=localhost
# MILVUS_PORT=19530
# COLLECTION_NAME=snail_store

# 确保 Milvus 已启动（默认 localhost:19530）

# 在仓库根目录启动服务
python python_study/rag/app/main.py
# 或: uvicorn python_study.rag.app.main:app --reload --host 0.0.0.0 --port 8100
```

访问：
- 前端页面：http://localhost:8100
- API 文档：http://localhost:8100/docs
- ReDoc：http://localhost:8100/redoc

**使用流程**：
1. 上传文档（POST `/api/v1/upload`）— 支持 PDF、TXT、Markdown、DOCX
2. 提问（POST `/api/v1/ask`）— 返回答案 + 引用来源（含文件名、页码、相似度得分）

---

### 3️⃣ 运行 LangChain 短期记忆示例（python_study/langchain）

```bash
# 公共依赖（langchain / langgraph / ChatTongyi 适配层）
pip install langchain langgraph langchain-community dashscope

# 按需追加：
# pip install langgraph-checkpoint-sqlite   # ShortTermMemory-2.py
# pip install psycopg2-binary               # ShortTermMemory-3.py（需另有 PostgreSQL）

export DASHSCOPE_API_KEY=your_api_key_here

python python_study/langchain/shortTermMemorry/ShortTermMemory-1.py
```

---

### 4️⃣ 启动 AI 聊天助手（snail-ai-oamr）

```bash
cd snail-ai-oamr/backend

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
export DASHSCOPE_API_KEY=your_api_key_here

# 确保 Milvus 已启动（默认 localhost:19530）

# 启动服务
python main.py
# 或: uvicorn main:app --reload --host 0.0.0.0 --port 8100
```

访问：http://localhost:8100

**接口**：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 聊天页面 |
| POST | `/chat` | SSE 流式聊天（`body: {"message": "你好"}`） |

---

## 🧠 核心亮点

### RAG 检索增强生成（python_study/rag）

- **场景化文档切分**：通用文档(500)/技术文档(800)/对话(300) 差异化 chunk_size，Markdown 标题层级分割
- **分批写入 + 容错**：batch_size=200 循环写入，控制内存峰值，支持局部重试
- **内容哈希去重**：SHA-256 判重 + 文件名定位旧版本，支持文档更新
- **混合检索**：向量语义(0.7) + BM25 关键词(0.3) 加权融合，补齐精确匹配盲区
- **Cross-encoder 重排序**：qwen3-rerank 模型精排，Precision@K 提升 30-50%
- **多模型 Embedding 支持**：DashScope / BGE-small / BGE-large / M3E-base 按需切换
- **结构化的引用来源**：每个回答附带文档名、相似度得分、重排序分数
- **优雅的依赖注入**：单例模式管理向量库连接和各服务实例
- **生命周期管理**：FastAPI lifespan 确保向量库在服务启动时初始化完毕

### LangGraph 工作流（snail-ai-oamr）

- 问题先经过**分类 Agent**（5 类），再路由到对应的**专业 Agent**
- `MemorySaver` 实现对话记忆隔离（thread_id 隔离会话）
- 支持扩展工具调用（业务操作 Agent 示例）

### 混合检索（snail-ai-oamr RAG）

- **向量检索**：Milvus 语义相似度搜索
- **BM25 检索**：jieba 分词 + 关键词匹配
- **混合策略**：加权合并，互补语义和精确匹配的各自优势

### LangChain 短期记忆（python_study/langchain）

- **Checkpointer + State** 机制实现线程级对话持久化（`thread_id` 隔离会话）
- **三种检查点后端**：`MemorySaver`（内存）→ `SqliteSaver`（SQLite 文件）→ `PostgresSaver`（PostgreSQL）渐进演进
- **三种上下文管理策略**：`trim_messages` 裁剪、`RemoveMessage` 删除、历史自动摘要（Summarize）
- 中间件钩子（`@before_model` / `@after_model` / `@dynamic_prompt`）与五种记忆访问方式在《第12天》文档中系统梳理，示例代码暂未覆盖

### 优雅的工程实践

- 统一异常处理（自定义业务异常、参数验证异常、数据库异常）
- 统一响应格式（`code + message + data`）
- Pydantic 双层验证（请求 + 响应）
- 完整的日志体系

---

## 📚 文档索引

### Python 学习文档（`docs/Python学习文档/`）

| 文档 | 路径 |
|------|------|
| 第1天：基础概念映射 | `docs/Python学习文档/第1天，基础概念映射.md` |
| 第2天：FastAPI | `docs/Python学习文档/第2天，FastAPI.md` |
| 第3天：SQLAlchemy | `docs/Python学习文档/第3天，SQLAlchemy.md` |
| 第4天：异步编程 | `docs/Python学习文档/第4天，异步编程.md` |
| 第5天：异常与日志 | `docs/Python学习文档/第5天，异常与日志.md` |
| 第6天：项目实战—用户管理 | `docs/Python学习文档/第6天，项目实战—用户管理.md` |
| 第7天：AI应用—基础概念 | `docs/Python学习文档/第7天，AI应用—基础概念.md` |
| 第8天：AI应用—Milvus向量数据库 | `docs/Python学习文档/第8天，AI应用—Milvus向量数据库.md` |
| 第9天：AI应用—RAG知识库 | `docs/Python学习文档/第9天，AI应用—RAG知识库.md` |
| 第10天：AI应用—RAG调优 | `docs/Python学习文档/第10天，AI应用—RAG调优.md` |
| 第11天：AI应用—LangChain（上） | `docs/Python学习文档/第11天，AI应用—LangChain（上）.md` |
| 第12天：AI应用—LangChain（短期记忆） | `docs/Python学习文档/第12天，AI应用—LangChain（短期记忆）.md` |

### RAG 专项

| 文档 | 路径 |
|------|------|
| RAG 检索问题分析 | `snail-ai-oamr/backend/rag/RAG检索问题.md` |

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

MIT License

---

## 📬 联系方式

- Gitee: https://gitee.com/WilliamWangmy/snail-ai
- 问题反馈: 请在 Gitee Issues 中提交
