## 1. LangChain介绍
### 1.1 什么是LangChain

​	**LangChain 是一个用于构建“大语言模型（LLM）驱动应用”的开源开发框架。**
​	它不是一个模型，也不是一个可以直接用的软件，而是一个**开发者工具包**，类似与**Spring**这样的框架。让你能像搭积木一样，把大模型和各种外部能力组合起来，构建复杂的AI应用。

- 可以把 LangChain 想象成 **给LLM装上的“四肢”和“记忆”**

### 1.2 LangChain核心价值

#### 1.2.1 组件化—让开发像搭积木

LangChain把LLM应用需要的各种能力，抽象成标准化的独立组件：

- **提示词模板**：管输入
- **模型接口**：管推理
- **记忆模块**：管历史
- **工具接口**：管行动
- **检索器**：管知识

每个组件都可以独立替换、升级、测试，不绑定某个特定模型或数据库，架构极其灵活。

#### 1.2.2 链式编排—让简单组件组合成复杂流程

单个组件能力有限，LangChain的核心创新是**Chain（链）**：

```
输入 → 提示词模板 → LLM调用 → 输出解析 → 下一步决策
```

一个典型的RAG应用至少包含：检索文档 → 拼入提示词 → 生成回答。这就是一个链。链让“线性流程”变得清晰、可复用、可调试。

#### 1.2.3 记忆管理—让AI记住你是谁

默认情况下，LLM每次对话都是“第一次见面”。LangChain的Memory组件解决了这个问题：

- 自动存储对话历史
- 按需检索相关记忆（短期、长期、摘要）
- 把记忆无缝注入提示词

这让AI从“金鱼记忆”变成有上下文的智能助手。

#### 1.2.4 工具调用—给AI装上手脚

这是LangChain最具突破性的能力之一。通过**Tools + Agent**机制：

- AI可以**自主决定**何时调用外部工具
- 能搜索互联网、查数据库、调API、执行代码
- 观察工具返回结果，继续推理

这让LLM从“语言处理器”升级为“行动引擎”。

#### 1.2.5 可组合性与生态—从Demo到生产

LangChain不仅管开发，还管落地：

| 能力       | 对应组件          |
| :--------- | :---------------- |
| 调试追踪   | LangSmith         |
| 复杂工作流 | LangGraph         |
| 模型切换   | 统一Model I/O接口 |

同样的代码，换一个模型、换一个向量数据库，几乎不需要重写。这种**可组合性**，让原型能快速进化为生产系统。

### 1.3 LangChain核心组件
- **Agents（智能体）** — 协调LLM、工具和记忆以执行任务的核心执行单元
- **Models（模型）** — 驱动智能体推理和决策的大语言模型
- **Messages（消息）** — 智能体与模型之间交互的消息格式与结构
- **Tools（工具）** — 智能体可调用的外部函数或API，用于执行实际操作
- **Short-term memory（短期记忆）** — 当前对话或任务执行期间的上下文管理
- **Event streaming（事件流）** — 实时流式传输智能体执行过程中的事件
- **Streaming（流式传输）** — 模型输出的逐词或逐块流式返回
- **Structured output（结构化输出）** — 以结构化格式（如JSON）输出模型结果，便于解析和处理

## 2. Models模型

### 2.1 模型的核心概念与作用

LLM（大语言模型）是能够像人类一样理解和生成文本的AI系统。它们具备多功能的通用性，无需针对每个具体任务重新训练即可完成写作、翻译、总结、问答等任务。

除了基础的文本生成，现代模型还扩展了以下能力：

- **工具调用（Tool Calling）**：模型可以主动请求调用外部工具，比如查询数据库、调用API接口、执行代码等，并将工具返回的结果整合到自己的回答中。
- **结构化输出（Structured Output）**：模型的回答可以被约束为遵循特定的格式（如JSON、Pydantic对象），方便后续程序解析和处理。
- **多模态（Multimodality）**：模型不仅能处理文本，还能接收和生成图像、音频、视频等多种类型的数据。
- **推理（Reasoning）**：模型能够进行多步骤的逻辑推理，将复杂问题拆解为更小的子问题，逐步推导出最终结论。

在LangChain的架构中，**模型是智能体的推理引擎**。智能体依靠模型来做出决策——决定调用哪些工具、如何解读工具返回的结果、以及何时给出最终答案。因此，所选模型的质量和功能直接决定了智能体的可靠性和性能表现。不同的模型各有专长：有的擅长遵循复杂指令，有的擅长结构化推理，还有的支持更大的上下文窗口以处理海量信息。

### 2.2 LangChain的模型集成体系

LangChain通过一套**标准化的模型接口**，让开发者可以访问众多主流模型提供商（如OpenAI、Anthropic、Azure、Google Gemini、AWS Bedrock、HuggingFace、OpenRouter等）。所有提供商都实现相同的接口规范，因此切换模型提供商时无需重写业务逻辑。新发布的模型也能立即使用，因为提供商包会将模型名称直接传递给提供商的API，无需等待LangChain版本更新。

#### 2.2.1 初始化模型

最便捷的方式是使用 `init_chat_model()` 函数：

```python
from langchain.chat_models import init_chat_model
model = init_chat_model(
    "claude-sonnet-4-6",
    # Kwargs passed to the model:
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
    max_retries=6,  # Default; increase for unreliable networks
)
response = model.invoke("为什么鹦鹉会说话？")
```

初始化时可以传入多种参数来配置模型行为，包括：

- `model`（必需）：模型名称或"提供商:模型名"格式
- `api_key`：认证密钥，通常通过环境变量设置
- `temperature`：控制随机性，值越高输出越有创意，越低越确定
- `max_tokens`：限制响应令牌总数
- `timeout`：请求超时时间（秒）
- `max_retries`：失败重试次数，默认6次，对不稳定网络可增至10-15次

#### 2.2.2 连接弹性

LangChain会自动处理API请求失败，采用指数退避策略进行重试。网络错误、速率限制（HTTP 429）和服务器错误（5xx）会自动重试；客户端错误（如401未授权、404不存在）则不会重试。对于长时间运行的智能体任务，建议增加 `max_retries` 并配合检查点（checkpointer）来保留进度。

### 2.3 模型调用方式

#### 2.3.1 invoke（调用）

最直接的同步调用方式，输入单条消息或消息列表，等待模型完整生成后返回最终结果。支持字典格式或LangChain消息对象（SystemMessage、HumanMessage、AIMessage）两种输入形式。

#### 2.3.2 stream（流式传输）

实时输出模型生成的内容，显著改善长文本响应的用户体验。`stream()` 返回迭代器，逐个产出 `AIMessageChunk` 对象。这些分块可以通过累加合并成完整的 `AIMessage`，与 `invoke()` 返回的对象用法一致。注意：流式传输要求整个处理链都支持分块处理。

#### 2.3.3 batch（批处理）

将多个独立请求批量提交，利用并行处理提升性能和降低成本。`batch()` 返回所有结果的列表。如果希望逐条接收完成的结果，可使用 `batch_as_completed()`（返回顺序可能乱序，但包含索引）。通过 `RunnableConfig` 的 `max_concurrency` 参数可以控制并行调用数量。

## 3. 消息 Message

### 3.1 消息的核心概念

在LangChain中，**消息（Messages）是模型交互的基本上下文单元**，代表模型的输入和输出。消息对象包含三个核心组成部分：

- **角色（Role）**：标识消息类型（如系统、用户、AI、工具）
- **内容（Content）**：消息的实际数据（文本、图像、音频、文档等）
- **元数据（Metadata）**：可选字段，如响应信息、消息ID、令牌用量等

LangChain提供了一套**标准化的消息类型**，可在所有模型提供商之间通用，确保行为一致性，无论调用哪个底层模型。

### 3.2 基本用法

最简单的使用方式是创建消息对象并传递给模型：

```python
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

model = init_chat_model("gpt-5-nano")

system_msg = SystemMessage("你是一个有用的助手。")
human_msg = HumanMessage("你好，你好吗？")

messages = [system_msg, human_msg]
response = model.invoke(messages)  # 返回 AIMessage
```

### 3.3 文本Prompt和Message Prompt

| 类型                           | 用法                                      | 适用场景                             |
| :----------------------------- | :---------------------------------------- | :----------------------------------- |
| **文本Prompt（字符串）**       | `model.invoke("写一首关于春天的俳句")`    | 单次独立请求，无需对话历史，代码简洁 |
| **Message Prompt（消息列表）** | 传入 `[SystemMessage, HumanMessage, ...]` | 多轮对话、多模态内容、包含系统指令   |

### 3.4 消息类型

#### 3.4.1 SystemMessage（系统消息）

定义模型的初始行为指令，用于设定语气、角色和响应准则。

```python
# 基本指令
system_msg = SystemMessage("你是一个有用的编程助手。")

# 详细角色设定
system_msg = SystemMessage("""
你是一位资深Python开发人员，擅长Web框架。
始终提供代码示例并解释你的推理。
解释要简洁但透彻。
""")
```

#### 3.4.2 HumanMessage（人类消息）

代表用户输入，可包含文本、图像、音频、文件等多模态内容。

```python
# 简单文本
human_msg = HumanMessage("什么是机器学习？")

# 带元数据
human_msg = HumanMessage(
    content="你好！",
    name="alice",      # 可选：标识不同用户
    id="msg_123"       # 可选：用于追踪的唯一标识符
)
```

> **注意**：`name`字段的行为因提供商而异——有些用于用户识别，有些忽略。

#### 3.4.3 AIMessage（AI消息）

代表模型的输出，可包含多模态数据、工具调用和提供商特定的元数据。

```python
response = model.invoke("解释AI")
print(type(response))  # <class 'langchain.messages.AIMessage'>
```

**手动创建AIMessage**用于对话历史：

```python
ai_msg = AIMessage("我很乐意帮助您解答这个问题！")
messages = [
    SystemMessage("你是一个有用的助手"),
    HumanMessage("你能帮我吗？"),
    ai_msg,  # 像是来自模型的回复
    HumanMessage("太好了！2+2等于几？")
]
```

**AIMessage的关键属性**：

- `tool_calls`：模型请求的工具调用列表
- `usage_metadata`：令牌用量统计
- `response_metadata`：提供商特定的元数据

#### 3.4.4 ToolMessage（工具消息）

用于将工具执行结果传回模型。必须包含匹配的`tool_call_id`。

```python
from langchain.messages import AIMessage
from langchain.messages import ToolMessage

# 模型发起一次工具调用之后
# （此处为精简演示，手动构造消息对象）
ai_message = AIMessage(
    content=[],
    tool_calls=[{
        "name": "get_weather",
        "args": {"location": "San Francisco"},
        "id": "call_123"
    }]
)

# 执行工具，生成结果消息
weather_result = "晴天，华氏72度"
tool_message = ToolMessage(
    content=weather_result,
    tool_call_id="call_123"  # 必须和调用ID保持一致
)

# 接续完整对话上下文
messages = [
    HumanMessage("旧金山天气怎么样？"),
    ai_message,  # 模型发出的工具调用请求
    tool_message,  # 工具执行返回结果
]
response = model.invoke(messages)  # 模型接收并处理工具返回结果
```

- **content（必填字符串）：**工具调用结果转为字符串形式的输出内容。

- **tool_call_id（必填字符串）：**当前这条消息所应答的那次工具调用编号，必须和 AIMessage 里工具调用的 ID 完全匹配。

- **name（必填字符串）：**被调用工具的名称。

- **artifact（字典类型，非必填）：**不会发送给大模型、仅可供程序代码读取的附加数据。

  `artifact` 字段用于存放补充数据，该数据不会送入模型上下文，只能通过代码访问。适合存储原始返回数据、调试信息、下游业务处理数据，避免污染模型对话上下文。

  **示例：**使用 artifact 存储检索元数据

  检索类工具常会从文档中提取一段文本供模型参考：消息主体内容存放模型需要阅读的文本，`artifact` 可以存放文档编号、页码等元数据，供应用程序后续使用（比如渲染原文页面），示例如下：

  ```python
  from langchain.messages import ToolMessage
  
  # 发给模型阅读的正文内容
  message_content = "这是最好的时代，也是最坏的时代。"
  
  # 下游程序可用的附属元数据
  artifact = {"document_id": "doc_123", "page": 0}
  
  tool_message = ToolMessage(
      content=message_content,
      tool_call_id="call_123",
      name="search_books",
      artifact=artifact,
  )
  ```

## 4、消息内容（Message Content）

### 4.1 什么是消息内容？

消息的`content`可以理解为**发送给模型的数据载荷（payload）**。LangChain的消息对象有一个`content`属性，它是**弱类型**的，支持：

- **字符串（String）**：纯文本内容
- **未类型化对象的列表（List of untyped objects）**：如字典列表

这种设计允许LangChain聊天模型直接支持**提供商原生结构**，例如多模态内容和其他数据类型，无需强制转换。

### 4.2 content的三种输入格式

#### 4.2.1 纯字符串

最简形式，适用于纯文本交互：

```python
from langchain.messages import HumanMessage

human_message = HumanMessage("你好，你好吗？")
```

#### 4.2.2 提供商原生格式

直接使用提供商定义的格式（如OpenAI的多模态结构）：

```python
human_message = HumanMessage(content=[
    {"type": "text", "text": "描述这张图片的内容。"},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
])
```

#### 5.2.3 LangChain标准内容块

使用LangChain统一的标准块格式：

```python
human_message = HumanMessage(content_blocks=[
    {"type": "text", "text": "描述这张图片的内容。"},
    {"type": "image", "url": "https://example.com/image.jpg"}
])
```

> **关键点**：使用`content_blocks`初始化消息时，会自动填充`content`属性，同时提供**类型安全**的接口。

### 4.3 标准内容块（Standard Content Blocks）

LangChain提供**跨提供商统一的内容表示**，解决不同提供商返回格式各异的问题。

`content_blocks`属性：消息对象实现了`content_blocks`属性，它会**惰性解析（lazily parse）**`content`属性，将其转换为标准、类型安全的表示。

**实际案例**：Anthropic的`thinking`块被统一解析为`ReasoningContentBlock`

```python
from langchain.messages import AIMessage

# Anthropic模型返回的消息（原生格式）
message = AIMessage(
    content=[
        {"type": "thinking", "thinking": "让我思考一下这个问题...", "signature": "WaUjzkyp..."},
        {"type": "text", "text": "答案是..."},
    ],
    response_metadata={"model_provider": "anthropic"}
)

# 通过content_blocks获取标准格式
message.content_blocks
# [
#   {'type': 'reasoning', 'reasoning': '让我思考一下这个问题...', 'extras': {'signature': 'WaUjzkyp...'}},
#   {'type': 'text', 'text': '答案是...'}
# ]
```

**跨提供商一致性**：

- 无论底层使用Anthropic（返回`thinking`块）、OpenAI（返回`reasoning`块）还是其他提供商
- `content_blocks`都会将它们统一为`reasoning`类型
- 应用程序无需针对不同提供商编写不同的解析逻辑
- 不用的内容块可参考官方文档说明：https://docs.langchain.com/oss/python/langchain/messages#content-block-reference

#### 4.3.1 **序列化标准内容**

如果LangChain外部的应用需要访问标准内容块表示，可以启用序列化：

- **方式一：环境变量**

```bash
LC_OUTPUT_VERSION=v1
```

- **方式二：初始化模型时指定**

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5-nano", output_version="v1")
```

启用后，标准内容块会被存储到消息的`content`属性中。

### 4.4 多模态（Multimodal）

多模态指处理不同形式数据的能力：**文本、音频、图像、视频**等。LangChain包含跨提供商通用的标准类型。

#### 4.4.1 图像输入

**1. 通过URL引用**

```python
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "描述这张图片的内容。"},
        {"type": "image", "url": "https://example.com/path/to/image.jpg"},
    ]
}
```

**2. 通过Base64数据**

```python
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "描述这张图片的内容。"},
        {
            "type": "image",
            "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
            "mime_type": "image/jpeg",
        },
    ]
}
```

**3. 通过提供商托管的文件ID**

```python
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "描述这张图片的内容。"},
        {"type": "image", "file_id": "file-abc123"},
    ]
}
```

额外键值

可以在内容块顶层或嵌套在`"extras"`中添加额外键：

```python
# 顶层添加
{"type": "image", "url": "...", "filename": "photo.jpg"}

# 嵌套在extras中
{"type": "image", "url": "...", "extras": {"filename": "photo.jpg"}}
```

> **注意**：不同提供商有特定要求，如OpenAI处理PDF时需要提供文件名。请查阅具体模型提供商文档。

