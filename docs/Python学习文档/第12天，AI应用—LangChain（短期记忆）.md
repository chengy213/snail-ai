# Short-term memory短期记忆

## 第一部分：基础概念与原理

### 1. 短期记忆概述

#### 1.1 什么是记忆系统？

记忆是一个能记住**过往交互信息**的系统。对于AI智能体来说，记忆至关重要，因为它能让智能体：

- **记住之前的互动** — 保存当前对话中的所有消息
- **从反馈中学习** — 从用户的纠正中调整行为
- **适应用户偏好** — 根据对话中的偏好调整回复

当智能体处理涉及大量用户交互的复杂任务时，这种能力对于**效率和用户满意度**都变得不可或缺。

#### 1.2 短期记忆的作用

短期记忆（Short-term Memory）让你的应用能够记住**单个线程或对话中**的过往交互。对话历史是最常见的短期记忆形式。

| 作用             | 说明                     | 示例                                 |
| :--------------- | :----------------------- | :----------------------------------- |
| **记住交互**     | 保存当前对话中的所有消息 | 用户说"我叫Bob"，智能体记住这个名字  |
| **学习反馈**     | 从用户的纠正中调整行为   | 用户说"不要用敬语"，后续回复调整语气 |
| **适应用户偏好** | 根据对话中的偏好调整回复 | 用户喜欢简洁回复，后续回复变得简短   |

#### 1.3 短期记忆 vs 长期记忆

```mermaid
flowchart TB
    subgraph Memory["🧠 记忆系统"]
        direction LR
        STM["短期记忆<br/>(Short-term Memory)"]
        LTM["长期记忆<br/>(Long-term Memory)"]
    end

    subgraph STM_Detail["短期记忆详情"]
        STM_Scope["作用范围: 单个线程内"]
        STM_Content["存储内容: 对话消息历史"]
        STM_Life["生命周期: 对话进行期间"]
        STM_Example["示例: 记住用户名字"]
        STM_Impl["实现: Checkpointer + State"]
    end

    subgraph LTM_Detail["长期记忆详情"]
        LTM_Scope["作用范围: 跨线程、跨会话"]
        LTM_Content["存储内容: 用户档案、知识库"]
        LTM_Life["生命周期: 持久化存储"]
        LTM_Example["示例: 用户画像、历史行为"]
        LTM_Impl["实现: 向量数据库、键值存储"]
    end

    STM --> STM_Detail
    LTM --> LTM_Detail
```

> **关键区别**：短期记忆是**线程级**的（同一对话内），长期记忆是**跨线程/跨会话**的（不同对话间共享）。

#### 1.4 线程（Thread）的概念

线程是组织对话的基本单位，类似于电子邮件中的"对话线程"。它在一个会话中组织多次交互，类似于电子邮件将消息分组到单个对话中的方式。

```mermaid
sequenceDiagram
    participant User1 as 用户A (thread_id: "user_123")
    participant Agent as 智能体
    participant User2 as 用户B (thread_id: "user_456")

    Note over User1,Agent: 线程1: 独立对话
    User1->>Agent: "你好，我叫Bob"
    Agent->>User1: "你好Bob！"
    User1->>Agent: "我的名字是什么？"
    Agent->>User1: "你的名字是Bob" ✅

    Note over User2,Agent: 线程2: 完全隔离
    User2->>Agent: "我的名字是什么？"
    Agent->>User2: "我不知道你的名字" ❌
```

> **需要跨对话记住信息？** 使用长期记忆在不同线程和会话中存储和回忆用户特定或应用级别的数据。

### 2. 核心挑战：对话历史与上下文窗口

#### 2.1 上下文窗口的限制

LLM（大语言模型）有一个重要的技术限制——**上下文窗口**（Context Window），它决定了模型一次能处理的最大令牌数。

```mermaid
pie title LLM上下文窗口大小对比
    "GPT-4 (8K-128K)" : 40
    "Claude 3 (200K)" : 30
    "Gemini 1.5 (1M)" : 30
```

#### 2.2 长对话带来的问题

当对话持续进行，消息列表不断增长，会出现以下问题：

```mermaid
flowchart TD
    Start[开始长对话] --> Growth[消息列表持续增长]
    Growth --> Problem1[超出上下文窗口]
    Growth --> Problem2[响应速度变慢]
    Growth --> Problem3[成本升高]
    Growth --> Problem4[模型注意力分散]

    Problem1 --> Result1[信息被截断/上下文丢失]
    Problem2 --> Result2[用户体验下降]
    Problem3 --> Result3[运营成本上升]
    Problem4 --> Result4[回答质量降低]

    Result1 & Result2 & Result3 & Result4 --> Need[需要上下文管理策略]
```

具体来说，长对话对今天的大语言模型构成了以下挑战：

| 挑战 | 说明 |
| :--- | :--- |
| **上下文溢出** | 完整的历史可能无法放入LLM的上下文窗口，导致上下文丢失或错误 |
| **长上下文性能差** | 即使模型支持完整的上下文长度，大多数LLM在处理长上下文时表现仍然不佳 |
| **注意力分散** | 模型会被过时或无关的内容"分散注意力" |
| **成本与延迟** | 响应速度变慢，成本更高 |

#### 2.3 消息结构的演变

聊天模型使用消息作为上下文，其中包括：**指令（系统消息）**、**输入（用户消息）**。在聊天应用中，消息在用户输入和模型响应之间交替，导致消息列表随时间推移越来越长。由于上下文窗口有限，许多应用可以通过使用技术来**移除或"遗忘"陈旧信息**而受益。

```mermaid
graph LR
    subgraph Round1["第1轮对话"]
        M1["System: 你是一个助手"]
        M2["User: 你好"]
        M3["Assistant: 你好！"]
    end

    subgraph Round10["第10轮对话"]
        N1["System: 你是一个助手"]
        N2["User: 你好"]
        N3["Assistant: 你好！"]
        N4["User: 我叫Bob"]
        N5["Assistant: 你好Bob！"]
        N6["...更多消息..."]
        N7["⚠️ 可能超出窗口"]
    end

    Round1 --> Round10
```

---

## 第二部分：使用方法

### 1. 核心机制：Checkpointer + State

要为智能体添加短期记忆（线程级持久化），你需要在创建智能体时指定一个**检查点器（checkpointer）**。

LangChain的智能体将短期记忆作为其**状态（state）**的一部分进行管理：

```mermaid
flowchart LR
    subgraph Mechanism["短期记忆工作机制"]
        direction TB
        A["状态(State)存储对话上下文"] --> B["检查点器(Checkpointer)持久化状态"]
        B --> C["不同线程之间保持隔离"]
        C --> D["线程可以随时恢复"]
    end
```

- 通过将信息存储在图的**状态**中，智能体可以访问给定对话的完整上下文，同时保持不同线程之间的隔离
- 状态使用**检查点器**持久化到数据库（或内存）中，以便线程可以随时恢复
- 短期记忆在智能体被调用或完成一个步骤（如工具调用）时**更新**
- 状态在每一步开始时被**读取**

### 2. 基础示例：InMemorySaver

使用内存检查点器快速体验短期记忆：

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

def get_user_info() -> str:
    """查找当前用户的信息。"""
    return "No user profile on file."

agent = create_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[get_user_info],
    checkpointer=InMemorySaver(),  # 内存检查点器，重启后丢失
)

thread_config = {"configurable": {"thread_id": "1"}}
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
    thread_config,
)["messages"][-1].content

print(response)  # "Hi Bob! Nice to see you here. How are you doing?"

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    thread_config,
)["messages"][-1].content

print(response)  # "You are Bob!"
```

> **注意**：`InMemorySaver` 仅在内存中保存数据，进程重启后所有对话历史会丢失，仅适合开发和测试。

### 3. 生产环境配置：PostgresSaver

在生产环境中，使用由**数据库**支持的检查点器来持久化存储：

```bash
pip install langgraph-checkpoint-postgres
```

```python
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver

def get_user_info() -> str:
    """查找当前用户的信息。"""
    return "No user profile on file."

DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # 在PostgreSQL中自动创建表
    agent = create_agent(
        "gpt-5.5",
        tools=[get_user_info],
        checkpointer=checkpointer,
    )
```

> **其他检查点器选项**：除了 Postgres，LangGraph 还支持 **SQLite**、**Postgres** 和 **Azure Cosmos DB** 等，请参阅 LangGraph 持久化文档中的检查点器库列表。

### 4. 自定义智能体记忆

默认情况下，智能体使用 `AgentState` 来管理短期记忆，特别是通过 `messages` 键管理对话历史。

你可以**扩展 AgentState** 来添加额外的字段。自定义的状态模式通过 `state_schema` 参数传递给 `create_agent`：

```python
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver

class CustomAgentState(AgentState):
    user_id: str
    preferences: dict

agent = create_agent(
    "gpt-5.5",
    tools=[get_user_info],
    state_schema=CustomAgentState,  # 传入自定义状态模式
    checkpointer=InMemorySaver(),
)

# 自定义状态可以在invoke中传递
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user_123",
        "preferences": {"theme": "dark"}
    },
    {"configurable": {"thread_id": "1"}})
```

---

## 第三部分：常见模式 — 管理上下文窗口

启用短期记忆后，长对话可能会超过LLM的上下文窗口。常见的解决方案有三种：

```mermaid
flowchart TD
    Problem[长对话超出上下文窗口] --> Strategy1[消息修剪 Trim]
    Problem --> Strategy2[消息删除 Delete]
    Problem --> Strategy3[消息总结 Summarize]

    Strategy1 --> S1_Desc["按token数量截断，保留最近的N条消息"]
    Strategy2 --> S2_Desc["精确删除特定消息或清空全部历史"]
    Strategy3 --> S3_Desc["用LLM对历史消息进行总结压缩"]
```

### 1. 修剪消息（Trim Messages）

大多数LLM都有最大支持的上下文窗口（以token计）。一种决定何时截断消息的方法是计算消息历史中的token数量，并在接近该限制时进行截断。

如果你使用LangChain，可以使用 `trim_messages` 工具，并指定：
- 要从列表中保留的 **token数量**
- 用于处理边界的**策略**（例如，保留最后的 `max_tokens` 条消息）

要在智能体中修剪消息历史，请使用 `@before_model` 中间件装饰器：

```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from typing import Any

@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """只保留最后几条消息以适应上下文窗口。"""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # 无需更改

    first_msg = messages[0]  # 保留首条消息（通常是系统消息）
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),  # 清空所有旧消息
            *new_messages                           # 只保留指定消息
        ]
    }

agent = create_agent(
    "gpt-5.5",
    tools=[...],
    middleware=[trim_messages],       # 注册中间件
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
```

输出：
```
================================== Ai Message ==================================

Your name is Bob. You told me that earlier.
If you'd like me to call you a nickname or use a different name, just say the word.
```

> **工作原理**：`@before_model` 中间件在每次模型调用**之前**执行，检查消息数量并截断旧消息，确保上下文始终在窗口限制内。

### 2. 删除消息（Delete Messages）

你可以从图状态中删除消息来管理消息历史。这对于**删除特定消息**或**清空整个消息历史**非常有用。

要从图状态中删除消息，你需要使用 `RemoveMessage`。要使 `RemoveMessage` 生效，你需要使用带有 `add_messages` 归约器的状态键。默认的 `AgentState` 提供了这个功能。

#### 删除特定消息

```python
from langchain.messages import RemoveMessage

def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 2:
        # 删除最早的两条消息
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
```

#### 删除所有消息

```python
from langgraph.graph.message import REMOVE_ALL_MESSAGES

def delete_messages(state):
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
```

> ⚠️ **重要提醒**：删除消息时，请确保生成的消息历史是**有效**的。检查你正在使用的LLM提供商的限制。例如：
> - 一些提供商期望消息历史以用户消息开始
> - 大多数提供商要求带有工具调用的助手消息后面跟上相应的工具结果消息

#### 完整示例（使用 @after_model）

```python
from langchain.messages import RemoveMessage
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig

@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """删除旧消息以保持对话可管理。"""
    messages = state["messages"]
    if len(messages) > 2:
        # 删除最早的两条消息
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    return None

agent = create_agent(
    "gpt-5-nano",
    tools=[...],
    system_prompt="请简洁扼要。",
    middleware=[delete_old_messages],
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
    config,
    version="v3",
)
for snapshot in stream.values:
    print([(message.type, message.content) for message in snapshot["messages"]])

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "write a short poem about cats"}]},
    config,
    version="v3",
)
for snapshot in stream.values:
    print([(message.type, message.content) for message in snapshot["messages"]])

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "what's my name?"}]},
    config,
    version="v3",
)
for snapshot in stream.values:
    print([(message.type, message.content) for message in snapshot["messages"]])
```

输出示例（注意第3轮时最早的对话已被删除，模型忘记了名字）：

```
[('human', "hi! I'm bob")]
[('human', "hi! I'm bob"), ('ai', 'Hi Bob! Nice to meet you...')]
[('human', "hi! I'm bob"), ('ai', 'Hi Bob! Nice to meet you...'), ('human', "write a short poem about cats")]
[('human', "hi! I'm bob"), ('ai', 'Hi Bob! Nice to meet you...'), ('human', "write a short poem about cats"), ('ai', 'There once was a cat on a wall...')]
[('human', 'write a short poem about cats'), ('ai', 'There once was a cat on a wall...')]  ← 旧消息被删除
[('human', 'write a short poem about cats'), ('ai', 'There once was a cat on a wall...'), ('human', "what's my name?")]
[('human', 'write a short poem about cats'), ('ai', 'There once was a cat on a wall...'), ('human', "what's my name?"), ('ai', "I don't know your name - you haven't told me!")]  ← 丢失了名字
[('human', "what's my name?"), ('ai', "I don't know your name - you haven't told me!")]
```

### 3. 总结消息（Summarize Messages）

如上所示的修剪或删除消息的问题在于，你可能会因为从消息队列中剔除而**丢失信息**。因此，一些应用受益于使用聊天模型对消息历史进行总结这种**更复杂的方法**。

要在智能体中总结消息历史，请使用内置的 `SummarizationMiddleware`：

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

checkpointer = InMemorySaver()

agent = create_agent(
    model="gpt-5.5",
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.4-mini",       # 用于总结的模型（可用更便宜的模型）
            trigger=("tokens", 4000),    # token数触发阈值：超过4000 token时触发总结
            keep=("messages", 20)        # 保留最近20条消息不被总结
        )
    ],
    checkpointer=checkpointer,
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}
agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
```

输出：
```
================================== Ai Message ==================================

Your name is Bob!
```

> **三种策略对比**：
>
> | 策略 | 优点 | 缺点 | 适用场景 |
> | :--- | :--- | :--- | :--- |
> | **修剪(Trim)** | 简单高效，保证不超窗口 | 可能丢失重要早期信息 | token预算严格，信息时效性强 |
> | **删除(Delete)** | 精确控制，可针对特定消息 | 需要手动管理删除逻辑 | 需要精确控制消息生命周期 |
> | **总结(Summarize)** | 保留信息精华，不丢失关键内容 | 额外LLM调用成本，可能有信息偏差 | 需要保留历史脉络的复杂对话 |

---

## 第四部分：访问记忆

你可以通过多种方式访问和修改智能体的短期记忆（状态）。LangChain提供了5种不同的访问方式，适用于不同的场景。

```mermaid
flowchart TD
    subgraph Access["五种访问记忆的方式"]
        direction LR
        A["1. 工具中读取<br/>(ToolRuntime)"]
        B["2. 工具中写入<br/>(Command)"]
        C["3. 提示中访问<br/>(@dynamic_prompt)"]
        D["4. 模型前访问<br/>(@before_model)"]
        E["5. 模型后访问<br/>(@after_model)"]
    end
```

### 1. 在工具中读取短期记忆

在工具中使用 `runtime` 参数（类型为 `ToolRuntime`）访问短期记忆（状态）。`runtime` 参数对工具签名是**隐藏的**（因此模型看不到它），但工具可以通过它访问状态。

```python
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime

class CustomState(AgentState):
    user_id: str

@tool
def get_user_info(
    runtime: ToolRuntime   # runtime对模型隐藏，但工具可访问
) -> str:
    """查找用户信息。"""
    user_id = runtime.state["user_id"]
    return "User is John Smith" if user_id == "user_123" else "Unknown user"

agent = create_agent(
    model="gpt-5-nano",
    tools=[get_user_info],
    state_schema=CustomState,
)

result = agent.invoke({
    "messages": "look up user information",
    "user_id": "user_123"
})
print(result["messages"][-1].content)
# > User is John Smith.
```

### 2. 从工具写入短期记忆

要在执行期间修改智能体的短期记忆（状态），你可以直接从工具返回状态更新。这对于**持久化中间结果**或**使信息可供后续工具或提示访问**非常有用。

```python
from langchain.tools import tool, ToolRuntime
from langchain_core.runnables import RunnableConfig
from langchain.messages import ToolMessage
from langchain.agents import create_agent, AgentState
from langgraph.types import Command
from pydantic import BaseModel

class CustomState(AgentState):
    user_name: str

class CustomContext(BaseModel):
    user_id: str

@tool
def update_user_info(
    runtime: ToolRuntime[CustomContext, CustomState],
) -> Command:
    """查找并更新用户信息。"""
    user_id = runtime.context.user_id
    name = "John Smith" if user_id == "user_123" else "Unknown user"
    return Command(update={
        "user_name": name,        # 更新自定义状态字段
        "messages": [             # 更新消息历史
            ToolMessage(
                "Successfully looked up user information",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

@tool
def greet(
    runtime: ToolRuntime[CustomContext, CustomState]
) -> str | Command:
    """找到用户信息后，用它来问候用户。"""
    user_name = runtime.state.get("user_name", None)
    if user_name is None:
       return Command(update={
            "messages": [
                ToolMessage(
                    "Please call the 'update_user_info' tool it will get and update the user's name.",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        })
    return f"Hello {user_name}!"

agent = create_agent(
    model="gpt-5-nano",
    tools=[update_user_info, greet],
    state_schema=CustomState,
    context_schema=CustomContext,
)

agent.invoke(
    {"messages": [{"role": "user", "content": "greet the user"}]},
    context=CustomContext(user_id="user_123"),
)
```

> **关键点**：使用 `Command(update={...})` 返回值可以直接修改状态，包括自定义字段和消息历史。这对于工具间的**协作**非常有用 — 一个工具写入信息，另一个工具读取使用。

### 3. 在提示中访问记忆

在中间件中访问短期记忆（状态），以基于对话历史或自定义状态字段创建**动态提示**。

```python
from langchain.agents import create_agent
from typing import TypedDict
from langchain.agents.middleware import dynamic_prompt, ModelRequest

class CustomContext(TypedDict):
    user_name: str

def get_weather(city: str) -> str:
    """获取城市的天气。"""
    return f"The weather in {city} is always sunny!"

@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context["user_name"]
    system_prompt = f"You are a helpful assistant. Address the user as {user_name}."
    return system_prompt

agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
    middleware=[dynamic_system_prompt],
    context_schema=CustomContext,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    context=CustomContext(user_name="John Smith"),
)
for msg in result["messages"]:
    msg.pretty_print()
```

输出示例：
```
================================ Human Message =================================
What is the weather in SF?

================================== Ai Message ==================================
Tool Calls:
  get_weather (call_WFQlOGn4b2yoJrv7cih342FG)
 Call ID: call_WFQlOGn4b2yoJrv7cih342FG
  Args:
    city: San Francisco

================================= Tool Message =================================
Name: get_weather
The weather in San Francisco is always sunny!

================================== Ai Message ==================================
Hi John Smith, the weather in San Francisco is always sunny!  ← 动态提示生效
```

### 4. 在模型前（@before_model）访问记忆

在 `@before_model` 中间件中访问短期记忆（状态），以便在模型调用**之前**处理消息。这是实现**消息修剪**等上下文管理策略的主要方式。

```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from typing import Any

@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """只保留最后几条消息以适应上下文窗口。"""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # 无需更改

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

agent = create_agent(
    "gpt-5-nano",
    tools=[],
    middleware=[trim_messages],
    checkpointer=InMemorySaver()
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
```

输出：
```
================================== Ai Message ==================================

Your name is Bob. You told me that earlier.
If you'd like me to call you a nickname or use a different name, just say the word.
```

### 5. 在模型后（@after_model）访问记忆

在 `@after_model` 中间件中访问短期记忆（状态），以便在模型调用**之后**处理消息。适用于**验证响应**、**过滤敏感内容**等场景。

```python
from langchain.messages import RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.runtime import Runtime

@after_model
def validate_response(state: AgentState, runtime: Runtime) -> dict | None:
    """移除包含敏感词的消息。"""
    STOP_WORDS = ["password", "secret"]
    last_message = state["messages"][-1]
    if any(word in last_message.content for word in STOP_WORDS):
        return {"messages": [RemoveMessage(id=last_message.id)]}
    return None

agent = create_agent(
    model="gpt-5-nano",
    tools=[],
    middleware=[validate_response],
    checkpointer=InMemorySaver(),
)
```

---

## 第五部分：五种记忆访问方式总结

```mermaid
flowchart TD
    subgraph Timeline["请求处理时间线"]
        direction LR
        Before["1️⃣ @before_model<br/>模型调用前处理消息"] --> Model["🤖 LLM 模型调用"]
        Model --> After["5️⃣ @after_model<br/>模型调用后验证/过滤"]

        Tools["2️⃣ 工具中读取<br/>3️⃣ 工具中写入<br/>(ToolRuntime / Command)"] -.-> Model

        Prompt["4️⃣ @dynamic_prompt<br/>动态构造系统提示"] -.-> Model
    end
```

| 方式 | 时机 | 典型用途 | 中间件装饰器 |
| :--- | :--- | :--- | :--- |
| **工具中读取** | 工具执行时 | 获取用户偏好、配置信息 | `ToolRuntime` |
| **工具中写入** | 工具执行时 | 持久化中间结果、工具协作 | `Command(update={...})` |
| **提示中访问** | 构造提示时 | 动态系统提示、个性化问候 | `@dynamic_prompt` |
| **模型前访问** | 模型调用前 | 消息修剪、上下文管理 | `@before_model` |
| **模型后访问** | 模型调用后 | 响应验证、敏感内容过滤 | `@after_model` |

---

## 学习要点

1. **短期记忆**是AI智能体的核心能力，通过 **Checkpointer + State** 机制实现线程级的对话持久化
2. **线程（Thread）** 是组织对话的基本单位，不同线程间的记忆完全隔离
3. **上下文窗口限制**是实际应用中的核心挑战，需要通过修剪、删除或总结来管理
4. **生产环境**应使用数据库支持的检查点器（如 PostgresSaver），而非内存检查点器
5. **三种上下文管理策略**各有适用场景：修剪简单高效、删除精确可控、总结保留信息精华
6. **五种记忆访问方式**覆盖了从工具、提示到中间件的完整场景，选择合适的方式取决于具体需求
