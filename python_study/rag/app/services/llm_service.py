"""
大模型服务模块

 封装与千问大模型的交互逻辑，提供：
    - 基于系统提示词的问答生成
    - 检索增强生成（RAG）的上下文整合
    - 防止模型编造信息的规则约束
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from python_study.rag.app.core.config import settings


class LLMService:

    def __init__(self):
        """
        初始化通义千问大模型服务

        配置说明：
        - model: 模型名称（从配置文件读取）
        - temperature: 温度参数，控制输出的随机性（0-1之间）
        - api_key: DashScope API密钥
        - base_url: API服务地址（OpenAI兼容模式）
        """
        # ============================================================
        # 初始化千问大模型（通过 DashScope 的 OpenAI 兼容接口）
        # ============================================================
        # DashScope 提供了与 OpenAI API 兼容的接口格式，
        # 因此可以直接使用 LangChain 的 ChatOpenAI 来调用千问模型。
        self.model = ChatOpenAI(
            model=settings.DASHSCOPE_MODEL,      # 模型名称，如 qwen-plus / qwen-max
            temperature=0.3,                     # 温度参数：0~1，越低输出越确定、越高越随机
                                                 # 0.3 适合知识问答，既能准确引用又不至于太死板
            api_key=settings.DASHSCOPE_API_KEY,  # DashScope API 密钥
            base_url=settings.DASHSCOPE_BASE_URL # API 地址（OpenAI 兼容端点）
        )

        # ============================================================
        # 系统提示词（System Prompt）
        # ============================================================
        # 系统提示词定义了模型的角色和行为约束，是 RAG 系统防止"幻觉"的关键：
        # - 规则1：有答案就引用原文，保证答案有据可查
        # - 规则2：没有就明确说没有，防止模型编造信息（这是RAG的核心价值）
        # - 规则3：注明来源文档，方便用户溯源验证
        # - 规则4：禁止编造，即使模型"知道"相关知识也不能脱离文档回答
        self.system_prompt = """你是一个基于文档的问答助手。
请严格根据提供的文档内容回答用户问题。

规则：
1. 如果文档中有明确答案，请引用原文并回答。
2. 如果文档中没有相关信息，请明确说"根据现有资料，无法回答该问题"。
3. 回答时注明引用了哪份文档。
4. 不要编造文档中没有的信息。
"""

    def generate(self, query: str, context: str) -> str:
        """
        基于检索到的文档上下文生成答案

        这是一个 RAG（检索增强生成）的核心方法：
        1. 将检索到的相关文档作为上下文
        2. 结合系统提示词约束模型行为
        3. 调用大模型生成基于事实的答案

        参数:
            query: str
                用户提出的问题文本
            context: str
                从向量数据库检索到的相关文档上下文
                通常由 RetrievalService.build_context() 方法生成

        返回:
            str: 模型生成的答案文本
                如果文档中有相关信息，返回基于文档的答案
                如果没有相关信息，返回"无法回答"的提示

        示例:
            service = LLMService()
            query = "什么是RAG？"
            context = "RAG（检索增强生成）是一种结合检索和生成的技术..."
            answer = service.generate(query, context)
            print(answer)  # 输出: 根据文档[xxx.pdf]所述，RAG是...

        注意:
            模型的行为受 temperature=0.3 影响，输出相对确定，
            但同一问题多次调用可能产生略有差异的回答。
        """
        # ============================================================
        # 构建消息列表（System + Human 双消息结构）
        # ============================================================
        # SystemMessage  : 系统级指令，定义模型角色和行为规则（不会被用户感知）
        # HumanMessage   : 用户级输入，包含参考上下文和具体问题
        #
        # 消息格式说明：
        #   【参考资料】→ 从向量库检索到的相关文档片段（RAG 的增强部分）
        #   【用户问题】→ 用户的原始问题
        #   通过明确的标签区分两部分，帮助模型准确理解上下文边界
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"【参考资料】\n{context}\n\n【用户问题】\n{query}")
        ]

        # 调用模型生成响应
        # invoke() 是 LangChain 的同步调用方法，返回 AIMessage 对象
        response = self.model.invoke(messages)
        # response.content 就是模型生成的文本答案
        return response.content