"""
上下文窗口管理器模块

功能：
1. 控制输入给 LLM 的文档总长度不超过 token 上限
2. 按相似度排序，优先保留高相关度文档
3. 超出限制时自动截断末尾文档
4. 多条文档时添加冲突提示，引导模型综合判断
"""
class ContextManager:
    """上下文窗口管理器，用于控制输入LLM的文档长度"""

    def __init__(self, max_tokens: int = 6000):
        # 最大token限制（千问Turbo上限8K，预留2K给模型输出）
        self.max_tokens = max_tokens

    def build_context(self, documents: list) -> str:
        """构建上下文：排序、截断、去重、标注冲突"""
        if not documents:
            return "暂无相关参考资料。"

        # 1. 按相似度从高到低排序
        sorted_docs = sorted(documents, key=lambda x: x.get("score", 0), reverse=True)

        parts = []          # 存储最终的文本片段
        current_length = 0  # 当前累计token数

        # 2. 遍历文档，逐个加入上下文直到达到长度上限
        for i, doc in enumerate(sorted_docs):
            # 格式化单条文档：编号+来源+页码+内容
            text = f"[文档{i + 1} 来源:{doc['source']} 第{doc.get('page', '-')}页]\n{doc['content']}"

            # 估算token数（中文场景：约1.5字符/token）
            estimated_tokens = len(text) / 1.5

            # 若加上当前文档会超限，则尝试截断最后一条文档
            if current_length + estimated_tokens > self.max_tokens:
                remaining = self.max_tokens - current_length
                if remaining > 100:   # 确保至少保留100个token的内容
                    text = text[:int(remaining * 1.5)] + "\n...[内容截断]"
                    parts.append(text)
                break   # 超出上限，停止添加更多文档

            parts.append(text)
            current_length += estimated_tokens

        # 3. 合并所有片段
        context = "\n\n".join(parts)

        # 4. 当有多条文档时，增加冲突提示，引导模型注意信息一致性
        if len(sorted_docs) >= 2:
            context += "\n\n【注意】参考资料可能包含多条相关信息，请综合分析。如有矛盾请指出。"

        return context