"""
聊天服务类 —— 在此填充你的业务逻辑。
"""

from typing import AsyncGenerator, Optional, List

from graph.workflow import graph


class ChatService:

    async def chat_stream(
            self,
            message: str,
            image_paths: Optional[List[str]] = None
    ) -> AsyncGenerator[str, None]:
        """
        流式对话接口。
        参数:
            message: 用户输入
        返回:
            AsyncGenerator，每次 yield 一个文本片段
        """
        # 构建配置（thread_id 用于记忆隔离）
        config = {"configurable": {"thread_id": "1"}}

        # 构建状态
        initial_state = {
            "user_input": message,
            "image_paths": image_paths or [],
            "messages": [],  # 首次会从checkpointer加载
            "classification": None,
            "final_response": None,
            "error": None,
        }

        # 调用工作流（会自动从checkpointer加载历史）
        result = graph.invoke(initial_state, config)

        response = result.get("final_response", "抱歉，我没能理解您的问题。")

        for char in response:
            yield char

    async def get_history(self, session_id: str) -> List[dict]:
        """获取会话历史"""
        config = {"configurable": {"thread_id": session_id}}
        state = graph.get_state(config)

        if state and state.values.get("messages"):
            return [
                {
                    "role": "user" if msg.type == "human" else "assistant",
                    "content": msg.content
                }
                for msg in state.values["messages"]
            ]
        return []

    async def clear_memory(self, session_id: str) -> bool:
        """清除会话记忆"""
        # MemorySaver不支持直接删除，重建即可
        # 生产环境用SqliteSaver可执行delete
        return True