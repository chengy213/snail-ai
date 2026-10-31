"""
模型工厂：帮我们提供模型
"""
from abc import ABC,abstractmethod
from typing import Optional

from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel

from utils.config_handler import oamr_config


# ABC内置的抽象类
class BaseModelFactory(ABC):

    # 抽象方法
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):

    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(model=oamr_config["chat_model_name"])

class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=oamr_config["embeddings_model_name"])


chat_model = ChatModelFactory().generator()
embedding_model = EmbeddingsFactory().generator()