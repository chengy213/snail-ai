"""
向量存储
"""
from langchain_community.vectorstores import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from model.factory import embedding_model
from utils.config_handler import vector_store_config



class VectorStoreService:

    def __init__(self):

        self.vector_store = Milvus(
            embedding_function=embedding_model,
            collection_name=vector_store_config["collection_name"],
            connection_args={
                "host": vector_store_config["host"],  # 根据你的实际配置修改
                "port": vector_store_config["port"],  # 根据你的实际配置修改
            },
            # 兼容之前的集合就要指定字段名
            vector_field="embedding",  # 添加这一行，指定向量字段名
            text_field="content",  # 文本字段名
            drop_old=False,
        )
        # 文本分割器
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=vector_store_config["chunk_size"],
            chunk_overlap=vector_store_config["chunk_overlap"],
            separators=vector_store_config["separators"],
            length_function=len
        )


    def get_retriever(self):
        """
        向量检索
        :return:
        """
        return self.vector_store.as_retriever(search_kwargs={"k": vector_store_config["k"],"score_threshold": vector_store_config["score_threshold"]})

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算md5去重
        :return: None
        """
        return  None




vector_store = VectorStoreService()
retriever = vector_store.get_retriever()
invoke = retriever.invoke("在项目中的限流和熔断是怎么实现的？")
print(invoke)