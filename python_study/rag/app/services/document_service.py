"""
文档处理服务模块

核心功能：统一解析各类上传文档，将不同格式的文件转化为可直接向量化的文本块，
提取元数据，为后续向量存储和检索做预处理。

模块基于 LangChain 开发，支持 PDF、TXT、MD、DOCX 四种主流格式。
所有文本分割参数均从全局配置读取，统一且易维护。

处理流水线：
    1. 选择加载器（根据文件扩展名）
    2. 加载文档内容（读取文件，提取原始文本）
    3. 切分文档（按语义边界将长文档切分成小块）
    4. 提取元数据（文件名、页码、分块序号等）
"""
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter


class DocumentService:
    """
    文档处理服务

    负责文档的加载、切分和元数据提取，是 RAG 流水线的第一道工序。

    支持的文件格式：
        - PDF   : 使用 PyPDFLoader 加载
        - TXT   : 使用 TextLoader 加载（UTF-8编码）
        - MD    : 先用 TextLoader 加载，再用 MarkdownHeaderTextSplitter 按标题切分
        - DOCX  : 使用 Docx2txtLoader 加载

    文本分割策略：
        - 通用文档 : chunk=500, overlap=50（默认，适合大多数中文文档）
        - 技术文档 : chunk=800, overlap=100（代码块需要更多上下文）
        - 对话文本 : chunk=300, overlap=30（短文本小块更精准）
    """

    def __init__(self):
        """
        初始化文档处理服务

        预配置三种场景的文本分割器：
            general_splitter — 通用文档，500字/块，50字重叠（默认推荐）
            tech_splitter    — 技术文档，800字/块，100字重叠，优先按标题切割
            chat_splitter    — 对话/短文本，300字/块，30字重叠
            md_splitter      — Markdown 专用，按 H1/H2/H3 标题层级结构化分割
        """
        # ===== 场景1：通用文档（推荐起点） =====
        # chunk_size=500   : 每块约500个字符，适合大多数中文文档
        # chunk_overlap=50 : 相邻块之间重叠50字符，避免语义在边界处断裂
        # separators       : 分割优先级：双换行 > 单换行 > 句号 > 空格
        self.general_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", ".", " "],
            length_function=len,
        )

        # ===== 场景2：技术文档（代码块需要更多上下文） =====
        # 更大的 chunk_size 和 overlap，避免代码示例被截断
        # 优先按 Markdown 标题（##、###）分割，保持代码块的完整性
        self.tech_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n## ", "\n### ", "\n\n", "\n", "。", " "],
        )

        # ===== 场景3：对话/短文本（小块更精准） =====
        # 对话内容较短，用更小的块可以精准定位到具体问答对
        self.chat_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30,
            separators=["\n\n", "\n", "。", ".", " "],
        )

        # ===== Markdown 专用：按标题层级结构化分割 =====
        # 将 H1/H2/H3 标题映射为层级标签，切分后保留原标题在内容中
        self.headers_to_split_on = [
            ("#", "h1"),    # 一级标题（文档标题）
            ("##", "h2"),   # 二级标题（章节标题）
            ("###", "h3"),  # 三级标题（小节标题）
        ]

        # strip_headers=False 表示切分后仍在内容中保留标题文本
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False,
        )

    async def process_file(self, file_path: str, file_name: str, doc_id: str = None):
        """
        处理上传的文档文件

        完整的文档处理流水线：
            Step 1: 根据文件扩展名选择合适的加载器
            Step 2: 加载文档内容，提取原始文本
            Step 3: 将长文档切分成适合向量化的文本块
            Step 4: 提取每个文本块的元数据（文件名、分块序号、页码、文档哈希）

        参数:
            file_path: 文件在磁盘上的完整路径
            file_name: 原始文件名（用于判断类型和记录来源）
            doc_id   : 文档内容的 SHA256 哈希（用于去重和溯源，可选）

        返回:
            tuple: (texts, metadata_list)
                - texts         : 文本块内容列表，每个元素是一段文本
                - metadata_list : 对应的元数据列表，每个元素是一个字典

        异常:
            ValueError: 文件格式不支持或加载器创建失败
        """

        try:
            # ============================================================
            # Step 1: 根据文件扩展名选择合适的文档加载器
            # ============================================================
            # 每种格式有对应的 LangChain 加载器：
            #   PyPDFLoader     → PDF 文件
            #   TextLoader      → TXT/MD 文件（纯文本，UTF-8编码）
            #   Docx2txtLoader  → DOCX 文件（Word文档）
            if file_name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_name.endswith(".txt") or file_name.endswith(".md"):
                # Markdown 文件本质是纯文本，先用 TextLoader 加载原始内容
                # 后续再用 MarkdownHeaderTextSplitter 按标题结构化切分
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_name.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            else:
                raise ValueError(f"不支持的文件类型: {file_name}，仅支持 .pdf, .txt, .md, .docx 格式")
            # 备注：unstructured 库 >= 0.12 版本支持自动识别文件类型，
            # 可用 UnstructuredLoader 替代上面所有分支，但需要额外安装依赖
            # loader = UnstructuredLoader(file_path)
        except Exception as e:
            raise ValueError(f"选择文档加载器失败，文件: {file_name}，错误: {e}")

        # ============================================================
        # Step 2: 加载文档内容
        # ============================================================
        # loader.load() 返回 Document 对象列表
        # 每个 Document 包含：
        #   - page_content: 文本内容字符串
        #   - metadata    : 元数据字典（页码、来源等）
        documents = loader.load()

        # ============================================================
        # Step 3: 将文档切分成文本块（Chunking）
        # ============================================================
        # Markdown 文件特殊处理：先合并所有页面内容，再按标题层级结构化切分
        # 这样每个章节标题会作为独立的 chunk 边界，结构更清晰
        if file_name.endswith(".md"):
            txt_document = "\n\n".join([doc.page_content for doc in documents])
            documents = self.md_splitter.split_text(txt_document)

        # 通用递归分割：按 separators 的优先级逐级切分
        # 尽可能在语义边界处断开（段落、句子），而不是在词中间截断
        chunks = self.general_splitter.split_documents(documents)

        # ============================================================
        # Step 4: 提取文本内容和元数据
        # ============================================================
        texts = []          # 文本块内容列表
        metadata_list = []  # 对应的元数据列表

        for i, chunk in enumerate(chunks):
            # 提取文本块的内容
            texts.append(chunk.page_content)

            # 构建该文本块的元数据字典
            # 这些元数据会随着向量一起存入 Milvus，方便后续检索时溯源
            metadata_list.append({
                "source": file_path,                        # 文件完整路径
                "file_name": file_name,                     # 原始文件名
                "chunk_index": i,                           # 该块在文档中的序号（从0开始）
                "page": chunk.metadata.get("page", -1),     # 原始页码（PDF专用，无页码则为-1）
                "doc_id": doc_id                            # 文档内容哈希，用于去重和文档级溯源
            })

        return texts, metadata_list

