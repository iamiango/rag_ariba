from langchain.text_splitter import RecursiveCharacterTextSplitter
import config

def split_documents(raw_documents):
    """切割文档为语义完整的小块"""
    # 配置切割参数（针对英文技术文档优化）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,          # 每个块的字符数
        chunk_overlap=config.CHUNK_OVERLAP,        # 块之间的重叠字符数（保证上下文连贯）
        length_function=len,      # 长度计算方式
        separators=["\n\n", "\n", ". ", " ", ""]  # 优先分割符（英文适配）
    )
    
    # 执行切割
    split_docs = text_splitter.split_documents(raw_documents)
    print(f"切割后文档块数量: {len(split_docs)}")
    # 示例：打印第一个块的内容
    print(f"示例块内容: {split_docs[0].page_content[:200]}...")
    return split_docs