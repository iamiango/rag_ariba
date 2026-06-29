import os
from backend import config, document_loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
from typing import Dict

def enhance_metadata(doc, chunk_index):
    """
    增强文档元数据，添加显示标题和chunk索引
    :param doc: Document对象
    :param chunk_index: chunk在同一页中的索引
    :return: 增强后的元数据字典
    """
    metadata = doc.metadata.copy() if hasattr(doc, 'metadata') and doc.metadata else {}
    source = metadata.get('source', '')

    if source:
        filename = os.path.basename(source)
        parent_dir = os.path.basename(os.path.dirname(source))
        filename_without_ext = os.path.splitext(filename)[0]
        metadata['document_title'] = f"{parent_dir}/{filename_without_ext}"
        metadata['file_name'] = filename

    metadata['chunk_index'] = chunk_index
    return metadata

def indexing_process(type_of_query, embedding_model, collection, enable_image_ocr=False):
    all_chunks = []
    all_ids = []
    all_metadatas = []

    # 加载Document对象列表（enable_image_ocr 控制是否启用图片 OCR）
    documents = document_loader.load_document(type_of_query, enable_image_ocr=enable_image_ocr)
    if documents:
        total_chars = sum(len(doc.page_content) for doc in documents)
        print(f"文档 {type_of_query} 的总字符数: {total_chars}")

        # 优化分隔符优先级，保持技术文档的语义完整性
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.config.CHUNK_SIZE,
            chunk_overlap=config.config.CHUNK_OVERLAP,
            separators=[
                "\n\n## ",    # Markdown二级标题
                "\n\n### ",   # Markdown三级标题
                "\n\n",       # 段落
                "\n",         # 行
                ". ",         # 句子
                " ",          # 空格
                ""
            ]
        )
        # 使用split_documents保留元数据
        chunks = text_splitter.split_documents(documents)
        print(f"文档 {type_of_query} 分割的文本Chunk数量: {len(chunks)}")

        # 为每个chunk增强元数据
        chunk_index_counter = {}
        for chunk in chunks:
            # 为同一页的chunk编号
            page_key = f"{chunk.metadata.get('source', '')}_{chunk.metadata.get('page', 0)}"
            chunk_idx = chunk_index_counter.get(page_key, 0)
            chunk_index_counter[page_key] = chunk_idx + 1

            # 增强元数据
            enhanced_meta = enhance_metadata(chunk, chunk_idx)

            all_chunks.append(chunk.page_content)
            all_ids.append(str(uuid.uuid4()))
            all_metadatas.append(enhanced_meta)

    embeddings = [embedding_model.encode(chunk, normalize_embeddings=True).tolist() for chunk in all_chunks]

    # 分批添加到ChromaDB，避免超过批量大小限制
    batch_size = config.config.BATCH_SIZE  # 从配置文件读取批量大小
    total_chunks = len(all_chunks)

    print(f"开始分批添加 {total_chunks} 个文本块到向量数据库（包含元数据）...")

    for i in range(0, total_chunks, batch_size):
        end_idx = min(i + batch_size, total_chunks)
        batch_ids = all_ids[i:end_idx]
        batch_embeddings = embeddings[i:end_idx]
        batch_documents = all_chunks[i:end_idx]
        batch_metadatas = all_metadatas[i:end_idx]

        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas  # 添加元数据
        )
        print(f"已添加批次 {i//batch_size + 1}: {len(batch_ids)} 个文本块 (总进度: {end_idx}/{total_chunks})")

    print("嵌入生成完成，向量数据库存储完成（包含元数据）.")
    print("索引过程完成.")
    print("********************************************************")