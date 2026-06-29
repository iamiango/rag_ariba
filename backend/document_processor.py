import chromadb
from sentence_transformers import SentenceTransformer
import os
from backend import config, indexing_processor
from backend.bm25_indexer import BM25Indexer

# 初始化嵌入模型（使用本地模型）
EMBEDDING_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), config.config.EMBEDDING_MODEL_PATH)
embedding_model = None

def get_embedding_model():
    """获取或初始化嵌入模型"""
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
    return embedding_model

def get_chroma_client():
    """获取ChromaDB客户端"""
    return chromadb.PersistentClient(path=config.config.CHROMA_PATH)

def check_metadata_exists(collection):
    """
    检查collection是否包含元数据
    :param collection: ChromaDB collection对象
    :return: True如果元数据存在，False如果不存在
    """
    try:
        sample = collection.get(limit=1, include=['metadatas'])
        if sample['metadatas'] and sample['metadatas'][0]:
            # 检查是否有document_title字段（新元数据的标志）
            return 'document_title' in sample['metadatas'][0]
        return False
    except:
        return False

def process_and_embed_documents():
    """
    一次性处理并嵌入所有文档到ChromaDB
    为sourcing和integration分别创建独立的collection
    同时构建BM25索引
    自动检测并迁移无元数据的旧数据
    """
    print("开始文档嵌入处理...")

    # 获取嵌入模型
    model = get_embedding_model()

    # 获取ChromaDB客户端
    client = get_chroma_client()

    # 检查是否需要重新索引（元数据缺失）
    for collection_name in ["ariba_sourcing", "ariba_integration"]:
        try:
            collection = client.get_collection(collection_name)
            if not check_metadata_exists(collection):
                print(f"\n警告: 检测到 {collection_name} 无元数据，需要重新索引")
                print(f"正在删除旧collection...")
                client.delete_collection(collection_name)
                print(f"{collection_name} 已删除")
        except Exception as e:
            # Collection不存在，继续
            pass

    # 处理Sourcing文档
    print("\n处理Sourcing文档...")
    sourcing_collection = client.get_or_create_collection(name="ariba_sourcing")
    indexing_processor.indexing_process("sourcing", model, sourcing_collection)

    # 处理Integration文档
    print("\n处理Integration文档...")
    integration_collection = client.get_or_create_collection(name="ariba_integration")
    indexing_processor.indexing_process("integration", model, integration_collection)

    print("\n所有文档嵌入完成！")

    # 构建BM25索引
    print("\n" + "="*60)
    print("开始构建BM25索引...")
    print("="*60)

    for type_of_query in ["sourcing", "integration"]:
        print(f"\n构建 {type_of_query} 的BM25索引...")
        collection = get_collection(type_of_query)

        # 分页获取所有文档，避免 SQLite "too many SQL variables"
        batch_size = 500
        offset = 0
        all_docs_list = []
        all_ids_list = []

        while True:
            batch = collection.get(limit=batch_size, offset=offset)
            if not batch["ids"]:  # 没有更多数据
                break

            all_docs_list.extend(batch["documents"])
            all_ids_list.extend(batch["ids"])
            offset += len(batch["ids"])

        if all_docs_list:
            # 构建BM25索引
            bm25_indexer = BM25Indexer(
                f"ariba_{type_of_query}",
                index_path=config.config.BM25_INDEX_PATH
            )
            bm25_indexer.build_index(all_docs_list, all_ids_list)
            bm25_indexer.save_index()
        else:
            print(f"警告: {type_of_query} collection为空，跳过BM25索引构建")

    print("\n" + "="*60)
    print("BM25索引构建完成！")
    print("="*60)

def get_collection(type_of_query):
    """
    获取指定类型的collection用于检索
    :param type_of_query: 查询类型，sourcing或integration
    :return: ChromaDB collection对象
    """
    client = get_chroma_client()

    if type_of_query == "sourcing":
        collection_name = "ariba_sourcing"
    elif type_of_query == "integration":
        collection_name = "ariba_integration"
    else:
        raise ValueError(f"无效的查询类型: {type_of_query}. 必须是'sourcing'或'integration'")

    try:
        collection = client.get_collection(name=collection_name)
        return collection
    except Exception as e:
        raise Exception(f"无法获取collection '{collection_name}': {str(e)}. 请先运行process_and_embed_documents()进行嵌入处理。")

def search_documents(type_of_query, query_text, n_results=None):
    """
    在指定类型的文档中搜索相关内容
    :param type_of_query: 查询类型，sourcing或integration
    :param query_text: 查询文本
    :param n_results: 返回结果数量，默认使用config中的MAX_RESULTS
    :return: 搜索结果
    """
    if n_results is None:
        n_results = config.config.MAX_RESULTS

    # 获取嵌入模型
    model = get_embedding_model()

    # 获取collection
    collection = get_collection(type_of_query)

    # 生成查询文本的嵌入向量
    query_embedding = model.encode(query_text, normalize_embeddings=True).tolist()

    # 在collection中搜索
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results

if __name__ == "__main__":
    # 运行一次性嵌入处理
    process_and_embed_documents()
