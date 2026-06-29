from backend import document_processor, config
from backend.bm25_indexer import BM25Indexer

# 尝试导入 FlagEmbedding（可选依赖）
try:
    from FlagEmbedding import FlagReranker
    FLAGEMBEDDING_AVAILABLE = True
except ImportError:
    FLAGEMBEDDING_AVAILABLE = False
    print("警告: FlagEmbedding 未安装，rerank 功能将不可用")
    print("安装方法: pip install FlagEmbedding>=1.2.0")

# 全局reranker实例，避免重复加载模型
_reranker = None

def get_reranker():
    """
    获取或初始化reranker模型（单例模式）
    :return: FlagReranker实例或None
    """
    global _reranker

    # 检查 FlagEmbedding 是否可用
    if not FLAGEMBEDDING_AVAILABLE:
        return None

    if _reranker is None and config.config.RERANK_ENABLED:
        try:
            # 优先使用本地模型路径（离线模式）
            if config.config.RERANK_USE_LOCAL_MODEL:
                import os
                if os.path.exists(config.config.RERANK_LOCAL_MODEL_PATH):
                    print(f"正在加载本地Reranker模型: {config.config.RERANK_LOCAL_MODEL_PATH}")
                    _reranker = FlagReranker(
                        config.config.RERANK_LOCAL_MODEL_PATH,
                        use_fp16=True  # 使用半精度加速
                    )
                    print("✓ 本地Reranker模型加载成功（离线模式）")
                else:
                    print(f"警告: 本地模型路径不存在: {config.config.RERANK_LOCAL_MODEL_PATH}")
                    print(f"尝试从HuggingFace加载: {config.config.RERANK_MODEL_PATH}")
                    _reranker = FlagReranker(
                        config.config.RERANK_MODEL_PATH,
                        use_fp16=True
                    )
                    print("Reranker模型加载成功")
            else:
                # 使用HuggingFace模型名称（可能需要网络）
                print(f"正在加载Reranker模型: {config.config.RERANK_MODEL_PATH}")
                _reranker = FlagReranker(
                    config.config.RERANK_MODEL_PATH,
                    use_fp16=True  # 使用半精度加速
                )
                print("Reranker模型加载成功")
        except Exception as e:
            print(f"警告: Reranker模型加载失败: {str(e)}")
            print("将跳过reranking步骤")
            return None
    return _reranker

def rerank_documents(query, documents_with_metadata, top_k):
    """
    使用FlagReranker对检索到的文档进行重新排序
    :param query: 查询文本
    :param documents_with_metadata: 文档列表，每个元素是包含'content'和'metadata'的字典
    :param top_k: 返回前k个文档
    :return: 重新排序后的前top_k个文档（带元数据）
    """
    if not config.config.RERANK_ENABLED:
        return documents_with_metadata[:top_k]

    reranker = get_reranker()
    if reranker is None:
        print("Reranker不可用，返回原始检索结果")
        return documents_with_metadata[:top_k]

    if not documents_with_metadata:
        return []

    try:
        print(f"\n[Rerank] 对 {len(documents_with_metadata)} 个文档进行重新排序...")

        # 提取文本内容用于rerank
        texts = [doc['content'] for doc in documents_with_metadata]
        pairs = [[query, text] for text in texts]

        # 批量计算相关性分数
        scores = reranker.compute_score(pairs, batch_size=config.config.RERANK_BATCH_SIZE)

        # 如果只有一个文档，scores可能是单个值而不是列表
        if not isinstance(scores, list):
            scores = [scores]

        # 将分数添加到文档对象
        for doc, score in zip(documents_with_metadata, scores):
            doc['rerank_score'] = float(score)

        # 按rerank分数排序
        documents_with_metadata.sort(key=lambda x: x['rerank_score'], reverse=True)

        # 添加rerank位置
        for i, doc in enumerate(documents_with_metadata[:top_k]):
            doc['rerank_position'] = i + 1

        # 打印rerank结果（包含元数据）
        print(f"[Rerank] Top {min(top_k, len(documents_with_metadata))} 重排序结果:")
        for doc in documents_with_metadata[:top_k]:
            meta = doc.get('metadata', {})
            if meta and meta.get('document_title'):
                doc_title = meta.get('document_title', 'Unknown')
                page = meta.get('page', 0)
                print(f"  {doc['rerank_position']}. 分数: {doc['rerank_score']:.4f} | "
                      f"来源: 【{doc_title}】第{page+1}页")
            else:
                doc_preview = doc['content'][:100].replace('\n', ' ') + '...' if len(doc['content']) > 100 else doc['content']
                print(f"  {doc['rerank_position']}. 分数: {doc['rerank_score']:.4f} | 文档: {doc_preview}")

        return documents_with_metadata[:top_k]

    except Exception as e:
        print(f"警告: Reranking失败: {str(e)}")
        print("返回原始检索结果")
        return documents_with_metadata[:top_k]

def retrieval_process(query, type_of_query, top_k=6):
    embedding_model = document_processor.get_embedding_model()
    collection = document_processor.get_collection(type_of_query)

    # 如果启用rerank，初始检索更多文档
    initial_top_k = top_k * config.config.RERANK_TOP_K_MULTIPLIER if config.config.RERANK_ENABLED else top_k

    # 步骤1: 向量检索
    query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=initial_top_k,
        include=['documents', 'metadatas', 'distances']  # 包含元数据
    )

    print(f"查询: {query}")
    print(f"向量检索: 前{initial_top_k}个文本块")

    retrieved_docs = []
    vector_ids = set()

    # 添加向量检索结果（带元数据）
    for doc_id, doc, metadata, score in zip(
        vector_results['ids'][0],
        vector_results['documents'][0],
        vector_results['metadatas'][0],
        vector_results['distances'][0]
    ):
        print(f"[向量] 文本块ID: {doc_id}, 相似度: {score}")
        print(f"  文档内容长度: {len(doc) if doc else 0} 字符")
        if not doc or len(doc) < 50:
            print(f"  警告: 文档内容为空或过短！内容: {doc}")
        retrieved_docs.append({
            'content': doc,
            'metadata': metadata,
            'vector_score': float(score),
            'doc_id': doc_id
        })
        vector_ids.add(doc_id)

    # 步骤2: BM25关键词检索 (如果启用混合检索)
    if config.config.HYBRID_RETRIEVAL:
        try:
            bm25_indexer = BM25Indexer(
                f"ariba_{type_of_query}",
                index_path=config.config.BM25_INDEX_PATH
            )

            # 检查索引是否存在
            if bm25_indexer.index_exists():
                bm25_indexer.load_index()
                bm25_results = bm25_indexer.search(query, top_k=initial_top_k)

                print(f"\nBM25检索: 前{initial_top_k}个文本块")

                # 添加BM25结果（去重，带元数据）
                bm25_added = 0
                for doc_id, score in bm25_results:
                    if doc_id not in vector_ids:
                        # 从ChromaDB获取文档文本和元数据
                        doc_data = collection.get(ids=[doc_id], include=['documents', 'metadatas'])
                        if doc_data['documents']:
                            retrieved_docs.append({
                                'content': doc_data['documents'][0],
                                'metadata': doc_data['metadatas'][0] if doc_data['metadatas'] else {},
                                'bm25_score': float(score),
                                'doc_id': doc_id
                            })
                            vector_ids.add(doc_id)
                            bm25_added += 1
                            print(f"[BM25] 文本块ID: {doc_id}, BM25分数: {score:.4f}")

                            # 限制总结果数量
                            if len(retrieved_docs) >= initial_top_k * config.config.BM25_TOP_K_MULTIPLIER:
                                break

                print(f"\n混合检索完成: 向量={len(vector_results['ids'][0])}, BM25新增={bm25_added}, 总计={len(retrieved_docs)}")
            else:
                print(f"\n警告: BM25索引不存在，仅使用向量检索")
                print(f"提示: 运行 'python -m backend.document_processor' 构建BM25索引")

        except Exception as e:
            print(f"\n警告: BM25检索失败，仅使用向量检索: {str(e)}")
    else:
        print(f"\n混合检索已禁用，仅使用向量检索")

    # 步骤3: Rerank重新排序 (如果启用)
    if config.config.RERANK_ENABLED and len(retrieved_docs) > 0:
        print(f"\n[步骤3] 启用Rerank，对检索结果进行重新排序...")
        retrieved_docs = rerank_documents(query, retrieved_docs, top_k)
        print(f"Rerank完成，返回前{top_k}个文档")
    else:
        # 如果未启用rerank，直接截取top_k个
        retrieved_docs = retrieved_docs[:top_k]

    print("\n检索过程完成.")
    print("********************************************************")
    return retrieved_docs