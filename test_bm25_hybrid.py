#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试BM25混合检索功能
"""
import os
import sys

def test_bm25_indexer():
    """测试BM25索引器基本功能"""
    print("\n" + "="*80)
    print("测试1: BM25索引器基本功能")
    print("="*80)

    from backend.bm25_indexer import BM25Indexer

    # 测试文档
    documents = [
        "SAP ERP enables users to create an RFQ with transaction codes ME41 or ME41N",
        "The purchase order can be created using transaction code ME21N in SAP ERP",
        "Ariba Sourcing integration with SAP ERP for procurement processes",
        "Configure the RFQ process in SAP Ariba Sourcing for supplier management"
    ]
    doc_ids = ["doc1", "doc2", "doc3", "doc4"]

    # 构建索引
    indexer = BM25Indexer("test_collection", index_path="./test_bm25_index")
    indexer.build_index(documents, doc_ids)

    # 测试搜索
    test_queries = [
        "create RFQ transaction code",
        "ME41 ME41N",
        "purchase order ME21N",
        "Ariba Sourcing integration"
    ]

    for query in test_queries:
        print(f"\n查询: {query}")
        results = indexer.search(query, top_k=3)
        print("结果:")
        for doc_id, score in results:
            print(f"  {doc_id}: {score:.4f}")

    # 测试保存和加载
    print("\n测试保存和加载索引...")
    indexer.save_index()

    indexer2 = BM25Indexer("test_collection", index_path="./test_bm25_index")
    indexer2.load_index()
    results = indexer2.search("RFQ ME41", top_k=2)
    print(f"加载后搜索 'RFQ ME41': {len(results)} 个结果")

    # 清理测试文件
    import shutil
    if os.path.exists("./test_bm25_index"):
        shutil.rmtree("./test_bm25_index")
        print("✓ 测试文件已清理")

    print("\n✓ BM25索引器测试通过")


def test_hybrid_retrieval():
    """测试混合检索功能"""
    print("\n" + "="*80)
    print("测试2: 混合检索功能")
    print("="*80)

    # 检查BM25索引是否存在
    from backend import config
    from backend.bm25_indexer import BM25Indexer

    sourcing_indexer = BM25Indexer("ariba_sourcing", index_path=config.config.BM25_INDEX_PATH)
    integration_indexer = BM25Indexer("ariba_integration", index_path=config.config.BM25_INDEX_PATH)

    if not sourcing_indexer.index_exists():
        print("\n❌ Sourcing BM25索引不存在")
        print("请先运行: python -m backend.document_processor")
        return False

    if not integration_indexer.index_exists():
        print("\n❌ Integration BM25索引不存在")
        print("请先运行: python -m backend.document_processor")
        return False

    print("✓ BM25索引文件存在")

    # 测试检索
    from backend import retrieval_processor

    test_cases = [
        ("transaction code ME41 ME41N create RFQ", "sourcing"),
        ("SAP Integration Suite managed gateway", "integration"),
    ]

    for query, type_of_query in test_cases:
        print(f"\n查询: {query}")
        print(f"类型: {type_of_query}")
        print("-" * 80)

        try:
            results = retrieval_processor.retrieval_process(
                query=query,
                type_of_query=type_of_query,
                top_k=5
            )
            print(f"\n✓ 检索成功，返回 {len(results)} 个结果")

        except Exception as e:
            print(f"\n❌ 检索失败: {str(e)}")
            return False

    print("\n✓ 混合检索测试通过")
    return True


def test_rag_query():
    """测试完整RAG查询"""
    print("\n" + "="*80)
    print("测试3: 完整RAG查询")
    print("="*80)

    # 检查API key
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("\n⚠️  跳过RAG查询测试（未设置DASHSCOPE_API_KEY）")
        return True

    from app import rag_query

    test_query = "Which transaction codes can users use to create an RFQ in SAP ERP?"

    print(f"\n查询: {test_query}")
    print("-" * 80)

    try:
        result = rag_query(
            query_text=test_query,
            type_of_query="sourcing",
            n_results=5
        )

        print(f"\n✓ RAG查询成功")
        print(f"答案长度: {len(result['answer'])} 字符")
        print(f"检索文档数: {len(result['retrieved_docs'])}")
        print(f"\n答案预览: {result['answer'][:200]}...")

        return True

    except Exception as e:
        print(f"\n❌ RAG查询失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "="*80)
    print("BM25混合检索功能测试")
    print("="*80)

    # 测试1: BM25索引器基本功能
    try:
        test_bm25_indexer()
    except Exception as e:
        print(f"\n❌ 测试1失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # 测试2: 混合检索功能
    try:
        if not test_hybrid_retrieval():
            print("\n提示: 请先构建BM25索引")
            print("运行命令: python -m backend.document_processor")
            return
    except Exception as e:
        print(f"\n❌ 测试2失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # 测试3: 完整RAG查询
    try:
        test_rag_query()
    except Exception as e:
        print(f"\n❌ 测试3失败: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("测试完成")
    print("="*80)


if __name__ == "__main__":
    main()
