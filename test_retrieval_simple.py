#!/usr/bin/env python3
"""简单测试检索功能和元数据"""
from backend import retrieval_processor

# 测试检索
print("测试检索功能...")
print("="*60)

query = "What is SAP Ariba Contracts?"
print(f"查询: {query}\n")

retrieved_docs = retrieval_processor.retrieval_process(
    query=query,
    type_of_query="sourcing",
    top_k=3
)

print("\n" + "="*60)
print("检索结果:")
print("="*60)

for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n文档 {i}:")
    if isinstance(doc, dict):
        meta = doc.get('metadata', {})
        content = doc.get('content', '')

        print(f"  标题: {meta.get('document_title', 'N/A')}")
        print(f"  页码: 第{meta.get('page', 0) + 1}页")
        print(f"  文件名: {meta.get('file_name', 'N/A')}")

        if 'rerank_score' in doc:
            print(f"  相关性分数: {doc['rerank_score']:.4f}")

        print(f"  内容预览: {content[:200]}...")
    else:
        print(f"  内容: {doc[:200]}...")

print("\n" + "="*60)
print("格式化引用:")
print("="*60)

from app import format_sources
sources = format_sources(retrieved_docs)
for i, source in enumerate(sources, 1):
    print(f"{i}. {source}")
