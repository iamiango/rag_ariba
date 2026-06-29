#!/usr/bin/env python3
"""测试RAG系统的引用功能"""
import os
os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY", "")

from app import rag_query

# 测试查询
print("测试RAG系统的引用功能...")
print("="*60)

query = "What is SAP Ariba Contracts?"
print(f"查询: {query}\n")

result = rag_query(
    query_text=query,
    type_of_query="sourcing",
    n_results=3
)

print("\n" + "="*60)
print("回答:")
print("="*60)
print(result['answer'])

print("\n" + "="*60)
print("参考文档来源:")
print("="*60)
for i, source in enumerate(result.get('sources', []), 1):
    print(f"{i}. {source}")

print("\n" + "="*60)
print("检索到的文档详情:")
print("="*60)
for i, doc in enumerate(result['retrieved_docs'], 1):
    if isinstance(doc, dict):
        meta = doc.get('metadata', {})
        content_preview = doc.get('content', '')[:150]
        print(f"\n文档 {i}:")
        print(f"  标题: {meta.get('document_title', 'N/A')}")
        print(f"  页码: {meta.get('page', 'N/A') + 1}")  # 转换为1-based
        print(f"  文件名: {meta.get('file_name', 'N/A')}")
        if 'rerank_score' in doc:
            print(f"  相关性分数: {doc['rerank_score']:.4f}")
        print(f"  内容预览: {content_preview}...")
    else:
        print(f"\n文档 {i}: {doc[:150]}...")
