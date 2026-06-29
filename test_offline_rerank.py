#!/usr/bin/env python3
"""测试离线rerank模型"""
from backend import config

print("="*60)
print("测试离线Rerank模型配置")
print("="*60)

print(f"\nRerank配置:")
print(f"  启用状态: {config.config.RERANK_ENABLED}")
print(f"  使用本地模型: {config.config.RERANK_USE_LOCAL_MODEL}")
print(f"  模型路径: {config.config.RERANK_MODEL_PATH}")
print(f"  本地模型路径: {config.config.RERANK_LOCAL_MODEL_PATH}")

# 检查本地模型是否存在
import os
if os.path.exists(config.config.RERANK_LOCAL_MODEL_PATH):
    print(f"\n✓ 本地模型存在")
    print(f"  路径: {config.config.RERANK_LOCAL_MODEL_PATH}")

    # 列出模型文件
    files = os.listdir(config.config.RERANK_LOCAL_MODEL_PATH)
    print(f"\n  模型文件:")
    for f in sorted(files):
        file_path = os.path.join(config.config.RERANK_LOCAL_MODEL_PATH, f)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            size_mb = size / (1024 * 1024)
            print(f"    - {f} ({size_mb:.2f} MB)")
else:
    print(f"\n✗ 本地模型不存在")
    print(f"  路径: {config.config.RERANK_LOCAL_MODEL_PATH}")

print("\n" + "="*60)
print("测试Rerank功能")
print("="*60)

from backend import retrieval_processor

# 测试检索
query = "What is SAP Ariba Contracts?"
print(f"\n查询: {query}")

try:
    retrieved_docs = retrieval_processor.retrieval_process(
        query=query,
        type_of_query="sourcing",
        top_k=3
    )

    print("\n✓ Rerank测试成功！")
    print(f"\n检索到 {len(retrieved_docs)} 个文档:")
    for i, doc in enumerate(retrieved_docs, 1):
        if isinstance(doc, dict):
            meta = doc.get('metadata', {})
            print(f"\n  {i}. 【{meta.get('document_title', 'N/A')}】第{meta.get('page', 0) + 1}页")
            if 'rerank_score' in doc:
                print(f"     Rerank分数: {doc['rerank_score']:.4f}")

except Exception as e:
    print(f"\n✗ 测试失败: {str(e)}")
    import traceback
    traceback.print_exc()
