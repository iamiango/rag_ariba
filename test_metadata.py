#!/usr/bin/env python3
"""测试元数据是否正确存储"""
import chromadb
from backend import config

# 获取ChromaDB客户端
client = chromadb.PersistentClient(path=config.config.CHROMA_PATH)

# 检查sourcing collection
print("检查 ariba_sourcing collection 的元数据...")
print("="*60)
sourcing_collection = client.get_collection("ariba_sourcing")
sample = sourcing_collection.get(limit=3, include=['documents', 'metadatas'])

for i, (doc, meta) in enumerate(zip(sample['documents'], sample['metadatas'])):
    print(f"\n样本 {i+1}:")
    print(f"文档内容预览: {doc[:100]}...")
    print(f"元数据: {meta}")
    if meta and 'document_title' in meta:
        print(f"✓ 文档标题: {meta['document_title']}")
        print(f"✓ 页码: {meta.get('page', 'N/A')}")
    else:
        print("✗ 缺少元数据 (document_title)")

print("\n" + "="*60)
print("检查 ariba_integration collection 的元数据...")
print("="*60)
integration_collection = client.get_collection("ariba_integration")
sample = integration_collection.get(limit=3, include=['documents', 'metadatas'])

for i, (doc, meta) in enumerate(zip(sample['documents'], sample['metadatas'])):
    print(f"\n样本 {i+1}:")
    print(f"文档内容预览: {doc[:100]}...")
    print(f"元数据: {meta}")
    if meta and 'document_title' in meta:
        print(f"✓ 文档标题: {meta['document_title']}")
        print(f"✓ 页码: {meta.get('page', 'N/A')}")
    else:
        print("✗ 缺少元数据 (document_title)")
