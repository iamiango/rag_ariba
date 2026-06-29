#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试文档嵌入过程 - 详细输出版本"""

import sys
import os

print("="*60)
print("开始测试文档嵌入过程（详细模式）")
print("="*60)

# 步骤1: 测试导入
print("\n[步骤1] 测试模块导入...")
try:
    from backend import config, document_loader, document_processor
    print("✓ 模块导入成功")
except Exception as e:
    print(f"✗ 模块导入失败: {e}")
    sys.exit(1)

# 步骤2: 检查配置
print("\n[步骤2] 检查配置...")
print(f"  Sourcing 路径: {config.config.FILE_PATH_SOURCING}")
print(f"  Integration 路径: {config.config.FILE_PATH_INTEGRATION}")
print(f"  ChromaDB 路径: {config.config.CHROMA_PATH}")
print(f"  Embedding 模型路径: {config.config.EMBEDDING_MODEL_PATH}")
print(f"  批量大小: {config.config.BATCH_SIZE}")

# 步骤3: 统计文档数量
print("\n[步骤3] 统计文档数量...")
sourcing_files = []
for root, dirs, files in os.walk(config.config.FILE_PATH_SOURCING):
    for f in files:
        if f.endswith(('.pdf', '.docx', '.pptx', '.xlsx', '.txt')):
            sourcing_files.append(os.path.join(root, f))

print(f"  Sourcing 目录下找到 {len(sourcing_files)} 个文档文件")
if len(sourcing_files) > 0:
    print(f"  示例文件: {sourcing_files[0]}")

# 步骤4: 测试加载单个文档
print("\n[步骤4] 测试加载单个文档...")
if len(sourcing_files) > 0:
    test_file = sourcing_files[0]
    print(f"  测试文件: {test_file}")
    try:
        from langchain_community.document_loaders import PDFPlumberLoader
        loader = PDFPlumberLoader(test_file)
        docs = loader.load()
        print(f"  ✓ 成功加载，页数: {len(docs)}")
        if len(docs) > 0:
            print(f"  前100个字符: {docs[0].page_content[:100]}...")
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")

# 步骤5: 运行完整嵌入（仅 sourcing，用于测试）
print("\n[步骤5] 开始嵌入处理...")
print("  注意: 这可能需要几分钟时间，请耐心等待...")

try:
    document_processor.process_and_embed_documents()
    print("\n✓ 嵌入处理完成！")
except Exception as e:
    print(f"\n✗ 嵌入处理失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 步骤6: 验证结果
print("\n[步骤6] 验证 collections...")
try:
    client = document_processor.get_chroma_client()
    collections = client.list_collections()
    print(f"  现有的 collections: {[c.name for c in collections]}")

    for collection in collections:
        count = collection.count()
        print(f"  - {collection.name}: {count} 个文档块")

    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
except Exception as e:
    print(f"  ✗ 验证失败: {e}")
    sys.exit(1)
