#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查嵌入进程状态"""

import chromadb
import os

print("检查ChromaDB状态...")
print("="*60)

try:
    client = chromadb.PersistentClient(path='./chroma_db')
    collections = client.list_collections()

    if len(collections) == 0:
        print("❌ 没有找到任何collections")
        print("   嵌入进程可能还在运行中...")
    else:
        print(f"✓ 找到 {len(collections)} 个collections:\n")
        for c in collections:
            count = c.count()
            print(f"  - {c.name}: {count:,} 个文档块")

            if c.name == "ariba_sourcing":
                expected = 25102
                if count == 0:
                    print(f"    ⏳ 预期: {expected:,} 个文档块")
                    print(f"    状态: 正在生成embeddings...")
                elif count < expected:
                    progress = (count / expected) * 100
                    print(f"    ⏳ 进度: {progress:.1f}% ({count:,}/{expected:,})")
                else:
                    print(f"    ✓ 完成!")

except Exception as e:
    print(f"错误: {e}")

print("\n" + "="*60)
print("\n提示:")
print("- 如果显示0个文档块，说明正在生成embeddings")
print("- 这个过程可能需要10-20分钟")
print("- 可以运行此脚本随时检查进度")
