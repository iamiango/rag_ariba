#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""运行文档嵌入 - 带输出刷新"""

import sys
import os

# 强制禁用输出缓冲
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

def print_flush(msg):
    """打印并立即刷新"""
    print(msg)
    sys.stdout.flush()

print_flush("="*60)
print_flush("开始文档嵌入处理")
print_flush("="*60)

from backend import document_processor

try:
    print_flush("\n正在处理文档，这可能需要几分钟...")
    print_flush("提示：如果长时间没有输出，说明正在处理大文档\n")

    document_processor.process_and_embed_documents()

    print_flush("\n" + "="*60)
    print_flush("嵌入处理完成！")
    print_flush("="*60)

    # 验证结果
    print_flush("\n验证 collections...")
    client = document_processor.get_chroma_client()
    collections = client.list_collections()
    print_flush(f"现有 collections: {[c.name for c in collections]}")

    for c in collections:
        count = c.count()
        print_flush(f"  - {c.name}: {count} 个文档块")

except Exception as e:
    print_flush(f"\n错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
