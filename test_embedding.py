#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试文档嵌入过程"""

import sys
from backend import document_processor

if __name__ == "__main__":
    print("="*60)
    print("开始测试文档嵌入过程")
    print("="*60)

    try:
        # 运行嵌入处理
        document_processor.process_and_embed_documents()

        print("\n" + "="*60)
        print("嵌入处理完成！")
        print("="*60)

        # 验证 collections 是否创建成功
        print("\n验证 collections...")
        client = document_processor.get_chroma_client()
        collections = client.list_collections()
        print(f"现有的 collections: {[c.name for c in collections]}")

        for collection in collections:
            count = collection.count()
            print(f"  - {collection.name}: {count} 个文档块")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
