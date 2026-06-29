#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""简单测试 - 只加载少量文档"""

import sys
import os

print("="*60)
print("简单测试：加载前3个文档")
print("="*60)

from backend import config
import time

# 找到前3个 PDF 文件
print("\n查找文档...")
pdf_files = []
for root, dirs, files in os.walk(config.config.FILE_PATH_SOURCING):
    for f in files:
        if f.endswith('.pdf'):
            pdf_files.append(os.path.join(root, f))
            if len(pdf_files) >= 3:
                break
    if len(pdf_files) >= 3:
        break

print(f"找到 {len(pdf_files)} 个 PDF 文件")

# 逐个加载
from langchain_community.document_loaders import PDFPlumberLoader

for i, pdf_file in enumerate(pdf_files, 1):
    print(f"\n[{i}/{len(pdf_files)}] 加载: {os.path.basename(pdf_file)}")
    start = time.time()
    try:
        loader = PDFPlumberLoader(pdf_file)
        docs = loader.load()
        elapsed = time.time() - start
        total_chars = sum(len(doc.page_content) for doc in docs)
        print(f"  ✓ 成功 - {len(docs)} 页, {total_chars} 字符, 耗时 {elapsed:.2f}秒")
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ✗ 失败 - {e}, 耗时 {elapsed:.2f}秒")

print("\n" + "="*60)
print("测试完成")
print("="*60)
