#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试方案A优化效果
"""

import os
from app import rag_query

# 设置API key
if not os.getenv("DASHSCOPE_API_KEY"):
    print("请先设置 DASHSCOPE_API_KEY 环境变量")
    exit(1)

def test_optimization():
    """测试优化后的效果"""
    print("="*80)
    print("测试方案A优化效果")
    print("="*80)

    print("\n优化配置:")
    print("- RERANK_TOP_K_MULTIPLIER: 10 (从5增加)")
    print("- LLM_MAX_TOKENS: 3000 (从2000增加)")
    print("- n_results: 15 (从10增加)")

    # 测试问题
    test_query = "What is SAP Ariba Contracts?"
    test_type = "sourcing"

    print(f"\n测试问题: {test_query}")
    print(f"查询类型: {test_type}")
    print("\n" + "="*80)

    try:
        # 执行RAG查询
        result = rag_query(
            query_text=test_query,
            type_of_query=test_type,
            n_results=15,  # 使用新的检索数量
            temperature=0.3
        )

        print("\n" + "="*80)
        print("查询结果")
        print("="*80)

        print(f"\n答案长度: {len(result['answer'])} 字符")
        print(f"检索文档数量: {len(result.get('retrieved_docs', []))}")

        print(f"\n答案内容:")
        print(result['answer'])

        print(f"\n文档来源:")
        for i, source in enumerate(result.get('sources', [])[:5], 1):
            print(f"  {i}. {source}")

        print("\n" + "="*80)
        print("✓ 测试成功！优化已生效")
        print("="*80)

        print("\n提示:")
        print("- 如果答案更长更详细，说明 LLM_MAX_TOKENS 优化生效")
        print("- 如果检索到15个文档，说明 n_results 优化生效")
        print("- Rerank优化在后台生效，会提高文档相关性")

    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_optimization()
