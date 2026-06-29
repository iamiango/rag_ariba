#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整测试RAG流程，追踪每一步的输出
"""

import os
from app import rag_query

# 设置API key（如果未设置）
if not os.getenv("DASHSCOPE_API_KEY"):
    print("请先设置 DASHSCOPE_API_KEY 环境变量")
    exit(1)

def test_rag_flow():
    """测试完整的RAG流程"""
    print("="*80)
    print("RAG流程完整测试")
    print("="*80)

    # 测试问题
    test_query = "What is SAP Ariba Contracts?"
    test_type = "sourcing"

    print(f"\n测试问题: {test_query}")
    print(f"查询类型: {test_type}")
    print(f"温度: 0.2")
    print("\n" + "="*80)

    try:
        # 执行RAG查询
        result = rag_query(
            query_text=test_query,
            type_of_query=test_type,
            n_results=5,
            temperature=0.2
        )

        print("\n" + "="*80)
        print("最终返回结果分析")
        print("="*80)

        print(f"\n1. 答案内容:")
        print(f"   长度: {len(result['answer'])} 字符")
        print(f"   内容: {result['answer']}")

        print(f"\n2. 检索到的文档数量: {len(result.get('retrieved_docs', []))}")

        print(f"\n3. 文档来源:")
        for i, source in enumerate(result.get('sources', []), 1):
            print(f"   {i}. {source}")

        print(f"\n4. 检索文档详情:")
        for i, doc in enumerate(result.get('retrieved_docs', []), 1):
            if isinstance(doc, dict):
                content_len = len(doc.get('content', ''))
                meta = doc.get('metadata', {})
                print(f"   文档 {i}:")
                print(f"     - 内容长度: {content_len} 字符")
                print(f"     - 标题: {meta.get('document_title', 'N/A')}")
                print(f"     - 页码: {meta.get('page', 'N/A')}")
                if 'rerank_score' in doc:
                    print(f"     - 相关性分数: {doc['rerank_score']:.4f}")
                print(f"     - 内容预览: {doc.get('content', '')[:150]}...")

        # 分析问题
        print(f"\n" + "="*80)
        print("问题诊断")
        print("="*80)

        if result['answer'] == "文档中未找到相关信息，无法回答该问题。":
            print("\n❌ LLM返回了'未找到信息'的回答")
            print("\n可能原因:")
            print("1. 检索到的文档内容与问题不够匹配")
            print("2. Prompt模板过于严格，LLM不敢基于文档推理")
            print("3. 温度设置过低（0.2），导致LLM过于保守")
            print("4. 文档内容质量问题（格式、语言等）")

            print("\n建议解决方案:")
            print("1. 增加温度参数（0.5-0.7）让LLM更灵活")
            print("2. 优化Prompt模板，允许基于文档进行合理推理")
            print("3. 检查检索到的文档内容是否真的相关")
            print("4. 尝试用中文提问（如果文档是中文）")
        else:
            print("\n✓ LLM成功生成了答案")

    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag_flow()
