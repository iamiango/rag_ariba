#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单RAG测试脚本 - 验证文档嵌入和问答功能
不使用RAGAS框架，直接测试RAG管道
"""

import os
import json
import time

def test_rag_system():
    """测试RAG系统基本功能"""

    print("\n" + "="*80)
    print("RAG系统简单测试")
    print("="*80)

    if not os.getenv("DASHSCOPE_API_KEY"):
        print("\n错误: 请设置DASHSCOPE_API_KEY环境变量")
        return

    # 导入RAG模块
    from app import rag_query
    from backend import document_processor

    # 测试问题
    test_questions = [
        {
            "question": "What are the main event types available in SAP Ariba Sourcing?",
            "type": "sourcing"
        },
        {
            "question": "How does SAP Integration Suite connect with SAP Business Network?",
            "type": "integration"
        },
        {
            "question": "What is a reverse auction in SAP Ariba?",
            "type": "sourcing"
        }
    ]

    results = []

    # 检查ChromaDB集合状态
    print("\n[步骤1] 检查ChromaDB集合状态...")
    try:
        sourcing_collection = document_processor.get_collection("sourcing")
        integration_collection = document_processor.get_collection("integration")
        print(f"  ✓ Sourcing集合: {sourcing_collection.count()} 条记录")
        print(f"  ✓ Integration集合: {integration_collection.count()} 条记录")
    except Exception as e:
        print(f"  ✗ 获取集合失败: {e}")
        return

    # 运行测试问题
    print("\n[步骤2] 运行测试问题...")

    for i, q in enumerate(test_questions):
        print(f"\n{'='*60}")
        print(f"问题 {i+1}/{len(test_questions)}: {q['question'][:60]}...")
        print(f"类型: {q['type']}")
        print("="*60)

        start_time = time.time()

        try:
            result = rag_query(
                query_text=q['question'],
                type_of_query=q['type'],
                n_results=5,
                temperature=0.2
            )

            elapsed = time.time() - start_time

            print(f"\n[答案] ({elapsed:.1f}秒)")
            print("-"*40)
            # 截取答案前500字符
            answer = result.get('answer', '无答案')
            if len(answer) > 500:
                print(answer[:500] + "...")
            else:
                print(answer)

            print(f"\n[检索到的文档数]: {len(result.get('retrieved_docs', []))}")

            # 显示前2个检索文档的来源
            docs = result.get('retrieved_docs', [])
            if docs:
                print("\n[文档来源]:")
                for j, doc in enumerate(docs[:2]):
                    if isinstance(doc, dict):
                        source = doc.get('metadata', {}).get('source', '未知')
                    else:
                        source = "文本片段"
                    print(f"  {j+1}. {source[:80]}")

            results.append({
                "question": q['question'],
                "type": q['type'],
                "answer_length": len(answer),
                "docs_retrieved": len(docs),
                "elapsed_seconds": elapsed,
                "status": "success"
            })

        except Exception as e:
            print(f"\n[错误] {str(e)}")
            results.append({
                "question": q['question'],
                "type": q['type'],
                "error": str(e),
                "status": "failed"
            })

    # 打印总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)

    success_count = sum(1 for r in results if r.get('status') == 'success')
    print(f"\n成功: {success_count}/{len(results)}")

    if success_count > 0:
        avg_time = sum(r.get('elapsed_seconds', 0) for r in results if r.get('status') == 'success') / success_count
        avg_docs = sum(r.get('docs_retrieved', 0) for r in results if r.get('status') == 'success') / success_count
        print(f"平均响应时间: {avg_time:.1f}秒")
        print(f"平均检索文档数: {avg_docs:.1f}")

    # 保存结果
    with open('test_rag_simple_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: test_rag_simple_results.json")

    print("\n" + "="*80)
    print("RAG系统测试完成!")
    print("="*80)


if __name__ == "__main__":
    test_rag_system()
