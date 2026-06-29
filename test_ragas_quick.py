#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAGAS评估系统快速测试脚本
用于验证系统功能是否正常
"""

import os
from backend.ragas_dataset_generator import RagasDatasetGenerator
from backend.ragas_evaluator import RagasEvaluator


def test_dataset_generation():
    """测试数据集生成功能"""
    print("\n" + "="*60)
    print("测试1: 数据集生成")
    print("="*60)

    try:
        generator = RagasDatasetGenerator()

        # 生成小规模测试数据集
        print("\n生成3个Sourcing测试问题...")
        dataset = generator.generate_dataset(
            type_of_query="sourcing",
            n_samples=3,
            question_distribution={
                "simple": 1.0,
                "reasoning": 0.0,
                "multi_context": 0.0
            }
        )

        print(f"\n✓ 成功生成{len(dataset)}个问题")

        # 显示生成的问题
        for i, item in enumerate(dataset):
            print(f"\n问题{i+1}: {item['question']}")
            print(f"答案: {item['ground_truth'][:100]}...")

        # 保存测试数据集
        generator.save_dataset(dataset, "test_dataset.json")
        print("\n✓ 数据集生成测试通过")

        return dataset

    except Exception as e:
        print(f"\n✗ 数据集生成测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_rag_evaluation(dataset):
    """测试RAG评估功能"""
    print("\n" + "="*60)
    print("测试2: RAG评估")
    print("="*60)

    if not dataset:
        print("跳过评估测试（数据集生成失败）")
        return

    try:
        evaluator = RagasEvaluator()

        # 在数据集上运行RAG
        print("\n在测试数据集上运行RAG...")
        rag_results = evaluator.run_rag_on_dataset(dataset, n_results=3)

        print(f"\n✓ 成功处理{len(rag_results)}个问题")

        # 显示RAG结果
        for i, result in enumerate(rag_results):
            print(f"\n问题{i+1}: {result['question']}")
            print(f"RAG答案: {result['answer'][:100]}...")
            print(f"检索到{len(result['contexts'])}个文档")

        # 运行RAGAS评估
        print("\n" + "-"*60)
        print("运行RAGAS评估...")
        results = evaluator.evaluate_rag(rag_results)

        # 打印评估报告
        evaluator.print_evaluation_report(results)

        # 保存结果
        evaluator.save_results(results, "test_results.json")

        print("\n✓ RAG评估测试通过")

    except Exception as e:
        print(f"\n✗ RAG评估测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("RAGAS评估系统 - 快速测试")
    print("="*60)

    # 检查API密钥
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("\n错误: 请设置DASHSCOPE_API_KEY环境变量")
        print("示例: export DASHSCOPE_API_KEY='your-api-key'")
        return

    print("\n✓ API密钥已配置")

    # 测试数据集生成
    dataset = test_dataset_generation()

    # 测试RAG评估
    if dataset:
        test_rag_evaluation(dataset)

    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
    print("\n生成的测试文件:")
    print("  - test_dataset.json (测试数据集)")
    print("  - test_results.json (测试评估结果)")
    print("\n如果测试通过，可以运行完整评估:")
    print("  python evaluate_rag.py --mode all --n-samples-sourcing 20 --n-samples-integration 20")


if __name__ == "__main__":
    main()
