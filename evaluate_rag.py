#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG系统评估脚本
使用RAGAS框架评估Ariba RAG系统的性能
"""

import os
import argparse
from backend.ragas_dataset_generator import RagasDatasetGenerator
from backend.ragas_evaluator import RagasEvaluator


def generate_datasets(n_samples_sourcing=20, n_samples_integration=20):
    """
    生成评估数据集
    :param n_samples_sourcing: Sourcing数据集样本数
    :param n_samples_integration: Integration数据集样本数
    """
    print("\n" + "="*60)
    print("步骤1: 生成评估数据集")
    print("="*60)

    generator = RagasDatasetGenerator()

    # 生成Sourcing数据集
    print("\n生成Sourcing评估数据集...")
    sourcing_dataset = generator.generate_dataset(
        type_of_query="sourcing",
        n_samples=n_samples_sourcing,
        question_distribution={
            "simple": 0.5,
            "reasoning": 0.3,
            "multi_context": 0.2
        }
    )
    generator.save_dataset(sourcing_dataset, "evaluation_dataset_sourcing.json")

    # 生成Integration数据集
    print("\n生成Integration评估数据集...")
    integration_dataset = generator.generate_dataset(
        type_of_query="integration",
        n_samples=n_samples_integration,
        question_distribution={
            "simple": 0.5,
            "reasoning": 0.3,
            "multi_context": 0.2
        }
    )
    generator.save_dataset(integration_dataset, "evaluation_dataset_integration.json")

    print("\n✓ 数据集生成完成！")


def evaluate_rag_system(n_results=5):
    """
    评估RAG系统
    :param n_results: 检索的文档数量
    """
    print("\n" + "="*60)
    print("步骤2: 评估RAG系统")
    print("="*60)

    evaluator = RagasEvaluator()

    # 评估Sourcing RAG
    print("\n【评估Sourcing RAG系统】")
    print("="*60)
    sourcing_results = evaluator.evaluate_from_file(
        dataset_path="evaluation_dataset_sourcing.json",
        output_path="evaluation_results_sourcing.json",
        n_results=n_results
    )

    # 评估Integration RAG
    print("\n\n【评估Integration RAG系统】")
    print("="*60)
    integration_results = evaluator.evaluate_from_file(
        dataset_path="evaluation_dataset_integration.json",
        output_path="evaluation_results_integration.json",
        n_results=n_results
    )

    print("\n✓ 评估完成！")

    return sourcing_results, integration_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAG系统评估工具")
    parser.add_argument(
        "--mode",
        choices=["generate", "evaluate", "all"],
        default="all",
        help="运行模式: generate(生成数据集), evaluate(评估), all(全部)"
    )
    parser.add_argument(
        "--n-samples-sourcing",
        type=int,
        default=20,
        help="Sourcing数据集样本数量"
    )
    parser.add_argument(
        "--n-samples-integration",
        type=int,
        default=20,
        help="Integration数据集样本数量"
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help="RAG检索的文档数量"
    )

    args = parser.parse_args()

    # 检查API密钥
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("错误: 请设置DASHSCOPE_API_KEY环境变量")
        print("示例: export DASHSCOPE_API_KEY='your-api-key'")
        return

    print("\n" + "="*60)
    print("Ariba RAG系统 - RAGAS评估")
    print("="*60)

    try:
        if args.mode in ["generate", "all"]:
            generate_datasets(
                n_samples_sourcing=args.n_samples_sourcing,
                n_samples_integration=args.n_samples_integration
            )

        if args.mode in ["evaluate", "all"]:
            evaluate_rag_system(n_results=args.n_results)

        print("\n" + "="*60)
        print("所有任务完成！")
        print("="*60)
        print("\n生成的文件:")
        print("  - evaluation_dataset_sourcing.json (Sourcing评估数据集)")
        print("  - evaluation_dataset_integration.json (Integration评估数据集)")
        print("  - evaluation_results_sourcing.json (Sourcing评估结果)")
        print("  - evaluation_results_integration.json (Integration评估结果)")

    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
