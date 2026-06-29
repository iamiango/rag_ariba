#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速RAG评估脚本 - 跳过查询预处理以加速评估
"""

import os
import sys

# 临时禁用查询预处理以加速评估
os.environ['SKIP_QUERY_PREPROCESSING'] = '1'

from backend.ragas_evaluator import RagasEvaluator
import json


def quick_evaluate():
    """快速评估 - 使用v2数据集"""

    print("\n" + "="*80)
    print("快速RAG评估 (跳过查询预处理)")
    print("="*80)

    if not os.getenv("DASHSCOPE_API_KEY"):
        print("\n错误: 请设置DASHSCOPE_API_KEY环境变量")
        return

    evaluator = RagasEvaluator()

    # 评估Sourcing (使用较少样本加速)
    print("\n评估 Sourcing RAG...")
    try:
        sourcing_results = evaluator.evaluate_from_file(
            dataset_path="evaluation_dataset_sourcing_en_v2.json",
            output_path="evaluation_results_sourcing_quick.json",
            n_results=5
        )
    except Exception as e:
        print(f"Sourcing评估失败: {e}")
        sourcing_results = None

    # 评估Integration
    print("\n\n评估 Integration RAG...")
    try:
        integration_results = evaluator.evaluate_from_file(
            dataset_path="evaluation_dataset_integration_en_v2.json",
            output_path="evaluation_results_integration_quick.json",
            n_results=5
        )
    except Exception as e:
        print(f"Integration评估失败: {e}")
        integration_results = None

    print("\n评估完成!")


if __name__ == "__main__":
    quick_evaluate()
