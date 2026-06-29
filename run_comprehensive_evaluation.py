#!/usr/bin/env python3
"""
综合评估脚本 - 评估Sourcing和Integration数据集，生成综合报告
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.ragas_evaluator import RagasEvaluator


def generate_suggestions(sourcing_metrics: Dict, integration_metrics: Dict, overall_score: float) -> List[Dict]:
    """
    根据评估指标生成优化建议
    """
    suggestions = []

    # 计算平均指标
    avg_metrics = {}
    for key in sourcing_metrics.keys():
        avg_metrics[key] = (sourcing_metrics[key] + integration_metrics[key]) / 2

    # 答案正确性建议
    if avg_metrics.get('answer_correctness', 1.0) < 0.7:
        suggestions.append({
            "title": "提升答案正确性",
            "priority": "高",
            "reason": f"当前正确性 {avg_metrics['answer_correctness']:.4f}，答案与标准答案差异较大",
            "actions": [
                "检查检索文档的质量和完整性",
                "优化整体检索流程 (向量+BM25+Rerank)",
                "考虑增加文档预处理步骤",
                "评估是否需要更新或补充文档库"
            ]
        })

    # 忠实度建议
    if avg_metrics.get('faithfulness', 1.0) < 0.7:
        suggestions.append({
            "title": "提升答案忠实度",
            "priority": "高" if avg_metrics.get('faithfulness', 1.0) < 0.6 else "中",
            "reason": f"当前忠实度 {avg_metrics['faithfulness']:.4f}，LLM 生成的答案存在幻觉或偏离文档内容",
            "actions": [
                "降低 LLM 温度参数 (LLM_TEMPERATURE 从 0.7 降到 0.2)",
                "优化提示词模板，强调\"仅基于提供的文档回答\"",
                "添加答案验证步骤",
                "考虑使用更强的 LLM 模型"
            ]
        })

    # 答案相关性建议
    if avg_metrics.get('answer_relevancy', 1.0) < 0.7:
        suggestions.append({
            "title": "提升答案相关性",
            "priority": "中",
            "reason": f"当前相关性 {avg_metrics['answer_relevancy']:.4f}，答案未能直接回答用户问题",
            "actions": [
                "优化提示词模板，明确回答格式和要求",
                "改进问题理解和意图识别",
                "添加问题分类，针对不同类型使用不同模板",
                "考虑使用 Few-shot 示例"
            ]
        })

    # 上下文召回率建议
    if avg_metrics.get('context_recall', 1.0) < 0.7:
        suggestions.append({
            "title": "提升上下文召回率",
            "priority": "中",
            "reason": f"当前召回率 {avg_metrics['context_recall']:.4f}，检索的文档未能包含回答所需的关键信息",
            "actions": [
                "增加检索的文档数量 (n_results 从 5 增加到 10)",
                "优化文档切分策略 (调整 chunk_size 和 chunk_overlap)",
                "检查 embedding 模型是否适合当前领域",
                "考虑使用混合检索策略"
            ]
        })

    # 上下文精确度建议
    if avg_metrics.get('context_precision', 1.0) < 0.7:
        suggestions.append({
            "title": "提升上下文精确度",
            "priority": "中",
            "reason": f"当前精确度 {avg_metrics['context_precision']:.4f}，检索的文档包含过多不相关内容",
            "actions": [
                "优化查询预处理和关键词提取",
                "调整 BM25 和向量检索的权重",
                "优化 Rerank 模型的 top_k 参数",
                "考虑添加查询扩展或改写"
            ]
        })

    # 领域差异建议
    score_diff = abs(sourcing_metrics.get('answer_correctness', 0) - integration_metrics.get('answer_correctness', 0))
    if score_diff > 0.15:
        better_domain = "Sourcing" if sourcing_metrics.get('answer_correctness', 0) > integration_metrics.get('answer_correctness', 0) else "Integration"
        worse_domain = "Integration" if better_domain == "Sourcing" else "Sourcing"

        suggestions.append({
            "title": "平衡不同领域的表现",
            "priority": "中",
            "reason": f"{better_domain} 和 {worse_domain} 领域的表现存在明显差异 (差距 {score_diff:.4f})",
            "actions": [
                f"分析 {worse_domain} 领域的文档质量和覆盖度",
                f"考虑为 {worse_domain} 领域使用不同的检索参数",
                "检查文档分布是否均衡",
                f"针对性优化 {worse_domain} 领域的提示词"
            ]
        })

    return suggestions


def determine_level(overall_score: float) -> str:
    """
    根据总体评分确定评级
    """
    if overall_score >= 0.8:
        return "优秀 (Excellent)"
    elif overall_score >= 0.7:
        return "良好 (Good)"
    elif overall_score >= 0.6:
        return "中等 (Fair)"
    else:
        return "需要改进 (Needs Improvement)"


def run_comprehensive_evaluation(
    sourcing_dataset: str,
    integration_dataset: str,
    output_file: str,
    n_results: int = 5
):
    """
    运行综合评估
    """
    print("="*80)
    print("开始综合评估 - Sourcing 和 Integration 数据集")
    print("="*80)

    # 初始化评估器
    evaluator = RagasEvaluator()

    # 评估 Sourcing 数据集
    print("\n" + "="*80)
    print("第 1 步: 评估 Sourcing 数据集")
    print("="*80)

    sourcing_data = evaluator.load_dataset(sourcing_dataset)
    sourcing_rag_results = evaluator.run_rag_on_dataset(sourcing_data, n_results=n_results)
    sourcing_results = evaluator.evaluate_rag(sourcing_rag_results)

    # 提取 Sourcing 指标
    sourcing_df = sourcing_results.to_pandas()
    sourcing_metrics = {}
    for col in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'answer_correctness']:
        if col in sourcing_df.columns:
            sourcing_metrics[col] = float(sourcing_df[col].mean())

    sourcing_score = sum(sourcing_metrics.values()) / len(sourcing_metrics)

    print(f"\n✓ Sourcing 评估完成，总分: {sourcing_score:.4f}")

    # 评估 Integration 数据集
    print("\n" + "="*80)
    print("第 2 步: 评估 Integration 数据集")
    print("="*80)

    integration_data = evaluator.load_dataset(integration_dataset)
    integration_rag_results = evaluator.run_rag_on_dataset(integration_data, n_results=n_results)
    integration_results = evaluator.evaluate_rag(integration_rag_results)

    # 提取 Integration 指标
    integration_df = integration_results.to_pandas()
    integration_metrics = {}
    for col in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'answer_correctness']:
        if col in integration_df.columns:
            integration_metrics[col] = float(integration_df[col].mean())

    integration_score = sum(integration_metrics.values()) / len(integration_metrics)

    print(f"\n✓ Integration 评估完成，总分: {integration_score:.4f}")

    # 计算总体评分
    overall_score = (sourcing_score + integration_score) / 2

    # 生成建议
    suggestions = generate_suggestions(sourcing_metrics, integration_metrics, overall_score)

    # 确定评级
    level = determine_level(overall_score)

    # 构建综合报告
    comprehensive_report = {
        "timestamp": datetime.now().isoformat(),
        "overall_score": overall_score,
        "sourcing_score": sourcing_score,
        "integration_score": integration_score,
        "sourcing_metrics": sourcing_metrics,
        "integration_metrics": integration_metrics,
        "level": level,
        "suggestions": suggestions
    }

    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_report, f, ensure_ascii=False, indent=2)

    print("\n" + "="*80)
    print("综合评估报告")
    print("="*80)
    print(f"\n总体评分: {overall_score:.4f}")
    print(f"评级: {level}")
    print(f"\nSourcing 评分: {sourcing_score:.4f}")
    print(f"Integration 评分: {integration_score:.4f}")
    print(f"\n报告已保存到: {output_file}")
    print("="*80)

    return comprehensive_report


if __name__ == "__main__":
    # 运行综合评估
    run_comprehensive_evaluation(
        sourcing_dataset="evaluation_dataset_sourcing_en_v2.json",
        integration_dataset="evaluation_dataset_integration_en_v2.json",
        output_file="evaluation_comprehensive_report_v6.json",
        n_results=15
    )
