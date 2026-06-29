#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 平台评估脚本 - V2 版本

使用 evaluation_dataset_sourcing_en_v2.json 和 evaluation_dataset_integration_en_v2.json
对当前 RAG 平台进行全面评估，输出总体评价及优化建议
"""

import os
import json
from datetime import datetime
from backend.ragas_evaluator import RagasEvaluator


def evaluate_rag_platform():
    """
    评估 RAG 平台的完整流程
    """
    print("\n" + "="*80)
    print("RAG 平台评估 - V2 版本")
    print("="*80)
    print(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"评估数据集: evaluation_dataset_sourcing_en_v2.json (15 样本)")
    print(f"            evaluation_dataset_integration_en_v2.json (12 样本)")
    print("="*80)

    # 检查环境变量
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("\n错误: 请设置 DASHSCOPE_API_KEY 环境变量")
        print("示例: export DASHSCOPE_API_KEY='your-api-key'")
        return

    # 初始化评估器
    try:
        evaluator = RagasEvaluator()
    except Exception as e:
        print(f"\n错误: 评估器初始化失败: {str(e)}")
        return

    # 评估结果汇总
    all_results = {}

    # 1. 评估 Sourcing 数据集
    print("\n" + "="*80)
    print("【第 1 部分】评估 Sourcing RAG 系统")
    print("="*80)

    try:
        sourcing_results = evaluator.evaluate_from_file(
            dataset_path="evaluation_dataset_sourcing_en_v2.json",
            output_path="evaluation_results_sourcing_v2.json",
            n_results=8  # 使用当前配置的 MAX_RESULTS
        )
        all_results['sourcing'] = sourcing_results
    except Exception as e:
        print(f"\n错误: Sourcing 评估失败: {str(e)}")
        import traceback
        traceback.print_exc()

    # 2. 评估 Integration 数据集
    print("\n\n" + "="*80)
    print("【第 2 部分】评估 Integration RAG 系统")
    print("="*80)

    try:
        integration_results = evaluator.evaluate_from_file(
            dataset_path="evaluation_dataset_integration_en_v2.json",
            output_path="evaluation_results_integration_v2.json",
            n_results=8
        )
        all_results['integration'] = integration_results
    except Exception as e:
        print(f"\n错误: Integration 评估失败: {str(e)}")
        import traceback
        traceback.print_exc()

    # 3. 生成综合评估报告
    print("\n\n" + "="*80)
    print("【第 3 部分】综合评估报告")
    print("="*80)

    generate_comprehensive_report(all_results)

    print("\n" + "="*80)
    print("评估完成！")
    print("="*80)


def generate_comprehensive_report(all_results):
    """
    生成综合评估报告
    :param all_results: 所有评估结果
    """
    print("\n" + "="*80)
    print("RAG 平台综合评估报告")
    print("="*80)

    # 提取指标
    sourcing_metrics = extract_metrics(all_results.get('sourcing'))
    integration_metrics = extract_metrics(all_results.get('integration'))

    # 打印对比表格
    print("\n【指标对比】")
    print("-"*80)
    print(f"{'指标':<30} {'Sourcing':<20} {'Integration':<20} {'平均':<20}")
    print("-"*80)

    metric_names = {
        "faithfulness": "忠实度",
        "answer_relevancy": "答案相关性",
        "context_precision": "上下文精确度",
        "context_recall": "上下文召回率",
        "answer_correctness": "答案正确性"
    }

    overall_scores = []

    for key, name in metric_names.items():
        sourcing_score = sourcing_metrics.get(key, 0.0)
        integration_score = integration_metrics.get(key, 0.0)
        avg_score = (sourcing_score + integration_score) / 2 if sourcing_score and integration_score else 0.0

        if sourcing_score or integration_score:
            overall_scores.append(avg_score)

        print(f"{name:<30} {sourcing_score:>6.4f}           {integration_score:>6.4f}           {avg_score:>6.4f}")

    print("-"*80)

    # 计算总体评分
    if overall_scores:
        overall_avg = sum(overall_scores) / len(overall_scores)
        sourcing_avg = sum(sourcing_metrics.values()) / len(sourcing_metrics) if sourcing_metrics else 0.0
        integration_avg = sum(integration_metrics.values()) / len(integration_metrics) if integration_metrics else 0.0

        print(f"{'总体评分':<30} {sourcing_avg:>6.4f}           {integration_avg:>6.4f}           {overall_avg:>6.4f}")

        # 总体评价
        print("\n【总体评价】")
        print("-"*80)

        if overall_avg >= 0.8:
            level = "优秀 (Excellent)"
            comment = "RAG 平台表现出色，在 Sourcing 和 Integration 两个领域都展现了高质量的检索和生成能力。"
        elif overall_avg >= 0.7:
            level = "良好 (Good)"
            comment = "RAG 平台表现良好，大部分指标达标，系统基本满足业务需求。"
        elif overall_avg >= 0.6:
            level = "中等 (Fair)"
            comment = "RAG 平台基本可用，但在某些方面存在明显不足，需要针对性优化。"
        else:
            level = "需要改进 (Needs Improvement)"
            comment = "RAG 平台表现不佳，建议进行系统性优化。"

        print(f"等级: {level}")
        print(f"评分: {overall_avg:.4f} / 1.0")
        print(f"评语: {comment}")

        # 详细分析
        print("\n【详细分析】")
        print("-"*80)

        # 分析各个指标
        analyze_metric("忠实度", sourcing_metrics.get("faithfulness", 0), integration_metrics.get("faithfulness", 0))
        analyze_metric("答案相关性", sourcing_metrics.get("answer_relevancy", 0), integration_metrics.get("answer_relevancy", 0))
        analyze_metric("上下文精确度", sourcing_metrics.get("context_precision", 0), integration_metrics.get("context_precision", 0))
        analyze_metric("上下文召回率", sourcing_metrics.get("context_recall", 0), integration_metrics.get("context_recall", 0))
        analyze_metric("答案正确性", sourcing_metrics.get("answer_correctness", 0), integration_metrics.get("answer_correctness", 0))

        # 优化建议
        print("\n【优化建议】")
        print("-"*80)

        suggestions = generate_optimization_suggestions(sourcing_metrics, integration_metrics)
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n{i}. {suggestion['title']}")
            print(f"   优先级: {suggestion['priority']}")
            print(f"   原因: {suggestion['reason']}")
            print(f"   建议:")
            for action in suggestion['actions']:
                print(f"   - {action}")

        # 当前系统配置
        print("\n【当前系统配置】")
        print("-"*80)
        from backend import config
        print(f"• 文档切分大小: {config.config.CHUNK_SIZE}")
        print(f"• 文档重叠: {config.config.CHUNK_OVERLAP}")
        print(f"• 检索文档数: {config.config.MAX_RESULTS}")
        print(f"• LLM 温度: {config.config.LLM_TEMPERATURE}")
        print(f"• 混合检索: {'启用' if config.config.HYBRID_RETRIEVAL else '禁用'}")
        print(f"• Rerank: {'启用' if config.config.RERANK_ENABLED else '禁用'}")
        if config.config.RERANK_ENABLED:
            print(f"• Rerank 倍数: {config.config.RERANK_TOP_K_MULTIPLIER}")

        # 保存综合报告
        save_comprehensive_report({
            'timestamp': datetime.now().isoformat(),
            'overall_score': overall_avg,
            'sourcing_score': sourcing_avg,
            'integration_score': integration_avg,
            'sourcing_metrics': sourcing_metrics,
            'integration_metrics': integration_metrics,
            'level': level,
            'suggestions': suggestions
        })


def extract_metrics(results):
    """
    从评估结果中提取指标
    :param results: RAGAS 评估结果
    :return: 指标字典
    """
    if not results:
        return {}

    metrics = {}

    if hasattr(results, 'to_pandas'):
        df = results.to_pandas()
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            metrics[col] = float(df[col].mean())
    elif hasattr(results, '__dict__'):
        metrics = {k: float(v) for k, v in results.__dict__.items() if isinstance(v, (int, float))}
    else:
        metrics = results

    return metrics


def analyze_metric(name, sourcing_score, integration_score):
    """
    分析单个指标
    """
    avg_score = (sourcing_score + integration_score) / 2

    if avg_score >= 0.8:
        status = "✓ 优秀"
    elif avg_score >= 0.7:
        status = "○ 良好"
    elif avg_score >= 0.6:
        status = "△ 中等"
    else:
        status = "✗ 需改进"

    print(f"\n• {name}: {status} (平均 {avg_score:.4f})")

    if abs(sourcing_score - integration_score) > 0.15:
        if sourcing_score > integration_score:
            print(f"  注意: Sourcing 表现明显优于 Integration ({sourcing_score:.4f} vs {integration_score:.4f})")
        else:
            print(f"  注意: Integration 表现明显优于 Sourcing ({integration_score:.4f} vs {sourcing_score:.4f})")


def generate_optimization_suggestions(sourcing_metrics, integration_metrics):
    """
    生成优化建议
    """
    suggestions = []

    # 计算平均分数
    avg_metrics = {}
    for key in sourcing_metrics.keys():
        avg_metrics[key] = (sourcing_metrics.get(key, 0) + integration_metrics.get(key, 0)) / 2

    # 1. 上下文召回率
    if avg_metrics.get("context_recall", 1.0) < 0.7:
        suggestions.append({
            'title': '提升上下文召回率',
            'priority': '高',
            'reason': f'当前召回率 {avg_metrics.get("context_recall", 0):.4f}，检索的文档未能充分覆盖答案所需信息',
            'actions': [
                '增加检索文档数量 (MAX_RESULTS 从 8 增加到 10-12)',
                '优化文档切分策略 (增加 CHUNK_OVERLAP 到 250-300)',
                '考虑使用更大的 CHUNK_SIZE (1500-2000)',
                '启用查询扩展或查询重写'
            ]
        })

    # 2. 上下文精确度
    if avg_metrics.get("context_precision", 1.0) < 0.7:
        suggestions.append({
            'title': '提升上下文精确度',
            'priority': '高',
            'reason': f'当前精确度 {avg_metrics.get("context_precision", 0):.4f}，检索到的文档中包含较多不相关内容',
            'actions': [
                '确保 Rerank 功能已启用 (RERANK_ENABLED=True)',
                '调整 Rerank 倍数 (RERANK_TOP_K_MULTIPLIER=4-5)',
                '优化查询预处理 (关键词提取、拼写纠正)',
                '考虑使用更精确的 embedding 模型'
            ]
        })

    # 3. 忠实度
    if avg_metrics.get("faithfulness", 1.0) < 0.7:
        suggestions.append({
            'title': '提升答案忠实度',
            'priority': '中',
            'reason': f'当前忠实度 {avg_metrics.get("faithfulness", 0):.4f}，LLM 生成的答案存在幻觉或偏离文档内容',
            'actions': [
                '降低 LLM 温度参数 (LLM_TEMPERATURE 从 0.2 降到 0.1)',
                '优化提示词模板，强调"仅基于提供的文档回答"',
                '添加答案验证步骤',
                '考虑使用更强的 LLM 模型'
            ]
        })

    # 4. 答案相关性
    if avg_metrics.get("answer_relevancy", 1.0) < 0.7:
        suggestions.append({
            'title': '提升答案相关性',
            'priority': '中',
            'reason': f'当前相关性 {avg_metrics.get("answer_relevancy", 0):.4f}，答案未能直接回答用户问题',
            'actions': [
                '优化提示词模板，明确回答格式和要求',
                '改进问题理解和意图识别',
                '添加问题分类，针对不同类型使用不同模板',
                '考虑使用 Few-shot 示例'
            ]
        })

    # 5. 答案正确性
    if avg_metrics.get("answer_correctness", 1.0) < 0.7:
        suggestions.append({
            'title': '提升答案正确性',
            'priority': '高',
            'reason': f'当前正确性 {avg_metrics.get("answer_correctness", 0):.4f}，答案与标准答案差异较大',
            'actions': [
                '检查检索文档的质量和完整性',
                '优化整体检索流程 (向量+BM25+Rerank)',
                '考虑增加文档预处理步骤',
                '评估是否需要更新或补充文档库'
            ]
        })

    # 6. 领域差异
    if abs(sum(sourcing_metrics.values()) / len(sourcing_metrics) -
           sum(integration_metrics.values()) / len(integration_metrics)) > 0.1:
        suggestions.append({
            'title': '平衡不同领域的表现',
            'priority': '中',
            'reason': 'Sourcing 和 Integration 领域的表现存在明显差异',
            'actions': [
                '分析表现较差领域的文档质量',
                '考虑为不同领域使用不同的检索参数',
                '检查文档分布是否均衡',
                '针对性优化表现较差领域的提示词'
            ]
        })

    # 按优先级排序
    priority_order = {'高': 0, '中': 1, '低': 2}
    suggestions.sort(key=lambda x: priority_order.get(x['priority'], 3))

    return suggestions


def save_comprehensive_report(report_data):
    """
    保存综合报告
    """
    output_path = "evaluation_comprehensive_report_v2.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print(f"\n综合报告已保存到: {output_path}")


if __name__ == "__main__":
    evaluate_rag_platform()
