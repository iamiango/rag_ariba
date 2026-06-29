import os
import json
from typing import List, Dict
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
import dashscope
from app import rag_query
from backend import config


class RagasEvaluator:
    """
    RAGAS评估器
    使用生成的数据集评估RAG系统性能，输出定量指标和整体评价
    """

    def __init__(self, qwen_api_key=None, qwen_model="qwen-plus"):
        """
        初始化评估器
        :param qwen_api_key: Qwen API密钥
        :param qwen_model: Qwen模型名称
        """
        self.qwen_api_key = qwen_api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.qwen_model = qwen_model

        if not self.qwen_api_key:
            raise ValueError("请设置DASHSCOPE_API_KEY环境变量")

        dashscope.api_key = self.qwen_api_key

        # 配置Qwen作为评估LLM
        self.llm = ChatOpenAI(
            model=self.qwen_model,
            openai_api_key=self.qwen_api_key,
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0
        )

        # 配置本地embedding模型用于RAGAS评估
        embedding_model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            config.config.EMBEDDING_MODEL_PATH
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def load_dataset(self, dataset_path: str) -> List[Dict[str, str]]:
        """
        加载评估数据集
        :param dataset_path: 数据集文件路径
        :return: 数据集列表
        """
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        print(f"加载了{len(dataset)}个评估样本")
        return dataset

    def run_rag_on_dataset(self, dataset: List[Dict[str, str]], n_results: int = 5) -> List[Dict]:
        """
        在数据集上运行RAG系统，收集答案和检索的上下文
        :param dataset: 评估数据集
        :param n_results: 检索的文档数量
        :return: 包含RAG输出的数据集
        """
        print("="*60)
        print("在数据集上运行RAG系统...")
        print("="*60)

        rag_results = []

        for i, item in enumerate(dataset):
            question = item['question']
            ground_truth = item['ground_truth']
            type_of_query = item.get('type', 'sourcing')

            print(f"\n[{i+1}/{len(dataset)}] 处理问题: {question[:80]}...")

            try:
                # 运行RAG查询
                result = rag_query(
                    query_text=question,
                    type_of_query=type_of_query,
                    n_results=n_results,
                    temperature=0.7
                )

                # 提取文档内容作为contexts（RAGAS需要字符串列表）
                contexts = []
                for doc in result['retrieved_docs']:
                    if isinstance(doc, dict):
                        contexts.append(doc.get('content', ''))
                    else:
                        contexts.append(str(doc))

                # 构建RAGAS评估所需的格式
                rag_results.append({
                    "question": question,
                    "answer": result['answer'],
                    "contexts": contexts,  # RAGAS需要contexts是字符串列表
                    "ground_truth": ground_truth
                })

                print(f"✓ 答案: {result['answer'][:100]}...")
                print(f"✓ 检索到 {len(contexts)} 个文档片段")

            except Exception as e:
                print(f"✗ RAG查询失败: {str(e)}")
                # 添加空结果以保持数据集完整性
                rag_results.append({
                    "question": question,
                    "answer": "查询失败",
                    "contexts": [],
                    "ground_truth": ground_truth
                })

        print("\n" + "="*60)
        print(f"RAG查询完成！共处理{len(rag_results)}个问题")
        print("="*60)

        return rag_results

    def evaluate_rag(self, rag_results: List[Dict]) -> Dict:
        """
        使用RAGAS评估RAG系统
        :param rag_results: RAG系统的输出结果
        :return: 评估指标字典
        """
        print("\n" + "="*60)
        print("开始RAGAS评估...")
        print("="*60)

        # 转换为RAGAS Dataset格式
        dataset_dict = {
            "question": [item["question"] for item in rag_results],
            "answer": [item["answer"] for item in rag_results],
            "contexts": [item["contexts"] for item in rag_results],
            "ground_truth": [item["ground_truth"] for item in rag_results]
        }

        ragas_dataset = Dataset.from_dict(dataset_dict)

        # 配置评估指标
        metrics = [
            faithfulness,           # 忠实度：答案是否基于检索的上下文
            answer_relevancy,       # 答案相关性：答案是否回答了问题
            context_precision,      # 上下文精确度：检索的上下文是否相关
            context_recall,         # 上下文召回率：检索的上下文是否包含ground truth所需的信息
            answer_correctness      # 答案正确性：答案与ground truth的相似度
        ]

        # 运行评估
        print("\n正在计算评估指标...")
        try:
            results = evaluate(
                dataset=ragas_dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embeddings
            )

            print("\n✓ 评估完成！")
            return results

        except Exception as e:
            print(f"\n✗ 评估失败: {str(e)}")
            raise

    def print_evaluation_report(self, results: Dict):
        """
        打印评估报告
        :param results: RAGAS评估结果
        """
        print("\n" + "="*60)
        print("RAGAS评估报告")
        print("="*60)

        # 提取指标 - 处理RAGAS返回的对象
        # results可能是EvaluationResult对象，需要转换为字典
        metrics = {}

        if hasattr(results, 'to_pandas'):
            # 转换为DataFrame然后只提取数值列的平均值
            df = results.to_pandas()
            # 只选择数值类型的列
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                metrics[col] = df[col].mean()
        elif hasattr(results, '__dict__'):
            metrics = results.__dict__
        else:
            metrics = results

        print("\n【定量指标】")
        print("-"*60)

        metric_names = {
            "faithfulness": "忠实度 (Faithfulness)",
            "answer_relevancy": "答案相关性 (Answer Relevancy)",
            "context_precision": "上下文精确度 (Context Precision)",
            "context_recall": "上下文召回率 (Context Recall)",
            "answer_correctness": "答案正确性 (Answer Correctness)"
        }

        scores = {}
        for key, name in metric_names.items():
            try:
                if key in metrics:
                    score = float(metrics[key])
                    scores[key] = score
                    print(f"{name:40s}: {score:.4f}")
            except (KeyError, TypeError, ValueError):
                print(f"{name:40s}: N/A (计算失败)")

        # 计算总体评分
        if scores:
            overall_score = sum(scores.values()) / len(scores)
            print("-"*60)
            print(f"{'总体评分 (Overall Score)':40s}: {overall_score:.4f}")

        print("\n【指标说明】")
        print("-"*60)
        print("• 忠实度: 衡量答案是否基于检索的上下文，避免幻觉")
        print("• 答案相关性: 衡量答案是否直接回答了用户的问题")
        print("• 上下文精确度: 衡量检索的文档是否与问题相关")
        print("• 上下文召回率: 衡量检索的文档是否包含回答问题所需的信息")
        print("• 答案正确性: 衡量答案与标准答案的相似度")
        print("\n所有指标范围: 0.0 - 1.0，分数越高越好")

        # 整体评价
        print("\n【整体评价】")
        print("-"*60)

        if scores:
            overall_score = sum(scores.values()) / len(scores)

            if overall_score >= 0.8:
                evaluation = "优秀 (Excellent)"
                comment = "RAG系统表现出色，检索准确，答案质量高，忠实于文档内容。"
            elif overall_score >= 0.7:
                evaluation = "良好 (Good)"
                comment = "RAG系统表现良好，大部分指标达标，但仍有优化空间。"
            elif overall_score >= 0.6:
                evaluation = "中等 (Fair)"
                comment = "RAG系统基本可用，但在检索质量或答案生成方面需要改进。"
            else:
                evaluation = "需要改进 (Needs Improvement)"
                comment = "RAG系统表现不佳，建议优化检索策略、文档切分或提示词模板。"

            print(f"等级: {evaluation}")
            print(f"评语: {comment}")

            # 具体建议
            print("\n【优化建议】")
            print("-"*60)

            if scores.get("context_recall", 1.0) < 0.7:
                print("• 上下文召回率较低，建议：")
                print("  - 增加检索的文档数量 (n_results)")
                print("  - 优化文档切分策略 (chunk_size, chunk_overlap)")
                print("  - 检查embedding模型是否适合当前领域")

            if scores.get("context_precision", 1.0) < 0.7:
                print("• 上下文精确度较低，建议：")
                print("  - 优化查询重写策略")
                print("  - 使用混合检索 (向量检索 + 关键词检索)")
                print("  - 添加重排序 (reranking) 步骤")

            if scores.get("faithfulness", 1.0) < 0.7:
                print("• 忠实度较低，建议：")
                print("  - 优化提示词模板，强调基于文档回答")
                print("  - 降低LLM的temperature参数")
                print("  - 添加答案验证步骤")

            if scores.get("answer_relevancy", 1.0) < 0.7:
                print("• 答案相关性较低，建议：")
                print("  - 优化提示词模板，明确回答格式")
                print("  - 改进问题理解和意图识别")

            if scores.get("answer_correctness", 1.0) < 0.7:
                print("• 答案正确性较低，建议：")
                print("  - 检查检索的文档质量")
                print("  - 优化LLM的生成策略")
                print("  - 考虑使用更强大的LLM模型")

        print("\n" + "="*60)

    def save_results(self, results: Dict, output_path: str):
        """
        保存评估结果到文件
        :param results: 评估结果
        :param output_path: 输出文件路径
        """
        # 转换为可序列化的格式
        metrics = {}

        if hasattr(results, 'to_pandas'):
            # 转换为DataFrame然后只提取数值列的平均值
            df = results.to_pandas()
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                metrics[col] = float(df[col].mean())
        elif hasattr(results, '__dict__'):
            metrics = results.__dict__
        else:
            metrics = results

        results_dict = {
            "metrics": {k: float(v) if hasattr(v, '__float__') else str(v) for k, v in metrics.items()}
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)

        print(f"\n评估结果已保存到: {output_path}")

    def evaluate_from_file(
        self,
        dataset_path: str,
        output_path: str = None,
        n_results: int = 5
    ):
        """
        从文件加载数据集并完成完整的评估流程
        :param dataset_path: 数据集文件路径
        :param output_path: 评估结果输出路径
        :param n_results: 检索的文档数量
        """
        # 加载数据集
        dataset = self.load_dataset(dataset_path)

        # 运行RAG
        rag_results = self.run_rag_on_dataset(dataset, n_results=n_results)

        # 评估
        results = self.evaluate_rag(rag_results)

        # 打印报告
        self.print_evaluation_report(results)

        # 保存结果
        if output_path:
            self.save_results(results, output_path)

        return results


if __name__ == "__main__":
    # 示例用法
    evaluator = RagasEvaluator()

    # 评估sourcing数据集
    print("\n" + "="*60)
    print("评估Sourcing RAG系统")
    print("="*60)
    evaluator.evaluate_from_file(
        dataset_path="evaluation_dataset_sourcing.json",
        output_path="evaluation_results_sourcing.json",
        n_results=5
    )

    # 评估integration数据集
    print("\n\n" + "="*60)
    print("评估Integration RAG系统")
    print("="*60)
    evaluator.evaluate_from_file(
        dataset_path="evaluation_dataset_integration.json",
        output_path="evaluation_results_integration.json",
        n_results=5
    )
