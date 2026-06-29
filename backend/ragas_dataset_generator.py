import os
import random
from typing import List, Dict
from backend import document_processor, config
from langchain_community.chat_models import ChatOpenAI
import dashscope
from dashscope import Generation

class RagasDatasetGenerator:
    """
    RAGAS评估数据集生成器
    从ChromaDB中采样文档，使用Qwen-plus生成问题-答案对
    """

    def __init__(self, qwen_api_key=None, qwen_model="qwen-plus"):
        """
        初始化数据集生成器
        :param qwen_api_key: Qwen API密钥
        :param qwen_model: Qwen模型名称
        """
        self.qwen_api_key = qwen_api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.qwen_model = qwen_model

        if not self.qwen_api_key:
            raise ValueError("请设置DASHSCOPE_API_KEY环境变量")

        dashscope.api_key = self.qwen_api_key

        # 配置Qwen作为LLM（使用OpenAI兼容接口）
        # 注意：RAGAS需要langchain的LLM接口，这里使用通义千问的兼容模式
        self.llm = self._create_qwen_llm()
        self.embedding_model = document_processor.get_embedding_model()

    def _create_qwen_llm(self):
        """
        创建Qwen LLM实例（使用OpenAI兼容接口）
        """
        # 使用DashScope的OpenAI兼容接口
        return ChatOpenAI(
            model=self.qwen_model,
            openai_api_key=self.qwen_api_key,
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.7
        )

    def _call_qwen_for_generation(self, prompt: str, temperature=0.7) -> str:
        """
        直接调用Qwen生成内容
        :param prompt: 提示词
        :param temperature: 温度参数
        :return: 生成的文本
        """
        try:
            response = Generation.call(
                model=self.qwen_model,
                messages=[
                    {"role": "system", "content": "你是一个专业的Ariba系统专家，负责生成高质量的问题和答案。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=2000,
                result_format='message'
            )

            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                raise Exception(f"API调用失败: {response.code} - {response.message}")
        except Exception as e:
            print(f"调用Qwen API时出错: {str(e)}")
            raise

    def sample_documents_from_chroma(self, type_of_query: str, n_samples: int = 50) -> List[str]:
        """
        从ChromaDB中随机采样文档
        :param type_of_query: 查询类型（sourcing或integration）
        :param n_samples: 采样数量
        :return: 文档文本列表
        """
        print(f"从{type_of_query}集合中采样{n_samples}个文档...")

        collection = document_processor.get_collection(type_of_query)
        total_docs = collection.count()

        if total_docs == 0:
            raise ValueError(f"{type_of_query}集合为空，请先运行嵌入处理")

        # 随机采样文档ID
        sample_size = min(n_samples, total_docs)

        # 获取所有文档（分批获取以避免内存问题）
        all_docs = []
        batch_size = 1000

        for offset in range(0, total_docs, batch_size):
            limit = min(batch_size, total_docs - offset)
            results = collection.get(limit=limit, offset=offset)
            all_docs.extend(results['documents'])

        # 随机采样
        sampled_docs = random.sample(all_docs, sample_size)

        print(f"成功采样{len(sampled_docs)}个文档")
        return sampled_docs

    def generate_qa_from_context(self, context: str, question_type: str = "simple") -> Dict[str, str]:
        """
        基于上下文生成问题和答案
        :param context: 文档上下文
        :param question_type: 问题类型（simple, reasoning, multi_context）
        :return: 包含question和ground_truth的字典
        """
        if question_type == "simple":
            prompt = f"""基于以下Ariba文档内容，生成一个简单的事实性问题和对应的答案。

文档内容：
{context}

请生成：
1. 一个可以直接从文档中找到答案的问题
2. 该问题的准确答案

格式：
问题：[你的问题]
答案：[准确答案]"""

        elif question_type == "reasoning":
            prompt = f"""基于以下Ariba文档内容，生成一个需要推理的问题和对应的答案。

文档内容：
{context}

请生成：
1. 一个需要理解和推理才能回答的问题
2. 该问题的详细答案

格式：
问题：[你的问题]
答案：[详细答案]"""

        else:  # multi_context
            prompt = f"""基于以下Ariba文档内容，生成一个综合性问题和对应的答案。

文档内容：
{context}

请生成：
1. 一个需要综合多个信息点才能回答的问题
2. 该问题的完整答案

格式：
问题：[你的问题]
答案：[完整答案]"""

        response = self._call_qwen_for_generation(prompt, temperature=0.7)

        # 解析响应
        lines = response.strip().split('\n')
        question = ""
        answer = ""

        for line in lines:
            if line.startswith("问题：") or line.startswith("问题:"):
                question = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("答案：") or line.startswith("答案:"):
                answer = line.split("：", 1)[-1].split(":", 1)[-1].strip()

        return {
            "question": question,
            "ground_truth": answer,
            "context": context
        }

    def generate_dataset(
        self,
        type_of_query: str = "sourcing",
        n_samples: int = 20,
        question_distribution: Dict[str, float] = None
    ) -> List[Dict[str, str]]:
        """
        生成评估数据集
        :param type_of_query: 查询类型（sourcing或integration）
        :param n_samples: 生成的问题数量
        :param question_distribution: 问题类型分布，例如 {"simple": 0.5, "reasoning": 0.3, "multi_context": 0.2}
        :return: 评估数据集列表
        """
        if question_distribution is None:
            question_distribution = {
                "simple": 0.5,
                "reasoning": 0.3,
                "multi_context": 0.2
            }

        print("="*60)
        print(f"开始生成{type_of_query}类型的评估数据集")
        print(f"目标数量: {n_samples}个问题")
        print("="*60)

        # 采样文档
        sampled_docs = self.sample_documents_from_chroma(type_of_query, n_samples * 2)

        # 生成问题
        dataset = []
        question_types = []

        # 根据分布生成问题类型列表
        for q_type, ratio in question_distribution.items():
            count = int(n_samples * ratio)
            question_types.extend([q_type] * count)

        # 补齐到n_samples
        while len(question_types) < n_samples:
            question_types.append("simple")

        random.shuffle(question_types)

        for i, (doc, q_type) in enumerate(zip(sampled_docs[:n_samples], question_types)):
            print(f"\n生成第{i+1}/{n_samples}个问题（类型: {q_type}）...")

            try:
                qa_pair = self.generate_qa_from_context(doc, q_type)

                if qa_pair["question"] and qa_pair["ground_truth"]:
                    dataset.append({
                        "question": qa_pair["question"],
                        "ground_truth": qa_pair["ground_truth"],
                        "context": qa_pair["context"],
                        "type": type_of_query,
                        "question_type": q_type
                    })
                    print(f"✓ 问题: {qa_pair['question'][:100]}...")
                else:
                    print(f"✗ 生成失败，跳过")

            except Exception as e:
                print(f"✗ 生成失败: {str(e)}")
                continue

        print("\n" + "="*60)
        print(f"数据集生成完成！共生成{len(dataset)}个问题")
        print("="*60)

        return dataset

    def save_dataset(self, dataset: List[Dict[str, str]], output_path: str):
        """
        保存数据集到文件
        :param dataset: 数据集
        :param output_path: 输出文件路径
        """
        import json

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        print(f"数据集已保存到: {output_path}")

    def load_dataset(self, input_path: str) -> List[Dict[str, str]]:
        """
        从文件加载数据集
        :param input_path: 输入文件路径
        :return: 数据集
        """
        import json

        with open(input_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        print(f"从{input_path}加载了{len(dataset)}个问题")
        return dataset


if __name__ == "__main__":
    # 示例用法
    generator = RagasDatasetGenerator()

    # 生成sourcing数据集
    sourcing_dataset = generator.generate_dataset(
        type_of_query="sourcing",
        n_samples=10
    )
    generator.save_dataset(sourcing_dataset, "evaluation_dataset_sourcing.json")

    # 生成integration数据集
    integration_dataset = generator.generate_dataset(
        type_of_query="integration",
        n_samples=10
    )
    generator.save_dataset(integration_dataset, "evaluation_dataset_integration.json")
