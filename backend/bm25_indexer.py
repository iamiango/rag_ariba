import os
import pickle
from typing import List, Tuple
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class BM25Indexer:
    """
    BM25索引器，用于关键词检索
    支持构建、保存、加载和搜索BM25索引
    """

    def __init__(self, collection_name: str, index_path: str = "./bm25_index"):
        """
        初始化BM25索引器
        :param collection_name: 集合名称 (如 ariba_sourcing)
        :param index_path: 索引存储路径
        """
        self.collection_name = collection_name
        self.index_path = index_path
        self.bm25 = None
        self.doc_ids = []
        self.tokenized_corpus = []

        # 确保索引目录存在
        os.makedirs(index_path, exist_ok=True)

        # 初始化NLTK资源（如果需要）
        self._init_nltk()

        # 加载英文停用词
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("警告: 英文停用词未下载，将不使用停用词过滤")
            self.stop_words = set()

    def _init_nltk(self):
        """初始化NLTK资源"""
        try:
            # 尝试使用punkt tokenizer
            word_tokenize("test")
        except LookupError:
            print("下载NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)

        try:
            # 尝试加载停用词
            stopwords.words('english')
        except LookupError:
            print("下载NLTK stopwords...")
            nltk.download('stopwords', quiet=True)

    def _tokenize(self, text: str) -> List[str]:
        """
        对文本进行分词和预处理
        :param text: 输入文本
        :return: 分词后的token列表
        """
        # 转小写
        text = text.lower()

        # 使用NLTK分词
        tokens = word_tokenize(text)

        # 过滤停用词和非字母数字token
        tokens = [
            token for token in tokens
            if token.isalnum() and token not in self.stop_words
        ]

        return tokens

    def build_index(self, documents: List[str], doc_ids: List[str]):
        """
        构建BM25索引
        :param documents: 文档列表
        :param doc_ids: 文档ID列表
        """
        print(f"\n构建BM25索引: {self.collection_name}")
        print(f"文档数量: {len(documents)}")

        self.doc_ids = doc_ids

        # 对所有文档进行分词
        print("正在分词...")
        self.tokenized_corpus = [self._tokenize(doc) for doc in documents]

        # 构建BM25索引
        print("正在构建BM25索引...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(f"✓ BM25索引构建完成")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        使用BM25搜索
        :param query: 查询文本
        :param top_k: 返回前k个结果
        :return: [(doc_id, score), ...] 按分数降序排列
        """
        if self.bm25 is None:
            raise ValueError("BM25索引未构建或加载，请先调用build_index()或load_index()")

        # 对查询进行分词
        tokenized_query = self._tokenize(query)

        # 获取BM25分数
        scores = self.bm25.get_scores(tokenized_query)

        # 获取top_k结果
        top_indices = scores.argsort()[-top_k:][::-1]

        # 返回 (doc_id, score) 列表
        results = [
            (self.doc_ids[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0  # 只返回分数大于0的结果
        ]

        return results

    def save_index(self):
        """保存BM25索引到磁盘"""
        if self.bm25 is None:
            raise ValueError("没有可保存的索引，请先调用build_index()")

        index_file = os.path.join(self.index_path, f"{self.collection_name}_bm25.pkl")

        index_data = {
            'bm25': self.bm25,
            'doc_ids': self.doc_ids,
            'tokenized_corpus': self.tokenized_corpus,
            'collection_name': self.collection_name
        }

        with open(index_file, 'wb') as f:
            pickle.dump(index_data, f)

        print(f"✓ BM25索引已保存到: {index_file}")

    def load_index(self):
        """从磁盘加载BM25索引"""
        index_file = os.path.join(self.index_path, f"{self.collection_name}_bm25.pkl")

        if not os.path.exists(index_file):
            raise FileNotFoundError(
                f"BM25索引文件不存在: {index_file}\n"
                f"请先运行 python -m backend.document_processor 构建索引"
            )

        with open(index_file, 'rb') as f:
            index_data = pickle.load(f)

        self.bm25 = index_data['bm25']
        self.doc_ids = index_data['doc_ids']
        self.tokenized_corpus = index_data['tokenized_corpus']

        print(f"✓ BM25索引已加载: {self.collection_name} ({len(self.doc_ids)} 文档)")

    def index_exists(self) -> bool:
        """检查索引文件是否存在"""
        index_file = os.path.join(self.index_path, f"{self.collection_name}_bm25.pkl")
        return os.path.exists(index_file)


# 测试代码
if __name__ == "__main__":
    # 示例：构建和测试BM25索引
    documents = [
        "SAP ERP enables users to create an RFQ with transaction codes me41 or me41n",
        "The purchase order can be created using transaction code ME21N",
        "Ariba Sourcing integration with SAP ERP for procurement processes"
    ]
    doc_ids = ["doc1", "doc2", "doc3"]

    # 构建索引
    indexer = BM25Indexer("test_collection")
    indexer.build_index(documents, doc_ids)

    # 搜索
    query = "create RFQ transaction code"
    results = indexer.search(query, top_k=3)

    print(f"\n查询: {query}")
    print("结果:")
    for doc_id, score in results:
        print(f"  {doc_id}: {score:.4f}")

    # 保存和加载
    indexer.save_index()

    indexer2 = BM25Indexer("test_collection")
    indexer2.load_index()
    results2 = indexer2.search(query, top_k=3)
    print("\n加载后的搜索结果:")
    for doc_id, score in results2:
        print(f"  {doc_id}: {score:.4f}")
