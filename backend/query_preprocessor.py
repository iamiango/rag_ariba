import re
from typing import Dict, List, Tuple

class QueryPreprocessor:
    """
    查询预处理器：对用户输入的查询进行语法/拼写纠错和关键词提取
    """

    def __init__(self):
        """初始化预处理器"""
        self.spell_checker = None
        self.keyword_extractor = None
        self._init_tools()

    def _init_tools(self):
        """延迟初始化纠错和关键词提取工具"""
        try:
            # 尝试导入语法纠错工具
            from language_tool_python import LanguageTool
            self.spell_checker = LanguageTool('en-US')
            print("语法纠错工具初始化成功")
        except ImportError:
            print("警告: language_tool_python未安装，拼写纠错功能将被禁用")
            print("安装命令: pip install language-tool-python")
            self.spell_checker = None

        try:
            # 尝试导入中文分词工具
            import jieba
            import jieba.analyse
            self.keyword_extractor = jieba.analyse
            print("关键词提取工具初始化成功")
        except ImportError:
            print("警告: jieba未安装，关键词提取功能将被禁用")
            print("安装命令: pip install jieba")
            self.keyword_extractor = None

    def _correct_spelling(self, text: str) -> Tuple[str, List[str]]:
        """
        拼写和语法纠错
        :param text: 原始文本
        :return: (纠错后的文本, 纠错列表)
        """
        if not self.spell_checker:
            return text, []

        try:
            # 检测并纠正拼写错误
            matches = self.spell_checker.check(text)
            corrections = []

            # 应用纠错（从后往前，避免位置偏移）
            corrected_text = text
            for match in reversed(matches):
                if match.replacements:
                    original = corrected_text[match.offset:match.offset + match.errorLength]
                    replacement = match.replacements[0]
                    corrected_text = (
                        corrected_text[:match.offset] +
                        replacement +
                        corrected_text[match.offset + match.errorLength:]
                    )
                    corrections.append(f"{original} -> {replacement}")

            return corrected_text, corrections
        except Exception as e:
            print(f"拼写纠错时出错: {str(e)}")
            return text, []

    def _extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        提取关键词
        :param text: 输入文本
        :param top_k: 提取关键词数量
        :return: 关键词列表
        """
        if not self.keyword_extractor:
            # 如果jieba未安装，使用简单的基于频率的方法
            return self._simple_keyword_extraction(text, top_k)

        try:
            # 使用TF-IDF提取关键词
            keywords_tfidf = self.keyword_extractor.extract_tags(
                text,
                topK=top_k,
                withWeight=False
            )

            # 使用TextRank提取关键词
            keywords_textrank = self.keyword_extractor.textrank(
                text,
                topK=top_k,
                withWeight=False
            )

            # 合并两种方法的结果，去重
            keywords = list(set(keywords_tfidf + keywords_textrank))
            return keywords[:top_k]

        except Exception as e:
            print(f"关键词提取时出错: {str(e)}")
            return self._simple_keyword_extraction(text, top_k)

    def _simple_keyword_extraction(self, text: str, top_k: int = 10) -> List[str]:
        """
        简单的关键词提取（基于词频，作为备用方案）
        :param text: 输入文本
        :param top_k: 提取关键词数量
        :return: 关键词列表
        """
        # 移除标点符号
        text_clean = re.sub(r'[^\w\s]', ' ', text)

        # 分词（简单按空格分割）
        words = text_clean.lower().split()

        # 过滤停用词（简单版本）
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                      'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                      '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
                      '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有'}

        words = [w for w in words if w not in stop_words and len(w) > 1]

        # 统计词频
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # 按频率排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        return [word for word, freq in sorted_words[:top_k]]

    def _build_optimized_query(self, original_query: str, corrected_query: str,
                               keywords: List[str]) -> str:
        """
        构建优化后的查询
        :param original_query: 原始查询
        :param corrected_query: 纠错后的查询
        :param keywords: 提取的关键词
        :return: 优化后的查询
        """
        # 如果纠错后的查询与原始查询差异不大，使用纠错后的查询
        # 否则，结合关键词增强查询

        if corrected_query != original_query:
            # 有纠错，使用纠错后的查询
            optimized = corrected_query
        else:
            # 无纠错，使用原始查询
            optimized = original_query

        # 如果提取到了关键词，可以选择性地增强查询
        # 这里采用保守策略：只使用纠错后的查询，不额外添加关键词
        # 因为添加关键词可能改变原始语义

        return optimized

    def query_corrector(self, query: str, verbose: bool = True) -> Dict[str, any]:
        """
        查询纠错和优化的主方法
        :param query: 原始查询文本
        :param verbose: 是否打印详细信息
        :return: 包含优化后查询和元数据的字典
        """
        if not query or not query.strip():
            return {
                "original_query": query,
                "optimized_query": query,
                "corrections": [],
                "keywords": [],
                "changed": False
            }

        query = query.strip()

        if verbose:
            print("\n" + "="*60)
            print("[查询预处理] 开始处理查询")
            print("="*60)
            print(f"原始查询: {query}")

        # 步骤1: 拼写和语法纠错
        corrected_query, corrections = self._correct_spelling(query)

        if verbose and corrections:
            print(f"\n[纠错] 发现 {len(corrections)} 处需要纠正:")
            for correction in corrections:
                print(f"  - {correction}")
            print(f"纠错后查询: {corrected_query}")
        elif verbose:
            print("\n[纠错] 未发现拼写或语法错误")

        # 步骤2: 提取关键词
        keywords = self._extract_keywords(corrected_query)

        if verbose and keywords:
            print(f"\n[关键词提取] 提取到 {len(keywords)} 个关键词:")
            print(f"  {', '.join(keywords)}")

        # 步骤3: 构建优化后的查询
        optimized_query = self._build_optimized_query(query, corrected_query, keywords)

        if verbose:
            print(f"\n[优化结果] 最终查询: {optimized_query}")
            print("="*60 + "\n")

        return {
            "original_query": query,
            "optimized_query": optimized_query,
            "corrections": corrections,
            "keywords": keywords,
            "changed": query != optimized_query
        }


# 全局单例实例
_preprocessor_instance = None

def get_preprocessor() -> QueryPreprocessor:
    """获取查询预处理器的单例实例"""
    global _preprocessor_instance
    if _preprocessor_instance is None:
        _preprocessor_instance = QueryPreprocessor()
    return _preprocessor_instance


# 测试代码
if __name__ == "__main__":
    preprocessor = QueryPreprocessor()

    # 测试用例
    test_queries = [
        "How to creat a sourcing projet?",  # 拼写错误
        "What is the proces for supplier managment?",  # 拼写错误
        "如何创建采购项目？",  # 中文查询
        "Ariba integration API documentation",  # 正常查询
    ]

    print("\n" + "="*60)
    print("查询预处理器测试")
    print("="*60)

    for query in test_queries:
        result = preprocessor.query_corrector(query, verbose=True)
        print(f"\n结果摘要:")
        print(f"  原始: {result['original_query']}")
        print(f"  优化: {result['optimized_query']}")
        print(f"  是否改变: {result['changed']}")
        print("\n" + "-"*60)
