import os
from dashscope import Generation
import dashscope

# 设置dashscope API key
QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
if QWEN_API_KEY:
    dashscope.api_key = QWEN_API_KEY

def translate_query_to_english(chinese_query: str) -> str:
    """
    将中文查询翻译为英文，保留SAP Ariba技术术语
    :param chinese_query: 中文查询文本
    :return: 英文查询文本
    """
    if not QWEN_API_KEY:
        raise ValueError("请设置DASHSCOPE_API_KEY环境变量")

    # 检测是否已经是英文查询（简单启发式：如果大部分字符是ASCII，则认为是英文）
    ascii_ratio = sum(1 for c in chinese_query if ord(c) < 128) / len(chinese_query) if chinese_query else 0
    if ascii_ratio > 0.7:
        print(f"检测到英文查询，跳过翻译: {chinese_query}")
        return chinese_query

    prompt = f"""Translate this Chinese question to English. Keep SAP Ariba technical terms unchanged.

Requirements:
1. Preserve technical terms like "Sourcing Project", "RFx", "Supplier", "Integration", "API", etc.
2. Keep the question format and intent
3. Use professional business English
4. Only output the English translation, no explanations

Chinese: {chinese_query}
English:"""

    try:
        response = Generation.call(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # 低温度确保翻译稳定
            result_format='message'
        )

        if response.status_code == 200:
            english_query = response.output.choices[0].message.content.strip()
            # 移除可能的引号
            english_query = english_query.strip('"').strip("'")
            print(f"查询翻译: {chinese_query} -> {english_query}")
            return english_query
        else:
            print(f"翻译API调用失败: {response.code} - {response.message}")
            print("使用原始查询")
            return chinese_query

    except Exception as e:
        print(f"查询翻译时出错: {str(e)}")
        print("使用原始查询")
        return chinese_query

def detect_language(query: str) -> str:
    """
    检测查询语言
    :param query: 查询文本
    :return: 'zh' 或 'en'
    """
    # 统计中文字符和英文字母
    chinese_chars = sum(1 for c in query if '\u4e00' <= c <= '\u9fff')
    english_chars = sum(1 for c in query if c.isalpha() and ord(c) < 128)
    total_chars = len(query)

    if total_chars == 0:
        return 'en'

    # 如果有中文字符，且中文字符占比超过20%，则认为是中文查询
    chinese_ratio = chinese_chars / total_chars
    if chinese_ratio > 0.2:
        return 'zh'

    # 如果没有中文字符，但有英文字母，则认为是英文查询
    if english_chars > 0 and chinese_chars == 0:
        return 'en'

    # 默认返回中文（保守策略）
    return 'zh'
