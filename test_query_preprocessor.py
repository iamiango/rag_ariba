"""
测试查询预处理器功能
"""
from backend.query_preprocessor import QueryPreprocessor

def test_query_preprocessor():
    """测试查询预处理器的各种功能"""

    print("\n" + "="*80)
    print("查询预处理器功能测试")
    print("="*80)

    preprocessor = QueryPreprocessor()

    # 测试用例
    test_cases = [
        {
            "query": "How to creat a sourcing projet in Ariba?",
            "description": "英文拼写错误测试"
        },
        {
            "query": "What is the proces for supplier managment?",
            "description": "多个拼写错误测试"
        },
        {
            "query": "如何在Ariba系统中创建采购项目和管理供应商？",
            "description": "中文查询关键词提取测试"
        },
        {
            "query": "Ariba integration API documentation and setup guide",
            "description": "正常英文查询测试"
        },
        {
            "query": "采购寻源流程配置步骤",
            "description": "中文专业术语测试"
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"测试用例 {i}: {test_case['description']}")
        print(f"{'='*80}")

        result = preprocessor.query_corrector(test_case['query'], verbose=True)

        print(f"\n{'─'*80}")
        print("测试结果摘要:")
        print(f"{'─'*80}")
        print(f"原始查询:     {result['original_query']}")
        print(f"优化后查询:   {result['optimized_query']}")
        print(f"是否改变:     {result['changed']}")

        if result['corrections']:
            print(f"\n纠错详情:")
            for correction in result['corrections']:
                print(f"  • {correction}")

        if result['keywords']:
            print(f"\n提取的关键词: {', '.join(result['keywords'])}")

        print(f"{'─'*80}")

if __name__ == "__main__":
    print("\n注意: 首次运行需要安装依赖:")
    print("  pip install language-tool-python jieba")
    print("\n如果未安装这些依赖，预处理器将使用简化版本功能。\n")

    try:
        test_query_preprocessor()
        print("\n" + "="*80)
        print("测试完成!")
        print("="*80 + "\n")
    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("\n请确保已安装所需依赖:")
        print("  pip install language-tool-python jieba")
