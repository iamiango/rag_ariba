"""
快速测试Phase 1优化后的RAG系统

测试查询翻译和检索功能
"""

from app import rag_query
import os

# 确保API key已设置
if not os.getenv("DASHSCOPE_API_KEY"):
    print("错误: 请设置DASHSCOPE_API_KEY环境变量")
    exit(1)

print("="*70)
print("Phase 1 优化后的RAG系统测试")
print("="*70)

# 测试1: 中文查询 (应该触发翻译)
print("\n[测试1] 中文查询 - 应该触发查询翻译")
print("-"*70)
test_query_1 = "如何创建采购项目？"
print(f"查询: {test_query_1}\n")

try:
    result = rag_query(
        query_text=test_query_1,
        type_of_query="sourcing",
        n_results=3
    )

    print("\n回答:")
    print(result['answer'])
    print(f"\n检索到 {len(result['retrieved_docs'])} 个文档片段")

except Exception as e:
    print(f"错误: {str(e)}")

print("\n" + "="*70)
print("测试完成")
print("="*70)
print("\n如果看到查询翻译信息，说明Phase 1优化正常工作！")
print("\n下一步: 运行完整的RAGAS评估")
print("  python evaluate_rag.py --mode all --n-samples-sourcing 20 --n-samples-integration 20")
