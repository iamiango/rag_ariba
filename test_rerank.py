#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rerank 功能测试脚本

用于测试和验证 rerank 功能是否正常工作
"""

import os
from backend import config, retrieval_processor

def test_rerank_config():
    """测试 rerank 配置是否正确加载"""
    print("="*60)
    print("测试 1: Rerank 配置")
    print("="*60)

    print(f"RERANK_ENABLED: {config.config.RERANK_ENABLED}")
    print(f"RERANK_MODEL_PATH: {config.config.RERANK_MODEL_PATH}")
    print(f"RERANK_TOP_K_MULTIPLIER: {config.config.RERANK_TOP_K_MULTIPLIER}")
    print(f"RERANK_BATCH_SIZE: {config.config.RERANK_BATCH_SIZE}")

    assert hasattr(config.config, 'RERANK_ENABLED'), "缺少 RERANK_ENABLED 配置"
    assert hasattr(config.config, 'RERANK_MODEL_PATH'), "缺少 RERANK_MODEL_PATH 配置"
    assert hasattr(config.config, 'RERANK_TOP_K_MULTIPLIER'), "缺少 RERANK_TOP_K_MULTIPLIER 配置"

    print("\n✓ 配置加载成功\n")

def test_rerank_import():
    """测试 FlagEmbedding 是否正确导入"""
    print("="*60)
    print("测试 2: FlagEmbedding 导入")
    print("="*60)

    try:
        from FlagEmbedding import FlagReranker
        print("✓ FlagEmbedding 导入成功")
        print(f"FlagReranker 类: {FlagReranker}")
        print()
        return True
    except ImportError as e:
        print(f"✗ FlagEmbedding 导入失败: {e}")
        print("请运行: pip install FlagEmbedding>=1.2.0")
        print()
        return False

def test_rerank_functions():
    """测试 rerank 相关函数是否存在"""
    print("="*60)
    print("测试 3: Rerank 函数")
    print("="*60)

    assert hasattr(retrieval_processor, 'get_reranker'), "缺少 get_reranker 函数"
    assert hasattr(retrieval_processor, 'rerank_documents'), "缺少 rerank_documents 函数"
    assert hasattr(retrieval_processor, 'retrieval_process'), "缺少 retrieval_process 函数"

    print("✓ get_reranker 函数存在")
    print("✓ rerank_documents 函数存在")
    print("✓ retrieval_process 函数存在")
    print()

def test_rerank_model_loading():
    """测试 reranker 模型加载（可选）"""
    print("="*60)
    print("测试 4: Reranker 模型加载（可选）")
    print("="*60)

    if not config.config.RERANK_ENABLED:
        print("⊘ Rerank 已禁用，跳过模型加载测试")
        print()
        return

    print("尝试加载 reranker 模型...")
    print("注意: 首次加载需要下载模型（约 1.3GB），可能需要几分钟")
    print()

    try:
        reranker = retrieval_processor.get_reranker()
        if reranker is not None:
            print("✓ Reranker 模型加载成功")
            print(f"模型类型: {type(reranker)}")
        else:
            print("⊘ Reranker 模型加载失败（可能是依赖未安装）")
    except Exception as e:
        print(f"✗ Reranker 模型加载出错: {e}")
    print()

def test_rerank_documents_function():
    """测试 rerank_documents 函数"""
    print("="*60)
    print("测试 5: rerank_documents 函数")
    print("="*60)

    # 测试数据
    query = "如何创建采购项目"
    documents = [
        "采购项目创建步骤：首先登录系统，然后点击新建项目按钮。",
        "系统配置文档：本文档介绍系统的基本配置方法。",
        "Ariba Sourcing 提供了完整的采购项目管理功能。",
        "用户手册：如何使用采购模块创建新的采购项目。",
        "技术文档：API 接口说明和集成指南。"
    ]
    top_k = 3

    print(f"查询: {query}")
    print(f"文档数量: {len(documents)}")
    print(f"期望返回: {top_k} 个文档")
    print()

    try:
        # 临时启用 rerank 进行测试
        original_enabled = config.config.RERANK_ENABLED
        config.config.RERANK_ENABLED = True

        result = retrieval_processor.rerank_documents(query, documents, top_k)

        # 恢复原始设置
        config.config.RERANK_ENABLED = original_enabled

        print(f"返回文档数量: {len(result)}")

        if len(result) == top_k:
            print("✓ 返回文档数量正确")
        else:
            print(f"⚠ 返回文档数量不符合预期（期望 {top_k}，实际 {len(result)}）")

        print("\n重排序后的文档:")
        for i, doc in enumerate(result):
            print(f"  {i+1}. {doc[:50]}...")

    except Exception as e:
        print(f"✗ rerank_documents 函数测试失败: {e}")
    print()

def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("Rerank 功能测试")
    print("="*60 + "\n")

    try:
        # 基础测试
        test_rerank_config()
        flag_embedding_available = test_rerank_import()
        test_rerank_functions()

        # 高级测试（需要 FlagEmbedding）
        if flag_embedding_available:
            test_rerank_model_loading()
            test_rerank_documents_function()
        else:
            print("="*60)
            print("跳过高级测试（FlagEmbedding 未安装）")
            print("="*60)
            print()

        print("="*60)
        print("测试完成")
        print("="*60)
        print("\n提示:")
        print("- 如果所有测试通过，rerank 功能已正确集成")
        print("- 如果模型加载失败，请检查网络连接和磁盘空间")
        print("- 首次使用需要下载模型（约 1.3GB）")
        print()

    except Exception as e:
        print(f"\n✗ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
