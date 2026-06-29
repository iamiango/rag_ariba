#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BM25混合检索设置脚本
自动安装依赖、下载NLTK数据、构建索引
"""
import subprocess
import sys
import os


def run_command(command, description):
    """运行命令并显示进度"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"命令: {command}")
    print()

    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"\n❌ 失败: {description}")
        return False
    print(f"\n✓ 完成: {description}")
    return True


def check_dependencies():
    """检查必要的依赖是否已安装"""
    print("\n" + "="*80)
    print("检查依赖")
    print("="*80)

    dependencies = {
        'rank_bm25': 'rank-bm25',
        'nltk': 'nltk',
    }

    missing = []
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"✗ {package} 未安装")
            missing.append(package)

    return missing


def install_dependencies(missing):
    """安装缺失的依赖"""
    if not missing:
        print("\n✓ 所有依赖已安装")
        return True

    print(f"\n需要安装: {', '.join(missing)}")
    packages = ' '.join(missing)

    return run_command(
        f"pip install {packages}",
        f"安装依赖: {packages}"
    )


def download_nltk_data():
    """下载NLTK数据"""
    print("\n" + "="*80)
    print("下载NLTK数据")
    print("="*80)

    try:
        import nltk

        # 检查是否已下载
        try:
            nltk.data.find('tokenizers/punkt')
            print("✓ punkt tokenizer 已存在")
        except LookupError:
            print("下载 punkt tokenizer...")
            nltk.download('punkt', quiet=False)
            print("✓ punkt tokenizer 下载完成")

        try:
            nltk.data.find('corpora/stopwords')
            print("✓ stopwords 已存在")
        except LookupError:
            print("下载 stopwords...")
            nltk.download('stopwords', quiet=False)
            print("✓ stopwords 下载完成")

        return True

    except Exception as e:
        print(f"❌ NLTK数据下载失败: {str(e)}")
        return False


def build_bm25_index():
    """构建BM25索引"""
    print("\n" + "="*80)
    print("构建BM25索引")
    print("="*80)
    print("这可能需要几分钟时间，请耐心等待...")
    print()

    # 检查ChromaDB是否存在
    if not os.path.exists("./chroma_db"):
        print("⚠️  警告: ChromaDB不存在")
        print("将同时进行文档嵌入和BM25索引构建")
        print("这可能需要较长时间（10-30分钟）")
        print()

    return run_command(
        "python -m backend.document_processor",
        "构建BM25索引"
    )


def verify_setup():
    """验证设置"""
    print("\n" + "="*80)
    print("验证设置")
    print("="*80)

    # 检查BM25索引文件
    index_files = [
        "./bm25_index/ariba_sourcing_bm25.pkl",
        "./bm25_index/ariba_integration_bm25.pkl"
    ]

    all_exist = True
    for index_file in index_files:
        if os.path.exists(index_file):
            size = os.path.getsize(index_file) / (1024 * 1024)  # MB
            print(f"✓ {index_file} ({size:.2f} MB)")
        else:
            print(f"✗ {index_file} 不存在")
            all_exist = False

    return all_exist


def run_tests():
    """运行测试"""
    print("\n" + "="*80)
    print("运行测试")
    print("="*80)
    print("是否运行测试脚本？(y/n): ", end='')

    try:
        response = input().strip().lower()
        if response == 'y':
            return run_command(
                "python test_bm25_hybrid.py",
                "运行BM25混合检索测试"
            )
    except KeyboardInterrupt:
        print("\n跳过测试")

    return True


def main():
    """主函数"""
    print("\n" + "="*80)
    print("BM25混合检索设置向导")
    print("="*80)
    print("\n此脚本将:")
    print("1. 检查并安装必要的依赖 (rank-bm25, nltk)")
    print("2. 下载NLTK数据 (punkt, stopwords)")
    print("3. 构建BM25索引 (sourcing, integration)")
    print("4. 验证设置")
    print("5. 可选: 运行测试")

    print("\n按Enter继续，Ctrl+C取消...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n\n已取消")
        return

    # 步骤1: 检查依赖
    missing = check_dependencies()

    # 步骤2: 安装依赖
    if missing:
        if not install_dependencies(missing):
            print("\n❌ 依赖安装失败，请手动安装:")
            print(f"   pip install {' '.join(missing)}")
            return

    # 步骤3: 下载NLTK数据
    if not download_nltk_data():
        print("\n❌ NLTK数据下载失败")
        return

    # 步骤4: 构建BM25索引
    if not build_bm25_index():
        print("\n❌ BM25索引构建失败")
        return

    # 步骤5: 验证设置
    if not verify_setup():
        print("\n⚠️  设置验证失败，某些索引文件缺失")
        return

    # 步骤6: 运行测试（可选）
    run_tests()

    # 完成
    print("\n" + "="*80)
    print("设置完成！")
    print("="*80)
    print("\n混合检索已启用，可以开始使用:")
    print("\n1. 交互式RAG查询:")
    print("   python app.py")
    print("\n2. 程序化使用:")
    print("   from backend import retrieval_processor")
    print("   results = retrieval_processor.retrieval_process(query, type_of_query, top_k)")
    print("\n3. 查看文档:")
    print("   cat BM25_HYBRID_RETRIEVAL.md")
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已取消")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
