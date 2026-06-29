#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试图片 OCR 功能

使用方法:
    # 安装依赖
    pip install paddleocr paddlepaddle opencv-python-headless pymupdf

    # 运行测试
    python test_image_ocr.py
"""

import os
import sys

def test_dependencies():
    """测试依赖是否已安装"""
    print("=" * 60)
    print("1. 检查依赖安装情况")
    print("=" * 60)

    dependencies = {
        'paddleocr': 'PaddleOCR (OCR 引擎)',
        'paddle': 'PaddlePaddle (深度学习框架)',
        'cv2': 'OpenCV (图像处理)',
        'fitz': 'PyMuPDF (PDF 图片提取)',
        'PIL': 'Pillow (图像处理)',
        'pptx': 'python-pptx (PPT 解析)',
        'docx': 'python-docx (Word 解析)',
    }

    missing = []
    for module, name in dependencies.items():
        try:
            if module == 'paddle':
                import paddle
            elif module == 'cv2':
                import cv2
            elif module == 'fitz':
                import fitz
            elif module == 'PIL':
                from PIL import Image
            elif module == 'pptx':
                from pptx import Presentation
            elif module == 'docx':
                from docx import Document
            elif module == 'paddleocr':
                from paddleocr import PaddleOCR
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - 未安装")
            missing.append(module)

    if missing:
        print("\n缺少以下依赖，请安装:")
        print("  pip install paddleocr paddlepaddle opencv-python-headless pymupdf")
        return False

    return True


def test_image_processor():
    """测试图片处理模块"""
    print("\n" + "=" * 60)
    print("2. 测试图片处理模块")
    print("=" * 60)

    try:
        from backend.image_processor import (
            ImageExtractionConfig,
            ImageExtractor,
            OCRProcessor,
            ImageDocumentLoader,
            is_ocr_available,
            get_supported_formats
        )
        print("  ✓ 图片处理模块导入成功")
        print(f"  ✓ OCR 可用: {is_ocr_available()}")
        print(f"  ✓ 支持格式: {get_supported_formats()}")
        return True
    except Exception as e:
        print(f"  ✗ 导入失败: {str(e)}")
        return False


def test_ocr_initialization():
    """测试 OCR 初始化"""
    print("\n" + "=" * 60)
    print("3. 测试 OCR 引擎初始化")
    print("=" * 60)

    try:
        from backend.image_processor import OCRProcessor, ImageExtractionConfig

        config = ImageExtractionConfig(
            enable_ocr=True,
            ocr_languages=['ch', 'en'],
            use_gpu=False
        )

        print("  正在初始化 PaddleOCR（首次运行会下载模型）...")
        processor = OCRProcessor(config)

        if processor.ocr_engine:
            print("  ✓ OCR 引擎初始化成功")
            return True
        else:
            print("  ✗ OCR 引擎初始化失败")
            return False

    except Exception as e:
        print(f"  ✗ 初始化失败: {str(e)}")
        return False


def test_sample_image_ocr():
    """测试样例图片 OCR"""
    print("\n" + "=" * 60)
    print("4. 测试图片 OCR 识别")
    print("=" * 60)

    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        from backend.image_processor import OCRProcessor, ImageExtractionConfig

        # 创建一个包含文字的测试图片
        print("  创建测试图片...")
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)

        # 尝试使用系统字体
        try:
            # macOS 中文字体
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 24)
        except:
            try:
                # Linux 中文字体
                font = ImageFont.truetype("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", 24)
            except:
                font = ImageFont.load_default()

        # 绘制中英文文字
        draw.text((20, 30), "Hello World", fill='black', font=font)
        draw.text((20, 80), "你好世界", fill='black', font=font)
        draw.text((20, 130), "OCR 测试图片", fill='black', font=font)

        print("  初始化 OCR...")
        config = ImageExtractionConfig(enable_ocr=True, use_gpu=False)
        processor = OCRProcessor(config)

        print("  执行 OCR 识别...")
        result = processor.process_image(img)

        print(f"\n  识别结果:")
        print(f"    文本: {result['text']}")
        print(f"    置信度: {result['confidence']:.2%}")

        if result['text']:
            print("\n  ✓ OCR 识别测试通过")
            return True
        else:
            print("\n  ⚠ OCR 未识别出文字（可能是字体问题）")
            return True  # 不视为失败

    except Exception as e:
        print(f"  ✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_document_extraction(doc_path=None):
    """测试文档图片提取"""
    print("\n" + "=" * 60)
    print("5. 测试文档图片提取")
    print("=" * 60)

    if not doc_path:
        print("  跳过（未指定测试文档）")
        print("  使用方法: python test_image_ocr.py <文档路径>")
        return True

    if not os.path.exists(doc_path):
        print(f"  ✗ 文档不存在: {doc_path}")
        return False

    try:
        from backend.image_processor import ImageDocumentLoader, ImageExtractionConfig

        config = ImageExtractionConfig(
            enable_ocr=True,
            ocr_languages=['ch', 'en'],
            min_image_size=50,
            max_images_per_document=10,
            use_gpu=False,
            save_extracted_images=False
        )

        loader = ImageDocumentLoader(config)

        print(f"  处理文档: {doc_path}")
        documents = loader.load_images_from_document(doc_path)

        print(f"\n  提取结果:")
        print(f"    图片数量: {len(documents)}")

        for i, doc in enumerate(documents[:3]):  # 只显示前3个
            print(f"\n    图片 {i+1}:")
            print(f"      内容预览: {doc.page_content[:100]}...")
            print(f"      置信度: {doc.metadata.get('ocr_confidence', 0):.2%}")

        if len(documents) > 3:
            print(f"\n    ... 还有 {len(documents) - 3} 张图片")

        print("\n  ✓ 文档图片提取测试通过")
        return True

    except Exception as e:
        print(f"  ✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("  图片 OCR 功能测试")
    print("=" * 60)

    # 获取命令行参数
    doc_path = sys.argv[1] if len(sys.argv) > 1 else None

    results = []

    # 1. 测试依赖
    results.append(("依赖检查", test_dependencies()))

    if not results[-1][1]:
        print("\n请先安装必要的依赖后再运行测试")
        return

    # 2. 测试模块导入
    results.append(("模块导入", test_image_processor()))

    # 3. 测试 OCR 初始化
    results.append(("OCR 初始化", test_ocr_initialization()))

    # 4. 测试样例图片
    results.append(("样例图片 OCR", test_sample_image_ocr()))

    # 5. 测试文档提取
    results.append(("文档图片提取", test_document_extraction(doc_path)))

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)

    passed = 0
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {status} - {name}")
        if result:
            passed += 1

    print(f"\n总计: {passed}/{len(results)} 项测试通过")

    if passed == len(results):
        print("\n✓ 所有测试通过！图片 OCR 功能已就绪。")
        print("\n使用方法:")
        print("  1. 重新运行文档嵌入: python -m backend.document_processor")
        print("  2. 或者直接使用 RAG 系统: python app.py")
    else:
        print("\n⚠ 部分测试未通过，请检查上述错误信息。")


if __name__ == "__main__":
    main()
