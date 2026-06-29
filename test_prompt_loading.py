#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试从prompts.md加载模板
"""

from app import load_prompt_template

def test_template_loading():
    """测试模板加载功能"""
    print("="*80)
    print("测试从prompts.md加载Prompt模板")
    print("="*80)

    templates = ["default", "sourcing", "integration"]

    for template_type in templates:
        print(f"\n{'='*80}")
        print(f"加载 {template_type.upper()} 模板")
        print(f"{'='*80}")

        try:
            template = load_prompt_template(template_type)

            print(f"\n✓ 成功加载 {template_type} 模板")
            print(f"模板长度: {len(template)} 字符")
            print(f"\n模板预览（前500字符）:")
            print("-"*80)
            print(template[:500])
            print("-"*80)

            # 检查必需的占位符
            if "{context}" in template and "{question}" in template:
                print(f"✓ 模板包含必需的占位符: {{context}} 和 {{question}}")
            else:
                print(f"❌ 警告: 模板缺少必需的占位符")
                if "{context}" not in template:
                    print("  - 缺少 {context}")
                if "{question}" not in template:
                    print("  - 缺少 {question}")

            # 测试格式化
            test_context = "这是测试文档内容"
            test_question = "这是测试问题"
            formatted = template.format(context=test_context, question=test_question)
            print(f"✓ 模板格式化测试通过")

        except Exception as e:
            print(f"❌ 加载 {template_type} 模板失败: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("测试完成")
    print(f"{'='*80}")

if __name__ == "__main__":
    test_template_loading()
