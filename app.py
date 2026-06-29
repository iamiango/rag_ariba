import os
from backend import retrieval_processor, config, query_preprocessor
import dashscope
from dashscope import Generation

# Qwen API配置 - 使用dashscope
QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")

# 设置dashscope API key
if QWEN_API_KEY:
    dashscope.api_key = QWEN_API_KEY

def load_prompt_template(template_type="default"):
    """
    从prompts.md加载提示词模板
    :param template_type: 模板类型 (default, sourcing, integration)
    :return: 提示词模板字符串
    """
    try:
        # 读取prompts.md文件
        prompts_file = os.path.join(os.path.dirname(__file__), 'prompts.md')
        with open(prompts_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 解析模板
        # 使用 "## Template Name" 作为分隔符
        if template_type == "default":
            # 提取 "## Default Template" 到下一个 "---" 之间的内容
            start_marker = "## Default Template"
            end_marker = "\n---"
        elif template_type == "sourcing":
            start_marker = "## Sourcing Template"
            end_marker = "\n---"
        elif template_type == "integration":
            start_marker = "## Integration Template"
            end_marker = None  # 最后一个模板，读到文件末尾
        else:
            template_type = "default"
            start_marker = "## Default Template"
            end_marker = "\n---"

        # 查找模板内容
        start_idx = content.find(start_marker)
        if start_idx == -1:
            raise ValueError(f"未找到模板: {template_type}")

        # 跳过标题行
        start_idx = content.find('\n', start_idx) + 1

        # 查找结束位置
        if end_marker:
            end_idx = content.find(end_marker, start_idx)
            if end_idx == -1:
                end_idx = len(content)
        else:
            end_idx = len(content)

        # 提取并清理模板
        template = content[start_idx:end_idx].strip()

        return template

    except Exception as e:
        print(f"警告: 无法从prompts.md加载模板 ({str(e)})，使用默认模板")
        # 降级方案：使用简单的默认模板
        fallback_template = """You are a professional SAP Ariba assistant.

Please answer the question based on the provided documentation.

Provided Documentation:
{context}

User Question:
{question}

Please provide an answer in Chinese:"""
        return fallback_template

def format_sources(retrieved_docs):
    """
    格式化文档来源信息
    :param retrieved_docs: 检索到的文档列表（带元数据）
    :return: 格式化后的来源信息列表
    """
    sources = []
    for doc in retrieved_docs:
        # 处理新格式（字典）和旧格式（字符串）
        if isinstance(doc, dict):
            meta = doc.get('metadata', {})
            if meta and meta.get('document_title'):
                doc_title = meta['document_title']
                page = meta.get('page', 0) + 1  # 转换为1-based
                source_str = f"【{doc_title}】第{page}页"

                # 添加rerank信息（如果有）
                if 'rerank_score' in doc:
                    source_str += f" (相关性: {doc['rerank_score']:.4f})"
            else:
                source_str = "【来源未知】"
        else:
            # 向后兼容：旧格式只是字符串
            source_str = "【来源未知】"

        sources.append(source_str)

    return sources

def call_qwen_llm(prompt, temperature=None, max_tokens=None):
    """
    调用Qwen大语言模型获取回答
    :param prompt: 完整的提示词（包含上下文和问题）
    :param temperature: 温度参数，控制回答的随机性（None则使用配置文件默认值）
    :param max_tokens: 最大生成token数（None则使用配置文件默认值）
    :return: LLM生成的回答
    """
    if not QWEN_API_KEY:
        raise ValueError("请设置DASHSCOPE_API_KEY环境变量")

    # 使用配置文件的默认值
    if temperature is None:
        temperature = config.config.LLM_TEMPERATURE
    if max_tokens is None:
        max_tokens = config.config.LLM_MAX_TOKENS

    try:
        # 使用dashscope调用Qwen
        response = Generation.call(
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": "你是一个专业的Ariba系统助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            result_format='message'
        )

        if response.status_code == 200:
            answer = response.output.choices[0].message.content
            return answer
        else:
            raise Exception(f"API调用失败: {response.code} - {response.message}")

    except Exception as e:
        print(f"调用Qwen API时出错: {str(e)}")
        raise

def rag_query(query_text, type_of_query="sourcing", n_results=None, temperature=None):
    """
    RAG系统的主查询方法
    :param query_text: 用户问题
    :param type_of_query: 查询类型 (sourcing或integration)
    :param n_results: 检索文档数量
    :param temperature: LLM温度参数（None则使用配置文件默认值0.2）
    :return: 包含答案和相关文档的字典
    """
    print("="*60)
    print("RAG查询开始")
    print("="*60)

    # 步骤0: 查询预处理（纠错和关键词提取）
    # 可通过环境变量 SKIP_QUERY_PREPROCESSING=1 跳过预处理
    skip_preprocessing = os.getenv('SKIP_QUERY_PREPROCESSING', '0') == '1'

    if skip_preprocessing:
        print("\n[步骤0] 跳过查询预处理...")
        optimized_query = query_text
        preprocess_result = {
            "original_query": query_text,
            "optimized_query": query_text,
            "corrections": [],
            "keywords": [],
            "changed": False
        }
    else:
        print("\n[步骤0] 查询预处理...")
        preprocessor = query_preprocessor.get_preprocessor()
        preprocess_result = preprocessor.query_corrector(query_text, verbose=True)

        # 使用优化后的查询
        optimized_query = preprocess_result['optimized_query']

        if preprocess_result['changed']:
            print(f"\n查询已优化: {query_text} -> {optimized_query}")

    # 步骤1: 检索相关文档
    print("\n[步骤1] 检索相关文档...")
    if n_results is None:
        n_results = config.config.MAX_RESULTS

    retrieved_chunks = retrieval_processor.retrieval_process(
        query=optimized_query,
        type_of_query=type_of_query,
        top_k=n_results
    )

    if not retrieved_chunks:
        return {
            "answer": "抱歉，没有找到相关文档来回答您的问题。",
            "retrieved_docs": [],
            "query": query_text,
            "type": type_of_query
        }

    # 步骤2: 构建上下文
    print("\n[步骤2] 构建上下文...")
    # 处理新格式（字典）和旧格式（字符串）
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        if isinstance(chunk, dict):
            content = chunk.get('content', '')
        else:
            content = chunk
        context_parts.append(f"文档片段 {i+1}:\n{content}")
    context = "\n\n".join(context_parts)

    # 调试：打印context长度和预览
    print(f"构建的上下文总长度: {len(context)} 字符")
    print(f"上下文预览（前500字符）:\n{context[:500]}...")
    if len(context) < 100:
        print(f"警告: 上下文内容过短，可能存在问题！完整内容:\n{context}")

    # 步骤3: 加载提示词模板
    print("\n[步骤3] 加载提示词模板...")
    template = load_prompt_template(type_of_query)
    prompt = template.format(context=context, question=query_text)

    # 调试：打印完整prompt（可选，用于排查）
    print(f"\n完整Prompt长度: {len(prompt)} 字符")
    print(f"Prompt预览（前800字符）:\n{prompt[:800]}...")

    # 步骤4: 调用Qwen获取答案
    print("\n[步骤4] 调用Qwen生成答案...")
    answer = call_qwen_llm(prompt, temperature=temperature)

    # 调试：打印LLM返回的原始答案
    print(f"\n[LLM返回] 答案长度: {len(answer)} 字符")
    print(f"[LLM返回] 完整答案:\n{answer}")
    print(f"[LLM返回] 答案类型: {type(answer)}")

    print("\n" + "="*60)
    print("RAG查询完成")
    print("="*60)

    # 格式化来源信息
    sources = format_sources(retrieved_chunks)

    return {
        "answer": answer,
        "retrieved_docs": retrieved_chunks,
        "sources": sources,  # 新增
        "query": query_text,
        "optimized_query": optimized_query,
        "type": type_of_query,
        "context": context,
        "preprocessing": preprocess_result
    }

def interactive_mode():
    """
    交互式问答模式
    """
    print("\n" + "="*60)
    print("欢迎使用Ariba RAG问答系统")
    print("="*60)
    print("\n可用命令:")
    print("  - 输入问题进行查询")
    print("  - 输入 'sourcing' 或 'integration' 切换查询类型")
    print("  - 输入 'quit' 或 'exit' 退出")
    print("\n" + "="*60 + "\n")

    current_type = "sourcing"

    while True:
        print(f"\n当前查询类型: {current_type}")
        user_input = input("\n请输入您的问题: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n感谢使用，再见！")
            break

        if user_input.lower() == 'sourcing':
            current_type = 'sourcing'
            print("已切换到 Sourcing 模式")
            continue

        if user_input.lower() == 'integration':
            current_type = 'integration'
            print("已切换到 Integration 模式")
            continue

        try:
            # 执行RAG查询
            result = rag_query(
                query_text=user_input,
                type_of_query=current_type
            )

            # 显示结果
            print("\n" + "-"*60)
            print("回答:")
            print("-"*60)
            print(result['answer'])

            print("\n" + "-"*60)
            print("参考文档来源:")
            print("-"*60)
            for i, source in enumerate(result.get('sources', []), 1):
                print(f"{i}. {source}")

            print("\n" + "-"*60)
            print(f"共参考 {len(result['retrieved_docs'])} 个文档片段")
            print("-"*60)

        except Exception as e:
            print(f"\n错误: {str(e)}")
            print("请检查配置和网络连接")

if __name__ == "__main__":
    # 运行交互式模式
    interactive_mode()
