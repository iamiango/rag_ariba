# SAP Ariba RAG System

基于 RAG (Retrieval-Augmented Generation) 的 SAP Ariba 文档智能问答系统。

## 功能特点

- **多格式文档支持**: PDF, Word, PowerPoint, Excel, CSV, Markdown, XML, HTML
- **混合检索策略**: 向量检索 + BM25 关键词检索 + Rerank 重排序
- **图片 OCR 支持**: 提取 PDF/PPT/Word 中嵌入图片的文字 (PaddleOCR)
- **查询预处理**: 拼写纠错、语法检查、关键词提取
- **RAGAS 评估框架**: 全面评估 RAG 系统性能
- **Streamlit Web 界面**: 友好的交互式问答界面

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户查询                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      查询预处理                                  │
│              (拼写纠错 + 关键词提取)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       混合检索                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  向量检索     │  │  BM25检索    │  │   Rerank     │          │
│  │ (BGE-large)  │  │  (关键词)    │  │ (BGE-reranker)│          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLM 生成答案                                │
│                      (Qwen API)                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 使用的模型

| 模型 | 用途 | 说明 |
|------|------|------|
| BAAI/bge-large-en-v1.5 | 文本嵌入 | 本地运行，无需网络 |
| BAAI/bge-reranker-large | 文档重排序 | 本地运行，提升检索精度 |
| Qwen (通义千问) | 答案生成 | 通过 DashScope API 调用 |
| PaddleOCR | 图片文字识别 | 支持中英文 OCR |

## 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/iamiango/rag_ariba.git
cd rag_ariba

# 创建虚拟环境
python -m venv ariba_env
source ariba_env/bin/activate  # Linux/Mac
# ariba_env\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载模型

```bash
# 下载 embedding 模型 (约1.3GB)
# 将 BAAI/bge-large-en-v1.5 下载到 bge-large-en-v1.5/ 目录

# Reranker 模型会在首次使用时自动下载到 ~/.cache/huggingface/
```

### 3. 配置

编辑 `backend/config.py`，设置文档路径：

```python
FILE_PATH_SOURCING = "/path/to/your/sourcing/documents"
FILE_PATH_INTEGRATION = "/path/to/your/integration/documents"
```

设置环境变量：

```bash
export DASHSCOPE_API_KEY="your-dashscope-api-key"
```

### 4. 文档嵌入

首次运行需要处理文档并创建向量索引：

```bash
# 嵌入文档
python -m backend.document_processor

# 构建 BM25 索引
python setup_bm25.py
```

### 5. 运行系统

**命令行模式：**

```bash
python app.py
```

**Web 界面：**

```bash
streamlit run streamlit_app.py
# 访问 http://localhost:8501
```

## 项目结构

```
rag_ariba/
├── backend/
│   ├── config.py              # 配置文件
│   ├── document_loader.py     # 多格式文档加载
│   ├── document_processor.py  # 文档处理主模块
│   ├── indexing_processor.py  # 文档索引处理
│   ├── retrieval_processor.py # 检索处理（向量+BM25+Rerank）
│   ├── bm25_indexer.py        # BM25 索引构建
│   ├── image_processor.py     # 图片 OCR 处理
│   ├── query_preprocessor.py  # 查询预处理
│   ├── ragas_evaluator.py     # RAGAS 评估模块
│   └── ragas_dataset_generator.py  # 评估数据集生成
├── app.py                     # RAG 系统主入口
├── streamlit_app.py           # Streamlit Web 界面
├── prompts.md                 # 提示词模板
├── requirements.txt           # Python 依赖
├── evaluation_dataset_*.json  # 评估数据集
└── test_*.py                  # 测试脚本
```

## 配置参数

主要配置项 (`backend/config.py`)：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| CHUNK_SIZE | 1500 | 文本块大小 |
| CHUNK_OVERLAP | 300 | 文本块重叠 |
| MAX_RESULTS | 5 | 返回文档数量 |
| HYBRID_RETRIEVAL | True | 启用混合检索 |
| RERANK_ENABLED | True | 启用 Rerank |
| RERANK_TOP_K_MULTIPLIER | 5 | Rerank 检索倍数 |
| LLM_TEMPERATURE | 0.2 | LLM 温度参数 |

## 评估

系统使用 RAGAS 框架进行评估，支持以下指标：

- **Faithfulness**: 答案忠实度
- **Answer Relevancy**: 答案相关性
- **Context Precision**: 上下文精确度
- **Context Recall**: 上下文召回率
- **Answer Correctness**: 答案正确性

运行评估：

```bash
python run_comprehensive_evaluation.py
```

## API 使用示例

```python
from app import rag_query

# 查询 Sourcing 文档
result = rag_query(
    query_text="What are the main event types in SAP Ariba Sourcing?",
    type_of_query="sourcing",
    n_results=5,
    temperature=0.2
)

print(result['answer'])
print(f"检索到 {len(result['retrieved_docs'])} 个相关文档")
```

## 常见问题

**Q: 如何启用图片 OCR？**

图片 OCR 默认禁用（处理较慢）。启用方法：

```python
# 在 backend/config.py 中设置
IMAGE_OCR_ENABLED = True
```

**Q: Reranker 模型下载慢？**

使用镜像加速：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**Q: 如何添加新的文档类型？**

在 `backend/document_loader.py` 的 `LOADER_MAP` 中添加对应的加载器。

## License

MIT License

## 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - 文档处理框架
- [ChromaDB](https://github.com/chroma-core/chroma) - 向量数据库
- [BAAI](https://github.com/FlagOpen/FlagEmbedding) - Embedding 和 Reranker 模型
- [RAGAS](https://github.com/explodinggradients/ragas) - RAG 评估框架
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - OCR 引擎
