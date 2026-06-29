# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) system for Ariba documentation, built with Python. The system processes various document formats (PDF, Word, PowerPoint, Excel, etc.), creates embeddings using the BAAI/bge-large-en-v1.5 model, stores them in ChromaDB, and enables semantic search over Ariba documentation.

## Architecture

### Core Components

- **backend/config.py**: Central configuration for the RAG system
  - Document processing settings (chunk size: 1500, overlap: 300, batch size: 5000)
  - File paths for Sourcing and Integration documentation
  - ChromaDB storage path configuration
  - Embedding model path configuration (BAAI/bge-large-en-v1.5)
  - BM25 configuration (k1: 1.5, b: 0.75, hybrid retrieval enabled, top_k multiplier: 2)
  - Rerank configuration (BAAI/bge-reranker-large, top_k multiplier: 5, batch size: 32)
  - LLM parameters (temperature: 0.2 for reduced hallucination, max_tokens: 2000)

- **backend/document_loader.py**: Multi-format document loader
  - Supports PDF, Word, PowerPoint, Excel, CSV, Markdown, XML, HTML
  - Uses langchain_community loaders for each format
  - Recursively traverses all subdirectories under configured paths
  - Loads documents from configured paths based on query type (sourcing/integration)
  - **NEW**: Supports image OCR extraction from PDF/PPT/Word documents

- **backend/image_processor.py**: Image extraction and OCR processing module
  - `ImageExtractionConfig`: Configuration for image extraction and OCR settings
  - `ImageExtractor`: Extracts embedded images from PDF, PPT, Word documents
  - `OCRProcessor`: Performs OCR on extracted images using PaddleOCR (supports Chinese+English)
  - `ImageDocumentLoader`: Integrated loader that extracts images and creates LangChain Documents
  - Uses PyMuPDF for fast PDF image extraction (with pdfplumber fallback)
  - Uses python-pptx and python-docx for Office document image extraction
  - Singleton pattern for OCR processor to avoid repeated model loading

- **backend/indexing_processor.py**: Document indexing pipeline
  - Splits documents into chunks using RecursiveCharacterTextSplitter
  - Generates embeddings using sentence-transformers
  - Stores embeddings and text chunks in ChromaDB with unique IDs
  - Uses batch processing to avoid ChromaDB size limits (configurable batch size)

- **backend/document_processor.py**: Main orchestration layer for embedding and retrieval
  - `process_and_embed_documents()`: One-time embedding process for all documents
  - `get_collection(type_of_query)`: Retrieves ChromaDB collection for sourcing or integration
  - `search_documents(type_of_query, query_text, n_results)`: Semantic search in specified collection
  - Manages embedding model initialization and ChromaDB client
  - Creates separate collections: `ariba_sourcing` and `ariba_integration`

- **backend/retrieval_processor.py**: Document retrieval interface with reranking
  - `retrieval_process(query, type_of_query, top_k)`: Retrieves top-k most similar document chunks
  - `get_reranker()`: Initializes FlagReranker model (singleton pattern)
  - `rerank_documents(query, documents, top_k)`: Reranks retrieved documents using FlagReranker
  - Supports hybrid retrieval: vector search + BM25 + reranking
  - Returns list of reranked text chunks with similarity scores
  - Prints retrieval and reranking results for debugging

- **backend/document_splitter.py**: Empty placeholder (not yet implemented)

- **backend/bm25_indexer.py**: BM25 index builder for hybrid retrieval
  - Builds BM25 index from ChromaDB collections
  - Tokenizes documents using NLTK
  - Stores index in pickle format for fast loading
  - Supports both sourcing and integration collections

- **backend/ragas_evaluator.py**: RAGAS-based evaluation system
  - `RagasEvaluator`: Main evaluation class using RAGAS metrics
  - Evaluates faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
  - Uses Qwen LLM for evaluation via OpenAI-compatible API
  - Loads evaluation datasets and runs RAG system on test questions
  - Generates comprehensive evaluation reports with metrics

- **backend/ragas_dataset_generator.py**: Evaluation dataset generator
  - Generates synthetic evaluation datasets from documents
  - Creates question-answer pairs with ground truth
  - Supports both sourcing and integration document types

- **backend/query_preprocessor.py**: Query preprocessing and optimization
  - `QueryPreprocessor`: Main class for query preprocessing
  - `query_corrector(query, verbose)`: Corrects spelling/grammar and extracts keywords
  - Uses language-tool-python for English spelling/grammar correction
  - Uses jieba for Chinese/English keyword extraction
  - Returns optimized query with corrections and keywords
  - Gracefully degrades if dependencies not installed

- **app.py**: RAG system main entrance
  - `rag_query(query_text, type_of_query, n_results, temperature)`: Main RAG query method
  - `call_qwen_llm(prompt, temperature, max_tokens)`: Calls Qwen API using dashscope
  - `load_prompt_template(template_type)`: Loads prompt templates for different query types
  - `interactive_mode()`: Interactive CLI for asking questions
  - Orchestrates query preprocessing → retrieval → context building → LLM generation pipeline

- **prompts.md**: Prompt template definitions
  - Default template for general Ariba questions
  - Sourcing-specific template for procurement questions
  - Integration-specific template for technical questions

### Embedding and Reranking Models

**Embedding Model (BAAI/bge-large-en-v1.5):**
- Stored in `bge-large-en-v1.5/` directory (configured in `backend/config.py`)
- Large English embedding model (1.34GB model.safetensors)
- Includes ONNX optimized version in `onnx/` subdirectory
- No network dependency for embeddings

**Reranking Model (BAAI/bge-reranker-large):**
- Used for reranking retrieved documents to improve relevance
- Loaded via FlagEmbedding library's FlagReranker class
- Uses FP16 precision for faster inference
- Singleton pattern to avoid repeated model loading
- **Offline mode enabled by default**: Uses local cached model (no network required)
- Local model path: `~/.cache/huggingface/hub/models--BAAI--bge-reranker-large/`
- Model size: ~2.1GB
- Can be disabled via `RERANK_ENABLED` config flag
- Can switch to online mode via `RERANK_USE_LOCAL_MODEL = False` in config

### Data Flow

**One-time Embedding Process:**
1. Documents are loaded from configured paths (`FILE_PATH_SOURCING` or `FILE_PATH_INTEGRATION`)
2. Text is extracted using format-specific loaders
3. **NEW**: Images are extracted from PDF/PPT/Word and OCR'd using PaddleOCR
4. Text (including OCR results) is split into chunks (1200 chars with 200 char overlap)
5. Each chunk is embedded using the local BGE model
6. Embeddings and chunks are stored in separate ChromaDB collections with UUIDs
   - `ariba_sourcing` collection for Sourcing documents
   - `ariba_integration` collection for Integration documents
7. BM25 index is built for keyword-based retrieval (optional, for hybrid search)

**RAG Query Process:**
1. User submits a question via app.py
2. Query preprocessing: spelling/grammar correction and keyword extraction
3. Retrieval process (retrieval_processor):
   - Step 1: Vector search retrieves top-k * RERANK_TOP_K_MULTIPLIER documents (default: 3x)
   - Step 2: BM25 keyword search adds additional relevant documents (if HYBRID_RETRIEVAL enabled)
   - Step 3: Reranking using FlagReranker to reorder all retrieved documents (if RERANK_ENABLED)
   - Returns final top-k most relevant documents after reranking
4. Retrieved chunks are combined into context
5. Prompt template is loaded based on query type (sourcing/integration)
6. Context and question are formatted into the prompt
7. Qwen LLM generates answer based on the prompt
8. Answer, retrieved documents, and preprocessing metadata are returned to user

## Development Setup

### Virtual Environment

The project uses a Python virtual environment in `ariba_env/`:

```bash
source ariba_env/bin/activate
```

### Key Dependencies

- langchain (0.3.27) - Document processing and text splitting
- langchain-community (0.3.31) - Document loaders
- chromadb (1.5.0) - Vector database
- sentence-transformers (5.1.2) - Embedding generation
- transformers (4.57.6) - Model loading
- unstructured (0.18.3) - Document parsing
- dashscope - For Qwen API calls
- FlagEmbedding - For document reranking (FlagReranker class)
- language-tool-python (>=2.7.1) - Query spelling/grammar correction (optional)
- paddleocr - Image OCR engine (optional, for image text extraction)
- paddlepaddle - PaddleOCR deep learning framework (optional)
- pymupdf (fitz) - Fast PDF image extraction (optional, recommended)
- opencv-python-headless - Image preprocessing (optional)
- jieba (>=0.42.1) - Chinese/English keyword extraction (optional)

Note: Query preprocessing dependencies are optional. If not installed, the system will use simplified fallback methods.

### Configuration

Update paths in `backend/config.py`:
- `FILE_PATH_SOURCING`: Path to Sourcing documentation
- `FILE_PATH_INTEGRATION`: Path to Integration documentation
- `CHROMA_PATH`: ChromaDB storage location (default: `./chroma_db`)
- `EMBEDDING_MODEL_PATH`: Local embedding model directory (default: `bge-large-en-v1.5`)
- `BM25_INDEX_PATH`: BM25 index storage location (default: `./bm25_index`)
- `HYBRID_RETRIEVAL`: Enable/disable hybrid search (default: `True`)
- `RERANK_ENABLED`: Enable/disable reranking (default: `True`)
- `RERANK_USE_LOCAL_MODEL`: Use local cached model for offline operation (default: `True`)
- `RERANK_MODEL_PATH`: Reranker model name (default: `BAAI/bge-reranker-large`)
- `RERANK_LOCAL_MODEL_PATH`: Local reranker model path (auto-configured to HuggingFace cache)
- `RERANK_TOP_K_MULTIPLIER`: Retrieve multiplier before reranking (default: `5`)
- `RERANK_BATCH_SIZE`: Batch size for reranking (default: `32`)

### Environment Variables

Required for RAG system:
```bash
export DASHSCOPE_API_KEY="your-dashscope-api-key"  # Required for Qwen LLM calls
export QWEN_MODEL="qwen-plus"  # Optional, default provided
```

## Usage

### One-Time Embedding Process

Run this once to embed all documents into ChromaDB:

```bash
# Activate virtual environment
source ariba_env/bin/activate

# Run the embedding process
python -m backend.document_processor
```

Or programmatically:

```python
from backend import document_processor

# Process and embed all documents (one-time operation)
document_processor.process_and_embed_documents()
```

### Retrieving Collections for Search

```python
from backend import document_processor

# Get sourcing collection
sourcing_collection = document_processor.get_collection("sourcing")

# Get integration collection
integration_collection = document_processor.get_collection("integration")
```

### Running the RAG System

**Interactive Mode (Recommended):**

```bash
# Activate virtual environment
source ariba_env/bin/activate

# Set Qwen API key
export DASHSCOPE_API_KEY="your-api-key-here"

# Run interactive mode
python app.py
```

**Programmatic Usage:**

```python
from app import rag_query

# Set environment variable first
import os
os.environ["DASHSCOPE_API_KEY"] = "your-api-key-here"

# Query sourcing documents
result = rag_query(
    query_text="How to create a sourcing project?",
    type_of_query="sourcing",
    n_results=5,  # Optional, defaults to config.MAX_RESULTS
    temperature=0.7  # Optional, controls LLM randomness
)

# Access the answer
print(result['answer'])

# Access retrieved documents
for chunk in result['retrieved_docs']:
    print(chunk)
```

### Direct Document Search (Without LLM)

```python
from backend import document_processor

# Search in sourcing documents
results = document_processor.search_documents(
    type_of_query="sourcing",
    query_text="How to create a sourcing project?",
    n_results=5
)

# Access results
for i, doc in enumerate(results['documents'][0]):
    print(f"Result {i+1}: {doc}")
```

### Running the Streamlit Web Interface

**启动Web界面：**

```bash
source ariba_env/bin/activate
export DASHSCOPE_API_KEY="your-key"
streamlit run streamlit_app.py
```

或使用启动脚本：

```bash
export DASHSCOPE_API_KEY="your-key"
./run_streamlit.sh
```

**访问地址：** http://localhost:8501

**功能特性：**
- 左侧Tab切换Sourcing/Integration文档类型
- 实时问答交互
- 显示文档引用来源（标题+页码）
- 可选显示检索文档详情
- 对话历史记录（按Tab分别保存）
- 高级设置：调整检索数量和LLM温度

## Code Conventions

- Chinese comments are used throughout the codebase
- Import statements use comma-separated style: `import os,config,document_loader`
- Configuration uses dataclass pattern with class-level constants
- Document loaders use a mapping dictionary pattern for extensibility

## Typical Workflow

1. **First Time Setup:**
   ```bash
   source ariba_env/bin/activate
   python -m backend.document_processor  # One-time embedding
   python setup_bm25.py  # Build BM25 index (if using hybrid retrieval)
   ```

2. **Using the RAG System:**
   ```bash
   export DASHSCOPE_API_KEY="your-key"
   python app.py  # Interactive mode
   ```

3. **Development/Testing:**
   - Test retrieval without LLM: Use `document_processor.search_documents()`
   - Test with LLM: Use `app.rag_query()`
   - Switch between sourcing/integration by changing `type_of_query` parameter

## Testing and Evaluation

### Running Tests

**Quick System Test:**
```bash
python test_simple.py  # Basic embedding and retrieval test
python test_embedding.py  # Test embedding functionality
python check_status.py  # Check ChromaDB collection status
```

**BM25 Hybrid Retrieval Test:**
```bash
python test_bm25_hybrid.py  # Test hybrid vector + BM25 search
```

**Reranking Test:**
```bash
python test_rerank.py  # Test reranking functionality
./test_rerank_with_mirror.sh  # Test with model mirror download
```

**Query Preprocessing Test:**
```bash
python test_query_preprocessor.py  # Test spelling correction and keyword extraction
```

### RAGAS Evaluation

The system includes comprehensive RAGAS-based evaluation using pre-built datasets:

**Run Comprehensive Evaluation:**
```bash
export DASHSCOPE_API_KEY="your-key"
python run_comprehensive_evaluation.py
```

This evaluates both sourcing and integration datasets and generates:
- Individual metrics: faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
- Overall score and performance analysis
- Optimization suggestions based on metrics
- JSON report: `evaluation_comprehensive_report_v*.json`

**Evaluation Datasets:**
- `evaluation_dataset_sourcing_en_v2.json`: Sourcing questions (English)
- `evaluation_dataset_integration_en_v2.json`: Integration questions (English)
- `evaluation_dataset_sourcing.json`: Sourcing questions (Chinese)
- `evaluation_dataset_integration.json`: Integration questions (Chinese)

**Quick Evaluation:**
```bash
python evaluate_rag_quick.py  # Fast evaluation with fewer samples
python evaluate_rag_v2.py  # Full evaluation with detailed output
```

### Building BM25 Index

If hybrid retrieval is enabled (default), build the BM25 index:

```bash
python setup_bm25.py
```

This script:
1. Checks and installs dependencies (rank-bm25, nltk)
2. Downloads NLTK data (punkt tokenizer)
3. Builds BM25 index from ChromaDB collections
4. Saves index to `./bm25_index/` directory

## Troubleshooting

### ChromaDB Collection Issues

**Check collection status:**
```bash
python check_status.py
```

**Re-embed documents if corrupted:**
```bash
python re_embed_documents.py  # Manual re-embedding
python re_embed_documents_auto.py  # Automatic re-embedding
```

### BM25 Index Issues

**Rebuild BM25 index:**
```bash
rm -rf bm25_index/
python setup_bm25.py
```

### Model Download Issues

**Reranker model download with mirror:**
```bash
export HF_ENDPOINT=https://hf-mirror.com
./test_rerank_with_mirror.sh
```

**Note**: The system is configured to use local cached models by default. If the rerank model is already downloaded to `~/.cache/huggingface/hub/models--BAAI--bge-reranker-large/`, no network connection is required.

### Query Preprocessing Dependencies

If `language-tool-python` or `jieba` are not installed, the system will use simplified fallback methods. To enable full preprocessing:

```bash
pip install language-tool-python>=2.7.1 jieba>=0.42.1
```

### Common Issues

**Issue: "Collection not found"**
- Solution: Run `python -m backend.document_processor` to create collections

**Issue: "BM25 index not found"**
- Solution: Run `python setup_bm25.py` to build index or disable hybrid retrieval in config

**Issue: "Reranker model not found"**
- Solution: The system uses local cached model by default at `~/.cache/huggingface/hub/models--BAAI--bge-reranker-large/`
- If model is missing, download it with: `./test_rerank_with_mirror.sh`
- Or disable reranking: Set `RERANK_ENABLED = False` in config.py
- Or switch to online mode: Set `RERANK_USE_LOCAL_MODEL = False` in config.py

**Issue: "Image OCR not working" or "PaddleOCR not available"**
- Solution: Install PaddleOCR dependencies:
  ```bash
  pip install paddleocr paddlepaddle opencv-python-headless pymupdf
  ```
- Test OCR functionality: `python test_image_ocr.py`
- To disable image OCR: Set `IMAGE_OCR_ENABLED = False` in config.py or pass `enable_image_ocr=False` to `load_document()`

**Issue: "PaddleOCR model download slow"**
- First run will download OCR models (~150MB), this is normal
- Use mirror if in China: `export HF_ENDPOINT=https://hf-mirror.com`
- Models are cached at `~/.paddleocr/` after first download

**Issue: Low evaluation scores**
- Check `evaluation_comprehensive_report_*.json` for specific suggestions
- Consider adjusting `LLM_TEMPERATURE` (lower = less hallucination)
- Verify document quality and completeness
- Tune retrieval parameters (top_k, chunk_size, overlap)
