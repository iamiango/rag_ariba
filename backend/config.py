import os
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration settings for the RAG system"""

    # Anthropic API settings
 #   ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
 #   ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"

    # Embedding model settings - 使用离线模式避免网络依赖
  #  EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
   #OFFLINE_MODE: bool = os.getenv("OFFLINE_MODE", "true").lower() == "true"

    # Vector store mode - use TF-IDF based OfflineVectorStore when True (no network required)
    # Set to False to use neural embeddings (requires HuggingFace model download)
    #USE_OFFLINE_MODE: bool = os.getenv("USE_OFFLINE_MODE", "true").lower() == "true"

    # Document processing settings
    CHUNK_SIZE: int = 1500  # Size of text chunks for vector storage (优化: 800->1200)
    CHUNK_OVERLAP: int = 300  # Characters to overlap between chunks (优化: 100->200)
    MAX_RESULTS: int = 10  # Maximum search results to return
    MAX_HISTORY: int = 2  # Number of conversation messages to remember
    BATCH_SIZE: int = 5000  # Batch size for adding embeddings to ChromaDB

    # LLM parameters
    LLM_TEMPERATURE: float = 0.3  # LLM temperature for generation (优化: 0.2->0.5 提高灵活性)
    LLM_MAX_TOKENS: int = 3000  # Maximum tokens for LLM generation (优化: 2000->3000 支持更完整答案)

    # Database paths - 支持环境变量配置（用于Docker）
    CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./chroma_db")  # ChromaDB storage location

    # Embedding model path - 支持环境变量配置（用于Docker）
    EMBEDDING_MODEL_PATH: str = os.getenv("EMBEDDING_MODEL_PATH", "bge-large-en-v1.5")  # Local BGE model directory

    # 文档路径 - 支持环境变量配置（用于Docker）
    FILE_PATH_SOURCING: str = os.getenv("FILE_PATH_SOURCING", "/Users/i322171_1/Documents/Ariba/2511/Sourcing/")
    FILE_PATH_INTEGRATION: str = os.getenv("FILE_PATH_INTEGRATION", "/Users/i322171_1/Documents/Ariba/2511/Integration/")

    # BM25 configuration - 支持环境变量配置（用于Docker）
    BM25_INDEX_PATH: str = os.getenv("BM25_INDEX_PATH", "./bm25_index")  # BM25 index storage location
    BM25_K1: float = 1.5  # BM25 term frequency saturation parameter
    BM25_B: float = 0.75  # BM25 length normalization parameter
    HYBRID_RETRIEVAL: bool = True  # Enable/disable hybrid search
    BM25_TOP_K_MULTIPLIER: int = 2  # Return top_k * multiplier results in hybrid mode

    # Rerank configuration
    RERANK_ENABLED: bool = True  # Enable/disable reranking
    RERANK_USE_LOCAL_MODEL: bool = True  # Use local cached model (no network required)
    RERANK_MODEL_PATH: str = "BAAI/bge-reranker-large"  # Reranker model path (will use local cache if available)
    RERANK_LOCAL_MODEL_PATH: str = os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-reranker-large/snapshots/55611d7bca2a7133960a6d3b71e083071bbfc312")  # Local model path
    RERANK_TOP_K_MULTIPLIER: int = 10  # Retrieve top_k * multiplier before reranking (优化: 5->10 提高召回率)
    RERANK_BATCH_SIZE: int = 32  # Batch size for reranking

    # Image OCR configuration (图片文字识别配置)
    IMAGE_OCR_ENABLED: bool = True  # Enable/disable image OCR extraction
    IMAGE_OCR_ENGINE: str = 'paddleocr'  # OCR engine: paddleocr, tesseract
    IMAGE_OCR_LANGUAGES: list = None  # OCR languages, default ['ch', 'en'] for Chinese+English
    IMAGE_OCR_MIN_SIZE: int = 50  # Minimum image pixel size to process
    IMAGE_OCR_MAX_PER_DOC: int = 50  # Maximum images to extract per document
    IMAGE_OCR_CONFIDENCE: float = 0.3  # OCR confidence threshold
    IMAGE_OCR_USE_GPU: bool = False  # Use GPU for OCR (requires CUDA)
    IMAGE_OCR_SAVE_IMAGES: bool = False  # Save extracted images to disk
    IMAGE_OCR_OUTPUT_DIR: str = './extracted_images'  # Directory for saved images


config = Config()