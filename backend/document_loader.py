from langchain_community.document_loaders import (
    PDFPlumberLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredXMLLoader,
    UnstructuredHTMLLoader,
) # 从 langchain_community.document_loaders 模块中导入各种类型文档加载器类
import os
from backend import config # 引入操作系统库，后续配置环境变量与获得当前文件路径使用

# 图片 OCR 支持（可选功能）
try:
    from backend.image_processor import (
        ImageDocumentLoader,
        ImageExtractionConfig,
        is_ocr_available,
        get_supported_formats
    )
    IMAGE_OCR_AVAILABLE = is_ocr_available()
except ImportError:
    IMAGE_OCR_AVAILABLE = False

def load_document(type_of_query, enable_image_ocr=True):
    """
    解析多种文档格式的文件，返回Document对象列表（保留元数据）
    递归遍历指定路径下所有子目录中的文档

    :param type_of_query: 查询类型，sourcing或integration
    :param enable_image_ocr: 是否启用图片 OCR 提取（默认启用）
    :return: 返回Document对象列表
    """
    if type_of_query == "sourcing":
        file_path = config.config.FILE_PATH_SOURCING
    elif type_of_query == "integration":
        file_path = config.config.FILE_PATH_INTEGRATION
    else:
        raise ValueError(f"Invalid query type: {type_of_query}")
    # 定义文档解析加载器字典，根据文档类型选择对应的文档解析加载器类和输入参数
    DOCUMENT_LOADER_MAPPING = {
        ".pdf": (PDFPlumberLoader, {}),
        ".txt": (TextLoader, {"encoding": "utf8"}),
        ".doc": (UnstructuredWordDocumentLoader, {}),
        ".docx": (UnstructuredWordDocumentLoader, {}),
        ".ppt": (UnstructuredPowerPointLoader, {}),
        ".pptx": (UnstructuredPowerPointLoader, {}),
        ".xlsx": (UnstructuredExcelLoader, {}),
        ".csv": (CSVLoader, {}),
        ".md": (UnstructuredMarkdownLoader, {}),
        ".xml": (UnstructuredXMLLoader, {}),
        ".html": (UnstructuredHTMLLoader, {}),
    }

    # 支持图片 OCR 的文件格式
    IMAGE_OCR_FORMATS = {'.pdf', '.pptx', '.ppt', '.docx', '.doc'}

    # 初始化图片加载器（如果启用）
    image_loader = None
    if enable_image_ocr and IMAGE_OCR_AVAILABLE:
        image_config = ImageExtractionConfig(
            enable_ocr=True,
            ocr_languages=['ch', 'en'],
            min_image_size=50,
            max_images_per_document=50,
            ocr_confidence_threshold=0.3,
            use_gpu=False,
            save_extracted_images=False,
            include_image_context=True
        )
        image_loader = ImageDocumentLoader(image_config)
        print("✓ 图片 OCR 功能已启用")

    documents = []
    image_doc_count = 0

    # 使用os.walk递归遍历所有子目录
    for root, _, files in os.walk(file_path):
        for fileName in files:
            ext = os.path.splitext(fileName)[1].lower()  # 获取文件扩展名，确定文档类型
            loader_tuple = DOCUMENT_LOADER_MAPPING.get(ext)  # 获取文档对应的文档解析加载器类和参数元组
            if loader_tuple: # 判断文档格式是否在加载器支持范围
                loader_class, loader_args = loader_tuple  # 解包元组，获取文档解析加载器类和参数
                try:
                    full_path = os.path.join(root, fileName)  # 构建完整文件路径
                    loader = loader_class(full_path, **loader_args)  # 创建文档解析加载器实例，并传入文档文件路径
                    document = loader.load()  # 加载文档，返回Document对象列表
                    # 保留Document对象而非只提取文本
                    documents.extend(document)
                    content_preview = document[0].page_content[:100] if document else ""
                    print(f"文档 {full_path} 的部分内容为: {content_preview}...")  # 仅用来展示文档内容的前100个字符

                    # 提取文档中的图片并进行 OCR
                    if image_loader and ext in IMAGE_OCR_FORMATS:
                        try:
                            image_docs = image_loader.load_images_from_document(full_path)
                            if image_docs:
                                documents.extend(image_docs)
                                image_doc_count += len(image_docs)
                                print(f"  └─ 从 {fileName} 提取了 {len(image_docs)} 张图片的文字")
                        except Exception as e:
                            print(f"  └─ 图片提取失败: {str(e)}")

                except Exception as e:
                    print(f"警告: 无法加载文档 {os.path.join(root, fileName)}，错误: {str(e)}")
            else:
                # 跳过不支持的文件类型，不打印警告（避免输出过多）
                pass

    if image_doc_count > 0:
        print(f"\n✓ 图片 OCR 完成，共提取 {image_doc_count} 张图片的文字内容")

    # 返回Document对象列表
    return documents