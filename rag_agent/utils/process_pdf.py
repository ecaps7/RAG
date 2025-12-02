# 轻量包装：统一 utils 下的入口，复用 common/process_pdf
from ..common.process_pdf import (
    preprocess_pdfs,
    preprocess,
    extract_text_docs,
    extract_table_docs,
    chunk_text_docs,
)

__all__ = [
    "preprocess_pdfs",
    "preprocess",
    "extract_text_docs",
    "extract_table_docs",
    "chunk_text_docs",
]

if __name__ == "__main__":
    from ..common.process_pdf import main
    main()