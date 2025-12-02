from __future__ import annotations

import os
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain_community.vectorstores import FAISS
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False

try:
    from langchain_core.vectorstores import InMemoryVectorStore
except Exception:
    InMemoryVectorStore = None  # type: ignore

from .common.config import get_config
from .common.logging import get_logger

# 优先使用本地 utils（rag_agent），回退到 src
try:
    from .utils.embed_and_store import create_vector_store  # type: ignore
except Exception:
    try:
        from src.embed_and_store import create_vector_store  # type: ignore
    except Exception:
        create_vector_store = None  # type: ignore


_VECTOR_STORE_CACHE = None
_LOGGER = get_logger("VectorStore")


def get_or_create_vector_store() -> Any:
    """
    获取或创建向量存储：
    - 若可用，优先使用 src.embed_and_store.create_vector_store 以支持增量更新；
    - 否则尝试直接加载 FAISS；失败则回退创建新存储。
    """
    global _VECTOR_STORE_CACHE
    if _VECTOR_STORE_CACHE is not None:
        return _VECTOR_STORE_CACHE

    cfg = get_config()
    vs_path = cfg.vector_store_path

    # 仅在显式请求时重建/更新；默认优先加载本地索引
    rebuild_flag = bool(getattr(cfg, "rebuild_vector_store", False))

    # 若未请求重建且索引存在，直接加载持久化 FAISS
    if not rebuild_flag and os.path.isdir(vs_path) and os.listdir(vs_path):
        try:
            if not _FAISS_AVAILABLE:
                raise RuntimeError("FAISS backend not available. Install langchain-community + faiss-cpu")
            embeddings = HuggingFaceEmbeddings(model_name=cfg.hf_model)
            vector_store = FAISS.load_local(
                vs_path,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            _VECTOR_STORE_CACHE = vector_store
            _LOGGER.info(f"Loaded persisted vector store, ntotal={vector_store.index.ntotal}")
            return vector_store
        except Exception as e:
            _LOGGER.warning(f"Failed to load persisted store; will rebuild/update. err={e}")

    # 显式请求重建/更新：优先使用本地构建器
    if rebuild_flag and create_vector_store is not None:
        try:
            backend = "faiss" if _FAISS_AVAILABLE else "in_memory"
            # 传入 rebuild=True；若旧版本不支持则回退
            try:
                vector_store = create_vector_store(
                    input_files=[cfg.all_chunks_path],
                    backend=backend,
                    faiss_dir=vs_path,
                    hf_model=cfg.hf_model,
                    enable_dedup=True,
                    rebuild=True,
                )
            except TypeError:
                vector_store = create_vector_store(
                    input_files=[cfg.all_chunks_path],
                    backend=backend,
                    faiss_dir=vs_path,
                    hf_model=cfg.hf_model,
                    enable_dedup=True,
                )
            _VECTOR_STORE_CACHE = vector_store
            try:
                total = getattr(vector_store.index, 'ntotal', None)
            except Exception:
                total = None
            if total is not None:
                _LOGGER.info(f"Rebuilt/updated vector store, ntotal={total}")
            else:
                _LOGGER.info("Rebuilt/updated vector store")
            return vector_store
        except Exception as e:
            _LOGGER.warning(f"Explicit rebuild failed; will try direct load. err={e}")

    # 未显式请求重建，或重建失败：若存在持久化存储则尝试直接加载 FAISS
    if os.path.exists(vs_path):
        try:
            if not _FAISS_AVAILABLE:
                raise RuntimeError("FAISS backend not available. Install langchain-community + faiss-cpu")

            embeddings = HuggingFaceEmbeddings(model_name=cfg.hf_model)
            vector_store = FAISS.load_local(
                vs_path,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            _VECTOR_STORE_CACHE = vector_store
            _LOGGER.info(f"Loaded vector store, ntotal={vector_store.index.ntotal}")
            return vector_store
        except Exception as e:
            # 加载失败则继续回退到内存存储
            _LOGGER.warning(f"Direct load failed; will re-create. err={e}")

    # 最后回退：创建一个空的内存向量存储（若可用）
    _LOGGER.info("Creating new vector store (first build or fallback)...")
    if create_vector_store is not None:
        try:
            vector_store = create_vector_store(
                input_files=[cfg.all_chunks_path],
                backend="faiss" if _FAISS_AVAILABLE else "in_memory",
                faiss_dir=vs_path,
                hf_model=cfg.hf_model,
                enable_dedup=True,
                rebuild=True,
            )
        except TypeError:
            vector_store = create_vector_store(
                input_files=[cfg.all_chunks_path],
                backend="faiss" if _FAISS_AVAILABLE else "in_memory",
                faiss_dir=vs_path,
                hf_model=cfg.hf_model,
                enable_dedup=True,
            )
    else:
        # 最后回退：创建一个空的内存向量存储（若可用）
        embeddings = HuggingFaceEmbeddings(model_name=cfg.hf_model)
        if InMemoryVectorStore is not None:
            vector_store = InMemoryVectorStore(embeddings)
        else:
            raise RuntimeError("No vector store backend available.")

    if _FAISS_AVAILABLE and hasattr(vector_store, "save_local"):
        _LOGGER.info(f"Saving vector store to {vs_path}")
        try:
            vector_store.save_local(vs_path)
        except Exception:
            pass
        try:
            total = getattr(vector_store.index, 'ntotal', None)
            if total is not None:
                _LOGGER.info(f"Saved vector store, ntotal={total}")
            else:
                _LOGGER.info("Saved vector store")
        except Exception:
            _LOGGER.info("Saved vector store")

    _VECTOR_STORE_CACHE = vector_store
    return vector_store