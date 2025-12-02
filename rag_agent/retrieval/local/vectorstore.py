"""Vector store management for FAISS/in-memory backends."""

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

from ...config import get_config
from ...utils.logging import get_logger

# Try importing from utils
try:
    from ...utils.embed_and_store import create_vector_store
except Exception:
    create_vector_store = None  # type: ignore


_VECTOR_STORE_CACHE = None
_LOGGER = get_logger("VectorStore")


def get_or_create_vector_store() -> Any:
    """Get or create the vector store.
    
    Priority:
    1. Return cached instance if available
    2. Load persisted FAISS index if exists and rebuild not requested
    3. Rebuild using create_vector_store if rebuild requested
    4. Create new index as fallback
    
    Returns:
        Vector store instance (FAISS or InMemoryVectorStore)
    """
    global _VECTOR_STORE_CACHE
    if _VECTOR_STORE_CACHE is not None:
        return _VECTOR_STORE_CACHE

    cfg = get_config()
    vs_path = cfg.vector_store_path
    rebuild_flag = bool(getattr(cfg, "rebuild_vector_store", False))

    # Try loading existing index
    if not rebuild_flag and os.path.isdir(vs_path) and os.listdir(vs_path):
        try:
            if not _FAISS_AVAILABLE:
                raise RuntimeError("FAISS not available")
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
            _LOGGER.warning(f"Failed to load store; will rebuild. err={e}")

    # Rebuild if requested
    if rebuild_flag and create_vector_store is not None:
        try:
            backend = "faiss" if _FAISS_AVAILABLE else "in_memory"
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
            _LOGGER.info("Rebuilt vector store")
            return vector_store
        except Exception as e:
            _LOGGER.warning(f"Rebuild failed; will try direct load. err={e}")

    # Try direct load again
    if os.path.exists(vs_path):
        try:
            if not _FAISS_AVAILABLE:
                raise RuntimeError("FAISS not available")
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
            _LOGGER.warning(f"Direct load failed; will create new. err={e}")

    # Create new index
    _LOGGER.info("Creating new vector store...")
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
        embeddings = HuggingFaceEmbeddings(model_name=cfg.hf_model)
        if InMemoryVectorStore is not None:
            vector_store = InMemoryVectorStore(embeddings)
        else:
            raise RuntimeError("No vector store backend available.")

    # Save if possible
    if _FAISS_AVAILABLE and hasattr(vector_store, "save_local"):
        try:
            vector_store.save_local(vs_path)
            _LOGGER.info(f"Saved vector store to {vs_path}")
        except Exception:
            pass

    _VECTOR_STORE_CACHE = vector_store
    return vector_store
