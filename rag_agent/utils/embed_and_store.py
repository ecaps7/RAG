import os
import json
import argparse
from typing import List, Dict
from .logging import get_logger

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

try:
    from langchain_community.vectorstores import FAISS
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False

LOGGER = get_logger("EmbedStore")


def load_jsonl(paths: List[str]) -> List[Dict]:
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def to_documents(rows: List[Dict]) -> List[Document]:
    docs: List[Document] = []
    for r in rows:
        docs.append(Document(page_content=r["page_content"], metadata=r.get("metadata", {})))
    return docs


def dedup_by_sha1(docs: List[Document]) -> List[Document]:
    seen = set()
    uniq = []
    for d in docs:
        md = d.metadata or {}
        key = md.get("content_sha1")
        if not key:
            key = json.dumps({
                "t": d.page_content,
                "s": md.get("source_path"),
                "p": md.get("page"),
                "dt": md.get("doctype"),
                "ti": md.get("table_index", None),
            }, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(d)
    return uniq


def build_embeddings(hf_model: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=hf_model)


def build_in_memory_store(embeddings: HuggingFaceEmbeddings, docs: List[Document]) -> InMemoryVectorStore:
    store = InMemoryVectorStore(embeddings)
    store.add_documents(docs)
    return store


def build_or_update_faiss(
    embeddings: HuggingFaceEmbeddings,
    docs: List[Document],
    faiss_dir: str,
    rebuild: bool = False,
) -> "FAISS":
    if not _FAISS_AVAILABLE:
        raise RuntimeError("FAISS backend not available. Install: pip install faiss-cpu langchain-community")

    # 支持显式重建（避免重复添加导致膨胀）
    if rebuild:
        os.makedirs(faiss_dir, exist_ok=True)
        index = FAISS.from_documents(docs, embeddings)
    else:
        if os.path.isdir(faiss_dir) and os.listdir(faiss_dir):
            index = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
            index.add_documents(docs)
        else:
            os.makedirs(faiss_dir, exist_ok=True)
            index = FAISS.from_documents(docs, embeddings)

    index.save_local(faiss_dir)
    return index


def create_vector_store(
    input_files: List[str] = None,
    backend: str = "in_memory",
    faiss_dir: str = "outputs/vector_store",
    hf_model: str = "Qwen/Qwen3-Embedding-0.6B",
    enable_dedup: bool = False,
    rebuild: bool = False,
):
    if input_files is None:
        input_files = ["outputs/all_chunks.jsonl"]

    rows = load_jsonl(input_files)
    docs = to_documents(rows)
    LOGGER.info("Loaded %d chunks from %d file(s)", len(docs), len(input_files))

    if enable_dedup:
        before = len(docs)
        docs = dedup_by_sha1(docs)
        LOGGER.info("Deduplicated: %d -> %d", before, len(docs))

    embeddings = build_embeddings(hf_model)
    LOGGER.info("Embedding model: %s", hf_model)

    if backend == "in_memory":
        store = build_in_memory_store(embeddings, docs)
        LOGGER.info("Built InMemoryVectorStore (not persisted)")
    else:
        store = build_or_update_faiss(embeddings, docs, faiss_dir, rebuild=rebuild)
        LOGGER.info("FAISS index saved to %s (rebuild=%s)", faiss_dir, str(rebuild))

    return store


def parse_args():
    ap = argparse.ArgumentParser(description="Embed preprocessed chunks and store into vector DB (rag_agent)")
    ap.add_argument("--inputs", nargs="+", default=["outputs/all_chunks.jsonl"],
                    help="JSONL files produced by rag_agent.common.process_pdf (e.g., outputs/all_chunks.jsonl)")
    ap.add_argument("--backend", choices=["in_memory", "faiss"], default="in_memory",
                    help="Vector store backend. 'in_memory' (non-persistent) or 'faiss' (persistent).")
    ap.add_argument("--faiss_dir", type=str, default="outputs/vector_store",
                    help="Directory to save/load FAISS index when backend=faiss.")
    ap.add_argument("--hf_model", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                    help="HF embedding model name.")
    ap.add_argument("--dedup", action="store_true",
                    help="Enable deduplication by metadata.content_sha1 (if available).")
    ap.add_argument("--demo_query", type=str, default=None,
                    help="Optional query to run a quick similarity_search validation.")
    ap.add_argument("--top_k", type=int, default=3, help="k for demo query.")
    return ap.parse_args()


def demo_search(store, query: str, k: int = 3):
    LOGGER.info("[DEMO] similarity_search: %r", query)
    try:
        hits = store.similarity_search(query, k=k)
        for i, h in enumerate(hits, 1):
            src = h.metadata.get("source")
            page = h.metadata.get("page")
            dtype = h.metadata.get("doctype")
            LOGGER.info("[%d] source=%s page=%s type=%s", i, src, page, dtype)
            LOGGER.info(h.page_content[:400].replace("\n", " "))
    except Exception as e:
        LOGGER.warning("demo_search failed: %s", e)


def main():
    args = parse_args()
    rows = load_jsonl(args.inputs)
    docs = to_documents(rows)
    LOGGER.info("Loaded %d chunks from %d file(s)", len(docs), len(args.inputs))

    if args.dedup:
        before = len(docs)
        docs = dedup_by_sha1(docs)
        LOGGER.info("Deduplicated: %d -> %d", before, len(docs))

    embeddings = build_embeddings(args.hf_model)
    LOGGER.info("Embedding model: %s", args.hf_model)

    if args.backend == "in_memory":
        store = build_in_memory_store(embeddings, docs)
        LOGGER.info("Built InMemoryVectorStore (not persisted)")
    else:
        store = build_or_update_faiss(embeddings, docs, args.faiss_dir)
        LOGGER.info("FAISS index saved to %s", args.faiss_dir)

    if args.demo_query:
        demo_search(store, args.demo_query, k=args.top_k)

    LOGGER.info("[DONE] Embedding & storing complete.")


if __name__ == "__main__":
    main()