"""BM25 index for keyword-based retrieval."""

from __future__ import annotations

import json
import math
import os
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

from langchain_core.documents import Document

from ...config import get_config
from ...utils.text import tokenize_zh, normalize_terms, normalize_numbers, SYNONYM_MAP


def _sha1_of(obj: Any) -> str:
    """Compute SHA1 hash of a JSON-serializable object."""
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


@dataclass
class _BM25Doc:
    """Internal document representation for BM25."""
    tokens: List[str]
    length: int
    doc: Document


class BM25Index:
    """Simple BM25 index with Chinese tokenization and synonym normalization.
    
    Features:
    - Chinese text tokenization via jieba
    - Synonym normalization for domain terms
    - BM25(k1, b) scoring
    - Persistence support
    """

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        stopwords: Optional[set[str]] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.stopwords = stopwords or set()

        self.docs: List[_BM25Doc] = []
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0

        self._build(rows)

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text: normalize and tokenize."""
        normed = normalize_numbers(normalize_terms(text or ""))
        toks = [t for t in tokenize_zh(normed) if t and t not in self.stopwords]
        return toks

    def _build(self, rows: List[Dict[str, Any]]) -> None:
        """Build the index from rows."""
        docs: List[_BM25Doc] = []
        for r in rows:
            content = str(r.get("page_content", ""))
            meta = dict(r.get("metadata", {}) or {})
            toks = self._preprocess_text(content)
            d = Document(page_content=content, metadata=meta)
            drec = _BM25Doc(tokens=toks, length=len(toks), doc=d)
            docs.append(drec)
        self.docs = docs

        # Compute document frequencies
        df: Dict[str, int] = {}
        for d in docs:
            uniq = set(d.tokens)
            for t in uniq:
                df[t] = df.get(t, 0) + 1
        self.df = df

        N = len(docs)
        
        # Compute IDF (BM25 smoothed)
        idf: Dict[str, float] = {}
        for t, f in df.items():
            idf[t] = math.log((N - f + 0.5) / (f + 0.5) + 1.0)
        self.idf = idf

        # Average document length
        self.avgdl = (sum(d.length for d in docs) / max(N, 1)) if N > 0 else 0.0

    def _score(self, query_tokens: List[str], doc_rec: _BM25Doc) -> float:
        """Compute BM25 score for a document."""
        if not doc_rec.length:
            return 0.0
        
        score = 0.0
        tf: Dict[str, int] = {}
        for t in doc_rec.tokens:
            tf[t] = tf.get(t, 0) + 1
        
        for qt in query_tokens:
            if qt not in tf:
                continue
            f = tf[qt]
            idf = self.idf.get(qt, 0.0)
            denom = f + self.k1 * (1.0 - self.b + self.b * (doc_rec.length / max(self.avgdl, 1e-6)))
            score += idf * (f * (self.k1 + 1.0)) / max(denom, 1e-6)
        
        return score

    def search(self, query: str, k: int = 8) -> List[Tuple[Document, float]]:
        """Search the index.
        
        Args:
            query: The search query
            k: Maximum number of results
            
        Returns:
            List of (Document, score) tuples, sorted by score descending
        """
        q_tokens = self._preprocess_text(query or "")
        scored: List[Tuple[Document, float]] = []
        
        for d in self.docs:
            s = self._score(q_tokens, d)
            if s > 0.0:
                scored.append((d.doc, s))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # ===== Persistence =====
    
    def to_dict(self, src_path: str | None = None, src_mtime: float | None = None) -> Dict[str, Any]:
        """Serialize index to dictionary."""
        return {
            "version": "v1",
            "k1": self.k1,
            "b": self.b,
            "stopwords": sorted(list(self.stopwords)),
            "avgdl": self.avgdl,
            "df": self.df,
            "idf": self.idf,
            "docs": [
                {
                    "tokens": d.tokens,
                    "length": d.length,
                    "doc": {
                        "page_content": d.doc.page_content,
                        "metadata": d.doc.metadata,
                    },
                }
                for d in self.docs
            ],
            "source": {"path": src_path, "mtime": src_mtime},
            "synonyms_sha1": _sha1_of(SYNONYM_MAP),
            "preprocessor": "normalize_terms+normalize_numbers+tokenize_zh",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BM25Index":
        """Deserialize index from dictionary."""
        idx = cls(
            rows=[],
            stopwords=set(data.get("stopwords", [])),
            k1=float(data.get("k1", 1.5)),
            b=float(data.get("b", 0.75))
        )
        idx.df = dict(data.get("df", {}))
        idx.idf = {k: float(v) for k, v in (data.get("idf", {}).items())}
        idx.avgdl = float(data.get("avgdl", 0.0))
        
        docs = []
        for rec in data.get("docs", []):
            doc = Document(
                page_content=rec["doc"]["page_content"],
                metadata=rec["doc"].get("metadata", {})
            )
            docs.append(_BM25Doc(
                tokens=list(rec.get("tokens", [])),
                length=int(rec.get("length", 0)),
                doc=doc
            ))
        idx.docs = docs
        return idx

    def save(self, path: str, src_path: str | None = None, src_mtime: float | None = None) -> None:
        """Save index to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(src_path, src_mtime), f, ensure_ascii=False)

    @staticmethod
    def load(path: str) -> Optional["BM25Index"]:
        """Load index from file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return BM25Index.from_dict(data)
        except Exception:
            return None


# Global cache
_BM25_CACHE: Dict[str, Tuple[float, BM25Index]] = {}


def _load_rows_from_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load rows from JSONL file."""
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass
    return rows


def get_or_create_bm25_index() -> Optional[BM25Index]:
    """Get or create BM25 index with caching and persistence.
    
    Returns:
        BM25Index instance or None if source file doesn't exist
    """
    cfg = get_config()
    src = cfg.all_chunks_path
    idx_path = getattr(cfg, "bm25_index_path", os.path.join(os.path.dirname(src), "bm25_index.json"))
    
    try:
        mtime = os.path.getmtime(src)
    except Exception:
        return None

    key = os.path.abspath(src)
    cached = _BM25_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]

    # Try loading from persistence
    loaded = BM25Index.load(idx_path)
    if loaded is not None:
        try:
            with open(idx_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            src_meta = (meta.get("source") or {})
            ok = (
                src_meta.get("path") == src and
                float(src_meta.get("mtime", 0.0)) == float(mtime) and
                meta.get("synonyms_sha1") == _sha1_of(SYNONYM_MAP) and
                sorted(list(getattr(cfg, "zh_stopwords", set()))) == meta.get("stopwords", []) and
                float(meta.get("k1", 1.5)) == float(getattr(cfg, "bm25_k1", 1.5)) and
                float(meta.get("b", 0.75)) == float(getattr(cfg, "bm25_b", 0.75))
            )
            if ok:
                _BM25_CACHE[key] = (mtime, loaded)
                return loaded
        except Exception:
            pass

    # Rebuild index
    rows = _load_rows_from_jsonl(src)
    idx = BM25Index(
        rows,
        stopwords=cfg.zh_stopwords,
        k1=float(getattr(cfg, "bm25_k1", 1.5)),
        b=float(getattr(cfg, "bm25_b", 0.75))
    )
    
    try:
        idx.save(idx_path, src_path=src, src_mtime=mtime)
    except Exception:
        pass
    
    _BM25_CACHE[key] = (mtime, idx)
    return idx
