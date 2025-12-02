from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

from langchain_core.documents import Document

from ..common.config import get_config
from ..common.utils import tokenize_zh, _normalize_terms, _normalize_numbers
from ..common.utils import _SYNONYM_MAP

def _sha1_of(obj: Any) -> str:
    import hashlib, json as _json
    s = _json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


@dataclass
class _BM25Doc:
    tokens: List[str]
    length: int
    doc: Document


class BM25Index:
    """简单 BM25 索引（中文分词 + 轻量同义词规范化）。

    - 语料来源：cfg.all_chunks_path（JSONL，每行含 page_content/metadata）
    - 评分：BM25(k1, b)；停用词由 cfg.zh_stopwords 提供
    - 使用内存索引，适合几十万级别以内数据
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
        normed = _normalize_numbers(_normalize_terms(text or ""))
        toks = [t for t in tokenize_zh(normed) if t and t not in self.stopwords]
        return toks

    def _build(self, rows: List[Dict[str, Any]]) -> None:
        docs: List[_BM25Doc] = []
        for r in rows:
            content = str(r.get("page_content", ""))
            meta = dict(r.get("metadata", {}) or {})
            toks = self._preprocess_text(content)
            d = Document(page_content=content, metadata=meta)
            drec = _BM25Doc(tokens=toks, length=len(toks), doc=d)
            docs.append(drec)
        self.docs = docs

        # 统计 DF
        df: Dict[str, int] = {}
        for d in docs:
            uniq = set(d.tokens)
            for t in uniq:
                df[t] = df.get(t, 0) + 1
        self.df = df

        N = len(docs)
        # 计算 IDF（BM25 推荐的平滑方式）
        idf: Dict[str, float] = {}
        for t, f in df.items():
            idf[t] = math.log((N - f + 0.5) / (f + 0.5) + 1.0)
        self.idf = idf

        # 平均文档长度
        self.avgdl = (sum(d.length for d in docs) / max(N, 1)) if N > 0 else 0.0

    def _score(self, query_tokens: List[str], doc_rec: _BM25Doc) -> float:
        if not doc_rec.length:
            return 0.0
        score = 0.0
        # 统计 q 中各 term 的频次在 d 中的出现次数
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
        q_tokens = self._preprocess_text(query or "")
        scored: List[Tuple[Document, float]] = []
        for d in self.docs:
            s = self._score(q_tokens, d)
            if s > 0.0:
                scored.append((d.doc, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # ===== 持久化 =====
    def to_dict(self, src_path: str | None = None, src_mtime: float | None = None) -> Dict[str, Any]:
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
            "synonyms_sha1": _sha1_of(_SYNONYM_MAP),
            "preprocessor": "normalize_terms+normalize_numbers+tokenize_zh",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BM25Index":
        idx = cls(rows=[], stopwords=set(data.get("stopwords", [])), k1=float(data.get("k1", 1.5)), b=float(data.get("b", 0.75)))
        # 直接装载字段
        idx.df = dict(data.get("df", {}))
        idx.idf = {k: float(v) for k, v in (data.get("idf", {}).items())}
        idx.avgdl = float(data.get("avgdl", 0.0))
        docs = []
        for rec in data.get("docs", []):
            doc = Document(page_content=rec["doc"]["page_content"], metadata=rec["doc"].get("metadata", {}))
            docs.append(_BM25Doc(tokens=list(rec.get("tokens", [])), length=int(rec.get("length", 0)), doc=doc))
        idx.docs = docs
        return idx

    def save(self, path: str, src_path: str | None = None, src_mtime: float | None = None) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(src_path, src_mtime), f, ensure_ascii=False)

    @staticmethod
    def load(path: str) -> Optional["BM25Index"]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return BM25Index.from_dict(data)
        except Exception:
            return None


_BM25_CACHE: Dict[str, Tuple[float, BM25Index]] = {}


def _load_rows_from_jsonl(path: str) -> List[Dict[str, Any]]:
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
    """按需创建或加载 BM25 索引，支持持久化与增量刷新。

    加载逻辑：
    - 若存在持久化文件且与当前 src mtime、同义词表指纹、stopwords、k1/b 一致，则直接加载；
    - 否则重建索引并保存。
    - 内存也做缓存，避免重复构建。
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

    # 先尝试加载持久化
    loaded = BM25Index.load(idx_path)
    if loaded is not None:
        try:
            with open(idx_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            src_meta = (meta.get("source") or {})
            ok = (
                src_meta.get("path") == src and float(src_meta.get("mtime", 0.0)) == float(mtime) and
                meta.get("synonyms_sha1") == _sha1_of(_SYNONYM_MAP) and
                sorted(list(getattr(cfg, "zh_stopwords", set()))) == meta.get("stopwords", []) and
                float(meta.get("k1", 1.5)) == float(getattr(cfg, "bm25_k1", 1.5)) and
                float(meta.get("b", 0.75)) == float(getattr(cfg, "bm25_b", 0.75))
            )
            if ok:
                _BM25_CACHE[key] = (mtime, loaded)
                return loaded
        except Exception:
            pass

    # 重建并保存
    rows = _load_rows_from_jsonl(src)
    idx = BM25Index(rows, stopwords=cfg.zh_stopwords, k1=float(getattr(cfg, "bm25_k1", 1.5)), b=float(getattr(cfg, "bm25_b", 0.75)))
    try:
        idx.save(idx_path, src_path=src, src_mtime=mtime)
    except Exception:
        # 保存失败不影响使用
        pass
    _BM25_CACHE[key] = (mtime, idx)
    return idx