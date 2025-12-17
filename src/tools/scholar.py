import os
import requests
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# ---- env ----
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "deer_scholar_arxiv")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")

# 可调参数：默认阈值建议 0.72~0.78 之间试
DEFAULT_SCORE_THRESHOLD = float(os.getenv("SCHOLAR_SCORE_THRESHOLD", "0"))


def _ollama_embed_one(text: str) -> List[float]:
    """
    Call Ollama embeddings endpoint: POST /api/embed
    payload: {"model": "...", "input": "..."}  (Ollama also supports list input in some versions)
    response: {"embeddings": [[...]]}  (official) or {"embedding": [...]} (compat)
    """
    r = requests.post(
        f"{OLLAMA_BASE_URL.rstrip('/')}/api/embed",
        json={"model": OLLAMA_EMBED_MODEL, "input": text},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()

    if "embeddings" in data and data["embeddings"]:
        # embeddings: [[...]] (batch) -> take first
        return data["embeddings"][0]
    if "embedding" in data:
        # embedding: [...]
        return data["embedding"]

    raise ValueError(f"Unexpected Ollama embed response keys: {list(data.keys())}")


def _qdrant() -> QdrantClient:
    api_key = os.getenv("QDRANT_API_KEY")
    if api_key:
        return QdrantClient(url=QDRANT_URL, api_key=api_key)
    return QdrantClient(url=QDRANT_URL)


def _build_filter(
    category: Optional[str] = None,
    year_from: Optional[int] = None,
) -> Optional[qmodels.Filter]:
    """
    Build optional Qdrant filter. This will only work if these fields exist in payload.
    - categories: usually List[str]
    - year: int (optional)
    """
    must: List[qmodels.FieldCondition] = []

    if category:
        # 若 payload["categories"] 是 list[str]，使用 MatchAny 更合理
        must.append(
            qmodels.FieldCondition(
                key="categories",
                match=qmodels.MatchAny(any=[category]),
            )
        )

    if year_from is not None:
        must.append(
            qmodels.FieldCondition(
                key="year",
                range=qmodels.Range(gte=year_from),
            )
        )

    if not must:
        return None
    return qmodels.Filter(must=must)


def _extract_snippet(payload: Dict[str, Any], max_len: int = 800) -> str:
    """
    Try multiple payload keys to find something usable as evidence.
    """
    cand = (
        payload.get("chunk")
        or payload.get("text")
        or payload.get("summary")
        or payload.get("abstract")
        or ""
    )
    if not isinstance(cand, str):
        cand = str(cand)
    cand = cand.replace("\n", " ").strip()
    return cand[:max_len]


@tool("scholar_search")
def scholar_search_tool(
    query: str,
    top_k: int = 5,
    category: Optional[str] = None,
    year_from: Optional[int] = None,
    score_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Search local arXiv vector store (Qdrant) and return top papers as evidence.

    Behavior:
    - Embed query with Ollama embedding model
    - Search Qdrant with a score threshold
    - Deduplicate by arxiv_id (paper-level)
    - Return top_k papers with snippet evidence
    """
    if not query or not query.strip():
        return []

    vec = _ollama_embed_one(query.strip())

    flt = _build_filter(category=category, year_from=year_from)
    thr = DEFAULT_SCORE_THRESHOLD if score_threshold is None else float(score_threshold)

    # 多取一些候选，方便按 paper 去重后仍能得到 top_k
    raw_hits = _qdrant().search(
        collection_name=QDRANT_COLLECTION,
        query_vector=vec,
        limit=max(top_k * 5, top_k),
        with_payload=True,
        query_filter=flt,
        score_threshold=thr,
    )

    if not raw_hits and thr > 0:
        # fallback：避免空结果
        raw_hits = _qdrant().search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vec,
            limit=max(top_k * 5, top_k),
            with_payload=True,
            query_filter=flt,
        )

    # 去重：同一 arxiv_id 只保留最高分
    best_by_paper: Dict[str, Any] = {}
    for h in raw_hits:
        p = h.payload or {}
        arxiv_id = p.get("arxiv_id") or p.get("id") or ""
        if not arxiv_id:
            # 没有 id 就跳过，避免 “去重键缺失”
            continue
        prev = best_by_paper.get(arxiv_id)
        if prev is None or h.score > prev.score:
            best_by_paper[arxiv_id] = h

    # 组装输出：按 score 排序，取 top_k
    out: List[Dict[str, Any]] = []
    for h in sorted(best_by_paper.values(), key=lambda x: x.score, reverse=True)[:top_k]:
        p = h.payload or {}
        arxiv_id = p.get("arxiv_id") or p.get("id") or ""
        title = p.get("title") or p.get("paper_title") or ""
        snippet = _extract_snippet(p, max_len=800)

        out.append(
            {
                "score": float(h.score),
                "title": title,
                "arxiv_id": arxiv_id,
                "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
                "snippet": snippet,
                "categories": p.get("categories"),
                "published": p.get("published"),
            }
        )

    return out
