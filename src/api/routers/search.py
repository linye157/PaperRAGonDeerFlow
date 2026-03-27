"""独立检索 API：scholar_search 调用、相似论文推荐。"""

import logging
import time
from typing import List

import requests
from fastapi import APIRouter, Depends, HTTPException
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from src.api.dependencies import (
    get_ollama_base_url,
    get_ollama_embed_model,
    get_qdrant_client,
    get_qdrant_collection,
)
from src.api.middleware import record_search_latency
from src.api.schemas import (
    APIResponse,
    PaperResult,
    ScholarSearchRequest,
    SimilarSearchRequest,
)
from src.db.database import get_db
from src.db import crud
from src.tools.scholar import scholar_search_tool

logger = logging.getLogger("deer_scholar.api.search")

router = APIRouter(prefix="/api/v1/search", tags=["独立检索"])


@router.post("/scholar", summary="独立调用 scholar_search")
async def scholar_search(req: ScholarSearchRequest, db: Session = Depends(get_db)):
    """直接调用 scholar_search 工具进行论文检索。"""
    start = time.time()

    try:
        results = scholar_search_tool.invoke({
            "query": req.query,
            "top_k": req.top_k,
            "category": req.category,
            "year_from": req.year_from,
            "score_threshold": req.score_threshold,
        })
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"检索服务异常: {e}")

    latency_ms = (time.time() - start) * 1000
    record_search_latency(latency_ms)

    # 记录检索日志到数据库
    crud.create_search_log(
        db,
        query=req.query,
        top_k=req.top_k,
        threshold=req.score_threshold,
        result_count=len(results),
        latency_ms=latency_ms,
    )

    papers = [PaperResult(**r) for r in results]
    return APIResponse(data=papers)


@router.post("/similar", summary="相似论文推荐")
async def similar_papers(
    req: SimilarSearchRequest,
    db: Session = Depends(get_db),
    qdrant: QdrantClient = Depends(get_qdrant_client),
    collection: str = Depends(get_qdrant_collection),
    ollama_url: str = Depends(get_ollama_base_url),
    embed_model: str = Depends(get_ollama_embed_model),
):
    """基于指定论文的 embedding 查找相似论文。"""
    start = time.time()

    try:
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        hits, _ = qdrant.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="arxiv_id",
                        match=MatchValue(value=req.arxiv_id),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

        if not hits:
            raise HTTPException(status_code=404, detail=f"论文 {req.arxiv_id} 不存在")

        payload = hits[0].payload or {}
        title = payload.get("title", "")
        summary = payload.get("summary", "")
        query_text = f"{title} {summary}".strip()

        if not query_text:
            raise HTTPException(status_code=400, detail="论文缺少标题和摘要信息")

        r = requests.post(
            f"{ollama_url.rstrip('/')}/api/embed",
            json={"model": embed_model, "input": query_text},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        vec = data.get("embeddings", [[]])[0] or data.get("embedding", [])

        if not vec:
            raise HTTPException(status_code=500, detail="Embedding 生成失败")

        raw_hits = qdrant.search(
            collection_name=collection,
            query_vector=vec,
            limit=req.top_k * 5,
            with_payload=True,
        )

        best_by_paper: dict = {}
        for h in raw_hits:
            p = h.payload or {}
            aid = p.get("arxiv_id") or ""
            if not aid or aid == req.arxiv_id:
                continue
            prev = best_by_paper.get(aid)
            if prev is None or h.score > prev.score:
                best_by_paper[aid] = h

        results: List[PaperResult] = []
        for h in sorted(best_by_paper.values(), key=lambda x: x.score, reverse=True)[: req.top_k]:
            p = h.payload or {}
            aid = p.get("arxiv_id") or ""
            results.append(
                PaperResult(
                    score=float(h.score),
                    title=p.get("title") or "",
                    arxiv_id=aid,
                    url=f"https://arxiv.org/abs/{aid}" if aid else None,
                    snippet=(p.get("chunk") or p.get("summary") or "")[:800],
                    categories=p.get("categories"),
                    published=p.get("published"),
                )
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"检索服务异常: {e}")

    latency_ms = (time.time() - start) * 1000
    record_search_latency(latency_ms)

    # 记录检索日志
    crud.create_search_log(
        db,
        query=f"similar:{req.arxiv_id}",
        top_k=req.top_k,
        threshold=None,
        result_count=len(results),
        latency_ms=latency_ms,
    )

    return APIResponse(data=results)
