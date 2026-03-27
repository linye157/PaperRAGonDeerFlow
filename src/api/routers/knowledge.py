"""知识库管理 API：入库触发、任务状态、统计信息、论文删除。"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone

import requests
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from src.api.dependencies import (
    get_ollama_base_url,
    get_ollama_embed_model,
    get_qdrant_client,
    get_qdrant_collection,
)
from src.api.schemas import (
    APIResponse,
    IngestRequest,
    IngestTaskResponse,
    KnowledgeStatsResponse,
)

logger = logging.getLogger("deer_scholar.api.knowledge")

router = APIRouter(prefix="/api/v1/knowledge", tags=["知识库管理"])

# 内存中的任务存储（任务2会迁移到 SQLite）
_ingest_tasks: dict[str, dict] = {}


def _run_ingest_pipeline(task_id: str, req: IngestRequest) -> None:
    """执行论文入库 ETL 流水线（同步，由 BackgroundTasks 调用）。"""
    task = _ingest_tasks[task_id]
    task["status"] = "processing"

    try:
        from datasets import load_dataset
        from qdrant_client.http.models import VectorParams, Distance, PointStruct

        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        collection = os.getenv("QDRANT_COLLECTION", "deer_scholar_arxiv")
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")

        # 1) 加载数据集
        ds = load_dataset(req.dataset_name, split="train")
        if req.limit and req.limit < len(ds):
            ds = ds.select(range(req.limit))
        task["total_chunks"] = len(ds)

        # 2) 初始化 Qdrant
        api_key = os.getenv("QDRANT_API_KEY")
        qc = QdrantClient(url=qdrant_url, api_key=api_key) if api_key else QdrantClient(url=qdrant_url)

        # 3) 探测向量维度
        r = requests.post(
            f"{ollama_url.rstrip('/')}/api/embeddings",
            json={"model": embed_model, "prompt": "dimension probe"},
            timeout=60,
        )
        r.raise_for_status()
        dim = len(r.json()["embedding"])

        # 4) 创建/重建 collection
        if req.recreate and qc.collection_exists(collection):
            qc.delete_collection(collection)
        if not qc.collection_exists(collection):
            qc.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

        # 5) 批量 upsert
        buf_points = []
        buf_texts = []

        def flush():
            nonlocal buf_points, buf_texts
            if not buf_points:
                return
            vecs = []
            for t in buf_texts:
                resp = requests.post(
                    f"{ollama_url.rstrip('/')}/api/embeddings",
                    json={"model": embed_model, "prompt": t},
                    timeout=60,
                )
                resp.raise_for_status()
                vecs.append(resp.json()["embedding"])
            for p, v in zip(buf_points, vecs):
                p.vector = v
            qc.upsert(collection_name=collection, points=buf_points)
            task["processed_chunks"] += len(buf_points)
            buf_points, buf_texts = [], []

        for row in ds:
            arxiv_id = row.get("id")
            chunk_id = row.get("chunk-id")
            chunk_text = row.get("chunk") or ""
            title = row.get("title") or ""
            summary = row.get("summary") or ""

            doc_text = f"Title: {title}\nAbstract: {summary}\nChunk: {chunk_text}".strip()

            payload = {
                "arxiv_id": arxiv_id,
                "chunk_id": chunk_id,
                "title": title,
                "summary": summary,
                "categories": row.get("categories"),
                "primary_category": row.get("primary_category"),
                "authors": row.get("authors"),
                "published": row.get("published"),
                "updated": row.get("updated"),
                "pdf_url": row.get("source"),
                "comment": row.get("comment"),
            }

            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{arxiv_id}:{chunk_id}"))
            buf_points.append(PointStruct(id=point_id, vector=[], payload=payload))
            buf_texts.append(doc_text)

            if len(buf_points) >= req.batch_size:
                flush()

        flush()

        task["status"] = "completed"
        task["completed_at"] = datetime.now(timezone.utc).isoformat()

    except Exception as e:
        logger.error(f"Ingest pipeline failed: {e}", exc_info=True)
        task["status"] = "failed"
        task["error_message"] = str(e)


@router.post("/ingest", summary="触发论文入库（异步任务）")
async def trigger_ingest(req: IngestRequest):
    """创建异步入库任务，使用 BackgroundTasks 执行 ETL 流水线。"""
    from fastapi import BackgroundTasks

    task_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    task = {
        "id": task_id,
        "status": "pending",
        "source_type": req.source_type,
        "total_chunks": 0,
        "processed_chunks": 0,
        "error_message": None,
        "created_at": now.isoformat(),
        "completed_at": None,
    }
    _ingest_tasks[task_id] = task

    # 在后台线程中执行入库
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_ingest_pipeline, task_id, req)

    return APIResponse(data=IngestTaskResponse(**task))


@router.get("/tasks/{task_id}", summary="查询入库任务状态")
async def get_task_status(task_id: str):
    """查询指定入库任务的状态和进度。"""
    if task_id not in _ingest_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    return APIResponse(data=IngestTaskResponse(**_ingest_tasks[task_id]))


@router.get("/tasks/{task_id}/progress", summary="SSE 推送入库进度")
async def task_progress(task_id: str):
    """通过 SSE 流式推送入库任务进度。"""
    if task_id not in _ingest_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    async def event_generator():
        while True:
            task = _ingest_tasks[task_id]
            total = task["total_chunks"]
            processed = task["processed_chunks"]
            progress = processed / total if total > 0 else 0

            data = json.dumps({
                "task_id": task_id,
                "status": task["status"],
                "progress": round(progress, 4),
                "processed_chunks": processed,
                "total_chunks": total,
            })
            yield f"data: {data}\n\n"

            if task["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/stats", summary="知识库统计")
async def knowledge_stats(
    qdrant: QdrantClient = Depends(get_qdrant_client),
    collection: str = Depends(get_qdrant_collection),
):
    """获取知识库统计信息：论文数、chunk 数、collection 信息。"""
    try:
        info = qdrant.get_collection(collection)
        total_points = info.points_count or 0
        vector_dim = 0
        distance = "Cosine"

        if info.config and info.config.params and info.config.params.vectors:
            vec_cfg = info.config.params.vectors
            if hasattr(vec_cfg, "size"):
                vector_dim = vec_cfg.size
                distance = str(vec_cfg.distance) if vec_cfg.distance else "Cosine"

        # 估算去重论文数（通过 scroll 采样）
        unique_papers = total_points  # 默认等于 chunk 数
        try:
            sample, _ = qdrant.scroll(
                collection_name=collection,
                limit=1000,
                with_payload=["arxiv_id"],
            )
            if sample:
                arxiv_ids = {
                    p.payload.get("arxiv_id")
                    for p in sample
                    if p.payload and p.payload.get("arxiv_id")
                }
                if arxiv_ids:
                    ratio = len(arxiv_ids) / len(sample)
                    unique_papers = int(total_points * ratio)
        except Exception:
            pass

        return APIResponse(
            data=KnowledgeStatsResponse(
                collection_name=collection,
                total_points=total_points,
                unique_papers=unique_papers,
                vector_dimension=vector_dim,
                distance_metric=distance,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant 连接失败: {e}")


@router.delete("/papers/{arxiv_id}", summary="删除指定论文")
async def delete_paper(
    arxiv_id: str,
    qdrant: QdrantClient = Depends(get_qdrant_client),
    collection: str = Depends(get_qdrant_collection),
):
    """删除指定 arXiv ID 的所有 chunk。"""
    try:
        qdrant.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="arxiv_id",
                        match=MatchValue(value=arxiv_id),
                    )
                ]
            ),
        )
        return APIResponse(message=f"论文 {arxiv_id} 的所有 chunk 已删除")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {e}")
