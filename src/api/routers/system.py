"""系统管理 API：健康检查、基础指标。"""

import logging
import time

import httpx
from fastapi import APIRouter, Depends
from qdrant_client import QdrantClient

from src.api.dependencies import get_ollama_base_url, get_qdrant_client
from src.api.middleware import get_metrics
from src.api.schemas import (
    APIResponse,
    HealthResponse,
    MetricsResponse,
    ServiceStatus,
)

logger = logging.getLogger("deer_scholar.api.system")

router = APIRouter(prefix="/api/v1", tags=["系统管理"])

# 服务启动时间
_start_time = time.time()


@router.get("/health", summary="健康检查")
async def health_check(
    qdrant: QdrantClient = Depends(get_qdrant_client),
    ollama_url: str = Depends(get_ollama_base_url),
):
    """检查 Qdrant、Ollama 等外部服务的连通性。"""
    checks: dict[str, ServiceStatus] = {}

    # Qdrant 健康检查
    try:
        qdrant.get_collections()
        checks["qdrant"] = ServiceStatus(status="healthy")
    except Exception as e:
        checks["qdrant"] = ServiceStatus(status="unhealthy", error=str(e))

    # Ollama 健康检查
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{ollama_url.rstrip('/')}/api/tags")
            if resp.status_code == 200:
                checks["ollama"] = ServiceStatus(status="healthy")
            else:
                checks["ollama"] = ServiceStatus(
                    status="unhealthy", error=f"HTTP {resp.status_code}"
                )
    except Exception as e:
        checks["ollama"] = ServiceStatus(status="unhealthy", error=str(e))

    overall = (
        "healthy"
        if all(c.status == "healthy" for c in checks.values())
        else "degraded"
    )

    return APIResponse(data=HealthResponse(status=overall, checks=checks))


@router.get("/metrics", summary="基础指标")
async def metrics():
    """返回服务运行时间、请求数、检索 QPS 等基础指标。"""
    m = get_metrics()
    total_searches = m["total_searches"]
    avg_latency = (
        m["search_latency_sum_ms"] / total_searches if total_searches > 0 else 0
    )

    return APIResponse(
        data=MetricsResponse(
            uptime_seconds=round(time.time() - _start_time, 2),
            total_requests=m["total_requests"],
            total_searches=total_searches,
            avg_search_latency_ms=round(avg_latency, 2),
        )
    )
