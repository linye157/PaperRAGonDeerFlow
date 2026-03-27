"""中间件：请求日志、全局异常处理、CORS 配置。"""

import json
import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger("deer_scholar.api")


class JSONFormatter(logging.Formatter):
    """结构化 JSON 日志格式。"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])
        return json.dumps(log_entry, ensure_ascii=False)


# 全局请求计数与延迟统计
_metrics = {
    "total_requests": 0,
    "total_searches": 0,
    "search_latency_sum_ms": 0.0,
}


def get_metrics() -> dict:
    return _metrics.copy()


def record_search_latency(latency_ms: float) -> None:
    _metrics["total_searches"] += 1
    _metrics["search_latency_sum_ms"] += latency_ms


def setup_middleware(app: FastAPI) -> None:
    """为 FastAPI 应用注册所有中间件。"""

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # 请求日志 & 计时中间件
    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start = time.time()

        _metrics["total_requests"] += 1

        logger.info(
            f"[{request_id}] {request.method} {request.url.path}",
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed = (time.time() - start) * 1000
            logger.error(
                f"[{request_id}] {request.method} {request.url.path} "
                f"500 {elapsed:.1f}ms - {exc}",
            )
            return JSONResponse(
                status_code=500,
                content={
                    "code": 500,
                    "message": "Internal Server Error",
                    "data": None,
                },
            )

        elapsed = (time.time() - start) * 1000
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"{response.status_code} {elapsed:.1f}ms",
        )
        response.headers["X-Request-ID"] = request_id
        return response

    # 全局异常处理
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "message": str(exc),
                "data": None,
            },
        )
