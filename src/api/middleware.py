"""中间件：请求日志、全局异常处理、CORS 配置、结构化日志。"""

import contextvars
import json
import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger("deer_scholar.api")

# 请求 ID 上下文变量，供整个请求链路使用
_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


def get_request_id() -> str:
    """获取当前请求 ID（在路由/服务层中使用）。"""
    return _request_id_ctx.get()


class JSONFormatter(logging.Formatter):
    """结构化 JSON 日志格式，自动附带请求 ID。"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", _request_id_ctx.get()),
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])
        return json.dumps(log_entry, ensure_ascii=False)


class RequestIDFilter(logging.Filter):
    """日志过滤器：自动将当前请求 ID 注入日志记录。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_ctx.get()
        return True


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


def setup_logging() -> None:
    """配置全局结构化日志。"""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    handler.addFilter(RequestIDFilter())

    root_logger = logging.getLogger("deer_scholar")
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


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

    # 请求日志 & 计时 & 请求 ID 中间件
    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        _request_id_ctx.set(request_id)

        # 将 request_id 注入 request.state，供路由层使用
        request.state.request_id = request_id
        start = time.time()

        _metrics["total_requests"] += 1

        logger.info(
            f"[{request_id}] --> {request.method} {request.url.path}",
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
            f"[{request_id}] <-- {request.method} {request.url.path} "
            f"{response.status_code} {elapsed:.1f}ms",
        )
        response.headers["X-Request-ID"] = request_id
        return response

    # 全局异常处理
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "-")
        logger.error(
            f"[{request_id}] Unhandled exception: {exc}", exc_info=True
        )
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "message": str(exc),
                "data": None,
            },
        )
