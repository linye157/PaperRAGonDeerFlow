"""Deer-Scholar API 入口：挂载所有 router，配置中间件。"""

import logging

from fastapi import FastAPI

from src.api.middleware import JSONFormatter, setup_middleware
from src.api.routers import chat, knowledge, search, system

# 配置结构化日志
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.getLogger("deer_scholar").addHandler(handler)
logging.getLogger("deer_scholar").setLevel(logging.INFO)

app = FastAPI(
    title="Deer-Scholar API",
    description="论文 RAG 科研助手 — RESTful API 服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 注册中间件
setup_middleware(app)

# 挂载路由
app.include_router(chat.router)
app.include_router(knowledge.router)
app.include_router(search.router)
app.include_router(system.router)


@app.get("/", tags=["根"])
async def root():
    return {
        "service": "Deer-Scholar API",
        "version": "1.0.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
