"""依赖注入：Qdrant client、Ollama URL 等共享资源。"""

import os
from functools import lru_cache

from qdrant_client import QdrantClient

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "deer_scholar_arxiv")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")


@lru_cache()
def get_qdrant_client() -> QdrantClient:
    """单例 Qdrant 客户端。"""
    api_key = os.getenv("QDRANT_API_KEY")
    if api_key:
        return QdrantClient(url=QDRANT_URL, api_key=api_key)
    return QdrantClient(url=QDRANT_URL)


def get_qdrant_collection() -> str:
    return QDRANT_COLLECTION


def get_ollama_base_url() -> str:
    return OLLAMA_BASE_URL


def get_ollama_embed_model() -> str:
    return OLLAMA_EMBED_MODEL
