"""Query Embedding 缓存，避免相同查询重复调用 Ollama。"""

import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Callable, List

logger = logging.getLogger("deer_scholar.cache.embedding")


class EmbeddingCache:
    """线程安全的 LRU Embedding 缓存。

    生产环境可替换为 Redis 实现，接口保持不变。
    """

    def __init__(self, max_size: int = 1024):
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(text: str) -> str:
        return hashlib.md5(text.strip().lower().encode()).hexdigest()

    def get_or_compute(
        self, query: str, embed_fn: Callable[[str], List[float]]
    ) -> List[float]:
        """查询缓存，未命中则调用 embed_fn 计算并写入缓存。"""
        key = self._make_key(query)

        with self._lock:
            if key in self._cache:
                self._hits += 1
                self._cache.move_to_end(key)
                logger.debug(f"Embedding cache HIT: {key[:8]}...")
                return self._cache[key]

        # 缓存未命中，在锁外计算 embedding（避免阻塞其他请求）
        embedding = embed_fn(query)
        self._misses += 1

        with self._lock:
            if key not in self._cache:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)  # 淘汰最旧条目
                self._cache[key] = embedding
            logger.debug(f"Embedding cache MISS: {key[:8]}... (size={len(self._cache)})")

        return embedding

    def invalidate(self, query: str) -> bool:
        """手动失效指定 query 的缓存。"""
        key = self._make_key(query)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
        return False

    def clear(self) -> int:
        """清空所有缓存，返回被清除的条目数。"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            return count

    def stats(self) -> dict:
        """返回缓存统计信息。"""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0,
            }


# 全局单例
_embedding_cache: EmbeddingCache | None = None


def get_embedding_cache() -> EmbeddingCache:
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache(max_size=1024)
    return _embedding_cache
