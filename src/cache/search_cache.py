"""检索结果缓存，相同 query+参数 直接返回缓存结果。"""

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

logger = logging.getLogger("deer_scholar.cache.search")


class SearchCache:
    """线程安全的 LRU 检索结果缓存，支持 TTL 过期。

    生产环境可替换为 Redis 实现，接口保持不变。
    """

    def __init__(self, max_size: int = 256, ttl_seconds: int = 300):
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(
        query: str,
        top_k: int,
        category: Optional[str] = None,
        year_from: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> str:
        raw = json.dumps(
            {
                "q": query.strip().lower(),
                "k": top_k,
                "cat": category,
                "year": year_from,
                "thr": score_threshold,
            },
            sort_keys=True,
        )
        return hashlib.md5(raw.encode()).hexdigest()

    def get(
        self,
        query: str,
        top_k: int,
        category: Optional[str] = None,
        year_from: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """查询缓存，未命中或过期返回 None。"""
        key = self._make_key(query, top_k, category, year_from, score_threshold)

        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if time.time() - entry["ts"] < self._ttl:
                    self._hits += 1
                    self._cache.move_to_end(key)
                    logger.debug(f"Search cache HIT: {key[:8]}...")
                    return entry["data"]
                else:
                    # 过期，删除
                    del self._cache[key]

        self._misses += 1
        return None

    def put(
        self,
        query: str,
        top_k: int,
        results: List[Dict[str, Any]],
        category: Optional[str] = None,
        year_from: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> None:
        """写入缓存。"""
        key = self._make_key(query, top_k, category, year_from, score_threshold)

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = {"data": results, "ts": time.time()}
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
                self._cache[key] = {"data": results, "ts": time.time()}
            logger.debug(f"Search cache PUT: {key[:8]}... (size={len(self._cache)})")

    def invalidate_query(self, query: str) -> int:
        """失效包含指定 query 的所有缓存条目。"""
        prefix = hashlib.md5(query.strip().lower().encode()).hexdigest()[:8]
        removed = 0
        with self._lock:
            keys_to_remove = [
                k for k in self._cache if k.startswith(prefix)
            ]
            for k in keys_to_remove:
                del self._cache[k]
                removed += 1
        return removed

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
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0,
            }


# 全局单例
_search_cache: SearchCache | None = None


def get_search_cache() -> SearchCache:
    global _search_cache
    if _search_cache is None:
        _search_cache = SearchCache(max_size=256, ttl_seconds=300)
    return _search_cache
