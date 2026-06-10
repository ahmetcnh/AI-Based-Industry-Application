"""
CAG Context Cache for the Coiltech Industrial Automation Platform.

Implements a lightweight in-memory cache for retrieved RAG context so that
repeated or semantically identical queries within a short window reuse the
same retrieved chunks without hitting ChromaDB again.

Invalidation rules:
  - TTL-based: entries older than CAG_CACHE_TTL_SECONDS are silently dropped.
  - Version-based: each collection has a monotonic version counter; uploading a
    new document or the FileWatcher indexing a file bumps the counter, making
    all cache entries for that collection stale on next access.
  - Explicit: callers can call invalidate_collection() or clear() at any time.
"""

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)

# Cache key: (sorted-tuple of collection names, short query hash)
_CacheKey = Tuple[Tuple[str, ...], str]


class CAGContextCache:
    """In-memory context cache for Context-Augmented Generation."""

    def __init__(self, ttl_seconds: int = 300, max_entries: int = 200) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        # entry format: {"sources": [...], "context": str, "ts": float, "versions": dict}
        self._cache: Dict[_CacheKey, Dict[str, Any]] = {}
        # Per-collection monotonic version counter; bumped on every new ingestion.
        self._collection_versions: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query_hash(self, query: str) -> str:
        return hashlib.md5(query.strip().lower().encode()).hexdigest()[:16]

    def _make_key(self, collections: List[str], query: str) -> _CacheKey:
        return (tuple(sorted(collections)), self._query_hash(query))

    def _snapshot_versions(self, collections: List[str]) -> Dict[str, int]:
        return {c: self._collection_versions.get(c, 0) for c in collections}

    def _is_version_stale(self, entry: Dict[str, Any]) -> bool:
        for coll, version in entry.get("versions", {}).items():
            if self._collection_versions.get(coll, 0) != version:
                return True
        return False

    def _evict_oldest(self) -> None:
        if not self._cache:
            return
        oldest = min(self._cache, key=lambda k: self._cache[k]["ts"])
        del self._cache[oldest]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self, collections: List[str], query: str
    ) -> Optional[Dict[str, Any]]:
        """Return cached result or None on miss/expiry/staleness."""
        key = self._make_key(collections, query)
        entry = self._cache.get(key)
        if entry is None:
            logger.debug("CAG cache miss", collections=collections)
            return None

        if time.time() - entry["ts"] > self.ttl_seconds:
            del self._cache[key]
            logger.debug("CAG cache miss (TTL expired)", collections=collections)
            return None

        if self._is_version_stale(entry):
            del self._cache[key]
            logger.debug("CAG cache miss (collection updated)", collections=collections)
            return None

        logger.info(
            "CAG cache hit",
            collections=collections,
            query_hash=key[1],
            sources=len(entry["sources"]),
        )
        return entry

    def set(
        self,
        collections: List[str],
        query: str,
        sources: List[Dict[str, Any]],
        context: str,
    ) -> None:
        """Store retrieved context in the cache."""
        if len(self._cache) >= self.max_entries:
            self._evict_oldest()

        key = self._make_key(collections, query)
        self._cache[key] = {
            "sources": sources,
            "context": context,
            "ts": time.time(),
            "versions": self._snapshot_versions(collections),
        }
        logger.debug(
            "CAG cache populated",
            collections=collections,
            query_hash=key[1],
            sources=len(sources),
        )

    def invalidate_collection(self, collection: str) -> int:
        """
        Bump the version counter for *collection* and remove all cache entries
        that reference it.  Called after a new document is indexed.
        """
        self._collection_versions[collection] = (
            self._collection_versions.get(collection, 0) + 1
        )
        stale = [k for k in list(self._cache) if collection in k[0]]
        for k in stale:
            del self._cache[k]
        logger.info(
            "CAG cache invalidated",
            collection=collection,
            entries_removed=len(stale),
            new_version=self._collection_versions[collection],
        )
        return len(stale)

    def clear(self) -> None:
        """Remove all entries and reset version counters."""
        self._cache.clear()
        self._collection_versions.clear()
        logger.info("CAG cache cleared")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._cache)

    def stats(self) -> Dict[str, Any]:
        return {
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
            "collections_tracked": dict(self._collection_versions),
        }
