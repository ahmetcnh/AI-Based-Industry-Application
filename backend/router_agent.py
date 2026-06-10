import time
from typing import Any, Dict, List, Optional
import structlog

logger = structlog.get_logger(__name__)

CONFIDENCE_THRESHOLD = 0.55
CACHE_TTL = 300  # seconds


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class RouterAgent:
    def __init__(self, pool, llm, settings):
        self.pool = pool
        self.llm = llm
        self.settings = settings
        self._cache: Dict[str, tuple] = {}  # key -> (agent_row, expires_at)

    def invalidate_cache(self) -> None:
        self._cache.clear()
        logger.info("Routing cache invalidated")

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        agent_row, expires_at = entry
        if time.time() > expires_at:
            del self._cache[key]
            return None
        return agent_row

    def _cache_set(self, key: str, agent_row: Optional[Dict[str, Any]]) -> None:
        self._cache[key] = (agent_row, time.time() + CACHE_TTL)

    async def route(
        self,
        question: str,
        tenant_id: int = 1,
        user_id: int = 1,
    ) -> Optional[Dict[str, Any]]:
        cache_key = question.strip().lower()
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.info("Routing cache hit", agent=cached.get("name"))
            return cached

        question_emb = await self.llm.get_embedding(question)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, name, system_prompt, model, collection, type, embedding "
                "FROM agents WHERE tenant_id=$1 AND user_id=$2 AND embedding IS NOT NULL",
                tenant_id, user_id,
            )

        if not rows:
            self._cache_set(cache_key, None)
            return None

        scored = []
        for row in rows:
            raw_emb = row["embedding"]
            if raw_emb is None:
                continue
            # asyncpg returns the pgvector column as a string like "[0.1,0.2,...]"
            if isinstance(raw_emb, str):
                agent_emb = [float(x) for x in raw_emb.strip("[]").split(",")]
            else:
                agent_emb = list(raw_emb)
            score = _cosine(question_emb, agent_emb)
            scored.append((score, dict(row)))

        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored:
            logger.warning("No agent embeddings found after scoring")
            self._cache_set(cache_key, None)
            return None

        best_score, best_agent = scored[0]

        # No LLM fallback below the threshold: an extra model in VRAM costs a
        # multi-second swap on small GPUs, and the fallback nearly always
        # settled on the top cosine agent anyway.
        logger.info(
            "Routing decision",
            agent=best_agent["name"],
            score=round(best_score, 3),
            confident=best_score >= CONFIDENCE_THRESHOLD,
            top3=[(a["name"], round(s, 3)) for s, a in scored[:3]],
        )
        self._cache_set(cache_key, best_agent)
        return best_agent
