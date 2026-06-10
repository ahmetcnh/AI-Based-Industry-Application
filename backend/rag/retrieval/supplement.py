from typing import Any, Dict, List
import asyncpg
import structlog

logger = structlog.get_logger(__name__)


async def supplementary_search(
    conn: asyncpg.Connection,
    *,
    query: str,
    collection: str,
    tenant_id: int,
    user_id: int,
    top_score: float,
    threshold: float = 0.3,
    k: int = 5,
) -> List[Dict[str, Any]]:
    if top_score >= threshold:
        return []
    from backend.rag.store import ilike_search
    words = [w for w in query.split() if len(w) > 3]
    if not words:
        return []
    phrase = words[0]
    results = await ilike_search(
        conn, phrase=phrase, collection=collection,
        tenant_id=tenant_id, user_id=user_id, k=k,
    )
    logger.info("Supplementary ILIKE search triggered", phrase=phrase, found=len(results))
    return results
