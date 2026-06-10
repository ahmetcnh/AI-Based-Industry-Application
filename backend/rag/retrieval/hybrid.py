import asyncio
from typing import Any, Dict, List
import asyncpg
import structlog

logger = structlog.get_logger(__name__)


async def hybrid_search(
    pool: asyncpg.Pool,
    *,
    query: str,
    query_embedding: List[float],
    collection: str,
    tenant_id: int,
    user_id: int,
    k: int = 20,
    alpha: float = 0.3,
) -> List[Dict[str, Any]]:
    from backend.rag.store import vector_search, bm25_search

    # Vector and BM25 are independent; run them concurrently on
    # separate pool connections.
    async def _vec():
        async with pool.acquire() as conn:
            return await vector_search(
                conn, embedding=query_embedding, collection=collection,
                tenant_id=tenant_id, user_id=user_id, k=k,
            )

    async def _bm25():
        async with pool.acquire() as conn:
            return await bm25_search(
                conn, query=query, collection=collection,
                tenant_id=tenant_id, user_id=user_id, k=k,
            )

    vec_results, bm25_results = await asyncio.gather(_vec(), _bm25())

    scores: Dict[int, Dict[str, Any]] = {}

    for r in vec_results:
        rid = r["id"]
        scores[rid] = {**r, "final_score": (1 - alpha) * float(r["vector_score"])}

    max_bm25 = max((float(r["bm25_score"]) for r in bm25_results), default=1.0) or 1.0
    for r in bm25_results:
        rid = r["id"]
        norm_bm25 = float(r["bm25_score"]) / max_bm25
        if rid in scores:
            scores[rid]["final_score"] += alpha * norm_bm25
        else:
            scores[rid] = {**r, "vector_score": 0.0,
                           "final_score": alpha * norm_bm25}

    ranked = sorted(scores.values(), key=lambda x: x["final_score"], reverse=True)
    return ranked[:k]
