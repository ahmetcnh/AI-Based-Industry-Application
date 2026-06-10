"""Semantic answer cache backed by pgvector cosine similarity."""
import json
from typing import Any, Dict, List, Optional


async def cache_lookup(
    pool,
    llm,
    question: str,
    tenant_id: int,
    user_id: int,
    collection: str,
    ttl_seconds: int,
    threshold: float = 0.95,
) -> Optional[Dict[str, Any]]:
    """Return cached answer dict or None."""
    try:
        emb = await llm.get_embedding(question)
        emb_str = "[" + ",".join(str(v) for v in emb) + "]"
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT answer, sources, created_at
                FROM answer_cache
                WHERE tenant_id=$1 AND user_id=$2 AND collection=$3
                  AND 1 - (question_embedding <=> $4::vector) >= $5
                  AND created_at > NOW() - ($6 || ' seconds')::INTERVAL
                ORDER BY created_at DESC
                LIMIT 1
                """,
                tenant_id, user_id, collection, emb_str,
                threshold, str(ttl_seconds),
            )
        if row:
            sources = row["sources"]
            if isinstance(sources, str):
                sources = json.loads(sources)
            return {"answer": row["answer"], "sources": sources or []}
    except Exception:
        pass
    return None


async def cache_store(
    pool,
    llm,
    question: str,
    answer: str,
    sources: list,
    tenant_id: int,
    user_id: int,
    collection: str,
) -> None:
    try:
        emb = await llm.get_embedding(question)
        emb_str = "[" + ",".join(str(v) for v in emb) + "]"
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO answer_cache
                    (tenant_id, user_id, collection, question, question_embedding, answer, sources)
                VALUES ($1,$2,$3,$4,$5::vector,$6,$7)
                """,
                tenant_id, user_id, collection, question, emb_str,
                answer, json.dumps(sources),
            )
    except Exception:
        pass


async def cache_invalidate_collection(pool, collection: str) -> None:
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM answer_cache WHERE collection=$1", collection
            )
    except Exception:
        pass
