import hashlib
import json
from typing import Any, Dict, List, Optional
import asyncpg
import structlog

logger = structlog.get_logger(__name__)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:32]


async def chunk_exists(conn: asyncpg.Connection, h: str) -> bool:
    row = await conn.fetchval("SELECT 1 FROM chunks WHERE content_hash = $1", h)
    return row is not None


async def insert_chunk(
    conn: asyncpg.Connection,
    *,
    document_id: int,
    tenant_id: int,
    user_id: int,
    collection: str,
    content: str,
    embedding: List[float],
    metadata: Dict[str, Any],
) -> int:
    h = content_hash(content)
    if await chunk_exists(conn, h):
        row = await conn.fetchrow(
            "SELECT id FROM chunks WHERE content_hash = $1", h
        )
        return row["id"]
    emb_str = "[" + ",".join(str(v) for v in embedding) + "]"
    row = await conn.fetchrow(
        """
        INSERT INTO chunks
            (document_id, tenant_id, user_id, collection, content, content_hash, embedding, metadata)
        VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8)
        RETURNING id
        """,
        document_id, tenant_id, user_id, collection, content, h,
        emb_str, json.dumps(metadata),
    )
    return row["id"]


def _split_collection(collection: str) -> List[str]:
    """Support comma-separated multi-collection strings (e.g. 'col1,col2')."""
    return [c.strip() for c in collection.split(",") if c.strip()]


async def vector_search(
    conn: asyncpg.Connection,
    *,
    embedding: List[float],
    collection: str,
    tenant_id: int,
    user_id: int,
    k: int = 10,
) -> List[Dict[str, Any]]:
    emb_str = "[" + ",".join(str(v) for v in embedding) + "]"
    cols = _split_collection(collection)
    rows = await conn.fetch(
        """
        SELECT id, content, metadata,
               1 - (embedding <=> $1::vector) AS vector_score
        FROM chunks
        WHERE collection = ANY($2)
          AND tenant_id = $3
          AND user_id = $4
          AND embedding IS NOT NULL
        ORDER BY embedding <=> $1::vector
        LIMIT $5
        """,
        emb_str, cols, tenant_id, user_id, k,
    )
    return [dict(r) for r in rows]


async def bm25_search(
    conn: asyncpg.Connection,
    *,
    query: str,
    collection: str,
    tenant_id: int,
    user_id: int,
    k: int = 10,
) -> List[Dict[str, Any]]:
    cols = _split_collection(collection)
    rows = await conn.fetch(
        """
        SELECT id, content, metadata,
               ts_rank(content_tsv, plainto_tsquery('turkish', $1)) AS bm25_score
        FROM chunks
        WHERE collection = ANY($2)
          AND tenant_id = $3
          AND user_id = $4
          AND content_tsv @@ plainto_tsquery('turkish', $1)
        ORDER BY bm25_score DESC
        LIMIT $5
        """,
        query, cols, tenant_id, user_id, k,
    )
    return [dict(r) for r in rows]


async def ilike_search(
    conn: asyncpg.Connection,
    *,
    phrase: str,
    collection: str,
    tenant_id: int,
    user_id: int,
    k: int = 5,
) -> List[Dict[str, Any]]:
    cols = _split_collection(collection)
    rows = await conn.fetch(
        """
        SELECT id, content, metadata, 0.3::float AS vector_score
        FROM chunks
        WHERE collection = ANY($1) AND tenant_id = $2 AND user_id = $3
          AND content ILIKE $4
        LIMIT $5
        """,
        cols, tenant_id, user_id, f"%{phrase}%", k,
    )
    return [dict(r) for r in rows]
