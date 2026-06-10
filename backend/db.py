import asyncpg
import structlog
from typing import Optional

logger = structlog.get_logger(__name__)

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call init_pool() first.")
    return _pool


async def init_pool(dsn: str) -> asyncpg.Pool:
    global _pool
    _pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
    logger.info("Database pool created")
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("Database pool closed")


async def init_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tenants (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                tenant_id INTEGER REFERENCES tenants(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                tenant_id INTEGER NOT NULL DEFAULT 1,
                user_id INTEGER NOT NULL DEFAULT 1,
                original_filename TEXT,
                rag_collection TEXT,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                tenant_id INTEGER NOT NULL DEFAULT 1,
                user_id INTEGER NOT NULL DEFAULT 1,
                collection TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                embedding vector(768),
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        # HNSW instead of IVFFlat: IVFFlat trains its clusters from rows present
        # at index creation, and this index used to be built on an empty table.
        await conn.execute("DROP INDEX IF EXISTS chunks_embedding_idx")
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
            ON chunks USING hnsw (embedding vector_cosine_ops)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS chunks_hash_idx ON chunks(content_hash)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS chunks_tenant_idx ON chunks(tenant_id, user_id, collection)
        """)
        # Precomputed tsvector + GIN index so BM25 search doesn't recompute
        # to_tsvector over every row on each query.
        await conn.execute("""
            ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector
            GENERATED ALWAYS AS (to_tsvector('turkish'::regconfig, content)) STORED
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (content_tsv)
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id SERIAL PRIMARY KEY,
                tenant_id INTEGER NOT NULL DEFAULT 1,
                user_id INTEGER NOT NULL DEFAULT 1,
                name TEXT NOT NULL,
                system_prompt TEXT,
                model TEXT,
                collection TEXT,
                type TEXT DEFAULT 'chatbot',
                flow_config JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute(
            "ALTER TABLE agents ADD COLUMN IF NOT EXISTS embedding VECTOR(768)"
        )
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id SERIAL PRIMARY KEY,
                tenant_id INTEGER NOT NULL DEFAULT 1,
                user_id INTEGER NOT NULL DEFAULT 1,
                agent_id INTEGER REFERENCES agents(id) ON DELETE SET NULL,
                title TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                chat_id INTEGER REFERENCES chats(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
                id SERIAL PRIMARY KEY,
                tenant_id INTEGER NOT NULL DEFAULT 1,
                user_id INTEGER NOT NULL DEFAULT 1,
                query TEXT NOT NULL,
                collection TEXT,
                sources_count INTEGER,
                confidence FLOAT,
                duration_ms INTEGER,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS answer_cache (
                id SERIAL PRIMARY KEY,
                tenant_id INTEGER NOT NULL DEFAULT 1,
                user_id INTEGER NOT NULL DEFAULT 1,
                collection TEXT NOT NULL,
                question TEXT NOT NULL,
                question_embedding vector(768),
                answer TEXT NOT NULL,
                sources JSONB DEFAULT '[]',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS answer_cache_embedding_idx
            ON answer_cache USING hnsw (question_embedding vector_cosine_ops)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS answer_cache_tenant_idx
            ON answer_cache (tenant_id, user_id, collection, created_at DESC)
        """)
        await conn.execute(
            "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS cached BOOLEAN DEFAULT FALSE"
        )
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_agent_requests (
                id SERIAL PRIMARY KEY,
                tenant_id INTEGER NOT NULL DEFAULT 1,
                user_id INTEGER NOT NULL DEFAULT 1,
                raw_message TEXT NOT NULL,
                params JSONB DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                fulfilled_agent_id INTEGER,
                fulfilled_agent_name TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                fulfilled_at TIMESTAMPTZ
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS service_records (
                id SERIAL PRIMARY KEY,
                tespit_edilenler TEXT NOT NULL,
                yapilan_islem TEXT NOT NULL
            )
        """)
        # ASCII-folded shadow columns: queries match regardless of Turkish
        # vs ASCII spelling ("hatası" vs "hatasi") and DB locale quirks.
        await conn.execute("""
            ALTER TABLE service_records ADD COLUMN IF NOT EXISTS tespit_norm TEXT
            GENERATED ALWAYS AS (
                lower(translate(tespit_edilenler, 'ıİşŞğĞüÜöÖçÇ', 'iIsSgGuUoOcC'))
            ) STORED
        """)
        await conn.execute("""
            ALTER TABLE service_records ADD COLUMN IF NOT EXISTS islem_norm TEXT
            GENERATED ALWAYS AS (
                lower(translate(yapilan_islem, 'ıİşŞğĞüÜöÖçÇ', 'iIsSgGuUoOcC'))
            ) STORED
        """)
        row = await conn.fetchval("SELECT id FROM tenants WHERE id = 1")
        if not row:
            await conn.execute("INSERT INTO tenants (id, name) VALUES (1, 'default')")
            await conn.execute(
                "INSERT INTO users (id, tenant_id, name) VALUES (1, 1, 'default')"
            )
    logger.info("Database schema initialized")
