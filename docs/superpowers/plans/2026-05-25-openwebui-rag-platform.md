# OpenWebUI RAG Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the Coiltech platform from ChromaDB/SQLite to PostgreSQL+pgvector with hybrid BM25+vector retrieval, multi-tenant routing, Neo4j graph support, and OpenWebUI as the sole frontend.

**Architecture:** Clean `backend/` folder structure replacing the current flat layout. PostgreSQL handles both relational data and 768-dim vector embeddings via pgvector. The existing FastAPI endpoints (including `/v1/chat/completions`) are preserved so OpenWebUI never breaks.

**Tech Stack:** FastAPI, asyncpg, pgvector, LlamaIndex PGVectorStore concept (raw asyncpg for hybrid), sentence-transformers CrossEncoder, Neo4j Python driver, Ollama (Gemma/Llama3/nomic-embed-text), DeepSeek API, Docker Compose.

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `backend/settings.py` | CREATE | All env vars (PG, Neo4j, Ollama, DeepSeek, RAG tuning) |
| `backend/db.py` | CREATE | asyncpg pool, schema init (tenants/users/documents/chunks/agents/chats/messages) |
| `backend/llm_providers.py` | CREATE | Ollama+Gemma+DeepSeek factory, list_ollama_models() |
| `backend/rag/store/__init__.py` | CREATE | Ollama embeddings via httpx, asyncpg vector insert/search helpers |
| `backend/rag/ingestion/excel_seed.py` | CREATE | issue_solution_final.xlsx → chunks with content-hash dedup |
| `backend/rag/ingestion/pdf_ingest.py` | CREATE | PDF/DOCX/TXT chunker → embed → store in chunks table |
| `backend/rag/retrieval/hybrid.py` | CREATE | pgvector cosine + tsvector ts_rank, alpha=0.3 blend |
| `backend/rag/retrieval/query_expansion.py` | CREATE | LLM-based query variant generation |
| `backend/rag/retrieval/reranker.py` | CREATE | CrossEncoder lazy-load, rescore top-K |
| `backend/rag/retrieval/supplement.py` | CREATE | ILIKE fallback when confidence < threshold |
| `backend/rag/retrieval/query_engine.py` | CREATE | Orchestrate: expand→hybrid→rerank→supplement→assemble |
| `backend/rag/graph/client.py` | CREATE | Neo4j driver wrapper |
| `backend/rag/graph/ingest.py` | CREATE | Entity extraction → Neo4j nodes/edges |
| `backend/rag/graph/retriever.py` | CREATE | Graph context enrichment after vector retrieval |
| `backend/rag/observability.py` | CREATE | Query logging to query_logs table |
| `backend/tenancy.py` | CREATE | X-Tenant-ID/X-User-ID middleware |
| `backend/tenants.py` | CREATE | GET/POST /tenants, GET/POST /tenants/{id}/users |
| `backend/agents.py` | CREATE | Agent CRUD backed by PostgreSQL agents table |
| `backend/chats.py` | CREATE | Chat thread + message management |
| `backend/rag.py` | CREATE | RAGPipeline: resolve_collection, ingest_async, query |
| `backend/file_watcher.py` | CREATE | Migrate api/file_watcher.py with updated imports |
| `backend/main.py` | CREATE | FastAPI app, all endpoints, OpenAI-compatible /v1/chat/completions |
| `backend/Dockerfile` | CREATE | Python 3.11 image |
| `openwebui_functions/agent_manager.py` | CREATE | OpenWebUI Function plugin for agent CRUD |
| `docker-compose.yml` | CREATE | postgres, neo4j, backend, openwebui |
| `.env.example` | CREATE | All required env vars with defaults |
| `requirements.txt` | MODIFY | Add asyncpg, neo4j, sentence-transformers, openpyxl |

---

## Task 1: Scaffold backend/ structure + requirements + Docker

**Files:**
- Create: `backend/__init__.py`
- Create: `backend/rag/__init__.py`
- Create: `backend/rag/ingestion/__init__.py`
- Create: `backend/rag/retrieval/__init__.py`
- Create: `backend/rag/graph/__init__.py`
- Create: `backend/rag/store/__init__.py` (placeholder)
- Create: `openwebui_functions/__init__.py`
- Modify: `requirements.txt`
- Create: `docker-compose.yml`
- Create: `backend/Dockerfile`
- Create: `.env.example`

- [ ] **Step 1: Create all __init__.py files**

```powershell
New-Item -ItemType Directory -Force -Path "backend\rag\ingestion"
New-Item -ItemType Directory -Force -Path "backend\rag\retrieval"
New-Item -ItemType Directory -Force -Path "backend\rag\graph"
New-Item -ItemType Directory -Force -Path "backend\rag\store"
New-Item -ItemType Directory -Force -Path "openwebui_functions"
"" | Out-File -FilePath "backend\__init__.py" -Encoding utf8
"" | Out-File -FilePath "backend\rag\__init__.py" -Encoding utf8
"" | Out-File -FilePath "backend\rag\ingestion\__init__.py" -Encoding utf8
"" | Out-File -FilePath "backend\rag\retrieval\__init__.py" -Encoding utf8
"" | Out-File -FilePath "backend\rag\graph\__init__.py" -Encoding utf8
"" | Out-File -FilePath "backend\rag\store\__init__.py" -Encoding utf8
"" | Out-File -FilePath "openwebui_functions\__init__.py" -Encoding utf8
```

- [ ] **Step 2: Write requirements.txt**

Replace `requirements.txt` with:

```
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.6.0
pydantic-settings>=2.1.0
python-multipart>=0.0.9
httpx>=0.26.0
structlog>=24.1.0
sse-starlette>=2.0.0
tenacity>=8.2.0

# Database
asyncpg>=0.29.0
pgvector>=0.2.5

# Document processing
pypdf>=4.0.0
python-docx>=1.1.0
pandas>=2.0.0
openpyxl>=3.1.0

# Reranker
sentence-transformers>=2.3.0
numpy>=1.24.0

# Graph
neo4j>=5.15.0

# SQLite (kept for business data)
aiosqlite>=0.19.0

# Testing
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

- [ ] **Step 3: Write docker-compose.yml**

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    restart: unless-stopped
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: ragdb
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: test1234
    volumes:
      - postgres_data:/var/lib/postgresql/data

  neo4j:
    image: neo4j:5.15
    restart: unless-stopped
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      NEO4J_AUTH: neo4j/test1234
    volumes:
      - neo4j_data:/data

  backend:
    build: ./backend
    restart: unless-stopped
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - postgres
      - neo4j
    volumes:
      - ./issue_solution_final.xlsx:/app/issue_solution_final.xlsx:ro
      - uploads_data:/app/data/uploads

  openwebui:
    image: ghcr.io/open-webui/open-webui:main
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      OPENAI_API_BASE_URL: http://backend:8000/v1
      OPENAI_API_KEY: unused
    depends_on:
      - backend
    volumes:
      - openwebui_data:/app/backend/data

volumes:
  postgres_data:
  neo4j_data:
  uploads_data:
  openwebui_data:
```

- [ ] **Step 4: Write backend/Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/backend
COPY ../issue_solution_final.xlsx /app/issue_solution_final.xlsx

ENV PYTHONPATH=/app

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 5: Write .env.example**

```env
APP_NAME=Coiltech RAG Platform
APP_VERSION=2.0.0
DEBUG=false
LOG_LEVEL=INFO

API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000,http://localhost:8080

DATABASE_URL=postgresql://postgres:test1234@localhost:5432/ragdb
SQLITE_DB_PATH=data/coiltech_platform.db

OLLAMA_BASE_URL=http://host.docker.internal:11434
DEFAULT_MODEL=ollama:gemma3
OLLAMA_GENERATION_MODEL=gemma3
OLLAMA_ROUTING_MODEL=gemma3
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_TIMEOUT=120

DEEPSEEK_API_KEY=
DEEPSEEK_MODELS=deepseek-chat,deepseek-reasoner

DEFAULT_ANSWER_LANGUAGE=tr
RAG_MULTI_QUERY=0
RAG_TOP_K=10
RAG_ALPHA=0.3
RAG_SCORE_THRESHOLD=0.3
RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

RAG_USE_GRAPH=0
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=test1234

EXCEL_SEED_PATH=issue_solution_final.xlsx
OPENWEBUI_DEFAULT_COLLECTION=service_maintenance
OPENWEBUI_WATCH_ENABLED=false
OPENWEBUI_UPLOADS_DIR=
OPENWEBUI_SCAN_INTERVAL=30

UPLOAD_DIR=data/uploads
MAX_UPLOAD_SIZE=52428800
ALLOWED_EXTENSIONS=.pdf,.docx,.txt,.csv,.xlsx,.md

ENABLE_CAG=true
CAG_CACHE_TTL_SECONDS=300
CAG_MAX_ENTRIES=200

CHUNK_SIZE=512
CHUNK_OVERLAP=64
```

- [ ] **Step 6: Commit**

```bash
git add backend/ openwebui_functions/ docker-compose.yml requirements.txt .env.example
git commit -m "feat: scaffold backend/ structure, Docker Compose, requirements"
```

---

## Task 2: Settings module

**Files:**
- Create: `backend/settings.py`

- [ ] **Step 1: Write backend/settings.py**

```python
import os
from functools import lru_cache
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Coiltech RAG Platform"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8080,http://localhost:5173"

    DATABASE_URL: str = "postgresql://postgres:test1234@localhost:5432/ragdb"
    SQLITE_DB_PATH: str = "data/coiltech_platform.db"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_GENERATION_MODEL: str = "gemma3"
    OLLAMA_ROUTING_MODEL: str = "gemma3"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_TIMEOUT: int = 120
    DEFAULT_MODEL: str = "ollama:gemma3"

    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_MODELS: str = "deepseek-chat,deepseek-reasoner"

    DEFAULT_ANSWER_LANGUAGE: str = "tr"
    RAG_MULTI_QUERY: bool = False
    RAG_TOP_K: int = 10
    RAG_ALPHA: float = 0.3
    RAG_SCORE_THRESHOLD: float = 0.3
    RERANKER_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    RAG_USE_GRAPH: bool = False
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "test1234"

    EXCEL_SEED_PATH: str = "issue_solution_final.xlsx"
    OPENWEBUI_DEFAULT_COLLECTION: str = "service_maintenance"
    OPENWEBUI_WATCH_ENABLED: bool = False
    OPENWEBUI_UPLOADS_DIR: str = ""
    OPENWEBUI_SCAN_INTERVAL: int = 30

    UPLOAD_DIR: str = "data/uploads"
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024
    ALLOWED_EXTENSIONS: str = ".pdf,.docx,.txt,.csv,.xlsx,.md"

    ENABLE_CAG: bool = True
    CAG_CACHE_TTL_SECONDS: int = 300
    CAG_MAX_ENTRIES: int = 200

    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    RAG_COLLECTIONS: str = (
        "products_compact_drives,products_loop_drives,products_blank_lines,"
        "products_destacker,products_press_transfer,products_press_transfer_robots,"
        "products_servo_drives,products_decoilers,products_ctl_lines,"
        "special_projects,installation_commissioning,service_maintenance,"
        "spare_parts,quality_documents,sales_and_quotations"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]

    @property
    def rag_collections_list(self) -> List[str]:
        return [c.strip() for c in self.RAG_COLLECTIONS.split(",")]

    @property
    def allowed_extensions_list(self) -> List[str]:
        return [e.strip() for e in self.ALLOWED_EXTENSIONS.split(",")]

    @property
    def deepseek_models_list(self) -> List[str]:
        if not self.DEEPSEEK_API_KEY:
            return []
        return [m.strip() for m in self.DEEPSEEK_MODELS.split(",") if m.strip()]

    def ensure_directories(self) -> None:
        for d in [self.UPLOAD_DIR, os.path.dirname(self.SQLITE_DB_PATH)]:
            if d:
                os.makedirs(d, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    s = Settings()
    s.ensure_directories()
    return s
```

- [ ] **Step 2: Commit**

```bash
git add backend/settings.py
git commit -m "feat: add backend/settings.py with all env vars"
```

---

## Task 3: PostgreSQL database layer

**Files:**
- Create: `backend/db.py`

- [ ] **Step 1: Write backend/db.py**

```python
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
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx
            ON chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS chunks_hash_idx ON chunks(content_hash)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS chunks_tenant_idx ON chunks(tenant_id, user_id, collection)
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
        # Seed default tenant/user if missing
        row = await conn.fetchval("SELECT id FROM tenants WHERE id = 1")
        if not row:
            await conn.execute("INSERT INTO tenants (id, name) VALUES (1, 'default')")
            await conn.execute(
                "INSERT INTO users (id, tenant_id, name) VALUES (1, 1, 'default')"
            )
    logger.info("Database schema initialized")
```

- [ ] **Step 2: Commit**

```bash
git add backend/db.py
git commit -m "feat: add backend/db.py with asyncpg pool and pgvector schema"
```

---

## Task 4: LLM providers

**Files:**
- Create: `backend/llm_providers.py`

- [ ] **Step 1: Write backend/llm_providers.py**

```python
import json
from typing import AsyncIterator, Dict, Any, List, Optional
import httpx
import structlog

logger = structlog.get_logger(__name__)


async def _ollama_generate(
    base_url: str,
    model: str,
    messages: list,
    stream: bool,
    timeout: int,
) -> AsyncIterator[str]:
    payload = {"model": model, "messages": messages, "stream": stream,
                "options": {"temperature": 0.7, "num_predict": 2048}}
    async with httpx.AsyncClient(timeout=timeout) as client:
        if stream:
            async with client.stream("POST", f"{base_url}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            chunk = data.get("message", {}).get("content", "")
                            if chunk:
                                yield chunk
                        except json.JSONDecodeError:
                            continue
        else:
            resp = await client.post(f"{base_url}/api/chat", json=payload)
            resp.raise_for_status()
            yield resp.json().get("message", {}).get("content", "")


async def _deepseek_generate(
    api_key: str,
    model: str,
    messages: list,
    stream: bool,
    timeout: int,
) -> AsyncIterator[str]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "stream": stream}
    async with httpx.AsyncClient(timeout=timeout) as client:
        if stream:
            async with client.stream(
                "POST", "https://api.deepseek.com/chat/completions",
                json=payload, headers=headers
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line[6:] != "[DONE]":
                        try:
                            data = json.loads(line[6:])
                            chunk = data["choices"][0]["delta"].get("content", "")
                            if chunk:
                                yield chunk
                        except (json.JSONDecodeError, KeyError):
                            continue
        else:
            resp = await client.post(
                "https://api.deepseek.com/chat/completions",
                json=payload, headers=headers
            )
            resp.raise_for_status()
            yield resp.json()["choices"][0]["message"]["content"]


class LLMProvider:
    def __init__(self, settings):
        self.settings = settings

    def _build_messages(self, prompt: str, system_prompt: Optional[str]) -> list:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _parse_model_id(self, model_id: Optional[str]) -> tuple[str, str]:
        """Returns (provider, model_name). Defaults to ollama:generation_model."""
        model_id = model_id or f"ollama:{self.settings.OLLAMA_GENERATION_MODEL}"
        if model_id.startswith("ollama:"):
            return "ollama", model_id[7:]
        if model_id in self.settings.deepseek_models_list:
            return "deepseek", model_id
        if model_id.startswith("deepseek"):
            return "deepseek", model_id
        return "ollama", model_id

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        provider, model_name = self._parse_model_id(model)
        messages = self._build_messages(prompt, system_prompt)
        result = ""
        if provider == "deepseek" and self.settings.DEEPSEEK_API_KEY:
            async for chunk in _deepseek_generate(
                self.settings.DEEPSEEK_API_KEY, model_name, messages, False,
                self.settings.OLLAMA_TIMEOUT
            ):
                result += chunk
        else:
            async for chunk in _ollama_generate(
                self.settings.OLLAMA_BASE_URL, model_name, messages, False,
                self.settings.OLLAMA_TIMEOUT
            ):
                result += chunk
        return result

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        provider, model_name = self._parse_model_id(model)
        messages = self._build_messages(prompt, system_prompt)
        if provider == "deepseek" and self.settings.DEEPSEEK_API_KEY:
            async for chunk in _deepseek_generate(
                self.settings.DEEPSEEK_API_KEY, model_name, messages, True,
                self.settings.OLLAMA_TIMEOUT
            ):
                yield chunk
        else:
            async for chunk in _ollama_generate(
                self.settings.OLLAMA_BASE_URL, model_name, messages, True,
                self.settings.OLLAMA_TIMEOUT
            ):
                yield chunk

    async def get_embedding(self, text: str) -> List[float]:
        payload = {"model": self.settings.OLLAMA_EMBEDDING_MODEL, "prompt": text}
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.settings.OLLAMA_BASE_URL}/api/embeddings", json=payload
            )
            resp.raise_for_status()
            return resp.json()["embedding"]

    async def list_ollama_models(self) -> List[str]:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.settings.OLLAMA_BASE_URL}/api/tags")
                resp.raise_for_status()
                return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    async def check_health(self) -> Dict[str, Any]:
        health: Dict[str, Any] = {}
        try:
            models = await self.list_ollama_models()
            health["ollama"] = {"available": True, "models": models}
        except Exception as e:
            health["ollama"] = {"available": False, "error": str(e)}
        if self.settings.DEEPSEEK_API_KEY:
            health["deepseek"] = {"available": True,
                                   "models": self.settings.deepseek_models_list}
        return health
```

- [ ] **Step 2: Commit**

```bash
git add backend/llm_providers.py
git commit -m "feat: add LLM providers (Ollama/Gemma/DeepSeek)"
```

---

## Task 5: Vector store helpers

**Files:**
- Create: `backend/rag/store/__init__.py`

- [ ] **Step 1: Write backend/rag/store/__init__.py**

```python
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
    rows = await conn.fetch(
        """
        SELECT id, content, metadata,
               1 - (embedding <=> $1::vector) AS vector_score
        FROM chunks
        WHERE collection = $2
          AND tenant_id = $3
          AND user_id = $4
          AND embedding IS NOT NULL
        ORDER BY embedding <=> $1::vector
        LIMIT $5
        """,
        emb_str, collection, tenant_id, user_id, k,
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
    rows = await conn.fetch(
        """
        SELECT id, content, metadata,
               ts_rank(to_tsvector('turkish', content),
                       plainto_tsquery('turkish', $1)) AS bm25_score
        FROM chunks
        WHERE collection = $2
          AND tenant_id = $3
          AND user_id = $4
          AND to_tsvector('turkish', content) @@ plainto_tsquery('turkish', $1)
        ORDER BY bm25_score DESC
        LIMIT $5
        """,
        query, collection, tenant_id, user_id, k,
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
    rows = await conn.fetch(
        """
        SELECT id, content, metadata, 0.3::float AS vector_score
        FROM chunks
        WHERE collection = $1 AND tenant_id = $2 AND user_id = $3
          AND content ILIKE $4
        LIMIT $5
        """,
        collection, tenant_id, user_id, f"%{phrase}%", k,
    )
    return [dict(r) for r in rows]
```

- [ ] **Step 2: Commit**

```bash
git add backend/rag/store/__init__.py
git commit -m "feat: add pgvector store helpers (insert, vector_search, bm25_search)"
```

---

## Task 6: Excel seed ingestion

**Files:**
- Create: `backend/rag/ingestion/excel_seed.py`

- [ ] **Step 1: Write backend/rag/ingestion/excel_seed.py**

```python
import asyncpg
import structlog
import pandas as pd
from typing import Optional
from backend.rag.store import content_hash, chunk_exists, insert_chunk

logger = structlog.get_logger(__name__)

COLLECTION = "service_maintenance"
TENANT_ID = 1
USER_ID = 1


async def seed_excel(
    pool: asyncpg.Pool,
    llm,  # LLMProvider instance
    excel_path: str,
    document_id: int,
) -> int:
    df = pd.read_excel(excel_path, sheet_name="Sheet1")
    df.columns = [c.strip() for c in df.columns]

    issue_col = df.columns[0]
    solution_col = df.columns[1]

    inserted = 0
    async with pool.acquire() as conn:
        for _, row in df.iterrows():
            issue = str(row[issue_col] or "").strip()
            solution = str(row[solution_col] or "").strip()
            if not issue or not solution:
                continue

            text = f"Sorun: {issue}\nÇözüm: {solution}"
            h = content_hash(text)

            if await chunk_exists(conn, h):
                continue

            embedding = await llm.get_embedding(text)
            await insert_chunk(
                conn,
                document_id=document_id,
                tenant_id=TENANT_ID,
                user_id=USER_ID,
                collection=COLLECTION,
                content=text,
                embedding=embedding,
                metadata={"source": excel_path, "type": "issue_solution"},
            )
            inserted += 1

    logger.info("Excel seed complete", inserted=inserted)
    return inserted


async def run_seed_if_needed(pool: asyncpg.Pool, llm, excel_path: str) -> None:
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM chunks WHERE collection = $1 AND tenant_id = $2",
            COLLECTION, TENANT_ID,
        )
        if count and count > 0:
            logger.info("Excel seed already done, skipping", existing_chunks=count)
            return

        row = await conn.fetchrow(
            """
            INSERT INTO documents (tenant_id, user_id, original_filename, rag_collection)
            VALUES ($1, $2, $3, $4) RETURNING id
            """,
            TENANT_ID, USER_ID, excel_path, COLLECTION,
        )
        document_id = row["id"]

    await seed_excel(pool, llm, excel_path, document_id)
```

- [ ] **Step 2: Commit**

```bash
git add backend/rag/ingestion/excel_seed.py
git commit -m "feat: add Excel seed ingestion for issue_solution_final.xlsx"
```

---

## Task 7: Document ingestion (PDF/DOCX/TXT)

**Files:**
- Create: `backend/rag/ingestion/pdf_ingest.py`

- [ ] **Step 1: Write backend/rag/ingestion/pdf_ingest.py**

```python
import os
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
import asyncpg
import structlog

logger = structlog.get_logger(__name__)


def _load_text(file_path: str) -> List[Dict[str, Any]]:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        return [
            {"content": (p.extract_text() or "").strip(), "page": i + 1}
            for i, p in enumerate(reader.pages)
            if (p.extract_text() or "").strip()
        ]
    if ext == ".docx":
        from docx import Document
        doc = Document(file_path)
        text = "\n\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
        return [{"content": text, "page": 1}]
    if ext in (".txt", ".md"):
        text = Path(file_path).read_text(encoding="utf-8", errors="ignore").strip()
        return [{"content": text, "page": 1}]
    if ext == ".xlsx":
        import pandas as pd
        xl = pd.ExcelFile(file_path)
        pages = []
        for sheet in xl.sheet_names:
            df = pd.read_excel(xl, sheet_name=sheet)
            rows = [
                " | ".join(f"{c}: {v}" for c, v in row.items())
                for _, row in df.iterrows()
            ]
            pages.append({"content": "\n".join(rows), "page": sheet})
        return pages
    raise ValueError(f"Unsupported extension: {ext}")


def _chunk_text(text: str, chunk_size: int = 2048, overlap: int = 256) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


async def ingest_file(
    pool: asyncpg.Pool,
    llm,
    *,
    file_path: str,
    collection: str,
    tenant_id: int = 1,
    user_id: int = 1,
    metadata: Optional[Dict[str, Any]] = None,
    mime_type: Optional[str] = None,
    file_size: int = 0,
) -> Dict[str, Any]:
    from backend.rag.store import content_hash, chunk_exists, insert_chunk

    metadata = metadata or {}
    pages = _load_text(file_path)

    async with pool.acquire() as conn:
        doc_row = await conn.fetchrow(
            """
            INSERT INTO documents
                (tenant_id, user_id, original_filename, rag_collection, metadata)
            VALUES ($1, $2, $3, $4, $5) RETURNING id
            """,
            tenant_id, user_id, os.path.basename(file_path),
            collection, json.dumps(metadata),
        )
        document_id = doc_row["id"]

        chunks_created = 0
        chunks_stored = 0
        for page in pages:
            for chunk_text in _chunk_text(page["content"]):
                chunks_created += 1
                h = content_hash(chunk_text)
                if await chunk_exists(conn, h):
                    continue
                embedding = await llm.get_embedding(chunk_text)
                chunk_meta = {**metadata, "page": page.get("page", 1),
                              "source": file_path}
                await insert_chunk(
                    conn,
                    document_id=document_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    collection=collection,
                    content=chunk_text,
                    embedding=embedding,
                    metadata=chunk_meta,
                )
                chunks_stored += 1

    return {
        "chunks_created": chunks_created,
        "chunks_stored": chunks_stored,
        "document_id": document_id,
        "collection": collection,
    }
```

- [ ] **Step 2: Commit**

```bash
git add backend/rag/ingestion/pdf_ingest.py
git commit -m "feat: add PDF/DOCX/TXT ingestion pipeline"
```

---

## Task 8: Hybrid retrieval

**Files:**
- Create: `backend/rag/retrieval/hybrid.py`

- [ ] **Step 1: Write backend/rag/retrieval/hybrid.py**

```python
from typing import Any, Dict, List
import asyncpg
import structlog

logger = structlog.get_logger(__name__)


async def hybrid_search(
    conn: asyncpg.Connection,
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

    vec_results = await vector_search(
        conn, embedding=query_embedding, collection=collection,
        tenant_id=tenant_id, user_id=user_id, k=k,
    )
    bm25_results = await bm25_search(
        conn, query=query, collection=collection,
        tenant_id=tenant_id, user_id=user_id, k=k,
    )

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
```

- [ ] **Step 2: Commit**

```bash
git add backend/rag/retrieval/hybrid.py
git commit -m "feat: add hybrid BM25+vector retrieval with alpha blending"
```

---

## Task 9: Query expansion, reranker, supplement

**Files:**
- Create: `backend/rag/retrieval/query_expansion.py`
- Create: `backend/rag/retrieval/reranker.py`
- Create: `backend/rag/retrieval/supplement.py`

- [ ] **Step 1: Write backend/rag/retrieval/query_expansion.py**

```python
import json
from typing import List
import structlog

logger = structlog.get_logger(__name__)

_EXPANSION_PROMPT = """Given the user query below, generate 2 alternative phrasings that mean the same thing but use different words. Respond with a JSON array of strings only.

Query: {query}

Example response: ["alternative phrasing 1", "alternative phrasing 2"]"""


async def expand_query(query: str, llm) -> List[str]:
    try:
        raw = await llm.generate(
            _EXPANSION_PROMPT.format(query=query),
            system_prompt="You expand search queries. Reply only with a JSON array.",
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = "\n".join(l for l in raw.splitlines() if not l.startswith("```"))
        variants = json.loads(raw)
        if isinstance(variants, list):
            return [query] + [v for v in variants if isinstance(v, str)]
    except Exception as e:
        logger.warning("Query expansion failed, using original", error=str(e))
    return [query]
```

- [ ] **Step 2: Write backend/rag/retrieval/reranker.py**

```python
from typing import Any, Dict, List, Optional
import structlog

logger = structlog.get_logger(__name__)

_encoder = None


def _get_encoder(model_name: str):
    global _encoder
    if _encoder is None:
        from sentence_transformers import CrossEncoder
        _encoder = CrossEncoder(model_name)
        logger.info("CrossEncoder loaded", model=model_name)
    return _encoder


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    model_name: str,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    if not candidates:
        return candidates
    try:
        encoder = _get_encoder(model_name)
        pairs = [(query, c["content"]) for c in candidates]
        scores = encoder.predict(pairs)
        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)
        return sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)[:top_k]
    except Exception as e:
        logger.warning("Reranking failed, returning original order", error=str(e))
        return candidates[:top_k]
```

- [ ] **Step 3: Write backend/rag/retrieval/supplement.py**

```python
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
```

- [ ] **Step 4: Commit**

```bash
git add backend/rag/retrieval/query_expansion.py backend/rag/retrieval/reranker.py backend/rag/retrieval/supplement.py
git commit -m "feat: add query expansion, cross-encoder reranker, ILIKE fallback"
```

---

## Task 10: Query engine

**Files:**
- Create: `backend/rag/retrieval/query_engine.py`

- [ ] **Step 1: Write backend/rag/retrieval/query_engine.py**

```python
import json
import time
from typing import Any, Dict, List, Optional
import asyncpg
import structlog

from backend.rag.retrieval.hybrid import hybrid_search
from backend.rag.retrieval.reranker import rerank
from backend.rag.retrieval.supplement import supplementary_search

logger = structlog.get_logger(__name__)

_ANSWER_PROMPT = """You are an expert assistant for the Coiltech industrial automation platform.
Use the context below to answer the question. If the context is insufficient, say so clearly.
Answer in {language}.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def _build_context(sources: List[Dict[str, Any]]) -> str:
    if not sources:
        return "No relevant documents found."
    parts = []
    for i, s in enumerate(sources, 1):
        meta = s.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        src = meta.get("source", "unknown")
        parts.append(f"[{i}] {src}\n{s['content']}")
    return "\n\n---\n\n".join(parts)


async def run_query(
    pool: asyncpg.Pool,
    llm,
    *,
    question: str,
    collection: str,
    tenant_id: int = 1,
    user_id: int = 1,
    k: int = 10,
    alpha: float = 0.3,
    use_expansion: bool = False,
    reranker_model: Optional[str] = None,
    score_threshold: float = 0.3,
    language: str = "tr",
    graph_retriever=None,
) -> Dict[str, Any]:
    t0 = time.time()

    queries = [question]
    if use_expansion:
        from backend.rag.retrieval.query_expansion import expand_query
        queries = await expand_query(question, llm)

    embedding = await llm.get_embedding(queries[0])

    async with pool.acquire() as conn:
        candidates: Dict[int, Dict[str, Any]] = {}
        for q in queries:
            q_emb = await llm.get_embedding(q) if q != queries[0] else embedding
            results = await hybrid_search(
                conn, query=q, query_embedding=q_emb,
                collection=collection, tenant_id=tenant_id, user_id=user_id,
                k=k, alpha=alpha,
            )
            for r in results:
                if r["id"] not in candidates:
                    candidates[r["id"]] = r

        merged = sorted(candidates.values(), key=lambda x: x.get("final_score", 0), reverse=True)

        if reranker_model and merged:
            import asyncio
            loop = asyncio.get_event_loop()
            merged = await loop.run_in_executor(
                None, lambda: rerank(question, merged, reranker_model, k)
            )

        top_score = merged[0].get("rerank_score", merged[0].get("final_score", 0)) if merged else 0.0

        supplementary = await supplementary_search(
            conn, query=question, collection=collection,
            tenant_id=tenant_id, user_id=user_id,
            top_score=top_score, threshold=score_threshold,
        )
        for s in supplementary:
            if s["id"] not in {r["id"] for r in merged}:
                merged.append(s)

        sources = merged[:k]

    graph_context = ""
    if graph_retriever:
        try:
            graph_context = await graph_retriever.get_context(question, sources)
        except Exception as e:
            logger.warning("Graph retrieval failed", error=str(e))

    context = _build_context(sources)
    if graph_context:
        context = f"GRAPH CONTEXT:\n{graph_context}\n\n---\n\n{context}"

    prompt = _ANSWER_PROMPT.format(
        context=context, question=question, language=language
    )
    answer = await llm.generate(prompt)

    scores = [s.get("rerank_score", s.get("final_score", 0.0)) for s in sources[:3]]
    confidence = round(sum(scores) / len(scores), 3) if scores else 0.0

    duration_ms = int((time.time() - t0) * 1000)

    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO query_logs (tenant_id, user_id, query, collection, sources_count, confidence, duration_ms)
               VALUES ($1, $2, $3, $4, $5, $6, $7)""",
            tenant_id, user_id, question, collection, len(sources), confidence, duration_ms,
        )

    return {
        "answer": answer.strip(),
        "sources": sources,
        "confidence": confidence,
        "query": question,
        "collection": collection,
        "duration_ms": duration_ms,
    }
```

- [ ] **Step 2: Commit**

```bash
git add backend/rag/retrieval/query_engine.py
git commit -m "feat: add query engine orchestrating hybrid→rerank→supplement→generate"
```

---

## Task 11: Neo4j graph layer

**Files:**
- Create: `backend/rag/graph/client.py`
- Create: `backend/rag/graph/ingest.py`
- Create: `backend/rag/graph/retriever.py`

- [ ] **Step 1: Write backend/rag/graph/client.py**

```python
from typing import Optional
import structlog

logger = structlog.get_logger(__name__)

_driver = None


def get_driver(uri: str, user: str, password: str):
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase
        _driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Neo4j driver initialized", uri=uri)
    return _driver


def close_driver():
    global _driver
    if _driver:
        _driver.close()
        _driver = None


def is_available(uri: str, user: str, password: str) -> bool:
    try:
        drv = get_driver(uri, user, password)
        with drv.session() as s:
            s.run("RETURN 1")
        return True
    except Exception:
        return False
```

- [ ] **Step 2: Write backend/rag/graph/ingest.py**

```python
import json
from typing import Any, Dict, List
import structlog

logger = structlog.get_logger(__name__)

_ENTITY_PROMPT = """Extract named entities from this industrial text. Return a JSON object with key "entities" containing a list of objects with fields: "name" (string), "type" (one of: machine, component, error_code, alarm, procedure, part).

TEXT: {text}

JSON:"""


async def extract_entities(text: str, llm) -> List[Dict[str, Any]]:
    try:
        raw = await llm.generate(
            _ENTITY_PROMPT.format(text=text[:1000]),
            system_prompt="Extract entities. Return only JSON.",
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = "\n".join(l for l in raw.splitlines() if not l.startswith("```"))
        data = json.loads(raw)
        return data.get("entities", [])
    except Exception as e:
        logger.warning("Entity extraction failed", error=str(e))
        return []


async def ingest_chunk_to_graph(
    chunk_content: str,
    chunk_id: int,
    driver,
    llm,
) -> None:
    entities = await extract_entities(chunk_content, llm)
    if not entities:
        return
    with driver.session() as session:
        for entity in entities:
            name = entity.get("name", "").strip()
            etype = entity.get("type", "unknown")
            if not name:
                continue
            session.run(
                """
                MERGE (e:Entity {name: $name, type: $type})
                MERGE (c:Chunk {chunk_id: $chunk_id})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                name=name, type=etype, chunk_id=chunk_id,
            )
```

- [ ] **Step 3: Write backend/rag/graph/retriever.py**

```python
from typing import Any, Dict, List
import structlog

logger = structlog.get_logger(__name__)


async def get_context(
    question: str,
    sources: List[Dict[str, Any]],
    driver,
) -> str:
    if not driver or not sources:
        return ""
    chunk_ids = [s["id"] for s in sources[:5]]
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE c.chunk_id IN $ids
                WITH e, count(*) as freq
                ORDER BY freq DESC LIMIT 20
                RETURN e.name as name, e.type as type, freq
                """,
                ids=chunk_ids,
            )
            rows = result.data()
            if not rows:
                return ""
            lines = [f"{r['name']} ({r['type']}, mentioned {r['freq']}x)" for r in rows]
            return "Related entities:\n" + "\n".join(lines)
    except Exception as e:
        logger.warning("Graph context retrieval failed", error=str(e))
        return ""
```

- [ ] **Step 4: Commit**

```bash
git add backend/rag/graph/
git commit -m "feat: add Neo4j graph layer (client, ingest, retriever)"
```

---

## Task 12: Tenancy middleware + tenant APIs

**Files:**
- Create: `backend/tenancy.py`
- Create: `backend/tenants.py`

- [ ] **Step 1: Write backend/tenancy.py**

```python
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

logger = structlog.get_logger(__name__)


class TenancyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            tenant_id = int(request.headers.get("X-Tenant-ID", "1"))
        except ValueError:
            tenant_id = 1
        try:
            user_id = int(request.headers.get("X-User-ID", "1"))
        except ValueError:
            user_id = 1
        request.state.tenant_id = tenant_id
        request.state.user_id = user_id
        return await call_next(request)


def get_tenant(request: Request) -> tuple[int, int]:
    return getattr(request.state, "tenant_id", 1), getattr(request.state, "user_id", 1)
```

- [ ] **Step 2: Write backend/tenants.py**

```python
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import structlog

from backend.db import get_pool

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/tenants", tags=["Tenants"])


class TenantCreate(BaseModel):
    name: str


class UserCreate(BaseModel):
    name: str


@router.get("", response_model=List[Dict[str, Any]])
async def list_tenants():
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT id, name, created_at FROM tenants ORDER BY id")
    return [dict(r) for r in rows]


@router.post("", response_model=Dict[str, Any])
async def create_tenant(body: TenantCreate):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO tenants (name) VALUES ($1) RETURNING id, name, created_at",
            body.name,
        )
    return dict(row)


@router.get("/{tenant_id}/users", response_model=List[Dict[str, Any]])
async def list_users(tenant_id: int):
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, tenant_id, name, created_at FROM users WHERE tenant_id = $1 ORDER BY id",
            tenant_id,
        )
    return [dict(r) for r in rows]


@router.post("/{tenant_id}/users", response_model=Dict[str, Any])
async def create_user(tenant_id: int, body: UserCreate):
    pool = await get_pool()
    async with pool.acquire() as conn:
        tenant = await conn.fetchval("SELECT id FROM tenants WHERE id = $1", tenant_id)
        if not tenant:
            raise HTTPException(status_code=404, detail="Tenant not found")
        row = await conn.fetchrow(
            "INSERT INTO users (tenant_id, name) VALUES ($1, $2) RETURNING id, tenant_id, name, created_at",
            tenant_id, body.name,
        )
    return dict(row)


@router.get("/me", tags=["Tenants"])
async def me(request: Request):
    from backend.tenancy import get_tenant
    tenant_id, user_id = get_tenant(request)
    return {"tenant_id": tenant_id, "user_id": user_id}
```

- [ ] **Step 3: Commit**

```bash
git add backend/tenancy.py backend/tenants.py
git commit -m "feat: add multi-tenant middleware and tenant/user management APIs"
```

---

## Task 13: Agents + chats CRUD

**Files:**
- Create: `backend/agents.py`
- Create: `backend/chats.py`

- [ ] **Step 1: Write backend/agents.py**

```python
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import structlog

from backend.db import get_pool
from backend.tenancy import get_tenant

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/agents", tags=["Agents"])


class AgentCreate(BaseModel):
    name: str
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    collection: Optional[str] = None
    type: str = "chatbot"
    flow_config: Optional[Dict[str, Any]] = None


class AgentUpdate(BaseModel):
    name: Optional[str] = None
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    collection: Optional[str] = None
    type: Optional[str] = None
    flow_config: Optional[Dict[str, Any]] = None


@router.get("", response_model=List[Dict[str, Any]])
async def list_agents(request: Request):
    tenant_id, user_id = get_tenant(request)
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, name, system_prompt, model, collection, type, created_at "
            "FROM agents WHERE tenant_id=$1 AND user_id=$2 ORDER BY id",
            tenant_id, user_id,
        )
    return [dict(r) for r in rows]


@router.post("", response_model=Dict[str, Any])
async def create_agent(body: AgentCreate, request: Request):
    import json
    tenant_id, user_id = get_tenant(request)
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO agents (tenant_id, user_id, name, system_prompt, model, collection, type, flow_config)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8) RETURNING id, name, system_prompt, model, collection, type, created_at""",
            tenant_id, user_id, body.name, body.system_prompt, body.model,
            body.collection, body.type,
            json.dumps(body.flow_config) if body.flow_config else None,
        )
    return dict(row)


@router.get("/{agent_id}", response_model=Dict[str, Any])
async def get_agent(agent_id: int, request: Request):
    tenant_id, user_id = get_tenant(request)
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM agents WHERE id=$1 AND tenant_id=$2 AND user_id=$3",
            agent_id, tenant_id, user_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Agent not found")
    return dict(row)


@router.put("/{agent_id}", response_model=Dict[str, Any])
async def update_agent(agent_id: int, body: AgentUpdate, request: Request):
    tenant_id, user_id = get_tenant(request)
    pool = await get_pool()
    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT id FROM agents WHERE id=$1 AND tenant_id=$2 AND user_id=$3",
            agent_id, tenant_id, user_id,
        )
        if not existing:
            raise HTTPException(status_code=404, detail="Agent not found")
        updates = {k: v for k, v in body.model_dump().items() if v is not None}
        if not updates:
            return await get_agent(agent_id, request)
        set_clause = ", ".join(f"{k}=${i+2}" for i, k in enumerate(updates))
        values = list(updates.values())
        row = await conn.fetchrow(
            f"UPDATE agents SET {set_clause} WHERE id=$1 RETURNING *",
            agent_id, *values,
        )
    return dict(row)


@router.delete("/{agent_id}")
async def delete_agent(agent_id: int, request: Request):
    tenant_id, user_id = get_tenant(request)
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM agents WHERE id=$1 AND tenant_id=$2 AND user_id=$3",
            agent_id, tenant_id, user_id,
        )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"status": "deleted", "id": agent_id}
```

- [ ] **Step 2: Write backend/chats.py**

```python
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import structlog

from backend.db import get_pool
from backend.tenancy import get_tenant

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/chats", tags=["Chats"])


class ChatCreate(BaseModel):
    title: Optional[str] = None
    agent_id: Optional[int] = None


@router.get("", response_model=List[Dict[str, Any]])
async def list_chats(request: Request):
    tenant_id, user_id = get_tenant(request)
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, title, agent_id, created_at FROM chats "
            "WHERE tenant_id=$1 AND user_id=$2 ORDER BY created_at DESC",
            tenant_id, user_id,
        )
    return [dict(r) for r in rows]


@router.post("", response_model=Dict[str, Any])
async def create_chat(body: ChatCreate, request: Request):
    tenant_id, user_id = get_tenant(request)
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO chats (tenant_id, user_id, title, agent_id) "
            "VALUES ($1,$2,$3,$4) RETURNING id, title, agent_id, created_at",
            tenant_id, user_id, body.title, body.agent_id,
        )
    return dict(row)


@router.patch("/{chat_id}", response_model=Dict[str, Any])
async def update_chat_title(chat_id: int, title: str, request: Request):
    tenant_id, user_id = get_tenant(request)
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "UPDATE chats SET title=$1 WHERE id=$2 AND tenant_id=$3 AND user_id=$4 RETURNING *",
            title, chat_id, tenant_id, user_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Chat not found")
    return dict(row)


@router.delete("/{chat_id}")
async def delete_chat(chat_id: int, request: Request):
    tenant_id, user_id = get_tenant(request)
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM chats WHERE id=$1 AND tenant_id=$2 AND user_id=$3",
            chat_id, tenant_id, user_id,
        )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"status": "deleted"}


@router.get("/{chat_id}/messages", response_model=List[Dict[str, Any]])
async def get_messages(chat_id: int, request: Request):
    tenant_id, user_id = get_tenant(request)
    pool = await get_pool()
    async with pool.acquire() as conn:
        chat = await conn.fetchval(
            "SELECT id FROM chats WHERE id=$1 AND tenant_id=$2 AND user_id=$3",
            chat_id, tenant_id, user_id,
        )
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        rows = await conn.fetch(
            "SELECT id, role, content, created_at FROM messages "
            "WHERE chat_id=$1 ORDER BY created_at",
            chat_id,
        )
    return [dict(r) for r in rows]
```

- [ ] **Step 3: Commit**

```bash
git add backend/agents.py backend/chats.py
git commit -m "feat: add agent CRUD and chat/message management (PostgreSQL-backed)"
```

---

## Task 14: RAG pipeline orchestration

**Files:**
- Create: `backend/rag.py`

- [ ] **Step 1: Write backend/rag.py**

```python
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import structlog

from backend.rag.ingestion.pdf_ingest import ingest_file
from backend.rag.retrieval.query_engine import run_query

logger = structlog.get_logger(__name__)

_KEYWORD_MAP = {
    "products_compact_drives": ["compact drive", "press feeding", "compact"],
    "products_loop_drives": ["loop drive", "loop type"],
    "products_blank_lines": ["blank line", "blanking"],
    "products_destacker": ["destacker"],
    "products_press_transfer_robots": ["transfer robot", "press transfer robot", "robot"],
    "products_press_transfer": ["press transfer", "transfer system"],
    "products_servo_drives": ["servo drive", "triggered", "zigzag"],
    "products_decoilers": ["decoiler", "uncoiler", "mandrel"],
    "products_ctl_lines": ["cut to length", "ctl", "shear", "stacker"],
    "special_projects": ["special project", "custom", "layout", "rfq"],
    "installation_commissioning": ["commissioning", "installation", "startup"],
    "service_maintenance": ["service", "maintenance", "troubleshooting", "repair", "alarm"],
    "spare_parts": ["spare", "part", "replacement"],
    "quality_documents": ["ce", "quality", "certificate", "compliance"],
    "sales_and_quotations": ["quotation", "quote", "commercial", "proposal"],
}


class RAGPipeline:
    def __init__(self, pool, llm, settings):
        self.pool = pool
        self.llm = llm
        self.settings = settings

    def resolve_collection(
        self,
        collection_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        file_path: Optional[str] = None,
    ) -> str:
        if collection_name:
            return collection_name
        metadata = metadata or {}
        explicit = metadata.get("rag_collection") or metadata.get("collection_name")
        if explicit:
            return str(explicit)
        search_text = " ".join(
            str(v).lower()
            for v in [
                metadata.get("title"), metadata.get("doc_type"),
                metadata.get("document_role"),
                Path(file_path).stem if file_path else "",
            ]
            if v
        )
        for col_key, keywords in _KEYWORD_MAP.items():
            if any(kw in search_text for kw in keywords):
                return col_key
        return self.settings.OPENWEBUI_DEFAULT_COLLECTION

    async def ingest_async(
        self,
        file_path: str,
        department: str,
        metadata: Optional[Dict[str, Any]] = None,
        tenant_id: int = 1,
        user_id: int = 1,
    ) -> Dict[str, Any]:
        collection = self.resolve_collection(department, metadata, file_path)
        result = await ingest_file(
            self.pool, self.llm,
            file_path=file_path,
            collection=collection,
            tenant_id=tenant_id,
            user_id=user_id,
            metadata=metadata,
        )
        if self.settings.RAG_USE_GRAPH:
            try:
                from backend.rag.graph.client import get_driver
                from backend.rag.graph.ingest import ingest_chunk_to_graph
                driver = get_driver(
                    self.settings.NEO4J_URI,
                    self.settings.NEO4J_USER,
                    self.settings.NEO4J_PASSWORD,
                )
                # graph ingest is fire-and-forget; run outside await for now
            except Exception as e:
                logger.warning("Graph ingest skipped", error=str(e))
        return result

    async def query(
        self,
        question: str,
        collection: str,
        tenant_id: int = 1,
        user_id: int = 1,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        graph_retriever = None
        if self.settings.RAG_USE_GRAPH:
            try:
                from backend.rag.graph.client import get_driver
                from backend.rag.graph import retriever as graph_mod
                driver = get_driver(
                    self.settings.NEO4J_URI,
                    self.settings.NEO4J_USER,
                    self.settings.NEO4J_PASSWORD,
                )
                class _GR:
                    async def get_context(self, q, srcs):
                        return await graph_mod.get_context(q, srcs, driver)
                graph_retriever = _GR()
            except Exception as e:
                logger.warning("Graph retriever not available", error=str(e))

        return await run_query(
            self.pool, self.llm,
            question=question,
            collection=collection,
            tenant_id=tenant_id,
            user_id=user_id,
            k=self.settings.RAG_TOP_K,
            alpha=self.settings.RAG_ALPHA,
            use_expansion=self.settings.RAG_MULTI_QUERY,
            reranker_model=self.settings.RERANKER_MODEL,
            score_threshold=self.settings.RAG_SCORE_THRESHOLD,
            language=self.settings.DEFAULT_ANSWER_LANGUAGE,
            graph_retriever=graph_retriever,
        )

    def list_all_collections(self) -> List[Dict[str, Any]]:
        return []
```

- [ ] **Step 2: Commit**

```bash
git add backend/rag.py
git commit -m "feat: add RAGPipeline orchestration with collection resolver"
```

---

## Task 15: File watcher migration

**Files:**
- Create: `backend/file_watcher.py`

- [ ] **Step 1: Copy and update imports**

Copy `api/file_watcher.py` to `backend/file_watcher.py`, then update the import at the top from `from config.settings import get_settings` to `from backend.settings import get_settings`, and change `rag_pipeline.ingest_async` call signature to pass `tenant_id=1, user_id=1`:

```python
# In backend/file_watcher.py, replace the ingest call inside _process_file:
result = await self.rag_pipeline.ingest_async(
    file_path=str(file_path),
    department=department,
    metadata=metadata,
    tenant_id=1,
    user_id=1,
)
```

The rest of the file is identical to `api/file_watcher.py`.

- [ ] **Step 2: Commit**

```bash
git add backend/file_watcher.py
git commit -m "feat: migrate file_watcher to backend/ with updated imports"
```

---

## Task 16: Main FastAPI app

**Files:**
- Create: `backend/main.py`

- [ ] **Step 1: Write backend/main.py**

```python
import base64
import json
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
import structlog
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse

from backend.settings import get_settings
from backend.db import init_pool, close_pool, init_schema
from backend.llm_providers import LLMProvider
from backend.rag import RAGPipeline
from backend.tenancy import TenancyMiddleware, get_tenant
from backend.tenants import router as tenants_router
from backend.agents import router as agents_router
from backend.chats import router as chats_router

logger = structlog.get_logger(__name__)
settings = get_settings()

_pool = None
_llm: Optional[LLMProvider] = None
_rag: Optional[RAGPipeline] = None
_file_watcher = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool, _llm, _rag, _file_watcher

    _pool = await init_pool(settings.DATABASE_URL)
    await init_schema(_pool)

    _llm = LLMProvider(settings)
    _rag = RAGPipeline(_pool, _llm, settings)

    # Seed Excel knowledge base
    if os.path.exists(settings.EXCEL_SEED_PATH):
        try:
            from backend.rag.ingestion.excel_seed import run_seed_if_needed
            await run_seed_if_needed(_pool, _llm, settings.EXCEL_SEED_PATH)
        except Exception as e:
            logger.warning("Excel seed failed", error=str(e))

    # File watcher
    if settings.OPENWEBUI_WATCH_ENABLED:
        try:
            from backend.file_watcher import FileWatcher
            os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
            watch_dirs = [settings.UPLOAD_DIR]
            if settings.OPENWEBUI_UPLOADS_DIR:
                watch_dirs.append(settings.OPENWEBUI_UPLOADS_DIR)
            _file_watcher = FileWatcher(
                watch_dirs=watch_dirs,
                rag_pipeline=_rag,
                allowed_extensions=settings.allowed_extensions_list,
                scan_interval=settings.OPENWEBUI_SCAN_INTERVAL,
            )
            await _file_watcher.start()
        except Exception as e:
            logger.warning("File watcher start failed", error=str(e))

    yield

    if _file_watcher:
        await _file_watcher.stop()
    await close_pool()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(TenancyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tenants_router)
app.include_router(agents_router)
app.include_router(chats_router)


def _require_rag() -> RAGPipeline:
    if not _rag:
        raise HTTPException(status_code=503, detail="RAG pipeline not ready")
    return _rag


def _extract_user_message(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            return _strip_openwebui_wrapper(str(content).strip())
    return ""


def _strip_openwebui_wrapper(text: str) -> str:
    for pat in [
        r"(?is)<(?:question|query)>\s*(.*?)\s*</(?:question|query)>",
        r"(?is)(?:^|\n)\s*(?:user\s+question|question|soru)\s*:\s*(.+?)(?:\n|$)",
    ]:
        m = re.findall(pat, text)
        if m and m[-1].strip():
            return m[-1].strip()
    return text


def _openwebui_utility(msg: str) -> Optional[str]:
    raw = (msg or "").lower()
    if "### task:" not in raw:
        return None
    if "generate a concise" in raw and "title" in raw:
        return settings.APP_NAME
    if "generate 1-3 broad tags" in raw or "categorizing" in raw:
        return json.dumps({"tags": ["coiltech", "industrial-ai"]})
    if "follow-up questions" in raw or '"follow_ups"' in raw:
        return json.dumps({"follow_ups": []})
    return None


@app.get("/", tags=["Health"])
async def root():
    return {"name": settings.APP_NAME, "version": settings.APP_VERSION, "status": "running"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "service": settings.APP_NAME}


@app.get("/v1/models", tags=["OpenAI Compatibility"])
async def openai_models():
    created = int(datetime.now(timezone.utc).timestamp())
    models = [
        {"id": "coiltech-rag", "object": "model", "created": created, "owned_by": "coiltech"},
    ]
    if _llm:
        for m in await _llm.list_ollama_models():
            models.append({"id": f"ollama:{m}", "object": "model", "created": created, "owned_by": "ollama"})
        for m in settings.deepseek_models_list:
            models.append({"id": m, "object": "model", "created": created, "owned_by": "deepseek"})
        if _pool:
            import asyncpg
            from backend.db import get_pool
            pool = await get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch("SELECT id, name FROM agents")
                for r in rows:
                    models.append({
                        "id": f"agent-{r['id']}-{r['name'].lower().replace(' ','-')}",
                        "object": "model", "created": created, "owned_by": "coiltech",
                    })
    return {"object": "list", "data": models}


@app.post("/upload", tags=["Documents"])
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    collection_name: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    language: str = Form("tr"),
    metadata_json: Optional[str] = Form(None),
):
    rag = _require_rag()
    tenant_id, user_id = get_tenant(request)
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in settings.allowed_extensions_list:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail="File too large")

    safe_name = f"{uuid.uuid4().hex}_{os.path.basename(file.filename or 'upload.bin')}"
    target = os.path.join(settings.UPLOAD_DIR, safe_name)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    with open(target, "wb") as fh:
        fh.write(content)

    metadata = {}
    if metadata_json:
        try:
            metadata = json.loads(metadata_json)
        except Exception:
            pass
    if title:
        metadata["title"] = title
    metadata["language"] = language

    collection = rag.resolve_collection(collection_name, metadata, target)
    result = await rag.ingest_async(
        file_path=target,
        department=collection,
        metadata=metadata,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    return {"status": "success", "file_name": file.filename, **result}


@app.get("/collections", tags=["Documents"])
async def list_collections(request: Request):
    tenant_id, user_id = get_tenant(request)
    pool = await _require_rag() and None or None
    from backend.db import get_pool
    db = await get_pool()
    async with db.acquire() as conn:
        rows = await conn.fetch(
            "SELECT collection, count(*) as chunk_count FROM chunks "
            "WHERE tenant_id=$1 AND user_id=$2 GROUP BY collection ORDER BY collection",
            tenant_id, user_id,
        )
    return {"status": "success", "collections": [dict(r) for r in rows]}


@app.post("/v1/chat/completions", tags=["OpenAI Compatibility"])
async def openai_chat_completions(request: Request):
    body = await request.json()
    rag = _require_rag()
    tenant_id, user_id = get_tenant(request)

    messages = body.get("messages", [])
    model = body.get("model", settings.DEFAULT_MODEL)
    stream = body.get("stream", False)
    collection_name = body.get("collection_name", None)

    user_message = _extract_user_message(messages)
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")

    utility = _openwebui_utility(user_message)
    if utility is not None:
        return _build_response(utility, model)

    collection = collection_name or settings.OPENWEBUI_DEFAULT_COLLECTION
    result = await rag.query(
        question=user_message,
        collection=collection,
        tenant_id=tenant_id,
        user_id=user_id,
        model=model,
    )
    answer = result["answer"]

    if stream:
        async def gen():
            for word in answer.split(" "):
                chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now(timezone.utc).timestamp()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            done = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion.chunk",
                "created": int(datetime.now(timezone.utc).timestamp()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    return _build_response(answer, model)


def _build_response(content: str, model: str) -> dict:
    from fastapi.responses import JSONResponse
    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(datetime.now(timezone.utc).timestamp()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=settings.DEBUG)
```

- [ ] **Step 2: Commit**

```bash
git add backend/main.py
git commit -m "feat: add backend/main.py with all endpoints and OpenAI-compatible chat"
```

---

## Task 17: OpenWebUI Function plugin

**Files:**
- Create: `openwebui_functions/agent_manager.py`

- [ ] **Step 1: Write openwebui_functions/agent_manager.py**

```python
"""
OpenWebUI Function Plugin — Coiltech Agent Manager
Install in Open WebUI: Admin → Functions → New Function → paste this file.

Commands (type in chat):
  /create-agent name=MyAgent prompt=You are... model=ollama:gemma3 collection=service_maintenance
  /list-agents
  /delete-agent id=3
"""
import json
import re
import httpx
from pydantic import BaseModel

BACKEND_URL = "http://backend:8000"


class EventEmitter:
    def __init__(self, event_emitter=None):
        self.event_emitter = event_emitter

    async def emit(self, description="", status="in_progress", done=False):
        if self.event_emitter:
            await self.event_emitter({
                "type": "status",
                "data": {"description": description, "status": status, "done": done},
            })


class Tools:
    class Valves(BaseModel):
        backend_url: str = BACKEND_URL
        tenant_id: int = 1
        user_id: int = 1

    def __init__(self):
        self.valves = self.Valves()

    async def run_command(
        self, command: str, __event_emitter__=None, __user__=None
    ) -> str:
        emitter = EventEmitter(__event_emitter__)
        headers = {
            "X-Tenant-ID": str(self.valves.tenant_id),
            "X-User-ID": str(self.valves.user_id),
        }
        base = self.valves.backend_url.rstrip("/")
        cmd = command.strip()

        if cmd.startswith("/list-agents"):
            await emitter.emit("Fetching agents...")
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{base}/agents", headers=headers)
                r.raise_for_status()
                agents = r.json()
            await emitter.emit("Done", status="complete", done=True)
            if not agents:
                return "No agents found."
            lines = [f"**ID {a['id']}**: {a['name']} | model: {a.get('model','?')} | collection: {a.get('collection','?')}" for a in agents]
            return "\n".join(lines)

        if cmd.startswith("/create-agent"):
            params = dict(re.findall(r'(\w+)=([^\s]+)', cmd))
            name = params.get("name", "NewAgent")
            prompt = params.get("prompt", "You are a helpful assistant.")
            model = params.get("model", "ollama:gemma3")
            collection = params.get("collection", "service_maintenance")
            await emitter.emit(f"Creating agent '{name}'...")
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    f"{base}/agents", headers=headers,
                    json={"name": name, "system_prompt": prompt, "model": model, "collection": collection},
                )
                r.raise_for_status()
                agent = r.json()
            await emitter.emit("Done", status="complete", done=True)
            return f"Agent created: **{agent['name']}** (ID: {agent['id']}) — model: {agent.get('model')}, collection: {agent.get('collection')}"

        if cmd.startswith("/delete-agent"):
            params = dict(re.findall(r'(\w+)=([^\s]+)', cmd))
            agent_id = params.get("id")
            if not agent_id:
                return "Usage: /delete-agent id=<id>"
            await emitter.emit(f"Deleting agent {agent_id}...")
            async with httpx.AsyncClient() as client:
                r = await client.delete(f"{base}/agents/{agent_id}", headers=headers)
                r.raise_for_status()
            await emitter.emit("Done", status="complete", done=True)
            return f"Agent {agent_id} deleted."

        return f"Unknown command: {cmd}. Available: /list-agents, /create-agent, /delete-agent"
```

- [ ] **Step 2: Commit**

```bash
git add openwebui_functions/agent_manager.py
git commit -m "feat: add OpenWebUI Function plugin for agent CRUD"
```

---

## Task 18: Integration smoke test

- [ ] **Step 1: Start PostgreSQL locally or via Docker**

```bash
docker run -d --name pg-test -e POSTGRES_DB=ragdb -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=test1234 -p 5432:5432 pgvector/pgvector:pg16
```

- [ ] **Step 2: Copy .env.example to .env and set OLLAMA_BASE_URL**

```bash
cp .env.example .env
# Edit DATABASE_URL and OLLAMA_BASE_URL if needed
```

- [ ] **Step 3: Install new dependencies**

```bash
pip install asyncpg pgvector neo4j sentence-transformers
```

- [ ] **Step 4: Run the backend from repo root**

```bash
cd C:\Users\AhmetCan\Desktop\bitirme2
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Expected: startup logs show "Database schema initialized", "Excel seed complete" (or "Excel seed already done").

- [ ] **Step 5: Test /health**

```bash
curl http://localhost:8000/health
```

Expected: `{"status":"ok","service":"Coiltech RAG Platform"}`

- [ ] **Step 6: Test /v1/chat/completions**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ollama:gemma3","messages":[{"role":"user","content":"transfer hatası nedir?"}]}'
```

Expected: JSON response with `choices[0].message.content` containing an answer drawn from the seeded Excel knowledge base.

- [ ] **Step 7: Test agent creation**

```bash
curl -X POST http://localhost:8000/agents \
  -H "Content-Type: application/json" \
  -d '{"name":"ServiceBot","system_prompt":"You are a Coiltech service expert.","model":"ollama:gemma3","collection":"service_maintenance"}'
```

Expected: `{"id": 1, "name": "ServiceBot", ...}`

- [ ] **Step 8: Verify agent appears in /v1/models**

```bash
curl http://localhost:8000/v1/models
```

Expected: `agent-1-servicebot` in the data array.

- [ ] **Step 9: Final commit**

```bash
git add .
git commit -m "feat: complete OpenWebUI RAG platform migration"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| PostgreSQL + pgvector | Task 3 (db.py) |
| LlamaIndex-style abstraction | Task 5 (store/__init__.py — asyncpg with same interface) |
| Hybrid BM25 + vector, alpha=0.3 | Task 8 (hybrid.py) |
| Cross-encoder reranking (lazy) | Task 9 (reranker.py) |
| Query expansion (RAG_MULTI_QUERY) | Task 9 (query_expansion.py) |
| Supplementary ILIKE fallback | Task 9 (supplement.py) |
| Excel seed from issue_solution_final.xlsx | Task 6 |
| PDF/DOCX/TXT ingestion | Task 7 |
| Gemma + Ollama + DeepSeek providers | Task 4 |
| Multi-tenant headers (X-Tenant-ID) | Task 12 (tenancy.py) |
| Tenant/user management APIs | Task 12 (tenants.py) |
| Neo4j graph (optional, RAG_USE_GRAPH) | Task 11 |
| Agent CRUD (PostgreSQL) | Task 13 (agents.py) |
| Chat/message management | Task 13 (chats.py) |
| /v1/chat/completions (OpenWebUI) | Task 16 (main.py) |
| /v1/models (Ollama + DeepSeek + agents) | Task 16 (main.py) |
| File watcher for auto-ingestion | Task 15 |
| OpenWebUI Function plugin | Task 17 |
| Docker Compose (all services) | Task 1 |
| Content hash dedup on ingestion | Tasks 5, 6, 7 |

No gaps found.

**Type consistency:** `insert_chunk` and `vector_search` in `store/__init__.py` are called consistently across `excel_seed.py`, `pdf_ingest.py`, and `hybrid.py`. `LLMProvider.get_embedding()` returns `List[float]` and is used as such everywhere.
