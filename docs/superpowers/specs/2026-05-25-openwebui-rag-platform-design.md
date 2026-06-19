# Design Spec: OpenWebUI-Integrated RAG Platform

**Date:** 2026-05-25  
**Status:** Approved  

---

## Overview

Rebuild the Coiltech Industrial Automation Platform backend to match the architecture described in `PROJECT_OVERVIEW.md`, using Open WebUI as the sole chat frontend. The custom React frontend is removed. The backend exposes an OpenAI-compatible `/v1/chat/completions` endpoint that Open WebUI connects to directly.

---

## 1. Project Structure

The current flat layout (`api/`, `rag/`, `llm/`, `database/`, `agents/`, `config/`) is migrated into a clean `backend/` folder and then the old directories are removed.

```
bitirme2/
├── backend/
│   ├── main.py                        # FastAPI app + lifespan
│   ├── db.py                          # PostgreSQL pool + schema init
│   ├── rag.py                         # RAG pipeline orchestration entry point
│   ├── agents.py                      # Agent CRUD (DB layer)
│   ├── chats.py                       # Chat thread/message management
│   ├── tenancy.py                     # Multi-tenant context resolution
│   ├── tenants.py                     # Tenant/user management APIs
│   ├── llm_providers.py               # LLM abstraction (Ollama, Gemma, DeepSeek)
│   ├── prompts.py                     # System prompts and formatting rules
│   ├── flows/
│   │   └── executor.py                # Visual flow node pipeline executor
│   └── rag/
│       ├── ingestion/
│       │   ├── excel_seed.py          # One-time seed from issue_solution_final.xlsx
│       │   ├── pdf_ingest.py          # Chunking, embedding, metadata for PDFs/DOCX/etc.
│       │   ├── pdf_extract.py         # Text, table, image extraction
│       │   └── connectors/            # SQL and REST API loaders
│       ├── retrieval/
│       │   ├── hybrid.py              # Vector + BM25 blended search (alpha=0.3)
│       │   ├── query_engine.py        # Main RAG pipeline
│       │   ├── reranker.py            # Cross-encoder scoring (lazy-loaded)
│       │   ├── query_expansion.py     # Multi-query generation
│       │   ├── supplement.py          # Fallback retrieval (ILIKE phrase search)
│       │   └── scoring.py             # TOC/junk filtering
│       ├── graph/
│       │   ├── ingest.py              # Entity extraction → Neo4j nodes/edges
│       │   ├── retriever.py           # Graph context enrichment
│       │   └── client.py              # Neo4j driver wrapper
│       ├── store/                     # LlamaIndex PGVectorStore wrapper
│       └── observability.py           # Query logging and analytics
├── openwebui_functions/
│   └── agent_manager.py               # Open WebUI Function plugin for agent CRUD
├── docker-compose.yml
├── .env.example
├── issue_solution_final.xlsx          # Seed knowledge base (Turkish issue/solution pairs)
└── PROJECT_OVERVIEW.md
```

---

## 2. Data Layer

### Database: PostgreSQL 16 + pgvector

One PostgreSQL instance replaces both SQLite (relational) and ChromaDB (vector). The `pgvector` extension handles 768-dim embeddings. LlamaIndex `PGVectorStore` is the abstraction layer — swapping to Qdrant/Weaviate later requires only a config change.

### Schema

```sql
CREATE TABLE tenants (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER REFERENCES tenants(id),
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER REFERENCES tenants(id),
    user_id INTEGER REFERENCES users(id),
    original_filename TEXT,
    rag_collection TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    content TEXT NOT NULL,
    embedding vector(768),
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('turkish', content)) STORED,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON chunks USING GIN (tsv);

CREATE TABLE agents (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER REFERENCES tenants(id),
    user_id INTEGER REFERENCES users(id),
    name TEXT NOT NULL,
    system_prompt TEXT,
    model TEXT,
    collection TEXT,
    type TEXT DEFAULT 'chatbot',
    flow_config JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE chats (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER REFERENCES tenants(id),
    user_id INTEGER REFERENCES users(id),
    agent_id INTEGER REFERENCES agents(id),
    title TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE messages (
    id SERIAL PRIMARY KEY,
    chat_id INTEGER REFERENCES chats(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Collection Namespacing

Collections are namespaced as `t{tenant_id}:u{user_id}:{name}` in chunk metadata, e.g. `t1:u1:service_maintenance`. Every retrieval query filters by tenant_id and user_id.

---

## 3. Ingestion Pipeline

### Seed Data: issue_solution_final.xlsx

The Excel file (`issue_solution_final.xlsx`) contains two columns:
- `tespit_edilenler` — issue/finding description (Turkish)
- `yapilan_islem` — action taken / solution (Turkish)

**Startup seed script** (`backend/rag/ingestion/excel_seed.py`):
1. Read all rows from `Sheet1`
2. Format each row as: `"Sorun: {tespit_edilenler}\nÇözüm: {yapilan_islem}"`
3. Generate 768-dim embedding via `nomic-embed-text` (Ollama)
4. Store in `chunks` table with `collection = "service_maintenance"`, `tenant_id=1`, `user_id=1`
5. Skip rows already ingested (idempotent — checks by content hash)

This runs once at backend startup if the `chunks` table is empty for the seed collection.

### Ongoing Ingestion

Files uploaded via Open WebUI or the `/upload` endpoint go through `pdf_ingest.py`:
1. Extract text (PDF, DOCX, PPTX, XLSX, images)
2. Semantic chunking (~100 words, respecting section boundaries)
3. Embed with `nomic-embed-text`
4. Store in `chunks` with tenant/user scoping

The existing `FileWatcher` is preserved: it watches Open WebUI's uploads directory and auto-ingests new files.

---

## 4. Retrieval Pipeline

All retrieval is scoped to the requesting tenant/user via headers.

**Step 1 — Query expansion** (optional, `RAG_MULTI_QUERY=1`):
Generate 2-3 paraphrased query variants via LLM, retrieve for each, deduplicate and merge results.

**Step 2 — Hybrid search** (always on):
- Vector: cosine similarity on `embedding` column via pgvector
- BM25: full-text rank on `tsv` column using `ts_rank`
- Blend: `score = alpha * bm25_score + (1 - alpha) * vector_score`, alpha=0.3 (tuned for Turkish)

**Step 3 — Reranking**:
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` rescores the top-K candidates. Loaded lazily on first use (no startup cost if unused).

**Step 4 — Supplementary fallback**:
If top reranked score < confidence threshold, fall back to `content ILIKE '%phrase%'` search.

**Step 5 — Context assembly + generation**:
Merge adjacent chunks from the same source document. Build prompt. Call LLM. Stream response to Open WebUI in OpenAI SSE format.

---

## 5. LLM Providers

`backend/llm_providers.py` exposes a single `get_llm(model_id)` factory.

| Provider | Model IDs | Transport |
|---|---|---|
| Ollama | `ollama:llama3`, `ollama:gemma3`, `ollama:gemma3:12b` | HTTP → `OLLAMA_BASE_URL` |
| Gemma (via Ollama) | `ollama:gemma3`, `ollama:gemma3:2b` | Same Ollama client |
| DeepSeek | `deepseek-chat`, `deepseek-reasoner` | REST API |

The `/v1/models` endpoint queries Ollama live for available models and appends DeepSeek models, so Open WebUI's model picker always reflects what is actually pulled.

**Default model**: `DEFAULT_MODEL=ollama:gemma3` (configurable via `.env`).

---

## 6. Multi-Tenant Architecture

- **Header routing**: every request reads `X-Tenant-ID` and `X-User-ID` headers
- **Defaults**: missing headers → tenant=1, user=1 (backward compatible with Open WebUI's default setup)
- **Middleware** (`tenancy.py`): extracts headers, validates tenant/user exist in DB, attaches to request state
- **Data isolation**: all queries to `chunks`, `agents`, `chats`, `messages` include `WHERE tenant_id = ? AND user_id = ?`
- **Management APIs** (`tenants.py`): `GET/POST /tenants`, `GET/POST /tenants/{id}/users`

---

## 7. Neo4j Knowledge Graph (Optional)

Enabled via `RAG_USE_GRAPH=1`. Silently skipped if Neo4j is unreachable or the var is `0`.

**Ingestion** (`graph/ingest.py`): during document ingestion, extract named entities (machine names, error codes, component names, alarm codes) using the LLM. Store as nodes and typed relationships in Neo4j.

**Retrieval** (`graph/retriever.py`): after hybrid search returns chunks, query Neo4j for entities mentioned in those chunks. Return related nodes (e.g. connected error codes, sibling components, past solutions for the same machine). Append graph context to the RAG prompt.

**Client** (`graph/client.py`): thin wrapper around the Neo4j Python driver. Connection details from `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` env vars.

---

## 8. Agent Management from Open WebUI

Two complementary paths:

### Path A — Open WebUI Native Model Editor
Open WebUI's built-in Models UI lets users set a name and system prompt. The system prompt is forwarded to the backend in the `messages` array on every request. No extra backend code required. Suitable for quick, UI-only persona customization.

### Path B — Open WebUI Function Plugin
`openwebui_functions/agent_manager.py` is a Python Function installed in Open WebUI. It exposes commands processed client-side:

- `/create-agent name=<n> prompt=<p> model=<m> collection=<c>` → `POST /agents`
- `/list-agents` → `GET /agents`
- `/delete-agent id=<id>` → `DELETE /agents/{id}`

Backend-registered agents appear in `/v1/models` as `agent-{id}-{name}` and are selectable in Open WebUI's model picker.

---

## 9. Docker Compose

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    ports: ["5432:5432"]
    environment:
      POSTGRES_DB: ragdb
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: test1234
    volumes: [postgres_data:/var/lib/postgresql/data]

  neo4j:
    image: neo4j:5.15
    ports: ["7474:7474", "7687:7687"]
    environment:
      NEO4J_AUTH: neo4j/test1234

  backend:
    build: ./backend
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [postgres, neo4j]

  openwebui:
    image: ghcr.io/open-webui/open-webui:main
    ports: ["3000:3000"]
    environment:
      OPENAI_API_BASE_URL: http://backend:8000/v1
      OPENAI_API_KEY: unused
    depends_on: [backend]

volumes:
  postgres_data:
```

---

## 10. Environment Variables

```env
# App
APP_NAME=Coiltech RAG Platform
APP_VERSION=2.0.0
API_HOST=0.0.0.0
API_PORT=8000

# LLM
OLLAMA_BASE_URL=http://host.docker.internal:11434
DEFAULT_MODEL=ollama:gemma3
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_MODELS=deepseek-chat,deepseek-reasoner

# RAG
DEFAULT_ANSWER_LANGUAGE=tr
RAG_MULTI_QUERY=0
RAG_TOP_K=10
RAG_ALPHA=0.3
RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

# Graph
RAG_USE_GRAPH=0
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=test1234

# Database
DATABASE_URL=postgresql://postgres:test1234@postgres:5432/ragdb

# Open WebUI integration
OPENWEBUI_WATCH_ENABLED=true
OPENWEBUI_UPLOADS_DIR=/path/to/openwebui/uploads
OPENWEBUI_DEFAULT_COLLECTION=service_maintenance
OPENWEBUI_SCAN_INTERVAL=30
```

---

## Out of Scope

- Custom React frontend (removed — Open WebUI replaces it)
- Visual flow builder UI (flow execution in backend is kept; UI is not rebuilt)
- Authentication/JWT (deferred — Open WebUI handles its own auth)
