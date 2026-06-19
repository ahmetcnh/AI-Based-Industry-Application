# Router Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `coiltech-router` model to OpenWebUI that routes questions to the best-matching specialized agent using embedding similarity + LLM fallback, streams `[→ AgentName]` as the first response token, and auto-creates agents when new documents are ingested.

**Architecture:** `RouterAgent` holds an in-memory routing cache (300s TTL) and computes cosine similarity between the question embedding and each agent's stored `VECTOR(768)` embedding. If max score ≥ 0.55 it picks that agent directly; otherwise a fast `llama3.2:1b` prompt breaks the tie. Agent embeddings are written to PostgreSQL on every `POST /agents` or `PUT /agents/{id}`. Auto-agent creation runs after both `POST /upload` and `FileWatcher` ingests.

**Tech Stack:** asyncpg + pgvector, `nomic-embed-text` 768-dim embeddings, `llama3.2:1b` for classification, FastAPI SSE streaming, Python in-memory dict cache.

---

## File Map

| File | Action |
|------|--------|
| `backend/db.py` | Add `embedding VECTOR(768)` column to `agents` table |
| `backend/deps.py` | **Create** — shared `_llm` / `_router` singletons (mirrors `db.py` pattern) |
| `backend/router_agent.py` | **Create** — `RouterAgent` class |
| `backend/agents.py` | Compute + store embedding after INSERT/UPDATE; call `invalidate_cache()` |
| `backend/rag/retrieval/query_engine.py` | Add `system_prompt` param to `run_query()` and `retrieve_context()` |
| `backend/pipeline.py` | Add `system_prompt` to `query()` / `_retrieve()`; add `maybe_create_agent()` public method |
| `backend/file_watcher.py` | Call `rag_pipeline.maybe_create_agent()` after successful ingest |
| `backend/main.py` | Init `RouterAgent` in lifespan; expose `coiltech-router`; routing branch; upload auto-create |

---

## Task 1: Add `embedding` column to `agents` table

**Files:**
- Modify: `backend/db.py:87-99`

- [ ] **Step 1: Add the ALTER TABLE statement to `init_schema()`**

Open `backend/db.py`. After the `CREATE TABLE IF NOT EXISTS agents (...)` block (which ends around line 99), add:

```python
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
```

The `ADD COLUMN IF NOT EXISTS` is idempotent — safe to run on every startup against an existing DB.

- [ ] **Step 2: Verify the column appears after backend restart**

```powershell
cd C:\Users\AhmetCan\Desktop\bitirme2
docker compose up -d --force-recreate backend
docker exec bitirme2-backend-1 python3 -c "
import asyncio, sys; sys.path.insert(0, '/app')
async def check():
    from backend.db import init_pool
    pool = await init_pool('postgresql://postgres:test1234@postgres:5432/ragdb')
    async with pool.acquire() as conn:
        row = await conn.fetchval(\"SELECT column_name FROM information_schema.columns WHERE table_name='agents' AND column_name='embedding'\")
        print('Column exists:', row)
asyncio.run(check())
"
```

Expected output: `Column exists: embedding`

- [ ] **Step 3: Commit**

```powershell
git add backend/db.py
git commit -m "feat: add embedding vector column to agents table"
```

---

## Task 2: Create `backend/deps.py` — shared singletons

**Files:**
- Create: `backend/deps.py`

This file stores `_llm` and `_router` as module-level globals (same pattern as `db.py` stores `_pool`). `agents.py` and other modules import from here to avoid circular imports with `main.py`.

- [ ] **Step 1: Create `backend/deps.py`**

```python
from typing import Optional

_llm = None
_router = None


def get_llm():
    if _llm is None:
        raise RuntimeError("LLM provider not initialized")
    return _llm


def set_llm(llm) -> None:
    global _llm
    _llm = llm


def get_router():
    if _router is None:
        raise RuntimeError("RouterAgent not initialized")
    return _router


def set_router(router) -> None:
    global _router
    _router = router
```

- [ ] **Step 2: Verify the module imports cleanly**

```powershell
docker exec bitirme2-backend-1 python3 -c "
import sys; sys.path.insert(0, '/app')
from backend.deps import get_llm, set_llm, get_router, set_router
print('deps OK')
"
```

Expected: `deps OK`

- [ ] **Step 3: Commit**

```powershell
git add backend/deps.py
git commit -m "feat: add shared deps module for llm/router singletons"
```

---

## Task 3: Create `backend/router_agent.py`

**Files:**
- Create: `backend/router_agent.py`

- [ ] **Step 1: Create the file**

```python
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
                "SELECT id, name, system_prompt, model, collection, embedding "
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
            agent_emb = list(raw_emb)
            score = _cosine(question_emb, agent_emb)
            scored.append((score, dict(row)))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_agent = scored[0]

        logger.info("Routing scores", top_agent=best_agent["name"], score=round(best_score, 3))

        if best_score >= CONFIDENCE_THRESHOLD:
            self._cache_set(cache_key, best_agent)
            return best_agent

        # LLM fallback: pick from top-3 by name
        top3 = [agent for _, agent in scored[:3]]
        choices = "\n".join(
            f"- {a['name']}: {(a.get('system_prompt') or '')[:80]}" for a in top3
        )
        prompt = (
            f"Aşağıdaki uzman agentlardan hangisi bu soruyu yanıtlamak için en uygun?\n\n"
            f"Soru: {question}\n\nAgentlar:\n{choices}\n\n"
            f"Sadece seçilen agentin tam adını yaz, başka hiçbir şey yazma:"
        )
        chosen_name = (await self.llm.generate(prompt, model="ollama:llama3.2:1b")).strip()

        selected = next(
            (a for a in top3 if a["name"].lower() == chosen_name.lower()),
            best_agent,  # fallback to top cosine if name not recognized
        )
        logger.info("LLM routing decision", chosen=chosen_name, selected=selected["name"])
        self._cache_set(cache_key, selected)
        return selected
```

- [ ] **Step 2: Verify the class imports cleanly (no syntax errors)**

```powershell
docker exec bitirme2-backend-1 python3 -c "
import sys; sys.path.insert(0, '/app')
from backend.router_agent import RouterAgent
print('RouterAgent OK')
"
```

Expected: `RouterAgent OK`

- [ ] **Step 3: Commit**

```powershell
git add backend/router_agent.py
git commit -m "feat: add RouterAgent with cosine routing and LLM fallback"
```

---

## Task 4: Embed agents on create/update in `backend/agents.py`

**Files:**
- Modify: `backend/agents.py`

- [ ] **Step 1: Add imports and the embedding helper at the top of `backend/agents.py`**

After the existing imports, add:

```python
from backend.db import get_pool
from backend.tenancy import get_tenant
```

These already exist. Add below them:

```python
async def _embed_and_store(conn, agent_id: int, name: str, system_prompt: Optional[str]) -> None:
    try:
        from backend.deps import get_llm, get_router
        llm = get_llm()
        text = f"{name}. {(system_prompt or '')[:200]}"
        emb = await llm.get_embedding(text)
        emb_str = "[" + ",".join(str(v) for v in emb) + "]"
        await conn.execute(
            "UPDATE agents SET embedding=$1::vector WHERE id=$2", emb_str, agent_id
        )
        try:
            get_router().invalidate_cache()
        except RuntimeError:
            pass  # router not yet initialized (startup)
    except Exception as e:
        logger.warning("Agent embedding failed", error=str(e))
```

- [ ] **Step 2: Call `_embed_and_store` after INSERT in `create_agent`**

Replace the existing `create_agent` function body:

```python
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
        agent_id = row["id"]
        await _embed_and_store(conn, agent_id, body.name, body.system_prompt)
    return dict(row)
```

- [ ] **Step 3: Call `_embed_and_store` after UPDATE in `update_agent`**

Replace the existing `update_agent` function body:

```python
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
        new_name = row["name"]
        new_prompt = row["system_prompt"]
        await _embed_and_store(conn, agent_id, new_name, new_prompt)
    return dict(row)
```

- [ ] **Step 4: Verify by creating a test agent via API and checking the embedding was stored**

```powershell
$body = '{"name":"TestBot","system_prompt":"Sen bir test uzmanisın.","model":"ollama:gemma3","collection":"service_maintenance"}'
Invoke-RestMethod -Uri http://localhost:8000/agents -Method POST -ContentType "application/json" -Body $body
```

Then check the DB:

```powershell
docker exec bitirme2-backend-1 python3 -c "
import asyncio, sys; sys.path.insert(0, '/app')
async def check():
    from backend.db import init_pool
    pool = await init_pool('postgresql://postgres:test1234@postgres:5432/ragdb')
    async with pool.acquire() as conn:
        row = await conn.fetchrow('SELECT id, name, embedding IS NOT NULL as has_emb FROM agents ORDER BY id DESC LIMIT 1')
        print(dict(row))
asyncio.run(check())
"
```

Expected: `{'id': ..., 'name': 'TestBot', 'has_emb': True}`

- [ ] **Step 5: Commit**

```powershell
git add backend/agents.py
git commit -m "feat: compute and store agent embeddings on create/update"
```

---

## Task 5: Add `system_prompt` param to `query_engine.py`

**Files:**
- Modify: `backend/rag/retrieval/query_engine.py`

- [ ] **Step 1: Replace the hardcoded `_ANSWER_PROMPT` with a parameterized version**

Replace lines 13-22:

```python
_DEFAULT_SYSTEM_ROLE = (
    "You are an expert assistant for the Coiltech industrial automation platform."
)

_ANSWER_PROMPT = """{system_role}
Use the context below to answer the question. If the context is insufficient, say so clearly.
Answer in {language}.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
```

- [ ] **Step 2: Add `system_prompt` parameter to `run_query()`**

Change the signature of `run_query()` (add after `graph_retriever=None`):

```python
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
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
```

And update the prompt building line (previously `_ANSWER_PROMPT.format(...)`):

```python
    prompt = _ANSWER_PROMPT.format(
        system_role=system_prompt or _DEFAULT_SYSTEM_ROLE,
        context=context,
        question=question,
        language=language,
    )
```

- [ ] **Step 3: Add `system_prompt` parameter to `retrieve_context()`**

Change the signature (add after `graph_retriever=None`):

```python
async def retrieve_context(
    pool: asyncpg.Pool,
    llm,
    *,
    question: str,
    collection: str,
    tenant_id: int = 1,
    user_id: int = 1,
    k: int = 5,
    alpha: float = 0.3,
    use_expansion: bool = False,
    reranker_model: Optional[str] = None,
    score_threshold: float = 0.3,
    language: str = "tr",
    graph_retriever=None,
    system_prompt: Optional[str] = None,
) -> tuple:
```

And update the prompt building line at the end of `retrieve_context()`:

```python
    prompt = _ANSWER_PROMPT.format(
        system_role=system_prompt or _DEFAULT_SYSTEM_ROLE,
        context=context,
        question=question,
        language=language,
    )
    return prompt, sources
```

- [ ] **Step 4: Verify no syntax error**

```powershell
docker exec bitirme2-backend-1 python3 -c "
import sys; sys.path.insert(0, '/app')
from backend.rag.retrieval.query_engine import run_query, retrieve_context
print('query_engine OK')
"
```

Expected: `query_engine OK`

- [ ] **Step 5: Commit**

```powershell
git add backend/rag/retrieval/query_engine.py
git commit -m "feat: add system_prompt override to run_query and retrieve_context"
```

---

## Task 6: Update `backend/pipeline.py` — system_prompt + `maybe_create_agent`

**Files:**
- Modify: `backend/pipeline.py`

- [ ] **Step 1: Add `system_prompt` parameter to `query()` and `_retrieve()`**

In `query()`, change the signature and `run_query()` call:

```python
    async def query(
        self,
        question: str,
        collection: str,
        tenant_id: int = 1,
        user_id: int = 1,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
```

Pass it through to `run_query()`:

```python
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
            system_prompt=system_prompt,
        )
```

In `_retrieve()`, change the signature and `retrieve_context()` call:

```python
    async def _retrieve(
        self,
        question: str,
        collection: str,
        tenant_id: int = 1,
        user_id: int = 1,
        system_prompt: Optional[str] = None,
    ):
        from backend.rag.retrieval.query_engine import retrieve_context
        return await retrieve_context(
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
            system_prompt=system_prompt,
        )
```

- [ ] **Step 2: Add `maybe_create_agent()` public method to `RAGPipeline`**

Add this method after `ingest_async()`:

```python
    async def maybe_create_agent(
        self,
        collection: str,
        tenant_id: int = 1,
        user_id: int = 1,
    ) -> None:
        try:
            async with self.pool.acquire() as conn:
                existing = await conn.fetchval(
                    "SELECT id FROM agents WHERE collection=$1 AND tenant_id=$2 AND user_id=$3 LIMIT 1",
                    collection, tenant_id, user_id,
                )
                if existing:
                    return
                name = collection.replace("_", " ").title() + "Bot"
                system_prompt = (
                    f"Sen Coiltech {name} uzmanısın. "
                    f"Yalnızca {collection} koleksiyonundaki belgelerden cevap ver. "
                    f"Cevaplarını Türkçe ver."
                )
                row = await conn.fetchrow(
                    """INSERT INTO agents (tenant_id, user_id, name, system_prompt, model, collection)
                       VALUES ($1,$2,$3,$4,$5,$6) RETURNING id""",
                    tenant_id, user_id, name, system_prompt,
                    self.settings.DEFAULT_MODEL, collection,
                )
                agent_id = row["id"]
                emb = await self.llm.get_embedding(f"{name}. {system_prompt[:200]}")
                emb_str = "[" + ",".join(str(v) for v in emb) + "]"
                await conn.execute(
                    "UPDATE agents SET embedding=$1::vector WHERE id=$2", emb_str, agent_id
                )
                try:
                    from backend.deps import get_router
                    get_router().invalidate_cache()
                except RuntimeError:
                    pass
            logger.info("Auto-created agent for collection", collection=collection, name=name)
        except Exception as e:
            logger.warning("Auto-agent creation failed", collection=collection, error=str(e))
```

- [ ] **Step 3: Call `maybe_create_agent` at the end of `ingest_async()`**

At the end of `ingest_async()`, after the graph ingest block, add:

```python
        await self.maybe_create_agent(collection, tenant_id, user_id)
        return result
```

The full updated `ingest_async()` tail:

```python
        if self.settings.RAG_USE_GRAPH:
            try:
                from backend.rag.graph.client import get_driver
                driver = get_driver(
                    self.settings.NEO4J_URI,
                    self.settings.NEO4J_USER,
                    self.settings.NEO4J_PASSWORD,
                )
            except Exception as e:
                logger.warning("Graph ingest skipped", error=str(e))
        await self.maybe_create_agent(collection, tenant_id, user_id)
        return result
```

- [ ] **Step 4: Verify import chain is clean**

```powershell
docker exec bitirme2-backend-1 python3 -c "
import sys; sys.path.insert(0, '/app')
from backend.pipeline import RAGPipeline
import inspect
sig = inspect.signature(RAGPipeline.query)
print('query params:', list(sig.parameters.keys()))
print('has maybe_create_agent:', hasattr(RAGPipeline, 'maybe_create_agent'))
"
```

Expected: query params includes `system_prompt`, `has maybe_create_agent: True`

- [ ] **Step 5: Commit**

```powershell
git add backend/pipeline.py
git commit -m "feat: add system_prompt override and maybe_create_agent to RAGPipeline"
```

---

## Task 7: Update `backend/file_watcher.py` — auto-create agent after ingest

**Files:**
- Modify: `backend/file_watcher.py:101-111`

- [ ] **Step 1: Call `maybe_create_agent` after successful ingest in `_process_file()`**

Replace lines 101-111 (the `result = await self.rag_pipeline.ingest_async(...)` block):

```python
            result = await self.rag_pipeline.ingest_async(
                file_path=str(file_path),
                department=department,
                metadata=metadata,
                tenant_id=1,
                user_id=1,
            )
            collection = self.rag_pipeline.resolve_collection(department, metadata, str(file_path))
            await self.rag_pipeline.maybe_create_agent(collection, tenant_id=1, user_id=1)
            self._processed_files[str_path] = file_hash
```

- [ ] **Step 2: Verify no syntax errors**

```powershell
docker exec bitirme2-backend-1 python3 -c "
import sys; sys.path.insert(0, '/app')
from backend.file_watcher import FileWatcher
print('FileWatcher OK')
"
```

Expected: `FileWatcher OK`

- [ ] **Step 3: Commit**

```powershell
git add backend/file_watcher.py
git commit -m "feat: auto-create agent after file watcher ingest"
```

---

## Task 8: Wire everything in `backend/main.py`

**Files:**
- Modify: `backend/main.py`

- [ ] **Step 1: Import `RouterAgent` and `deps` setters; initialize in lifespan**

Add to imports at top of `main.py`:

```python
from backend.router_agent import RouterAgent
from backend.deps import set_llm, set_router
```

In the `lifespan` function, after `_rag = RAGPipeline(_pool, _llm, settings)`, add:

```python
    _router = RouterAgent(_pool, _llm, settings)
    set_llm(_llm)
    set_router(_router)
```

Also add `_router` to the module-level globals at the top:

```python
_pool = None
_llm: Optional[LLMProvider] = None
_rag: Optional[RAGPipeline] = None
_router = None
_file_watcher = None
```

- [ ] **Step 2: Expose `coiltech-router` in `/v1/models`**

In `openai_models()`, after the `coiltech-rag` entry, add:

```python
    models = [
        {"id": "coiltech-rag", "object": "model", "created": created, "owned_by": "coiltech"},
        {"id": "coiltech-router", "object": "model", "created": created, "owned_by": "coiltech"},
    ]
```

- [ ] **Step 3: Add routing branch in `/v1/chat/completions`**

Replace the `if stream:` block (and the `result = await rag.query(...)` below it) with:

```python
    collection = collection_name or settings.OPENWEBUI_DEFAULT_COLLECTION

    # Determine collection + system_prompt via router when coiltech-router selected
    agent_system_prompt = None
    if model == "coiltech-router" and _router is not None:
        selected_agent = await _router.route(user_message, tenant_id, user_id)
        if selected_agent:
            collection = selected_agent["collection"] or collection
            agent_system_prompt = selected_agent.get("system_prompt")
            routing_label = f"[→ {selected_agent['name']}]\n\n"
        else:
            routing_label = ""
    else:
        routing_label = ""

    if stream:
        async def gen():
            if routing_label:
                label_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now(timezone.utc).timestamp()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": routing_label}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(label_chunk)}\n\n"
            prompt, sources = await rag._retrieve(
                question=user_message,
                collection=collection,
                tenant_id=tenant_id,
                user_id=user_id,
                system_prompt=agent_system_prompt,
            )
            cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            async for token in _llm.generate_stream(prompt, model=model if model != "coiltech-router" else settings.DEFAULT_MODEL):
                chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now(timezone.utc).timestamp()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            done = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": int(datetime.now(timezone.utc).timestamp()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    result = await rag.query(
        question=user_message,
        collection=collection,
        tenant_id=tenant_id,
        user_id=user_id,
        model=model if model != "coiltech-router" else settings.DEFAULT_MODEL,
        system_prompt=agent_system_prompt,
    )
    answer = routing_label + result["answer"]
    return _build_response(answer, model)
```

- [ ] **Step 4: Call `maybe_create_agent` in `POST /upload` after ingest**

In `upload_document()`, after `return {"status": "success", ...}` line, insert before the return:

```python
    await rag.maybe_create_agent(collection, tenant_id, user_id)
    return {"status": "success", "file_name": file.filename, **result}
```

- [ ] **Step 5: Rebuild Docker image and recreate container**

```powershell
cd C:\Users\AhmetCan\Desktop\bitirme2
docker compose build backend
docker compose up -d --force-recreate backend
```

Wait 10 seconds for startup, then:

```powershell
Start-Sleep -Seconds 10
docker logs bitirme2-backend-1 --tail 10
```

Expected: `Application startup complete.` with no errors.

- [ ] **Step 6: Verify `coiltech-router` appears in `/v1/models`**

```powershell
(Invoke-RestMethod -Uri http://localhost:8000/v1/models).data | Select-Object id
```

Expected output includes: `coiltech-router`

- [ ] **Step 7: Test routing with a question**

First seed a couple of agents so routing has something to choose from:

```powershell
$body1 = '{"name":"ServisBot","system_prompt":"Sen Coiltech servis ve bakim uzmanisın. Sadece bakim ve ariza kayitlarindan cevap ver.","model":"ollama:gemma3","collection":"service_maintenance"}'
Invoke-RestMethod -Uri http://localhost:8000/agents -Method POST -ContentType "application/json" -Body $body1

$body2 = '{"name":"SatisBot","system_prompt":"Sen Coiltech satis ve teklif uzmanisın. Sadece teklif belgelerinden cevap ver.","model":"ollama:gemma3","collection":"sales_and_quotations"}'
Invoke-RestMethod -Uri http://localhost:8000/agents -Method POST -ContentType "application/json" -Body $body2
```

Then send a chat request with the router model:

```powershell
$chatBody = '{"model":"coiltech-router","messages":[{"role":"user","content":"transfer hatasi nedir"}],"stream":false}'
$resp = Invoke-RestMethod -Uri http://localhost:8000/v1/chat/completions -Method POST -ContentType "application/json" -Body $chatBody -TimeoutSec 60
$resp.choices[0].message.content
```

Expected: response starts with `[→ ServisBot]` followed by the RAG answer.

- [ ] **Step 8: Commit**

```powershell
git add backend/main.py
git commit -m "feat: wire RouterAgent into chat completions endpoint with transparent routing label"
```

---

## Task 9: Test auto-agent creation on upload

**Files:** None (integration test only)

- [ ] **Step 1: Upload a document and verify an agent is auto-created**

```powershell
# Create a small test text file
"Bu belge kurulum ve devreye alma hakkindadir." | Out-File -FilePath test_doc.txt -Encoding utf8

# Upload it — collection should resolve to 'installation_commissioning' via keyword map
$form = @{ file = Get-Item .\test_doc.txt; collection_name = "installation_commissioning" }
Invoke-RestMethod -Uri http://localhost:8000/upload -Method POST -Form $form -TimeoutSec 30
```

- [ ] **Step 2: Verify agent was auto-created with embedding**

```powershell
(Invoke-RestMethod -Uri http://localhost:8000/agents).name
```

Expected: list includes `Installation CommissioningBot` (or similar derived name).

```powershell
docker exec bitirme2-backend-1 python3 -c "
import asyncio, sys; sys.path.insert(0, '/app')
async def check():
    from backend.db import init_pool
    pool = await init_pool('postgresql://postgres:test1234@postgres:5432/ragdb')
    async with pool.acquire() as conn:
        rows = await conn.fetch('SELECT name, collection, embedding IS NOT NULL as has_emb FROM agents ORDER BY id')
        for r in rows: print(dict(r))
asyncio.run(check())
"
```

Expected: all auto-created agents have `has_emb: True`.

- [ ] **Step 3: Clean up test file**

```powershell
Remove-Item test_doc.txt
```

---

## Self-Review Checklist

- [x] **Spec coverage:**
  - RouterAgent with embedding similarity + LLM fallback ✓ (Task 3)
  - Confidence threshold 0.55 ✓ (Task 3, `CONFIDENCE_THRESHOLD = 0.55`)
  - Agent embedding stored in `agents` table ✓ (Task 1 + Task 4)
  - Routing cache 300s TTL + invalidation on CRUD ✓ (Task 3 + Task 4)
  - `coiltech-router` in `/v1/models` ✓ (Task 8)
  - Streaming with `[→ AgentName]` label ✓ (Task 8)
  - Auto-create agent on `POST /upload` ✓ (Task 6 + Task 8)
  - Auto-create agent on FileWatcher ingest ✓ (Task 7)
  - Fallback to default collection when no agents ✓ (Task 3 returns `None`, Task 8 handles `None`)
  - `system_prompt` override in RAG pipeline ✓ (Task 5 + Task 6)

- [x] **Placeholder scan:** No TBD/TODO. All code blocks are complete.

- [x] **Type consistency:**
  - `route()` returns `Optional[Dict[str, Any]]` — used as such in Task 8
  - `maybe_create_agent()` is called consistently as `await rag.maybe_create_agent(collection, tenant_id, user_id)`
  - `_retrieve()` signature matches call in Task 8 (positional + `system_prompt` keyword)
  - `_embed_and_store(conn, agent_id, name, system_prompt)` — called with same 4 args in both Task 4 locations
