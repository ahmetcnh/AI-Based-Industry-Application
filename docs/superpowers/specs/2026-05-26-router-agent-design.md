# Router Agent Implementation Design

## Goal

Add a `coiltech-router` model to OpenWebUI that transparently routes user questions to the most relevant specialized agent (from the PostgreSQL `agents` table), generates a RAG-grounded response using that agent's system prompt and collection, and streams the result with a visible routing label (e.g. `[→ ServisBot]`).

## Architecture

```
OpenWebUI → POST /v1/chat/completions (model="coiltech-router")
                    ↓
            RouterAgent.route(question) → selected Agent row
            ┌──────────────────────────────────────────┐
            │ 1. Embed question (~0.5s, nomic-embed)   │
            │ 2. Cosine similarity vs. all agent       │
            │    embeddings (stored in agents table)   │
            │ 3. max_score ≥ 0.55?                     │
            │    YES → use that agent                  │
            │    NO  → LLM classify with top-3 names   │
            │          (llama3.2:1b, ~1-2s)            │
            └──────────────────────────────────────────┘
                    ↓
            RAGPipeline.query(
                collection = selected_agent.collection,
                system_prompt = selected_agent.system_prompt,
                model = selected_agent.model,
            )
                    ↓
            Stream: "[→ ServisBot]\n\n<RAG answer tokens...>"
```

## Tech Stack

- **Routing classifier:** `nomic-embed-text` (768-dim, already in use) + `llama3.2:1b` fallback
- **Confidence threshold:** 0.55 (matches presentation spec)
- **Agent embeddings:** stored in `agents` table as `VECTOR(768)` column
- **Routing label:** prepended to SSE stream, visible in OpenWebUI chat

---

## File Structure

| File | Change | Responsibility |
|------|--------|----------------|
| `backend/router_agent.py` | **Create** | `RouterAgent` class: embedding cache, `route()`, LLM fallback, routing decision cache |
| `backend/db.py` | **Modify** | Add `embedding VECTOR(768)` column to `agents` table in `init_schema()` |
| `backend/agents.py` | **Modify** | On `POST /agents` and `PUT /agents/{id}`, embed agent name+system_prompt and store; invalidate routing cache |
| `backend/pipeline.py` | **Modify** | `query()` accepts optional `system_prompt` override |
| `backend/main.py` | **Modify** | Expose `coiltech-router` in `/v1/models`; add routing branch in `/v1/chat/completions`; auto-create agent on `POST /upload` |

---

## Component Details

### `backend/router_agent.py` — RouterAgent

```python
class RouterAgent:
    def __init__(self, pool, llm, settings):
        ...

    async def route(self, question: str) -> dict:
        """Returns agent row dict or None (fallback to default collection)."""
        ...
```

**`route()` algorithm:**
1. Check routing cache: key = `question.strip().lower()`, TTL = 300s. If hit → return cached agent row immediately (skip steps 2-5).
2. Embed `question` via `llm.get_embedding()`
3. SQL query: `SELECT id, name, system_prompt, model, collection, embedding FROM agents WHERE embedding IS NOT NULL`
4. Compute cosine similarity in Python for each agent
5. If `max_score >= 0.55` → select that agent
6. Else → build classification prompt with top-3 agent names + one-line descriptions, call `llm.generate()` with `model="llama3.2:1b"`, parse the chosen name
7. Store result in routing cache with 300s TTL
8. Return the matched agent row (or `None` if no agents exist)

**Routing cache:** In-memory `dict` keyed by normalized question string, value is `(agent_row, expires_at)`. Invalidated for all entries when any agent is created or updated via the CRUD endpoints.

**Agent embedding text format:**
```
{agent.name}. {agent.system_prompt[:200]}
```
Truncated to 200 chars of system_prompt to keep the vector focused.

### `backend/db.py` — Schema Change

Add to `init_schema()`:
```sql
ALTER TABLE agents ADD COLUMN IF NOT EXISTS embedding VECTOR(768);
```

### `backend/agents.py` — Embedding on Write

After inserting/updating an agent row, compute and store embedding:
```python
text = f"{body.name}. {body.system_prompt[:200]}"
emb = await llm.get_embedding(text)
emb_str = "[" + ",".join(str(v) for v in emb) + "]"
await conn.execute("UPDATE agents SET embedding=$1::vector WHERE id=$2", emb_str, agent_id)
```

### `backend/pipeline.py` — system_prompt override

`query()` passes an optional `system_prompt` override to `run_query()`, which prepends it to the LLM prompt instead of the default "You are an expert assistant..." preamble.

### `backend/main.py` — Routing branch

In `/v1/chat/completions`:
```python
if model == "coiltech-router":
    agent = await _router.route(user_message)
    # stream "[→ AgentName]\n\n" then RAG stream
```

In `/v1/models`:
```python
{"id": "coiltech-router", "object": "model", "owned_by": "coiltech"}
```

### Auto-create agent on ingest — two trigger paths

Auto-agent creation fires from **both** ingest paths:

**Path 1 — `POST /upload` (API/Swagger uploads):**  
After `ingest_async()` completes, check if an agent exists for the resolved `collection`. If not, auto-create one.

**Path 2 — `FileWatcher` (OpenWebUI drag-and-drop):**  
OpenWebUI stores dragged files in its own uploads directory. The `FileWatcher` in `backend/file_watcher.py` monitors that directory (enabled via `OPENWEBUI_WATCH_ENABLED=true` + `OPENWEBUI_UPLOADS_DIR=<path>`). After the watcher calls `ingest_async()`, the same auto-create check runs.

To enable the drag-and-drop path, `.env` must have:
```env
OPENWEBUI_WATCH_ENABLED=true
OPENWEBUI_UPLOADS_DIR=/path/to/openwebui/uploads
```

**Shared auto-create logic** (extracted to a helper in `pipeline.py`):
```python
async def _maybe_create_agent(pool, llm, collection, tenant_id, user_id, settings):
    existing = await conn.fetchval(
        "SELECT id FROM agents WHERE collection=$1 AND tenant_id=$2 AND user_id=$3 LIMIT 1",
        collection, tenant_id, user_id
    )
    if not existing:
        name = collection.replace("_", " ").title() + "Bot"
        system_prompt = (
            f"Sen Coiltech {name} uzmanısın. "
            f"Yalnızca {collection} koleksiyonundaki belgelerden cevap ver."
        )
        # INSERT agent + compute embedding + store
```

The auto-created agent uses `model = settings.DEFAULT_MODEL`. Name is derived from collection name (e.g. `service_maintenance` → `Service MaintenanceBot`). The embedding is computed and stored immediately so the router can use it on the next query.

---

## Data Flow — Streaming Response

```
SSE chunk 1:  "[→ ServisBot]\n\n"
SSE chunk 2+: <tokens from Ollama stream>
SSE final:    {"finish_reason": "stop"}
data: [DONE]
```

The routing label is sent as the first content chunk so it appears immediately in OpenWebUI before the model starts generating.

---

## Fallback Behavior

| Condition | Behavior |
|-----------|----------|
| No agents in DB | Use default collection (`service_maintenance`), no label |
| All agent embeddings NULL (freshly added agent not yet embedded) | Skip that agent in cosine step |
| LLM classifier returns unrecognized name | Use top cosine-similarity agent |
| Ollama unreachable during classification | Use top cosine-similarity agent |

---

## Agent Setup (15 Specialized Agents)

The 15 agents from the presentation must be seeded via `POST /agents` API (or a seed script). Each agent needs:
- `name`: e.g. `"ServisBot"` 
- `system_prompt`: role description + instructions to answer only from documents
- `model`: `"ollama:gemma3"`
- `collection`: matching collection name from PostgreSQL

Once created via API, the embedding is computed automatically.

---

## Performance Budget

| Step | Estimated time |
|------|---------------|
| Routing cache hit | <1ms (skip all below) |
| Embedding the question | ~0.5s |
| Cosine similarity (15 agents, in-Python) | <1ms |
| LLM classifier (only when score < 0.55) | ~1-2s |
| RAG retrieval + generation | ~6-8s |
| **Total (cache hit)** | **~6-8s** (embedding step skipped) |
| **Total (embedding path)** | **~7-9s** |
| **Total (LLM fallback path)** | **~9-11s** |

---

## Out of Scope

- Agent-level rate limiting or ACLs
- Multi-hop routing (agent calling another agent)
