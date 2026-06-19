# Chat-Driven Agent Creation + NL-to-SQL Service Records

Date: 2026-06-10
Status: Approved by user

## Goal

Three features for the Coiltech RAG Platform (FastAPI + PostgreSQL/pgvector + Ollama, Open WebUI frontend):

1. **Agent creation from chat.** A user message like "Bir agent oluştur" or "I want to create agent"
   creates a specialist agent. Agents must be created **from a file**: if no document was uploaded,
   the system warns the user and remembers the request; when the user then uploads a file (even with
   no accompanying text), the agent is created using the parameters from the original message.
   Each created agent is stored in the DB **and** as a definition file in an `agents_store/` folder
   (volume-mounted to the host).
2. **Per-message routing.** The `coiltech-router` model selects the best agent for every message
   independently, so the router can switch agents mid-chat when the topic changes. (Already true;
   one fix needed: history condensation must not drag the old topic into an unrelated new question.)
3. **NL-to-SQL service records.** `issue_solution_final.xlsx` (781 rows: `tespit_edilenler`,
   `yapilan_islem`) is loaded into a real `service_records` table. Maintenance questions route to a
   `type='sql'` agent that generates a safe SELECT, executes it, and summarizes the rows — **not**
   RAG. If SQL yields nothing, fall back to the existing RAG search over `service_maintenance`.

## Architecture (approved approach: extend the existing router path)

```
User message → /v1/chat/completions
  ├─ 1. Agent-creation intent (regex, raw message)
  │    ├─ recent file ingested? → create/upsert agent now (+ definition file)
  │    └─ no file → Turkish warning + pending_agent_requests row
  │         FileWatcher ingest → fulfills newest open pending request
  │         next chat message → "✅ <Name> oluşturuldu" prefix
  ├─ 2. RouterAgent picks agent per message (cosine, as today)
  │    ├─ type='chatbot' → RAG pipeline (unchanged)
  │    └─ type='sql' → NL→SQL over service_records → stream summary
  │                    └─ 0 rows / invalid SQL → RAG fallback
  └─ 3. Stream answer with [→ AgentName] label, citations, answer cache
```

## Components

### backend/agent_creation.py (new)
- `detect_intent(text)` — regex: `agent oluştur`, `ajan oluştur`, `yeni (bir )?agent/ajan`,
  `create (an |a )?agent`, `i want to create agent`. Case/Turkish-char tolerant. No LLM call.
- `extract_params(message, llm)` — one LLM call returning `{name, role}`; fail-soft to None.
- `upsert_agent_for_collection(...)` — if an agent already exists for the collection (e.g. the
  auto-created one from ingest), UPDATE its name/prompt/embedding; else INSERT. Always writes
  `agents_store/<slug>.json` (name, system_prompt, model, collection, source_file, created_at)
  and invalidates the router cache.
- `find_recent_document(pool, since_seconds=120)` — newest `documents` row in the window.
- `create_pending(...)` / `fulfill_pending_after_ingest(...)` / `pop_unannounced(...)` —
  state machine over the `pending_agent_requests` table
  (status: pending → fulfilled → announced; TTL 30 min; newest-wins for tenant/user 1/1).

### backend/sql_agent.py (new)
- `seed_service_records(pool, excel_path)` — `CREATE TABLE IF NOT EXISTS service_records
  (id, tespit_edilenler TEXT, yapilan_islem TEXT)`; seed once (skip when count matches).
- `ensure_sql_agent(pool, llm)` — upsert agents row: name "Servis Kayıtları", `type='sql'`,
  embedding from a Turkish description of maintenance/issue/solution questions.
- `validate_safe_select(sql)` — single statement, starts with SELECT, only `service_records`,
  no semicolons/comments/mutations, LIMIT forced ≤ 20 (ported from legacy `_validate_safe_select`).
- `generate_sql(question, llm)` — schema + few-shot Turkish examples → SELECT with ILIKE filters.
- `run_sql_query(pool, sql)` — execute, return rows as dicts.
- Caller streams the summary: "Soru + rows → Türkçe cevap" via `llm.generate_stream`.

### Modified
- `backend/db.py` — `pending_agent_requests` table (id, tenant_id, user_id, raw_message,
  params JSONB, status, created_at, fulfilled_agent_id, fulfilled_at).
- `backend/pipeline.py` — `ingest_async`: fulfill pending requests BEFORE `maybe_create_agent`
  (fulfilled pending covers the collection, so auto-create then skips); `maybe_create_agent`
  also writes the definition file via agent_creation helpers.
- `backend/file_watcher.py` — `scan_now()` method for an immediate on-demand scan.
- `backend/main.py` — intent check before routing (raw message, after `_openwebui_utility`);
  force `scan_now()` on intent so a file attached to the same message is found; SQL-agent
  dispatch in `_handle_router_request`; "✅ oluşturuldu" confirmation prefix on the first
  message after async fulfillment; condensation prompt told to return topic-changing
  questions unchanged.
- `backend/settings.py` — `AGENTS_STORE_DIR` (default `agents_store`),
  `PENDING_AGENT_TTL_MINUTES` (default 30).
- `docker-compose.yml` — `./agents_store:/app/agents_store` volume on backend.
- Lifespan — call `seed_service_records` + `ensure_sql_agent` at startup.

## Error handling
Fail-soft everywhere: param extraction failure → filename-derived name + content-derived prompt;
SQL generation/validation failure or 0 rows → RAG fallback (`service_maintenance`); definition
file write failure → agent still created in DB (warning logged); pending fulfillment failure →
ingest still succeeds.

## Testing
Unit (no services): intent regex TR/EN positives + negatives ("agent nedir?" must NOT match);
`validate_safe_select` (UPDATE/DROP/DELETE/`;`/comments rejected, LIMIT clamped, other tables
rejected); definition file content (tmp dir); pending state transitions with a fake pool.
Live smoke test: create-agent message without file → warning; upload file → agent created +
definition file appears on host; maintenance question routes to [→ Servis Kayıtları] and
answers from SQL; topic switch mid-chat routes to a different agent.

## Out of scope
Multi-user pending queues (single pending per tenant/user, newest wins); SQL over arbitrary
tables (only `service_records`); deleting agents from chat.
