# Chat-Driven Agent Creation + NL-to-SQL Service Records Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the implementation of chat-driven agent creation and NL-to-SQL service records by adding the missing unit tests for the pending state machine.

**Architecture:** The core implementation is already complete — `agent_creation.py`, `sql_agent.py`, all DB schema changes, `main.py` routing, `pipeline.py` ingest hooks, `file_watcher.py` `scan_now()`, `settings.py` fields, and `docker-compose.yml` volume. The only gap vs the spec is unit tests for the pending state machine (`create_pending`, `fulfill_pending_after_ingest`, `pop_unannounced`) using a fake asyncpg pool.

**Tech Stack:** Python 3.14, pytest, pytest-asyncio, asyncpg (mocked with MagicMock), pydantic-settings

---

## Status Summary

The following are **already implemented and tested** (47 tests passing):
- `backend/agent_creation.py` — `detect_intent`, `extract_params`, `write_definition_file`, `upsert_agent_for_collection`, `find_recent_document`, `create_pending`, `fulfill_pending_after_ingest`, `pop_unannounced`
- `backend/sql_agent.py` — `seed_service_records`, `ensure_sql_agent`, `validate_safe_select`, `generate_sql`, `run_sql_query`, `build_summary_prompt`, `try_sql_path`, `clamp_limit`, `fold_literals`
- `backend/db.py` — `pending_agent_requests` table, `service_records` table with computed `tespit_norm`/`islem_norm` columns
- `backend/settings.py` — `AGENTS_STORE_DIR`, `PENDING_AGENT_TTL_MINUTES`
- `backend/pipeline.py` — `fulfill_pending_after_ingest` before `maybe_create_agent`; `write_definition_file` via helpers
- `backend/file_watcher.py` — `scan_now()` method
- `backend/main.py` — intent check, `scan_now()` on intent, SQL-agent dispatch, `pop_unannounced` confirmation prefix, condensation returns topic-switching questions unchanged
- `docker-compose.yml` — `./agents_store:/app/agents_store` volume
- `agents_store/` — directory exists and contains JSON definition files

---

## Task 1: Add Pending State Machine Unit Tests

**Files:**
- Modify: `tests/test_agent_features.py`

The spec requires: "pending state transitions with a fake pool". These functions need testing without a live database.

- [ ] **Step 1: Add fake pool helper and pending state tests**

Append to `tests/test_agent_features.py`:

```python
# ---------------------------------------------------------------------------
# Pending state machine (fake asyncpg pool)
# ---------------------------------------------------------------------------
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from backend.agent_creation import (
    create_pending,
    fulfill_pending_after_ingest,
    pop_unannounced,
)


def _make_pool(*fetchrow_returns):
    """Build a fake asyncpg pool that returns rows in sequence."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value=None)
    # fetchrow returns items from the queue in order
    conn.fetchrow = AsyncMock(side_effect=list(fetchrow_returns))
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=conn)
    return pool, conn


class FakeSettings:
    PENDING_AGENT_TTL_MINUTES = 30
    DEFAULT_MODEL = "ollama:gemma3"
    AGENTS_STORE_DIR = "/tmp/agents_store_test"


class FakeLLMForPending:
    async def get_embedding(self, text):
        return [0.0] * 768

    async def generate(self, prompt, system_prompt=None, model=None):
        return '{"name": "TestBot", "role": "test"}'


@pytest.mark.asyncio
async def test_create_pending_expires_old_and_inserts():
    pool, conn = _make_pool()
    await create_pending(
        pool,
        raw_message="Bir agent oluştur",
        params={"name": "TestBot", "role": "test"},
        ttl_minutes=30,
    )
    # First call expires old pending rows, second inserts the new one
    assert conn.execute.call_count == 2
    first_call_sql = conn.execute.call_args_list[0][0][0]
    assert "expired" in first_call_sql
    second_call_sql = conn.execute.call_args_list[1][0][0]
    assert "INSERT" in second_call_sql


@pytest.mark.asyncio
async def test_fulfill_pending_no_row_returns_none():
    pool, conn = _make_pool(None)  # fetchrow → None (no pending)
    result = await fulfill_pending_after_ingest(
        pool, FakeLLMForPending(), FakeSettings(),
        collection="test_collection",
    )
    assert result is None


@pytest.mark.asyncio
async def test_fulfill_pending_creates_agent():
    # First fetchrow: the pending request row
    pending_row = {"id": 42, "params": '{"name": "TestBot", "role": "test_role"}'}
    # Second fetchrow: upsert_agent_for_collection checks existing agent (None = not found)
    not_found = None
    # Third fetchrow: INSERT agents RETURNING id
    agent_row = {"id": 99}

    pool, conn = _make_pool(pending_row, not_found, agent_row)
    conn.execute = AsyncMock(return_value=None)

    with patch("backend.agent_creation.write_definition_file", return_value="/tmp/test.json"), \
         patch("backend.deps.get_router") as mock_get_router:
        mock_get_router.side_effect = RuntimeError("not ready")
        result = await fulfill_pending_after_ingest(
            pool, FakeLLMForPending(), FakeSettings(),
            collection="test_collection",
            source_file="test.pdf",
        )

    assert result is not None
    assert result["name"] == "TestBot"
    assert result["collection"] == "test_collection"


@pytest.mark.asyncio
async def test_pop_unannounced_returns_name():
    row = {"fulfilled_agent_name": "TestBot"}
    pool, conn = _make_pool(row)
    name = await pop_unannounced(pool)
    assert name == "TestBot"


@pytest.mark.asyncio
async def test_pop_unannounced_returns_none_when_no_row():
    pool, conn = _make_pool(None)
    name = await pop_unannounced(pool)
    assert name is None


@pytest.mark.asyncio
async def test_pop_unannounced_returns_none_on_exception():
    pool = MagicMock()
    pool.acquire = MagicMock(side_effect=Exception("db down"))
    name = await pop_unannounced(pool)
    assert name is None
```

- [ ] **Step 2: Run the new tests to verify they pass**

Run: `cd /home/ubuntu/bitirme2 && .venv_linux/bin/python -m pytest tests/test_agent_features.py -v -k "pending or pop_unannounced or fulfill"`

Expected: All new pending state tests PASS.

- [ ] **Step 3: Run full suite to confirm no regressions**

Run: `cd /home/ubuntu/bitirme2 && .venv_linux/bin/python -m pytest tests/test_agent_features.py -v`

Expected: All 53+ tests PASS.

- [ ] **Step 4: Commit**

```bash
cd /home/ubuntu/bitirme2
git add tests/test_agent_features.py
git commit -m "test: add pending state machine unit tests with fake pool"
```
