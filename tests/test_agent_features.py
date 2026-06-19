"""Tests for chat-driven agent creation and the NL-to-SQL service records agent."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.agent_creation import (
    detect_intent,
    extract_params,
    write_definition_file,
    _slugify,
)
from backend.sql_agent import (
    validate_safe_select,
    clamp_limit,
    fold_literals,
    generate_sql,
    build_summary_prompt,
)


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Bir agent oluştur",
    "bir agent oluşturmak istiyorum",
    "Agent oluştur lütfen",
    "Yeni bir ajan oluştur",
    "yeni agent istiyorum",
    "Bir agent olustur, adi SatisBot olsun",
    "ajan olustur",
    "I want to create agent",
    "I want to create an agent for sales documents",
    "create agent",
    "Can you create a new agent?",
    "make an agent from this file",
    "AJAN OLUŞTUR",
])
def test_intent_positive(text):
    assert detect_intent(text) is True


@pytest.mark.parametrize("text", [
    "Agent nedir?",
    "Hangi agentlar mevcut?",
    "Servo motor alarmı veriyor ne yapmalıyım?",
    "What does the router agent do?",
    "agent listesini göster",
    "",
])
def test_intent_negative(text):
    assert detect_intent(text) is False


# ---------------------------------------------------------------------------
# Param extraction (mocked LLM)
# ---------------------------------------------------------------------------

class FakeLLM:
    def __init__(self, response: str):
        self._response = response

    async def generate(self, prompt, system_prompt=None, model=None):
        return self._response


@pytest.mark.asyncio
async def test_extract_params_with_details():
    llm = FakeLLM('{"name": "SatışBot", "role": "satış sorularına cevap vermek"}')
    params = await extract_params("Bir agent oluştur, adı SatışBot olsun", llm)
    assert params["name"] == "SatışBot"
    assert "satış" in params["role"]


@pytest.mark.asyncio
async def test_extract_params_no_details():
    llm = FakeLLM('{"name": null, "role": null}')
    params = await extract_params("Bir agent oluştur", llm)
    assert params == {"name": None, "role": None}


@pytest.mark.asyncio
async def test_extract_params_garbage_response():
    llm = FakeLLM("I cannot help with that")
    params = await extract_params("create agent", llm)
    assert params == {"name": None, "role": None}


# ---------------------------------------------------------------------------
# Definition files
# ---------------------------------------------------------------------------

def test_slugify_turkish():
    assert _slugify("Satış Botu") == "satis_botu"
    assert _slugify("Servis Kayıtları") == "servis_kayitlari"
    assert _slugify("???") == "agent"


def test_write_definition_file(tmp_path):
    path = write_definition_file(
        str(tmp_path),
        agent_id=7,
        name="KaliteBot",
        system_prompt="Sen kalite uzmanısın.",
        model="ollama:gemma3",
        collection="quality_documents",
        source_file="ce_cert.pdf",
    )
    assert path is not None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["id"] == 7
    assert data["name"] == "KaliteBot"
    assert data["collection"] == "quality_documents"
    assert data["source_file"] == "ce_cert.pdf"
    assert data["type"] == "chatbot"


def test_write_definition_file_bad_dir_fails_soft():
    # Invalid path characters on Windows -> returns None, no raise
    assert write_definition_file(
        "\0invalid", agent_id=1, name="X", system_prompt="p",
        model="m", collection="c",
    ) is None


# ---------------------------------------------------------------------------
# SQL safety gate
# ---------------------------------------------------------------------------

def test_safe_select_accepts_valid():
    ok, _ = validate_safe_select(
        "SELECT tespit_edilenler, yapilan_islem FROM service_records "
        "WHERE tespit_edilenler ILIKE '%toprak%' LIMIT 10"
    )
    assert ok


def test_safe_select_accepts_count():
    ok, _ = validate_safe_select("SELECT COUNT(*) FROM service_records LIMIT 20")
    assert ok


def test_safe_select_accepts_group_by():
    ok, _ = validate_safe_select(
        "SELECT tespit_edilenler, COUNT(*) AS adet FROM service_records "
        "GROUP BY tespit_edilenler ORDER BY adet DESC LIMIT 10"
    )
    assert ok


def test_safe_select_accepts_union_all_stats():
    # The management-report agent emits category counts via UNION ALL
    ok, _ = validate_safe_select(
        "SELECT 'servo' AS tip, COUNT(*) AS adet FROM service_records "
        "WHERE tespit_norm LIKE '%servo%' "
        "UNION ALL SELECT 'vakum', COUNT(*) FROM service_records "
        "WHERE tespit_norm LIKE '%vakum%' ORDER BY adet DESC LIMIT 20"
    )
    assert ok


def test_safe_select_union_rejects_other_table():
    ok, _ = validate_safe_select(
        "SELECT 1 FROM service_records UNION ALL SELECT 1 FROM agents LIMIT 5"
    )
    assert not ok


@pytest.mark.parametrize("sql", [
    "UPDATE service_records SET yapilan_islem='x'",
    "DROP TABLE service_records",
    "DELETE FROM service_records",
    "INSERT INTO service_records VALUES ('a','b')",
    "SELECT * FROM service_records; DROP TABLE agents",
    "SELECT * FROM agents LIMIT 5",
    "SELECT * FROM service_records -- comment",
    "SELECT * FROM service_records /* x */",
    "SELECT * FROM service_records JOIN agents ON true",
    "TRUNCATE service_records",
    "",
])
def test_safe_select_rejects(sql):
    ok, _ = validate_safe_select(sql)
    assert not ok


def test_clamp_limit_adds_missing():
    assert clamp_limit("SELECT * FROM service_records").endswith("LIMIT 20")


def test_clamp_limit_reduces_large():
    out = clamp_limit("SELECT * FROM service_records LIMIT 5000")
    assert out.endswith("LIMIT 20")


def test_clamp_limit_keeps_small():
    out = clamp_limit("SELECT * FROM service_records LIMIT 5")
    assert out.endswith("LIMIT 5")


def test_relax_and_to_or():
    from backend.sql_agent import _relax_and_to_or
    sql = "SELECT * FROM service_records WHERE tespit_norm LIKE '%toprak%' AND tespit_norm LIKE '%alarmi%' LIMIT 20"
    relaxed = _relax_and_to_or(sql)
    assert " OR " in relaxed and " AND " not in relaxed.upper().replace(" OR ", " ")
    assert _relax_and_to_or("SELECT * FROM service_records WHERE tespit_norm LIKE '%x%' LIMIT 5") is None


def test_single_keyword_sql_picks_longest():
    from backend.sql_agent import _single_keyword_sql, validate_safe_select
    sql = "SELECT * FROM service_records WHERE tespit_norm LIKE '%topraklama%' AND islem_norm LIKE '%kablo%' LIMIT 20"
    out = _single_keyword_sql(sql)
    assert "'%topraklama%'" in out
    assert validate_safe_select(out)[0]


def test_fold_literals_turkish():
    out = fold_literals(
        "SELECT * FROM service_records WHERE tespit_norm LIKE '%Hatası%' OR islem_norm LIKE '%DEĞİŞTİ%'"
    )
    assert "'%hatasi%'" in out
    assert "'%degisti%'" in out
    # SQL keywords outside literals untouched
    assert out.startswith("SELECT")


# ---------------------------------------------------------------------------
# SQL generation parsing (mocked LLM)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_sql_strips_markdown():
    llm = FakeLLM("```sql\nSELECT * FROM service_records WHERE tespit_edilenler ILIKE '%servo%' LIMIT 10\n```")
    sql = await generate_sql("servo arızaları", llm)
    assert sql is not None
    assert sql.lower().startswith("select")
    assert "```" not in sql


@pytest.mark.asyncio
async def test_generate_sql_no_select_returns_none():
    llm = FakeLLM("Üzgünüm, bu soruya cevap veremem.")
    assert await generate_sql("soru", llm) is None


def test_build_summary_prompt_contains_rows_and_question():
    prompt = build_summary_prompt(
        "toprak hatası", [{"tespit_edilenler": "asi toprak hatası", "yapilan_islem": "modül reset"}]
    )
    assert "toprak hatası" in prompt
    assert "modül reset" in prompt


# ---------------------------------------------------------------------------
# Pending state machine (fake asyncpg pool)
# ---------------------------------------------------------------------------
from unittest.mock import AsyncMock, MagicMock, patch

from backend.agent_creation import (
    create_pending,
    fulfill_pending_after_ingest,
    pop_unannounced,
)


def _make_pool(*fetchrow_returns):
    """Build a fake asyncpg pool that returns rows from fetchrow in sequence."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value=None)
    conn.fetchrow = AsyncMock(side_effect=list(fetchrow_returns))
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=conn)
    return pool, conn


class _FakeSettings:
    PENDING_AGENT_TTL_MINUTES = 30
    DEFAULT_MODEL = "ollama:gemma3"
    AGENTS_STORE_DIR = "/tmp/agents_store_test"


class _FakeLLMForPending:
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
    assert conn.execute.call_count == 2
    first_sql = conn.execute.call_args_list[0][0][0]
    assert "expired" in first_sql
    second_sql = conn.execute.call_args_list[1][0][0]
    assert "INSERT" in second_sql


@pytest.mark.asyncio
async def test_fulfill_pending_no_row_returns_none():
    pool, conn = _make_pool(None)
    result = await fulfill_pending_after_ingest(
        pool, _FakeLLMForPending(), _FakeSettings(),
        collection="test_collection",
    )
    assert result is None


@pytest.mark.asyncio
async def test_fulfill_pending_creates_agent():
    pending_row = {"id": 42, "params": '{"name": "TestBot", "role": "test_role"}'}
    not_found = None
    agent_row = {"id": 99}

    pool, conn = _make_pool(pending_row, not_found, agent_row)
    conn.execute = AsyncMock(return_value=None)

    with patch("backend.agent_creation.write_definition_file", return_value="/tmp/test.json"), \
         patch("backend.deps.get_router") as mock_get_router:
        mock_get_router.side_effect = RuntimeError("not ready")
        result = await fulfill_pending_after_ingest(
            pool, _FakeLLMForPending(), _FakeSettings(),
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
