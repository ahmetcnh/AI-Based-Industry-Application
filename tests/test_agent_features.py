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
