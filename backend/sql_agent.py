"""NL-to-SQL agent over the service_records table.

issue_solution_final.xlsx is loaded into a real table; maintenance questions
routed to the 'Servis Kayıtları' agent (agents.type='sql') are answered by
generating a guarded SELECT and summarizing the rows — not by RAG. The caller
falls back to RAG when no rows match or generation fails.
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)

SQL_AGENT_NAME = "Servis Kayıtları"
SQL_AGENT_COLLECTION = "service_records_sql"
MAX_ROWS = 20

_SQL_AGENT_DESCRIPTION = (
    "Servis Kayıtları. Bakım, arıza, servis, alarm, sorun ve çözüm kayıtları veritabanı. "
    "Geçmiş arıza kayıtlarını arar: hangi sorun tespit edildi, hangi işlem yapıldı. "
    "Örnek sorular: toprak hatası nasıl çözüldü, servo alarmı kayıtları, "
    "enkoder arızasında ne yapılmış, kaç adet hidrolik arıza kaydı var."
)

_SQL_PROMPT = """Türkçe soruları PostgreSQL SELECT sorgusuna çevir.

Tablo: service_records
Kolonlar:
- tespit_edilenler (TEXT): tespit edilen sorun/arıza açıklaması
- yapilan_islem (TEXT): yapılan işlem / çözüm açıklaması
- tespit_norm (TEXT): tespit_edilenler'in küçük harfli, Türkçe karaktersiz hali (aramada bunu kullan)
- islem_norm (TEXT): yapilan_islem'in küçük harfli, Türkçe karaktersiz hali (aramada bunu kullan)

Kurallar:
- SADECE SELECT yaz, başka hiçbir şey yazma (açıklama yok, markdown yok).
- WHERE koşullarında DAİMA tespit_norm/islem_norm kolonlarını ve küçük harfli, Türkçe karaktersiz
  anahtar kelimeler kullan (ş->s, ı->i, ğ->g, ü->u, ö->o, ç->c). Ör: "toprak hatası" -> '%toprak%'.
- Kelimenin kökünü kullan, ekleri atma: "alarmı" -> '%alarm%'.
- SELECT listesinde tespit_edilenler ve yapilan_islem döndür (norm kolonlarını döndürme).
- Birden çok anahtar kelime varsa LIKE koşullarını AND/OR ile birleştir; genel kelimeleri
  (hata, sorun, arıza) koşula EKLEME, sadece ayırt edici kelimeleri kullan.
- Her zaman LIMIT {max_rows} ekle.

Örnekler:
Soru: Toprak hatası nasıl çözülmüş?
SELECT tespit_edilenler, yapilan_islem FROM service_records WHERE tespit_norm LIKE '%toprak%' LIMIT {max_rows}

Soru: Kaç adet servo arıza kaydı var?
SELECT COUNT(*) FROM service_records WHERE tespit_norm LIKE '%servo%' LIMIT {max_rows}

Soru: Enkoder sorunlarında ne yapılmış?
SELECT tespit_edilenler, yapilan_islem FROM service_records WHERE tespit_norm LIKE '%enkoder%' OR tespit_norm LIKE '%encoder%' LIMIT {max_rows}

Soru: {question}
"""

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|execute|call|do)\b",
    re.IGNORECASE,
)


def validate_safe_select(sql: str) -> Tuple[bool, str]:
    """Single guarded SELECT over service_records only."""
    if not sql or not sql.strip():
        return False, "empty"
    s = sql.strip().rstrip(";").strip()
    if ";" in s:
        return False, "multiple statements"
    if "--" in s or "/*" in s:
        return False, "comments not allowed"
    if not re.match(r"^select\b", s, re.IGNORECASE):
        return False, "must start with SELECT"
    if _FORBIDDEN.search(s):
        return False, "forbidden keyword"
    # Only the service_records table may be referenced
    tables = re.findall(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_.]*)", s, re.IGNORECASE)
    if not tables:
        return False, "no table"
    for t in tables:
        if t.lower() != "service_records":
            return False, f"table not allowed: {t}"
    return True, "ok"


_TURKISH_MAP = str.maketrans("ıİşŞğĞüÜöÖçÇ", "iIsSgGuUoOcC")


def fold_literals(sql: str) -> str:
    """Lowercase + ASCII-fold string literals so they match the *_norm columns
    even when the LLM writes Turkish characters in the keywords."""
    return re.sub(
        r"'([^']*)'",
        lambda m: "'" + m.group(1).translate(_TURKISH_MAP).lower() + "'",
        sql,
    )


def clamp_limit(sql: str, max_rows: int = MAX_ROWS) -> str:
    """Force a LIMIT of at most max_rows onto the query."""
    s = sql.strip().rstrip(";").strip()
    m = re.search(r"\blimit\s+(\d+)\s*$", s, re.IGNORECASE)
    if m:
        if int(m.group(1)) > max_rows:
            s = s[: m.start()] + f"LIMIT {max_rows}"
        return s
    return f"{s} LIMIT {max_rows}"


async def generate_sql(question: str, llm) -> Optional[str]:
    try:
        raw = await llm.generate(_SQL_PROMPT.format(question=question, max_rows=MAX_ROWS))
        # Strip markdown fences / prose; keep the first SELECT line block
        raw = raw.strip()
        raw = re.sub(r"^```(?:sql)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
        m = re.search(r"(select\b.*)", raw, re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        sql = m.group(1).strip().split("\n\n")[0].strip()
        return clamp_limit(fold_literals(sql))
    except Exception as e:
        logger.warning("SQL generation failed", error=str(e))
        return None


async def run_sql_query(pool, sql: str) -> Optional[List[Dict[str, Any]]]:
    ok, reason = validate_safe_select(sql)
    if not ok:
        logger.warning("SQL rejected by safety gate", reason=reason, sql=sql[:200])
        return None
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql)
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("SQL execution failed", error=str(e), sql=sql[:200])
        return None


def build_summary_prompt(question: str, rows: List[Dict[str, Any]], system_prompt: Optional[str] = None) -> str:
    rows_text = json.dumps(rows[:MAX_ROWS], ensure_ascii=False, default=str)
    role = system_prompt or (
        "Sen Coiltech servis kayıtları uzmanısın. Geçmiş arıza/çözüm kayıtlarına dayanarak cevap verirsin."
    )
    return (
        f"{role}\n"
        f"Aşağıdaki veritabanı kayıtlarını kullanarak soruyu Türkçe yanıtla. "
        f"Kayıtlardaki bilgilerin dışına çıkma; benzer kayıtları grupla ve net adımlar ver.\n\n"
        f"KAYITLAR:\n{rows_text}\n\n"
        f"SORU: {question}\n\n"
        f"CEVAP:"
    )


def _relax_and_to_or(sql: str) -> Optional[str]:
    """Small models over-AND generic keywords; OR the conditions as a retry."""
    if not re.search(r"\sAND\s", sql, re.IGNORECASE):
        return None
    return re.sub(r"\sAND\s", " OR ", sql, flags=re.IGNORECASE)


def _single_keyword_sql(sql: str) -> Optional[str]:
    """Last resort: search both columns for the longest keyword literal."""
    lits = re.findall(r"'%([^']{3,})%'", sql)
    if not lits:
        return None
    kw = max(lits, key=len)
    return (
        "SELECT tespit_edilenler, yapilan_islem FROM service_records "
        f"WHERE tespit_norm LIKE '%{kw}%' OR islem_norm LIKE '%{kw}%' LIMIT {MAX_ROWS}"
    )


async def try_sql_path(pool, llm, question: str) -> Optional[Dict[str, Any]]:
    """Full NL→SQL attempt. Returns {'sql', 'rows'} or None (caller falls back to RAG)."""
    sql = await generate_sql(question, llm)
    if not sql:
        logger.info("SQL generation produced no query", question=question[:100])
        return None
    attempts = [sql]
    relaxed = _relax_and_to_or(sql)
    if relaxed:
        attempts.append(relaxed)
    single = _single_keyword_sql(sql)
    if single and single not in attempts:
        attempts.append(single)
    for attempt in attempts:
        rows = await run_sql_query(pool, attempt)
        if rows:
            logger.info("SQL path answered", sql=attempt[:200], rows=len(rows))
            return {"sql": attempt, "rows": rows}
    logger.info("SQL returned no rows after relaxation", sql=sql[:200])
    return None


async def seed_service_records(pool, excel_path: str) -> int:
    """Load the Excel into service_records once (skipped when already seeded)."""
    import pandas as pd

    async with pool.acquire() as conn:
        existing = await conn.fetchval("SELECT COUNT(*) FROM service_records")
        if existing and existing > 0:
            logger.info("service_records already seeded", rows=existing)
            return existing

    df = pd.read_excel(excel_path, sheet_name="Sheet1")
    df.columns = [c.strip() for c in df.columns]
    issue_col, solution_col = df.columns[0], df.columns[1]
    records = [
        (str(r[issue_col]).strip(), str(r[solution_col]).strip())
        for _, r in df.iterrows()
        if str(r[issue_col] or "").strip() and str(r[solution_col] or "").strip()
    ]
    async with pool.acquire() as conn:
        await conn.executemany(
            "INSERT INTO service_records (tespit_edilenler, yapilan_islem) VALUES ($1, $2)",
            records,
        )
    logger.info("service_records seeded from Excel", rows=len(records))
    return len(records)


async def ensure_sql_agent(pool, llm, settings) -> None:
    """Upsert the 'Servis Kayıtları' router-selectable SQL agent."""
    try:
        async with pool.acquire() as conn:
            existing = await conn.fetchval(
                "SELECT id FROM agents WHERE type='sql' AND tenant_id=1 AND user_id=1 LIMIT 1"
            )
            if existing:
                return
            row = await conn.fetchrow(
                """INSERT INTO agents (tenant_id, user_id, name, system_prompt, model, collection, type)
                   VALUES (1, 1, $1, $2, $3, $4, 'sql') RETURNING id""",
                SQL_AGENT_NAME,
                "Sen Coiltech servis kayıtları uzmanısın. Geçmiş arıza ve çözüm kayıtları "
                "veritabanından SQL ile cevap verirsin. Cevaplarını Türkçe ver.",
                settings.DEFAULT_MODEL,
                SQL_AGENT_COLLECTION,
            )
            agent_id = row["id"]
            emb = await llm.get_embedding(_SQL_AGENT_DESCRIPTION)
            emb_str = "[" + ",".join(str(v) for v in emb) + "]"
            await conn.execute(
                "UPDATE agents SET embedding=$1::vector WHERE id=$2", emb_str, agent_id
            )
        from backend.agent_creation import write_definition_file
        write_definition_file(
            settings.AGENTS_STORE_DIR,
            agent_id=agent_id,
            name=SQL_AGENT_NAME,
            system_prompt="NL-to-SQL over service_records (issue_solution_final.xlsx)",
            model=settings.DEFAULT_MODEL,
            collection=SQL_AGENT_COLLECTION,
            agent_type="sql",
            source_file="issue_solution_final.xlsx",
        )
        logger.info("SQL agent created", name=SQL_AGENT_NAME)
    except Exception as e:
        logger.warning("SQL agent creation failed", error=str(e))
