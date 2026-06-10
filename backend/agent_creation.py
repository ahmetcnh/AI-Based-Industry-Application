"""Chat-driven agent creation.

A user message like "Bir agent oluştur" creates a specialist agent from the
most recently uploaded document. If no document exists yet, the request is
stored in pending_agent_requests and fulfilled when the next file is ingested.
Every created agent is also written as a JSON definition file under
settings.AGENTS_STORE_DIR so the "agent folder" mirrors the database.
"""
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import structlog

logger = structlog.get_logger(__name__)

# How far back an ingested document counts as "just uploaded" when the
# creation message and the file arrive together.
RECENT_DOCUMENT_WINDOW_SECONDS = 120

# Matched against ASCII-normalized text ("oluştur" -> "olustur"), so both
# Turkish and ASCII spellings work.
_INTENT_PATTERNS = [
    r"\bagent\s+olustur",
    r"\bajan\s+olustur",
    r"\byeni\s+(?:bir\s+)?(?:agent|ajan)\b",
    r"\b(?:agent|ajan)\s+yarat",
    r"\bcreate\s+(?:an?\s+)?(?:new\s+)?agent\b",
    r"\bi\s+want\s+to\s+create\s+(?:an?\s+)?agent\b",
    r"\bmake\s+(?:an?\s+)?(?:new\s+)?agent\b",
]

_TURKISH_MAP = str.maketrans("ıİşŞğĞüÜöÖçÇ", "iIsSgGuUoOcC")


def _normalize(text: str) -> str:
    return text.translate(_TURKISH_MAP).lower()


def detect_intent(text: str) -> bool:
    """True when the message asks to create an agent (TR or EN)."""
    if not text:
        return False
    norm = _normalize(text)
    return any(re.search(p, norm) for p in _INTENT_PATTERNS)


async def extract_params(message: str, llm) -> Dict[str, Optional[str]]:
    """Pull an agent name and role out of the creation message via one LLM call.

    Fail-soft: returns {"name": None, "role": None} when the message has no
    details or the call/parsing fails.
    """
    prompt = (
        "Kullanıcı bir agent oluşturmak istiyor. Mesajından agent adını ve görevini çıkar.\n"
        f"Mesaj: {message}\n\n"
        'Sadece JSON yaz: {"name": "...", "role": "..."}\n'
        'Mesajda ad veya görev belirtilmemişse o alanı null yap.'
    )
    try:
        raw = await llm.generate(prompt)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return {"name": None, "role": None}
        data = json.loads(m.group(0))
        name = data.get("name")
        role = data.get("role")
        return {
            "name": str(name).strip() if name else None,
            "role": str(role).strip() if role else None,
        }
    except Exception:
        return {"name": None, "role": None}


def _slugify(name: str) -> str:
    norm = unicodedata.normalize("NFKD", name.translate(_TURKISH_MAP))
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    norm = re.sub(r"[^a-zA-Z0-9]+", "_", norm).strip("_").lower()
    return norm or "agent"


def write_definition_file(
    agents_store_dir: str,
    *,
    agent_id: int,
    name: str,
    system_prompt: str,
    model: str,
    collection: str,
    agent_type: str = "chatbot",
    source_file: Optional[str] = None,
) -> Optional[str]:
    """Write agents_store/<slug>.json; returns the path or None on failure."""
    try:
        os.makedirs(agents_store_dir, exist_ok=True)
        path = os.path.join(agents_store_dir, f"{_slugify(name)}.json")
        definition = {
            "id": agent_id,
            "name": name,
            "type": agent_type,
            "system_prompt": system_prompt,
            "model": model,
            "collection": collection,
            "source_file": source_file,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(definition, fh, ensure_ascii=False, indent=2)
        return path
    except Exception as e:
        logger.warning("Agent definition file write failed", name=name, error=str(e))
        return None


async def upsert_agent_for_collection(
    pool,
    llm,
    settings,
    *,
    collection: str,
    name: Optional[str] = None,
    role: Optional[str] = None,
    source_file: Optional[str] = None,
    tenant_id: int = 1,
    user_id: int = 1,
) -> Dict[str, Any]:
    """Create the agent for a collection, or rename/re-prompt the existing one.

    The auto-created agent from ingest shares the collection, so a user-requested
    creation upgrades it in place instead of adding a near-duplicate the router
    would have to disambiguate.
    """
    name = name or collection.replace("_", " ").title() + "Bot"
    system_prompt = (
        f"Sen Coiltech {name} uzmanısın. "
        + (f"Görevin: {role}. " if role else "")
        + f"Yalnızca {collection} koleksiyonundaki belgelerden cevap ver. "
        f"Cevaplarını Türkçe ver."
    )
    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT id FROM agents WHERE collection=$1 AND tenant_id=$2 AND user_id=$3 "
            "AND type='chatbot' LIMIT 1",
            collection, tenant_id, user_id,
        )
        if existing:
            agent_id = existing["id"]
            await conn.execute(
                "UPDATE agents SET name=$1, system_prompt=$2 WHERE id=$3",
                name, system_prompt, agent_id,
            )
        else:
            row = await conn.fetchrow(
                """INSERT INTO agents (tenant_id, user_id, name, system_prompt, model, collection)
                   VALUES ($1,$2,$3,$4,$5,$6) RETURNING id""",
                tenant_id, user_id, name, system_prompt,
                settings.DEFAULT_MODEL, collection,
            )
            agent_id = row["id"]
        emb_text = (
            f"{name}. "
            + (f"{role}. " if role else "")
            + f"{collection.replace('_', ' ')}. {system_prompt[:200]}"
        )
        emb = await llm.get_embedding(emb_text)
        emb_str = "[" + ",".join(str(v) for v in emb) + "]"
        await conn.execute(
            "UPDATE agents SET embedding=$1::vector WHERE id=$2", emb_str, agent_id
        )

    write_definition_file(
        settings.AGENTS_STORE_DIR,
        agent_id=agent_id,
        name=name,
        system_prompt=system_prompt,
        model=settings.DEFAULT_MODEL,
        collection=collection,
        source_file=source_file,
    )
    try:
        from backend.deps import get_router
        get_router().invalidate_cache()
    except RuntimeError:
        pass
    logger.info("Agent upserted from chat request", name=name, collection=collection)
    return {"id": agent_id, "name": name, "collection": collection}


async def find_recent_document(
    pool,
    *,
    tenant_id: int = 1,
    user_id: int = 1,
    since_seconds: int = RECENT_DOCUMENT_WINDOW_SECONDS,
) -> Optional[Dict[str, Any]]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT id, original_filename, rag_collection FROM documents
               WHERE tenant_id=$1 AND user_id=$2
                 AND created_at > NOW() - ($3 || ' seconds')::INTERVAL
               ORDER BY created_at DESC LIMIT 1""",
            tenant_id, user_id, str(since_seconds),
        )
    return dict(row) if row else None


async def create_pending(
    pool,
    *,
    raw_message: str,
    params: Dict[str, Any],
    ttl_minutes: int,
    tenant_id: int = 1,
    user_id: int = 1,
) -> None:
    """Store the request; the newest pending request wins, older ones expire."""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE pending_agent_requests SET status='expired' "
            "WHERE tenant_id=$1 AND user_id=$2 AND status='pending'",
            tenant_id, user_id,
        )
        await conn.execute(
            """INSERT INTO pending_agent_requests (tenant_id, user_id, raw_message, params)
               VALUES ($1,$2,$3,$4)""",
            tenant_id, user_id, raw_message, json.dumps(params),
        )


async def fulfill_pending_after_ingest(
    pool,
    llm,
    settings,
    *,
    collection: str,
    source_file: Optional[str] = None,
    tenant_id: int = 1,
    user_id: int = 1,
) -> Optional[Dict[str, Any]]:
    """Called after a document ingest; creates the agent for an open pending request.

    Returns the created agent dict, or None when no valid pending request exists.
    """
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT id, params FROM pending_agent_requests
                   WHERE tenant_id=$1 AND user_id=$2 AND status='pending'
                     AND created_at > NOW() - ($3 || ' minutes')::INTERVAL
                   ORDER BY created_at DESC LIMIT 1""",
                tenant_id, user_id, str(settings.PENDING_AGENT_TTL_MINUTES),
            )
        if not row:
            return None
        params = row["params"]
        if isinstance(params, str):
            params = json.loads(params or "{}")
        agent = await upsert_agent_for_collection(
            pool, llm, settings,
            collection=collection,
            name=(params or {}).get("name"),
            role=(params or {}).get("role"),
            source_file=source_file,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        async with pool.acquire() as conn:
            await conn.execute(
                """UPDATE pending_agent_requests
                   SET status='fulfilled', fulfilled_agent_id=$1, fulfilled_agent_name=$2,
                       fulfilled_at=NOW()
                   WHERE id=$3""",
                agent["id"], agent["name"], row["id"],
            )
        return agent
    except Exception as e:
        logger.warning("Pending agent fulfillment failed", error=str(e))
        return None


async def pop_unannounced(
    pool,
    *,
    tenant_id: int = 1,
    user_id: int = 1,
) -> Optional[str]:
    """Return the name of a fulfilled-but-unannounced agent and mark it announced."""
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """UPDATE pending_agent_requests SET status='announced'
                   WHERE id = (
                       SELECT id FROM pending_agent_requests
                       WHERE tenant_id=$1 AND user_id=$2 AND status='fulfilled'
                       ORDER BY fulfilled_at DESC LIMIT 1
                   )
                   RETURNING fulfilled_agent_name""",
                tenant_id, user_id,
            )
        return row["fulfilled_agent_name"] if row else None
    except Exception:
        return None
