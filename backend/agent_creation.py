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
    meta = _COLLECTION_META.get(collection, {})
    name = name or meta.get("name") or collection.replace("_", " ").title() + "Bot"
    system_prompt = (
        meta.get("system_prompt")
        or (
            f"Sen Coiltech {name} uzmanısın. "
            + (f"Görevin: {role}. " if role else "")
            + f"Yalnızca {collection} koleksiyonundaki belgelerden cevap ver. "
            f"Cevaplarını Türkçe ver."
        )
    )
    if role:
        system_prompt = f"Görevin: {role}. " + system_prompt
    # Use the rich routing description for the embedding so the router can
    # match user questions to this agent with the same quality as built-in agents.
    routing_description = meta.get("routing_description") or f"{name}. {system_prompt[:300]}"
    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT id FROM agents WHERE collection=$1 AND tenant_id=$2 AND (user_id=$3 OR user_id=1) "
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
        emb_text = routing_description
        try:
            emb = await llm.get_embedding(emb_text)
            emb_str = "[" + ",".join(str(v) for v in emb) + "]"
            await conn.execute(
                "UPDATE agents SET embedding=$1::vector WHERE id=$2", emb_str, agent_id
            )
        except Exception as emb_err:
            logger.warning("Agent embedding unavailable (LLM down?), agent created without vector", error=str(emb_err))

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


# Turkish name + routing description for known collections (mirrors department_agents.py)
_COLLECTION_META: Dict[str, Dict[str, str]] = {
    "special_projects": {
        "name": "Özel Projeler",
        "system_prompt": (
            "Sen Coiltech özel projeler ve referans portföy uzmanısın. Müşteriye özel "
            "mühendislik projeleri, tamamlanan referans projeler (Kardemir, Ford Otosan, "
            "Erdemir vb.), turnkey çözümler ve özel makine tasarımları hakkında "
            "Türkçe cevap verirsin."
        ),
        "routing_description": (
            "Özel müşteri projeleri ve referans proje portföyü. Müşteriye özel tasarım, "
            "turnkey çözüm, anahtar teslim proje, özel mühendislik projesi, referans proje, "
            "geçmiş projeler, tamamlanan projeler, müşteri referansı. "
            "Kardemir modernizasyon, Ford Otosan pres hattı, Erdemir servis hattı gibi "
            "gerçekleştirilmiş endüstriyel projeler. Portföy, müşteri spesifik çözüm. "
            "Örnek sorular: daha önce hangi müşteriler için özel proje yapıldı, "
            "referans projeler neler, turnkey proje kapsamı nedir, bu projeyle ilgili bilgi ver."
        ),
    },
    "spare_parts": {
        "name": "Yedek Parça",
        "system_prompt": "Sen Coiltech yedek parça uzmanısın. Parça numaraları, stok durumu ve bakım malzemeleri hakkında Türkçe cevap verirsin.",
        "routing_description": "Yedek parça, spare part, parça numarası, stok, yedek parça siparişi.",
    },
    "installation_commissioning": {
        "name": "Kurulum ve Devreye Alma",
        "system_prompt": "Sen Coiltech kurulum ve devreye alma uzmanısın. Montaj talimatları ve devreye alma prosedürleri hakkında Türkçe cevap verirsin.",
        "routing_description": "Kurulum, devreye alma, installation, commissioning, montaj talimatı, startup.",
    },
    "quality_documents": {
        "name": "Kalite ve Uyumluluk",
        "system_prompt": "Sen Coiltech kalite belgeleri uzmanısın. CE belgeleri, sertifikalar ve uygunluk beyanları hakkında Türkçe cevap verirsin.",
        "routing_description": "Kalite belgesi, CE belgesi, sertifika, uygunluk beyanı, test raporu.",
    },
    "sales_and_quotations": {
        "name": "Satış ve Teklifler",
        "system_prompt": "Sen Coiltech satış ve teklif uzmanısın. Fiyat teklifleri ve ticari teklifler hakkında Türkçe cevap verirsin.",
        "routing_description": "Teklif, satış, fiyat, quotation, commercial, proposal, sipariş.",
    },
    "service_maintenance": {
        "name": "Servis ve Bakım",
        "system_prompt": "Sen Coiltech servis ve bakım uzmanısın. Arıza giderme, bakım prosedürleri ve onarım hakkında Türkçe cevap verirsin.",
        "routing_description": "Bakım, servis, arıza, troubleshooting, alarm, onarım, maintenance.",
    },
}

_COLLECTION_DESCRIPTIONS = {
    "special_projects":          "Özel müşteri projeleri, referans projeler, portföy, turnkey çözüm, müşteriye özel tasarım",
    "spare_parts":               "Yedek parça, spare part, parça numarası, stok, bakım malzemesi",
    "installation_commissioning":"Kurulum, devreye alma, installation, commissioning, montaj talimatı",
    "quality_documents":         "Kalite belgesi, CE belgesi, sertifika, uygunluk beyanı, test raporu",
    "sales_and_quotations":      "Teklif, satış, fiyat, quotation, commercial, proposal",
    "service_maintenance":       "Bakım, servis, arıza, troubleshooting, alarm, onarım, maintenance",
    "products_compact_drives":   "Compact drive, pres besleme sistemi, compact besleme",
    "products_loop_drives":      "Loop drive, loop tipi besleme",
    "products_blank_lines":      "Blank line, blanking hattı, sac kesim",
    "products_destacker":        "Destacker, yığın çözücü",
    "products_press_transfer_robots": "Transfer robot, pres transfer robotu",
    "products_press_transfer":   "Press transfer, transfer sistemi",
    "products_servo_drives":     "Servo drive, servo besleme, zigzag, triggered",
    "products_decoilers":        "Decoiler, uncoiler, mandrel, açıcı",
    "products_ctl_lines":        "Cut to length, CTL, makas hattı, kesme hattı",
    "general":                   "Genel, sınıflandırılamayan, diğer",
}


async def detect_collection_from_content(
    pool,
    llm,
    document_id: int,
    tenant_id: int,
    user_id: int,
    default_collection: str = "general",
) -> str:
    """Read the first few chunks of a document and use LLM to classify its collection."""
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT content FROM chunks
                   WHERE document_id=$1 AND tenant_id=$2 AND user_id=$3
                   ORDER BY id LIMIT 5""",
                document_id, tenant_id, user_id,
            )
            # When the same file was uploaded before, new upload has 0 chunks (already_ingested).
            # Fall back: find chunks from any document with the same base filename (strip UUID prefix).
            if not rows:
                filename_row = await conn.fetchrow(
                    "SELECT original_filename FROM documents WHERE id=$1", document_id
                )
                if filename_row:
                    fname = filename_row["original_filename"] or ""
                    # Strip leading UUID/hash prefix (e.g. "abc123_ozel.pdf" -> "ozel.pdf")
                    import re as _re
                    base = _re.sub(r'^[0-9a-f\-]{8,}_', '', fname)
                    if base and base != fname:
                        rows = await conn.fetch(
                            """SELECT c.content FROM chunks c
                               JOIN documents d ON d.id = c.document_id
                               WHERE d.original_filename ILIKE $1
                                 AND c.tenant_id=$2
                               ORDER BY c.id LIMIT 5""",
                            f"%{base}", tenant_id,
                        )
        if not rows:
            return default_collection

        sample_text = "\n\n---\n\n".join(r["content"] for r in rows)[:3000]
        options = "\n".join(
            f"- {key}: {desc}" for key, desc in _COLLECTION_DESCRIPTIONS.items()
        )
        prompt = (
            "Aşağıdaki belge içeriğine bakarak, bu belgenin hangi koleksiyona ait olduğunu belirle.\n\n"
            f"Koleksiyon seçenekleri:\n{options}\n\n"
            f"Belge içeriği:\n{sample_text}\n\n"
            "Yalnızca koleksiyon anahtar adını yaz (örn: special_projects). Başka hiçbir şey yazma."
        )
        result = await llm.generate(prompt)
        key = result.strip().lower().split()[0] if result.strip() else ""
        key = key.strip(".,;:\"'")
        if key in _COLLECTION_DESCRIPTIONS:
            logger.info("Content-based collection detected", document_id=document_id, collection=key)
            return key
        logger.warning("LLM returned unknown collection key", raw=result.strip())
        return default_collection
    except Exception as e:
        logger.warning("Content-based collection detection failed", error=str(e))
        return default_collection


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
        # Detect the collection from the document's actual content rather than
        # relying on the filename-based assignment from the ingest pipeline.
        async with pool.acquire() as conn:
            doc_id = await conn.fetchval(
                """SELECT id FROM documents
                   WHERE tenant_id=$1 AND user_id=$2
                   ORDER BY created_at DESC LIMIT 1""",
                tenant_id, user_id,
            )
        if doc_id:
            collection = await detect_collection_from_content(
                pool, llm,
                document_id=doc_id,
                tenant_id=tenant_id,
                user_id=user_id,
                default_collection=collection,
            )
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
