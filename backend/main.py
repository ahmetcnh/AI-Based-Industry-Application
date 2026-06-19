import base64
import json
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
import structlog
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from backend.settings import get_settings
from backend.db import init_pool, close_pool, init_schema
from backend.llm_providers import LLMProvider
from backend.pipeline import RAGPipeline
from backend.router_agent import RouterAgent
from backend.deps import set_llm, set_router
from backend.tenancy import TenancyMiddleware, get_tenant
from backend.tenants import router as tenants_router
from backend.agents import router as agents_router
from backend.chats import router as chats_router

logger = structlog.get_logger(__name__)
settings = get_settings()

_pool = None
_llm: Optional[LLMProvider] = None
_rag: Optional[RAGPipeline] = None
_router: Optional[RouterAgent] = None
_file_watcher = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool, _llm, _rag, _router, _file_watcher

    _pool = await init_pool(settings.DATABASE_URL)
    await init_schema(_pool)

    _llm = LLMProvider(settings)
    _rag = RAGPipeline(_pool, _llm, settings)
    _router = RouterAgent(_pool, _llm, settings)
    set_llm(_llm)
    set_router(_router)

    if os.path.exists(settings.EXCEL_SEED_PATH):
        try:
            from backend.rag.ingestion.excel_seed import run_seed_if_needed
            await run_seed_if_needed(_pool, _llm, settings.EXCEL_SEED_PATH)
        except Exception as e:
            logger.warning("Excel seed failed", error=str(e))
        try:
            from backend.sql_agent import seed_service_records
            await seed_service_records(_pool, settings.EXCEL_SEED_PATH)
        except Exception as e:
            logger.warning("service_records seed failed", error=str(e))
    try:
        from backend.sql_agent import ensure_sql_agent
        await ensure_sql_agent(_pool, _llm, settings)
    except Exception as e:
        logger.warning("SQL agent setup failed", error=str(e))
    try:
        from backend.department_agents import ensure_department_agents
        await ensure_department_agents(_pool, _llm, settings)
    except Exception as e:
        logger.warning("Department agent setup failed", error=str(e))
    try:
        from backend.digital_twin import ensure_digital_twin_agent, resolve_public_base_url
        resolve_public_base_url(settings)
        await ensure_digital_twin_agent(_pool, _llm, settings)
    except Exception as e:
        logger.warning("Digital twin agent setup failed", error=str(e))

    if settings.OPENWEBUI_WATCH_ENABLED:
        try:
            from backend.file_watcher import FileWatcher
            os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
            watch_dirs = [settings.UPLOAD_DIR]
            if settings.OPENWEBUI_UPLOADS_DIR:
                watch_dirs.append(settings.OPENWEBUI_UPLOADS_DIR)
            _file_watcher = FileWatcher(
                watch_dirs=watch_dirs,
                rag_pipeline=_rag,
                allowed_extensions=settings.allowed_extensions_list,
                scan_interval=settings.OPENWEBUI_SCAN_INTERVAL,
            )
            await _file_watcher.start()
        except Exception as e:
            logger.warning("File watcher start failed", error=str(e))

    yield

    if _file_watcher:
        await _file_watcher.stop()
    if _llm:
        await _llm.aclose()
    await close_pool()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(TenancyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tenants_router)
app.include_router(agents_router)
app.include_router(chats_router)

_DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "static", "dashboard")
_DASHBOARD_HTML = os.path.join(_DASHBOARD_DIR, "index.html")


@app.get("/dashboard", tags=["Digital Twin"])
async def dashboard():
    if not os.path.exists(_DASHBOARD_HTML):
        raise HTTPException(status_code=503, detail="Dashboard not found")
    return FileResponse(_DASHBOARD_HTML, media_type="text/html")


@app.get("/api/digital-twin/status", tags=["Digital Twin"])
async def digital_twin_status():
    from backend.digital_twin import get_status
    return JSONResponse(get_status())


def _require_rag() -> RAGPipeline:
    if not _rag:
        raise HTTPException(status_code=503, detail="RAG pipeline not ready")
    return _rag


def _extract_user_message(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            return _strip_openwebui_wrapper(str(content).strip())
    return ""


def _strip_openwebui_wrapper(text: str) -> str:
    for pat in [
        r"(?is)<(?:question|query)>\s*(.*?)\s*</(?:question|query)>",
        r"(?is)(?:^|\n)\s*(?:user\s+question|question|soru)\s*:\s*(.+?)(?:\n|$)",
    ]:
        m = re.findall(pat, text)
        if m and m[-1].strip():
            return m[-1].strip()
    return text


def _openwebui_utility(msg: str) -> Optional[str]:
    raw = (msg or "").lower()
    if "### task:" not in raw:
        return None
    if "generate a concise" in raw and "title" in raw:
        return settings.APP_NAME
    if "generate 1-3 broad tags" in raw or "categorizing" in raw:
        return json.dumps({"tags": ["coiltech", "industrial-ai"]})
    if "follow-up questions" in raw or '"follow_ups"' in raw:
        return json.dumps({"follow_ups": []})
    return None


async def _condense_question(
    messages: List[Dict[str, Any]],
    last_user_msg: str,
    llm: "LLMProvider",
) -> Optional[str]:
    """Rewrite last user message into a standalone question using prior history.

    Returns None if there is no prior history or if condensation fails.
    """
    non_system = [m for m in messages if m.get("role") in ("user", "assistant")]
    # Need at least 2 user turns to benefit from condensation
    user_turns = [m for m in non_system if m.get("role") == "user"]
    if len(user_turns) < 2:
        return None
    try:
        # Last 3 turns excluding the final user message (already in last_user_msg)
        prior = non_system[:-1][-6:]
        history_lines = []
        for m in prior:
            role = "Kullanıcı" if m["role"] == "user" else "Asistan"
            content = str(m.get("content") or "")
            if isinstance(m.get("content"), list):
                content = "\n".join(
                    p.get("text", "") for p in m["content"] if isinstance(p, dict)
                )
            history_lines.append(f"{role}: {content[:500]}")
        history_text = "\n".join(history_lines)
        prompt = (
            f"Sohbet geçmişi:\n{history_text}\n\n"
            f"Son soru: {last_user_msg}\n\n"
            "Sohbet geçmişini kullanarak son soruyu tek başına anlaşılır, bağımsız bir soruya dönüştür. "
            "Eğer son soru önceki konudan tamamen bağımsız yeni bir konuysa, soruyu değiştirmeden "
            "olduğu gibi geri ver. Sadece soruyu yaz."
        )
        condensed = await llm.generate(prompt)
        condensed = condensed.strip()
        return condensed if condensed else None
    except Exception:
        return None


@app.get("/", tags=["Health"])
async def root():
    return {"name": settings.APP_NAME, "version": settings.APP_VERSION, "status": "running"}


from backend.answer_cache import cache_lookup as _ac_lookup, cache_store as _ac_store
from backend import agent_creation
from backend import sql_agent as sql_agent_mod


def _static_response(text: str, model: str, stream: bool):
    """Return a fixed text as a normal or SSE response (warnings, confirmations)."""
    if not stream:
        return _build_response(text, model)

    async def g():
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(datetime.now(timezone.utc).timestamp())
        chunk = {
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield f"data: {json.dumps({**chunk, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(g(), media_type="text/event-stream")


async def _handle_agent_creation(
    user_message: str,
    model: str,
    stream: bool,
    tenant_id: int,
    user_id: int,
):
    """User asked to create an agent: bind to a just-uploaded file, or warn and wait."""
    llm = _require_llm()

    params = await agent_creation.extract_params(user_message, llm)
    # Check for an explicitly-uploaded document first (API /upload path).
    doc = await agent_creation.find_recent_document(
        _pool, tenant_id=tenant_id, user_id=user_id
    )
    # If nothing found yet, trigger file-watcher scan (Open WebUI attachment path)
    # and try once more.
    if not doc and _file_watcher:
        try:
            await _file_watcher.scan_now()
        except Exception as e:
            logger.warning("On-demand scan failed", error=str(e))
        doc = await agent_creation.find_recent_document(
            _pool, tenant_id=tenant_id, user_id=user_id
        )
    if doc:
        # Always detect collection from PDF content, not from the filename-based
        # ingest assignment. This ensures the agent maps to the correct collection
        # regardless of how the file was named.
        collection = await agent_creation.detect_collection_from_content(
            _pool, llm,
            document_id=doc["id"],
            tenant_id=tenant_id,
            user_id=user_id,
            default_collection=doc["rag_collection"],
        )
        # If detected collection differs from stored one, migrate chunks + documents
        # so that RAG queries on the new collection actually find the content.
        if collection != doc["rag_collection"]:
            try:
                async with _pool.acquire() as _conn:
                    await _conn.execute(
                        """UPDATE chunks SET collection=$1
                           WHERE document_id IN (
                               SELECT id FROM documents
                               WHERE original_filename=$2 AND tenant_id=$3
                           )""",
                        collection, doc["original_filename"], tenant_id,
                    )
                    await _conn.execute(
                        "UPDATE documents SET rag_collection=$1 WHERE original_filename=$2 AND tenant_id=$3",
                        collection, doc["original_filename"], tenant_id,
                    )
                logger.info("Migrated chunks to detected collection",
                            from_col=doc["rag_collection"], to_col=collection)
            except Exception as _e:
                logger.warning("Chunk migration failed", error=str(_e))
        agent = await agent_creation.upsert_agent_for_collection(
            _pool, llm, settings,
            collection=collection,
            name=params.get("name"),
            role=params.get("role"),
            source_file=doc.get("original_filename"),
            tenant_id=tenant_id,
            user_id=user_id,
        )
        return _static_response(
            f"✅ **{agent['name']}** agent'ı oluşturuldu (koleksiyon: {agent['collection']}).\n\n"
            f"Tanım dosyası `agents_store/` klasörüne yazıldı. "
            f"`coiltech-router` modeli artık ilgili soruları bu agent'a yönlendirecek.",
            model, stream,
        )

    await agent_creation.create_pending(
        _pool,
        raw_message=user_message,
        params=params,
        ttl_minutes=settings.PENDING_AGENT_TTL_MINUTES,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    return _static_response(
        "⚠️ Agent oluşturmak için bir doküman gerekli. Lütfen agent'ın kullanacağı dosyayı "
        "(PDF, DOCX, XLSX veya TXT) bu sohbete yükleyin.\n\n"
        "Dosyayı yüklediğinizde agent, az önceki isteğinizdeki bilgilerle otomatik olarak oluşturulacak.",
        model, stream,
    )


def _build_citation_chunk(sources: list, cid: str, model: str) -> str:
    """Build an Open WebUI-compatible citation SSE chunk."""
    seen = {}
    for s in sources:
        meta = s.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        name = meta.get("source") or meta.get("title") or "unknown"
        if name not in seen:
            seen[name] = s.get("content", "")
    owui_sources = [
        {
            "source": {"name": name},
            "document": [content],
            "metadata": [{"source": name}],
        }
        for name, content in seen.items()
    ]
    chunk = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(datetime.now(timezone.utc).timestamp()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}],
        "sources": owui_sources,
    }
    return f"data: {json.dumps(chunk)}\n\n"


async def _cache_lookup(pool, llm, question, tenant_id, user_id, collection):
    if not settings.ANSWER_CACHE_ENABLED:
        return None
    return await _ac_lookup(
        pool, llm, question, tenant_id, user_id, collection,
        settings.ANSWER_CACHE_TTL_SECONDS,
    )


async def _cache_store(pool, llm, question, answer, sources, tenant_id, user_id, collection):
    if not settings.ANSWER_CACHE_ENABLED:
        return
    await _ac_store(pool, llm, question, answer, sources, tenant_id, user_id, collection)


async def _log_cache_hit(pool, question: str, collection: str, tenant_id: int, user_id: int, duration_ms: int) -> None:
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO query_logs (tenant_id, user_id, query, collection, sources_count, confidence, duration_ms, cached)
                   VALUES ($1, $2, $3, $4, 0, 1.0, $5, TRUE)""",
                tenant_id, user_id, question, collection, duration_ms,
            )
    except Exception:
        pass


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "service": settings.APP_NAME}


@app.get("/v1/models", tags=["OpenAI Compatibility"])
async def openai_models():
    created = int(datetime.now(timezone.utc).timestamp())
    models = [
        {"id": "coiltech-rag", "object": "model", "created": created, "owned_by": "coiltech"},
        {"id": "coiltech-router", "object": "model", "created": created, "owned_by": "coiltech"},
    ]
    if _llm:
        for m in await _llm.list_ollama_models():
            models.append({"id": f"ollama:{m}", "object": "model", "created": created, "owned_by": "ollama"})
        for m in settings.deepseek_models_list:
            models.append({"id": m, "object": "model", "created": created, "owned_by": "deepseek"})
        for m in settings.gemini_models_list:
            models.append({"id": m, "object": "model", "created": created, "owned_by": "google"})
        if _pool:
            from backend.db import get_pool
            pool = await get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch("SELECT id, name FROM agents")
                for r in rows:
                    models.append({
                        "id": f"agent-{r['id']}-{r['name'].lower().replace(' ', '-')}",
                        "object": "model",
                        "created": created,
                        "owned_by": "coiltech",
                    })
    return {"object": "list", "data": models}


@app.get("/v1/llm-status", tags=["Health"])
async def llm_status():
    if not _llm:
        return {"status": "not_ready"}
    return await _llm.check_health()


@app.get("/metrics", tags=["Health"])
async def get_metrics(request: Request, hours: int = 24):
    if not _pool:
        raise HTTPException(status_code=503, detail="Database not ready")
    # Clamp hours to prevent full-table scans
    hours = max(1, min(hours, 720))
    tenant_id, user_id = get_tenant(request)
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                collection,
                COUNT(*) AS query_count,
                AVG(duration_ms) AS avg_duration_ms,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95_duration_ms,
                AVG(confidence) AS avg_confidence,
                AVG(sources_count) AS avg_sources_count,
                COUNT(*) FILTER (WHERE sources_count = 0 OR sources_count IS NULL) AS zero_source_queries,
                COUNT(*) FILTER (WHERE cached = TRUE) AS cache_hits
            FROM query_logs
            WHERE tenant_id=$1 AND user_id=$2
              AND created_at > NOW() - ($3 || ' hours')::INTERVAL
            GROUP BY ROLLUP(collection)
            ORDER BY collection NULLS LAST
            """,
            tenant_id, user_id, str(hours),
        )
    result = []
    for r in rows:
        total = r["query_count"] or 0
        hits = r["cache_hits"] or 0
        result.append({
            "collection": r["collection"],
            "query_count": total,
            "avg_duration_ms": round(r["avg_duration_ms"] or 0, 1),
            "p95_duration_ms": round(r["p95_duration_ms"] or 0, 1),
            "avg_confidence": round(r["avg_confidence"] or 0, 4),
            "avg_sources_count": round(r["avg_sources_count"] or 0, 2),
            "zero_source_queries": r["zero_source_queries"] or 0,
            "cache_hits": hits,
            "cache_hit_rate": round(hits / total, 4) if total else 0.0,
        })
    return {"hours": hours, "metrics": result}


@app.post("/upload", tags=["Documents"])
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    collection_name: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    language: str = Form("tr"),
    metadata_json: Optional[str] = Form(None),
):
    rag = _require_rag()
    tenant_id, user_id = get_tenant(request)
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in settings.allowed_extensions_list:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail="File too large")

    safe_name = f"{uuid.uuid4().hex}_{os.path.basename(file.filename or 'upload.bin')}"
    target = os.path.join(settings.UPLOAD_DIR, safe_name)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    with open(target, "wb") as fh:
        fh.write(content)

    metadata = {}
    if metadata_json:
        try:
            metadata = json.loads(metadata_json)
        except Exception:
            pass
    if title:
        metadata["title"] = title
    metadata["language"] = language

    collection = rag.resolve_collection(collection_name, metadata, target)
    result = await rag.ingest_async(
        file_path=target,
        department=collection,
        metadata=metadata,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    return {"status": "success", "file_name": file.filename, **result}


@app.get("/collections", tags=["Documents"])
async def list_collections(request: Request):
    tenant_id, user_id = get_tenant(request)
    from backend.db import get_pool
    db = await get_pool()
    async with db.acquire() as conn:
        rows = await conn.fetch(
            "SELECT collection, count(*) as chunk_count FROM chunks "
            "WHERE tenant_id=$1 AND user_id=$2 GROUP BY collection ORDER BY collection",
            tenant_id, user_id,
        )
    return {"status": "success", "collections": [dict(r) for r in rows]}


def _require_llm() -> "LLMProvider":
    if not _llm:
        raise HTTPException(status_code=503, detail="LLM provider not ready")
    return _llm


async def _handle_router_request(
    user_message: str,
    messages: List[Dict[str, Any]],
    stream: bool,
    tenant_id: int,
    user_id: int,
    prefix: str = "",
    original_question: Optional[str] = None,
):
    rag = _require_rag()
    llm = _require_llm()
    agent = None

    # If the last assistant turn came from Dijital İkiz AND the current question
    # looks like a live-status follow-up, pin routing to that agent.
    # Only pin when the message references machines, live metrics, or real-time state —
    # not for unrelated questions that happen to follow a DT response.
    _DT_FOLLOWUP_PATTERN = re.compile(
        r"\b(ctl[-\s]?\d+|ptr[-\s]?\d+|ack[-\s]?\d+|dst[-\s]?\d+|sbl[-\s]?\d+|lsd[-\s]?\d+"
        r"|su\s*an|simdiki|anlik|sicakl|basinc|oee|alarm|calis[iy]|durdu|bekle"
        r"|makine\s*durumu|sensor|vardiya|uretim\s*say|hiz\s*kac)\b",
        re.IGNORECASE,
    )
    _last_assistant = next(
        (m for m in reversed(messages) if m.get("role") == "assistant"), None
    )
    _last_was_dt = _last_assistant and "Dijital İkiz" in str(
        _last_assistant.get("content", "")
    )

    if _last_was_dt and _router:
        _q_for_dt_check = original_question if original_question else user_message
        _q_norm = _q_for_dt_check.translate(str.maketrans("ıİşŞğĞüÜöÖçÇ", "iIsSgGuUoOcC")).lower()
        if _DT_FOLLOWUP_PATTERN.search(_q_norm):
            try:
                async with _router.pool.acquire() as _conn:
                    _row = await _conn.fetchrow(
                        "SELECT id, name, system_prompt, model, collection, type, embedding "
                        "FROM agents WHERE name='Dijital İkiz' AND tenant_id=$1 AND user_id=$2 LIMIT 1",
                        tenant_id, user_id,
                    )
                    if _row:
                        agent = dict(_row)
                        logger.info("Follow-up pinned to Dijital İkiz")
            except Exception as e:
                logger.warning("DT follow-up pin failed", error=str(e))

    if agent is None and _router:
        # Route on the ORIGINAL question so condensed context (e.g. "CTL-01 enkoder...")
        # doesn't pull an unrelated question toward Dijital İkiz.
        _route_question = original_question if original_question else user_message
        try:
            agent = await _router.route(_route_question, tenant_id=tenant_id, user_id=user_id)
        except Exception as e:
            logger.warning("Router failed, using default collection", error=str(e))

    collection = agent["collection"] if agent else settings.OPENWEBUI_DEFAULT_COLLECTION
    agent_model = agent["model"] if agent else settings.DEFAULT_MODEL
    agent_system_prompt = agent["system_prompt"] if agent else None
    routing_label = prefix + (f"[→ {agent['name']}]\n\n" if agent else "")

    # Digital Twin answers are live data — never serve from cache.
    is_digital_twin = agent and agent.get("type") == "digital_twin"

    # Cache lookup for router path (skip for digital twin)
    if not is_digital_twin:
        cached = await _cache_lookup(_pool, llm, user_message, tenant_id, user_id, collection)
        if cached:
            import time as _time; _t0 = _time.time()
            if stream:
                async def router_cached_gen():
                    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                    if routing_label:
                        yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': int(datetime.now(timezone.utc).timestamp()), 'model': 'coiltech-router', 'choices': [{'index': 0, 'delta': {'content': routing_label}, 'finish_reason': None}]})}\n\n"
                    yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': int(datetime.now(timezone.utc).timestamp()), 'model': 'coiltech-router', 'choices': [{'index': 0, 'delta': {'content': cached['answer']}, 'finish_reason': None}]})}\n\n"
                    yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': int(datetime.now(timezone.utc).timestamp()), 'model': 'coiltech-router', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                    yield "data: [DONE]\n\n"
                    await _log_cache_hit(_pool, user_message, collection, tenant_id, user_id, int((_time.time()-_t0)*1000))
                return StreamingResponse(router_cached_gen(), media_type="text/event-stream")
            await _log_cache_hit(_pool, user_message, collection, tenant_id, user_id, int((_time.time()-_t0)*1000))
            return _build_response(routing_label + cached["answer"], "coiltech-router")

    # Digital Twin agent: inject live sensor data into the system prompt.
    if is_digital_twin:
        from backend.digital_twin import build_agent_context, _SYSTEM_PROMPT as _dt_sysprompt
        dt_context = build_agent_context()
        agent_system_prompt = _dt_sysprompt() + "\n\n" + dt_context

    # SQL agent: answer from service_records via NL→SQL; on no rows / bad SQL
    # fall back to RAG over the default maintenance collection.
    sql_result = None
    if agent and agent.get("type") == "sql":
        _sql_question = original_question if original_question else user_message
        sql_result = await sql_agent_mod.try_sql_path(_pool, llm, _sql_question)
        if sql_result is None:
            logger.info("SQL path empty, falling back to RAG")
            collection = settings.OPENWEBUI_DEFAULT_COLLECTION
            agent_system_prompt = None

    if stream:
        async def gen():
            cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            # First chunk: routing label
            if routing_label:
                label_chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now(timezone.utc).timestamp()),
                    "model": "coiltech-router",
                    "choices": [{"index": 0, "delta": {"content": routing_label}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(label_chunk)}\n\n"
            if is_digital_twin:
                prompt = f"{agent_system_prompt}\n\nKullanıcı sorusu: {user_message}"
                sources = []
                # stream with skip_ollama_fallback so 429 raises immediately
            elif sql_result:
                prompt = sql_agent_mod.build_summary_prompt(
                    _sql_question, sql_result["rows"], agent_system_prompt
                )
                sources = []
            else:
                prompt, sources = await rag._retrieve(
                    question=user_message,
                    collection=collection,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    system_prompt=agent_system_prompt,
                )
            accumulated = []
            try:
                async for token in llm.generate_stream(prompt, model=agent_model,
                                                       skip_ollama_fallback=is_digital_twin):
                    accumulated.append(token)
                    chunk = {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": int(datetime.now(timezone.utc).timestamp()),
                        "model": "coiltech-router",
                        "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            except Exception as e:
                err_str = str(e)
                if is_digital_twin:
                    from backend.digital_twin import build_direct_response
                    msg = build_direct_response(user_message)
                    logger.info("Digital twin LLM unavailable, using direct response")
                elif "429" in err_str:
                    msg = "Şu an yapay zeka servisine çok fazla istek gönderildi. Lütfen bir dakika bekleyip tekrar deneyin."
                else:
                    msg = f"Yanıt oluşturulurken bir hata oluştu."
                    logger.error("Streaming LLM failed", error=err_str[:200])
                err_chunk = {
                    "id": cid, "object": "chat.completion.chunk",
                    "created": int(datetime.now(timezone.utc).timestamp()),
                    "model": "coiltech-router",
                    "choices": [{"index": 0, "delta": {"content": msg}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(err_chunk)}\n\n"
            if sources:
                yield _build_citation_chunk(sources, cid, "coiltech-router")
            done = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": int(datetime.now(timezone.utc).timestamp()),
                "model": "coiltech-router",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"
            full_answer = "".join(accumulated)
            if full_answer and not is_digital_twin:
                await _cache_store(_pool, llm, user_message, full_answer, sources,
                                   tenant_id, user_id, collection)
        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        if is_digital_twin:
            prompt = f"{agent_system_prompt}\n\nKullanıcı sorusu: {user_message}"
            answer = await llm.generate(prompt, model=agent_model, skip_ollama_fallback=True)
            result = {"answer": answer.strip(), "sources": []}
        elif sql_result:
            prompt = sql_agent_mod.build_summary_prompt(
                _sql_question, sql_result["rows"], agent_system_prompt
            )
            answer = await llm.generate(prompt, model=agent_model)
            result = {"answer": answer.strip(), "sources": []}
        else:
            result = await rag.query(
                question=user_message,
                collection=collection,
                tenant_id=tenant_id,
                user_id=user_id,
                model=agent_model,
                system_prompt=agent_system_prompt,
            )
    except Exception as e:
        err_str = str(e)
        if is_digital_twin:
            from backend.digital_twin import build_direct_response
            msg = build_direct_response(user_message)
            logger.info("Digital twin LLM unavailable, using direct response")
        elif "429" in err_str:
            msg = "Şu an yapay zeka servisine çok fazla istek gönderildi. Lütfen bir dakika bekleyip tekrar deneyin."
        else:
            logger.error("LLM generate failed", error=err_str[:200])
            msg = f"Yanıt oluşturulurken bir hata oluştu: {err_str[:100]}"
        return _build_response(routing_label + msg, "coiltech-router")
    if not is_digital_twin:
        await _cache_store(_pool, llm, user_message, result["answer"],
                           result.get("sources", []), tenant_id, user_id, collection)
    response = _build_response(routing_label + result["answer"], "coiltech-router")
    response_data = json.loads(response.body)
    response_data["sources"] = result.get("sources", [])
    return JSONResponse(response_data)


@app.post("/v1/chat/completions", tags=["OpenAI Compatibility"])
async def openai_chat_completions(request: Request):
    body = await request.json()
    rag = _require_rag()
    tenant_id, user_id = get_tenant(request)

    messages = body.get("messages", [])
    model = body.get("model", settings.DEFAULT_MODEL)
    stream = body.get("stream", False)
    collection_name = body.get("collection_name", None)

    user_message = _extract_user_message(messages)
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")

    utility = _openwebui_utility(user_message)
    if utility is not None:
        return _build_response(utility, model)

    # "Bir agent oluştur" / "create agent" → agent creation flow (raw message,
    # before condensation so history can't rewrite the command)
    if agent_creation.detect_intent(user_message):
        return await _handle_agent_creation(
            user_message, model, stream, tenant_id, user_id
        )

    # A file upload may have fulfilled a pending agent request since the last
    # message; announce it once at the top of this response.
    confirm_prefix = ""
    announced = await agent_creation.pop_unannounced(
        _pool, tenant_id=tenant_id, user_id=user_id
    )
    if announced:
        confirm_prefix = f"✅ **{announced}** agent'ı oluşturuldu.\n\n"

    # Condense follow-up questions into standalone queries
    llm = _require_llm()
    condensed = await _condense_question(messages, user_message, llm)
    question = condensed if condensed else user_message

    if model in ("coiltech-router", "coiltech-rag", "default"):
        return await _handle_router_request(
            question, messages, stream, tenant_id, user_id, prefix=confirm_prefix,
            original_question=user_message,
        )

    # Virtual model ids (coiltech-rag, agent-*) are not real Ollama models;
    # resolve them to a real generation model before calling the LLM.
    collection = collection_name or settings.OPENWEBUI_DEFAULT_COLLECTION
    gen_model = model
    agent_system_prompt = None
    if model == "coiltech-rag":
        gen_model = settings.DEFAULT_MODEL
    else:
        agent_match = re.match(r"^agent-(\d+)-", model)
        if agent_match:
            gen_model = settings.DEFAULT_MODEL
            async with _pool.acquire() as conn:
                agent_row = await conn.fetchrow(
                    "SELECT system_prompt, model, collection FROM agents "
                    "WHERE id=$1 AND tenant_id=$2 AND user_id=$3",
                    int(agent_match.group(1)), tenant_id, user_id,
                )
            if agent_row:
                gen_model = agent_row["model"] or settings.DEFAULT_MODEL
                agent_system_prompt = agent_row["system_prompt"]
                collection = collection_name or agent_row["collection"] or collection

    # Cache lookup (uses embedding LRU so fast on repeated questions)
    import time as _time
    cached = await _cache_lookup(_pool, llm, question, tenant_id, user_id, collection)
    if cached:
        _t0 = _time.time()
        if stream:
            async def cached_gen():
                cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                chunk = {
                    "id": cid, "object": "chat.completion.chunk",
                    "created": int(datetime.now(timezone.utc).timestamp()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": confirm_prefix + cached["answer"]}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield f"data: {json.dumps({**chunk, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
                await _log_cache_hit(_pool, question, collection, tenant_id, user_id, int((_time.time()-_t0)*1000))
            return StreamingResponse(cached_gen(), media_type="text/event-stream")
        await _log_cache_hit(_pool, question, collection, tenant_id, user_id, int((_time.time()-_t0)*1000))
        return _build_response(confirm_prefix + cached["answer"], model)

    if stream:
        async def gen():
            prompt, sources = await rag._retrieve(
                question=question,
                collection=collection,
                tenant_id=tenant_id,
                user_id=user_id,
                system_prompt=agent_system_prompt,
            )
            cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            if confirm_prefix:
                pchunk = {
                    "id": cid, "object": "chat.completion.chunk",
                    "created": int(datetime.now(timezone.utc).timestamp()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": confirm_prefix}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(pchunk)}\n\n"
            accumulated = []
            async for token in llm.generate_stream(prompt, model=gen_model):
                accumulated.append(token)
                chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now(timezone.utc).timestamp()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            if sources:
                yield _build_citation_chunk(sources, cid, model)
            done = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": int(datetime.now(timezone.utc).timestamp()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"
            full_answer = "".join(accumulated)
            await _cache_store(_pool, llm, question, full_answer, sources,
                               tenant_id, user_id, collection)
        return StreamingResponse(gen(), media_type="text/event-stream")

    result = await rag.query(
        question=question,
        collection=collection,
        tenant_id=tenant_id,
        user_id=user_id,
        model=gen_model,
        system_prompt=agent_system_prompt,
    )
    await _cache_store(_pool, llm, question, result["answer"],
                       result.get("sources", []), tenant_id, user_id, collection)
    response = _build_response(confirm_prefix + result["answer"], model)
    response_data = json.loads(response.body)
    response_data["sources"] = result.get("sources", [])
    return JSONResponse(response_data)


def _build_response(content: str, model: str) -> JSONResponse:
    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(datetime.now(timezone.utc).timestamp()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


@app.get("/me", tags=["Tenants"])
async def me(request: Request):
    tenant_id, user_id = get_tenant(request)
    return {"tenant_id": tenant_id, "user_id": user_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=settings.DEBUG)
