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
from fastapi.responses import JSONResponse, StreamingResponse

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
            "Sadece yeniden yazılmış soruyu yaz."
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
):
    rag = _require_rag()
    llm = _require_llm()
    agent = None
    if _router:
        try:
            agent = await _router.route(user_message, tenant_id=tenant_id, user_id=user_id)
        except Exception as e:
            logger.warning("Router failed, using default collection", error=str(e))

    collection = agent["collection"] if agent else settings.OPENWEBUI_DEFAULT_COLLECTION
    agent_model = agent["model"] if agent else settings.DEFAULT_MODEL
    agent_system_prompt = agent["system_prompt"] if agent else None
    routing_label = f"[→ {agent['name']}]\n\n" if agent else ""

    # Cache lookup for router path
    cached = await _cache_lookup(_pool, llm, user_message, tenant_id, user_id, collection)
    if cached:
        if stream:
            async def router_cached_gen():
                cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                if routing_label:
                    yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': int(datetime.now(timezone.utc).timestamp()), 'model': 'coiltech-router', 'choices': [{'index': 0, 'delta': {'content': routing_label}, 'finish_reason': None}]})}\n\n"
                yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': int(datetime.now(timezone.utc).timestamp()), 'model': 'coiltech-router', 'choices': [{'index': 0, 'delta': {'content': cached['answer']}, 'finish_reason': None}]})}\n\n"
                yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': int(datetime.now(timezone.utc).timestamp()), 'model': 'coiltech-router', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(router_cached_gen(), media_type="text/event-stream")
        return _build_response(routing_label + cached["answer"], "coiltech-router")

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
            # RAG retrieve + stream
            prompt, sources = await rag._retrieve(
                question=user_message,
                collection=collection,
                tenant_id=tenant_id,
                user_id=user_id,
                system_prompt=agent_system_prompt,
            )
            accumulated = []
            async for token in llm.generate_stream(prompt, model=agent_model):
                accumulated.append(token)
                chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now(timezone.utc).timestamp()),
                    "model": "coiltech-router",
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
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
            await _cache_store(_pool, llm, user_message, full_answer, sources,
                               tenant_id, user_id, collection)
        return StreamingResponse(gen(), media_type="text/event-stream")

    result = await rag.query(
        question=user_message,
        collection=collection,
        tenant_id=tenant_id,
        user_id=user_id,
        model=agent_model,
        system_prompt=agent_system_prompt,
    )
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

    # Condense follow-up questions into standalone queries
    llm = _require_llm()
    condensed = await _condense_question(messages, user_message, llm)
    question = condensed if condensed else user_message

    if model == "coiltech-router":
        return await _handle_router_request(
            question, messages, stream, tenant_id, user_id
        )

    collection = collection_name or settings.OPENWEBUI_DEFAULT_COLLECTION

    # Cache lookup (uses embedding LRU so fast on repeated questions)
    cached = await _cache_lookup(_pool, llm, question, tenant_id, user_id, collection)
    if cached:
        if stream:
            async def cached_gen():
                cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                chunk = {
                    "id": cid, "object": "chat.completion.chunk",
                    "created": int(datetime.now(timezone.utc).timestamp()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": cached["answer"]}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield f"data: {json.dumps({**chunk, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(cached_gen(), media_type="text/event-stream")
        return _build_response(cached["answer"], model)

    if stream:
        async def gen():
            prompt, sources = await rag._retrieve(
                question=question,
                collection=collection,
                tenant_id=tenant_id,
                user_id=user_id,
            )
            cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            accumulated = []
            async for token in llm.generate_stream(prompt, model=model):
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
        model=model,
    )
    await _cache_store(_pool, llm, question, result["answer"],
                       result.get("sources", []), tenant_id, user_id, collection)
    response = _build_response(result["answer"], model)
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
