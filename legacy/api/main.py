"""
FastAPI application for the Coiltech Industrial Automation Platform.
"""

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
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse

from agents.excel_analyzer_agent import ExcelAnalyzerAgent
from agents.orchestrator import Orchestrator
from api.file_watcher import FileWatcher
from api.middleware import (
    general_exception_handler,
    setup_middleware,
    validation_exception_handler,
)
from api.models import (
    AgentsResponse,
    AlertListResponse,
    AgentInfo,
    ChatRequest,
    ChatResponse,
    CollectionInfo,
    CollectionsResponse,
    DatabaseQueryResponse,
    ExcelAnalysisResponse,
    InstalledBaseListResponse,
    OpenAIChatRequest,
    OpenAIChatResponse,
    ProductFamilyListResponse,
    ProjectDetailResponse,
    ProjectListResponse,
    QuotationListResponse,
    ServiceTicketListResponse,
    SessionInfo,
    SessionResponse,
    SparePartListResponse,
    StatusResponse,
    UploadResponse,
)
from config.settings import RAG_COLLECTION_REGISTRY, get_settings
from database.db_manager import SQLiteManager
from llm.llm_client import LLMClient
from rag.context_cache import CAGContextCache
from rag.rag_pipeline import RAGPipeline

logger = structlog.get_logger(__name__)

settings = get_settings()
db_manager: Optional[SQLiteManager] = None
rag_pipeline: Optional[RAGPipeline] = None
orchestrator: Optional[Orchestrator] = None
cag_cache: Optional[CAGContextCache] = None
_file_watcher: Optional[FileWatcher] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_manager, rag_pipeline, orchestrator, cag_cache, _file_watcher

    logger.info("Starting Coiltech platform")
    db_manager = SQLiteManager()
    db_manager.initialize_schema()

    rag_pipeline = RAGPipeline()

    # CAG context cache (disabled via ENABLE_CAG=false if not wanted)
    if settings.ENABLE_CAG:
        cag_cache = CAGContextCache(
            ttl_seconds=settings.CAG_CACHE_TTL_SECONDS,
            max_entries=settings.CAG_MAX_ENTRIES,
        )
        logger.info(
            "CAG context cache enabled",
            ttl=settings.CAG_CACHE_TTL_SECONDS,
            max_entries=settings.CAG_MAX_ENTRIES,
        )

    orchestrator = Orchestrator(
        db_manager=db_manager,
        rag_pipeline=rag_pipeline,
        cag_cache=cag_cache,
    )

    # File watcher for hot-reload RAG ingestion (enabled via OPENWEBUI_WATCH_ENABLED)
    if settings.OPENWEBUI_WATCH_ENABLED:
        try:
            os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
            watch_dirs = [settings.UPLOAD_DIR]
            if settings.OPENWEBUI_UPLOADS_DIR:
                watch_dirs.append(settings.OPENWEBUI_UPLOADS_DIR)
            _file_watcher = FileWatcher(
                watch_dirs=watch_dirs,
                rag_pipeline=rag_pipeline,
                allowed_extensions=settings.allowed_extensions_list,
                scan_interval=settings.OPENWEBUI_SCAN_INTERVAL,
                cag_cache=cag_cache,
            )
            await _file_watcher.start()
            logger.info("File watcher started", watch_dirs=watch_dirs)
        except Exception as exc:
            logger.warning("File watcher failed to start — continuing without it", error=str(exc))
            _file_watcher = None

    yield

    logger.info("Shutting down Coiltech platform")
    if _file_watcher is not None:
        try:
            await _file_watcher.stop()
        except Exception as exc:
            logger.warning("File watcher shutdown error", error=str(exc))

    if db_manager:
        await db_manager.async_close()
        db_manager.close()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Coiltech industrial automation backend for product knowledge, application "
        "engineering, project execution, service, spare parts, and digital twin monitoring."
    ),
    lifespan=lifespan,
)

setup_middleware(app, settings)
app.add_exception_handler(Exception, general_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)


def _require_services() -> tuple[SQLiteManager, RAGPipeline, Orchestrator]:
    if not db_manager or not rag_pipeline or not orchestrator:
        raise HTTPException(status_code=503, detail="Application services are not initialized")
    return db_manager, rag_pipeline, orchestrator


def _safe_json_loads(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {exc}") from exc


def _extract_text_from_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(part for part in parts if part).strip()
    return ""


def _get_last_user_message(messages: List[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            text = _extract_text_from_message_content(message.get("content"))
            if text:
                return _extract_actual_user_question(text)
    if messages:
        return _extract_actual_user_question(_extract_text_from_message_content(messages[-1].get("content")))
    return ""


def _extract_actual_user_question(text: str) -> str:
    """
    Open WebUI can wrap the user's message with retrieved document context before
    forwarding it to an OpenAI-compatible backend. Route the actual question, not
    the whole RAG wrapper, so a PDF chat can still switch to database/agent tasks.
    """
    stripped = (text or "").strip()
    if not stripped:
        return ""

    patterns = [
        r"(?is)<(?:question|query)>\s*(.*?)\s*</(?:question|query)>",
        r"(?is)(?:^|\n)\s*(?:user\s+question|question|query|soru)\s*:\s*(.+?)(?:\n\s*(?:answer|cevap)\s*:?\s*$|$)",
        r"(?is)(?:^|\n)\s*#{1,6}\s*(?:question|query|soru)\s*\n(.+?)(?:\n\s*#{1,6}\s|\Z)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, stripped)
        if matches:
            candidate = matches[-1].strip()
            if candidate:
                return candidate

    return stripped


def _build_upload_path(original_name: str) -> str:
    safe_name = os.path.basename(original_name or f"upload_{uuid.uuid4().hex}.bin")
    stored_name = f"{uuid.uuid4().hex}_{safe_name}"
    return os.path.join(settings.UPLOAD_DIR, stored_name)


async def _write_upload_file(file: UploadFile) -> tuple[str, int]:
    extension = os.path.splitext(file.filename or "")[1].lower()
    if extension not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Allowed: {settings.allowed_extensions_list}",
        )

    target_path = _build_upload_path(file.filename or "upload.bin")
    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail="Uploaded file exceeds maximum size")

    with open(target_path, "wb") as handle:
        handle.write(content)

    return target_path, len(content)


async def _decode_attachment_bytes(attachment: Dict[str, Any]) -> bytes:
    if attachment.get("data"):
        data = str(attachment["data"])
        if "base64," in data:
            data = data.split("base64,", 1)[1]
        return base64.b64decode(data)

    nested_file = attachment.get("file") or {}
    if nested_file.get("data"):
        data = str(nested_file["data"])
        if "base64," in data:
            data = data.split("base64,", 1)[1]
        return base64.b64decode(data)

    if attachment.get("content"):
        try:
            return base64.b64decode(str(attachment["content"]))
        except Exception:
            return str(attachment["content"]).encode("utf-8")

    if attachment.get("url"):
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(str(attachment["url"]))
            response.raise_for_status()
            return response.content

    raise ValueError("Unsupported attachment payload")


async def _register_technical_document(
    db: SQLiteManager,
    file_path: str,
    collection_name: str,
    original_filename: str,
    title: Optional[str],
    doc_type: Optional[str],
    language: str,
    metadata: Dict[str, Any],
) -> None:
    if not doc_type:
        return

    model_id = metadata.get("model_id")
    family_id = metadata.get("family_id")
    await db.async_execute_query(
        """
        INSERT INTO technical_documents (
            title, doc_type, model_id, family_id, language, file_path, rag_collection, version, metadata
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            title or os.path.splitext(original_filename)[0],
            doc_type,
            int(model_id) if model_id is not None else None,
            int(family_id) if family_id is not None else None,
            language,
            file_path,
            collection_name,
            metadata.get("version"),
            json.dumps(metadata, ensure_ascii=True),
        ),
    )


async def _ingest_saved_file(
    file_path: str,
    original_filename: str,
    mime_type: Optional[str],
    file_size: int,
    collection_name: Optional[str],
    title: Optional[str],
    doc_type: Optional[str],
    language: str,
    metadata: Optional[Dict[str, Any]] = None,
    upload_source: str = "api",
) -> Dict[str, Any]:
    db, rag, _ = _require_services()
    metadata = metadata or {}
    metadata.setdefault("title", title or os.path.splitext(original_filename)[0])
    metadata.setdefault("doc_type", doc_type or "manual")
    metadata.setdefault("language", language)
    metadata.setdefault("source", file_path)
    metadata.setdefault("upload_source", upload_source)

    resolved_collection = rag.resolve_collection(collection_name, metadata, file_path)
    ingestion_result = await rag.ingest_async(file_path=file_path, department=resolved_collection, metadata=metadata)

    upload_record = await db.register_uploaded_document(
        original_filename=original_filename,
        stored_filename=os.path.basename(file_path),
        file_path=file_path,
        mime_type=mime_type,
        file_size_bytes=file_size,
        document_role=metadata.get("document_role", "general_document"),
        rag_collection=resolved_collection,
        metadata=metadata,
    )

    await _register_technical_document(
        db=db,
        file_path=file_path,
        collection_name=resolved_collection,
        original_filename=original_filename,
        title=title,
        doc_type=doc_type,
        language=language,
        metadata=metadata,
    )

    # CAG: new content was indexed — invalidate cached context for this collection
    if cag_cache is not None:
        cag_cache.invalidate_collection(resolved_collection)

    return {
        "collection_name": resolved_collection,
        "chunks_created": ingestion_result["chunks_created"],
        "chunks_stored": ingestion_result["chunks_stored"],
        "uploaded_document_id": upload_record.get("id"),
    }


def _to_chat_response(result: Any, session_id: Optional[str]) -> ChatResponse:
    return ChatResponse(
        status="success",
        content=result.content,
        intent=result.intent,
        agent_used=result.agent_used,
        sources=result.sources,
        data=result.data,
        confidence=result.confidence,
        session_id=session_id,
        timestamp=result.timestamp,
    )


@app.get("/", tags=["Health"])
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "chat_endpoint": "/chat",
        "openai_endpoint": "/v1/chat/completions",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "service": settings.APP_NAME}


@app.get("/status", response_model=StatusResponse, tags=["Health"])
async def system_status():
    db, rag, orch = _require_services()
    components: Dict[str, Dict[str, Any]] = {}

    try:
        db.execute_query("SELECT 1 AS ok", fetch_one=True)
        components["database"] = {"status": "healthy", "type": "sqlite"}
    except Exception as exc:
        components["database"] = {"status": "unhealthy", "error": str(exc)}

    try:
        collections = rag.list_all_collections()
        components["vector_store"] = {
            "status": "healthy",
            "type": "chromadb",
            "collections": len(collections),
        }
    except Exception as exc:
        components["vector_store"] = {"status": "unhealthy", "error": str(exc)}

    try:
        llm_health = await LLMClient().check_health()
        available = [name for name, meta in llm_health.items() if meta.get("available")]
        components["llm"] = {
            "status": "healthy" if available else "degraded",
            "available_providers": available,
        }
    except Exception as exc:
        components["llm"] = {"status": "degraded", "error": str(exc)}

    components["orchestrator"] = {
        "status": "healthy",
        "agents": len(orch.agents),
        "sessions": len(orch.sessions),
    }

    overall = "healthy"
    if any(component["status"] == "unhealthy" for component in components.values()):
        overall = "degraded"

    return StatusResponse(status=overall, version=settings.APP_VERSION, components=components)


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest, http_request: Request):
    _, _, orch = _require_services()
    session_id = request.session_id or getattr(http_request.state, "session_id", None)

    if request.stream:
        return await chat_stream(request, http_request)

    result = await orch.route_and_execute(
        user_message=request.message,
        session_id=session_id,
        collection_hint=request.collection_hint,
    )
    return _to_chat_response(result, session_id)


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest, http_request: Request):
    _, _, orch = _require_services()
    session_id = request.session_id or getattr(http_request.state, "session_id", None)

    async def event_generator() -> AsyncIterator[Dict[str, str]]:
        async for chunk in orch.route_and_execute_stream(
            user_message=request.message,
            session_id=session_id,
            collection_hint=request.collection_hint,
        ):
            yield {"event": chunk.get("type", "message"), "data": json.dumps(chunk)}

    return EventSourceResponse(event_generator())


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    collection_name: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    doc_type: Optional[str] = Form(None),
    language: str = Form("en"),
    metadata_json: Optional[str] = Form(None),
):
    metadata = _safe_json_loads(metadata_json)
    saved_path, file_size = await _write_upload_file(file)
    result = await _ingest_saved_file(
        file_path=saved_path,
        original_filename=file.filename or os.path.basename(saved_path),
        mime_type=file.content_type,
        file_size=file_size,
        collection_name=collection_name,
        title=title,
        doc_type=doc_type,
        language=language,
        metadata=metadata,
        upload_source="api_upload",
    )
    return UploadResponse(
        status="success",
        file_name=file.filename or os.path.basename(saved_path),
        collection_name=result["collection_name"],
        chunks_created=result["chunks_created"],
        chunks_stored=result["chunks_stored"],
        uploaded_document_id=result["uploaded_document_id"],
        message="Document uploaded, ingested into RAG, and registered in uploaded_documents.",
    )


@app.post("/analyze-excel", response_model=ExcelAnalysisResponse, tags=["Excel"])
async def analyze_excel(
    file: UploadFile = File(...),
    question: Optional[str] = Form(None),
    sheet_name: Optional[str] = Form(None),
    operation: Optional[str] = Form(None),
    value_column: Optional[str] = Form(None),
    group_by: Optional[str] = Form(None),
    filter_column: Optional[str] = Form(None),
    filter_value: Optional[str] = Form(None),
    top_n: int = Form(10),
):
    extension = os.path.splitext(file.filename or "")[1].lower()
    if extension not in {".xlsx", ".csv"}:
        raise HTTPException(status_code=400, detail="Only .xlsx and .csv files are supported for table analysis")

    saved_path, _ = await _write_upload_file(file)
    try:
        analysis = ExcelAnalyzerAgent().analyze(
            file_path=saved_path,
            question=question,
            sheet_name=sheet_name,
            operation=operation,
            value_column=value_column,
            group_by=group_by,
            filter_column=filter_column,
            filter_value=filter_value,
            top_n=max(1, min(top_n, 100)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Excel analysis failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Excel analysis failed: {exc}") from exc

    return ExcelAnalysisResponse(status="success", **analysis.to_dict())


@app.get("/collections", response_model=CollectionsResponse, tags=["Documents"])
async def list_collections():
    _, rag, _ = _require_services()
    actual = {item["collection_key"]: item for item in rag.list_all_collections()}
    items: List[CollectionInfo] = []

    for collection_key in settings.rag_collections_list:
        current = actual.get(collection_key, {})
        registry = RAG_COLLECTION_REGISTRY.get(collection_key, {})
        items.append(
            CollectionInfo(
                name=f"{settings.CHROMA_COLLECTION_PREFIX}_{collection_key}",
                collection_key=collection_key,
                document_count=current.get("count", 0),
                metadata={**registry, **current.get("metadata", {})},
            )
        )

    return CollectionsResponse(status="success", collections=items)


@app.get("/agents", response_model=AgentsResponse, tags=["System"])
async def list_agents():
    _, _, orch = _require_services()
    agents = [AgentInfo(**agent) for agent in orch.list_agents()]
    return AgentsResponse(status="success", agents=agents)


@app.get("/departments", tags=["System"])
async def list_departments():
    """Return supported department/collection keys with their labels and domains."""
    items = [
        {
            "collection_key": key,
            "label": meta.get("label", key),
            "domain": meta.get("domain", ""),
            "description": meta.get("description", ""),
        }
        for key, meta in RAG_COLLECTION_REGISTRY.items()
    ]
    return {
        "status": "success",
        "count": len(items),
        "departments": items,
    }


@app.get("/v1/models", tags=["OpenAI Compatibility"])
async def openai_models():
    created = int(datetime.now(timezone.utc).timestamp())
    return {
        "object": "list",
        "data": [
            {
                "id": "coiltech-platform",
                "object": "model",
                "created": created,
                "owned_by": "coiltech",
            },
            {
                "id": "industrial-ai-platform",
                "object": "model",
                "created": created,
                "owned_by": "coiltech",
            },
        ],
    }


@app.get("/product-families", response_model=ProductFamilyListResponse, tags=["Catalog"])
async def get_product_families(category: Optional[str] = None):
    db, _, _ = _require_services()
    items = await db.get_product_families(category=category)
    return ProductFamilyListResponse(status="success", items=items, count=len(items))


@app.get("/projects", response_model=ProjectListResponse, tags=["Projects"])
async def get_projects(status: Optional[str] = None, customer_id: Optional[int] = None):
    db, _, _ = _require_services()
    items = await db.get_projects(status=status, customer_id=customer_id)
    return ProjectListResponse(status="success", items=items, count=len(items))


@app.get("/projects/{project_id}", response_model=ProjectDetailResponse, tags=["Projects"])
async def get_project_detail(project_id: int):
    db, _, _ = _require_services()
    item = await db.get_project_detail(project_id)
    if not item:
        raise HTTPException(status_code=404, detail="Project not found")
    return ProjectDetailResponse(status="success", item=item)


@app.get("/quotations", response_model=QuotationListResponse, tags=["Sales"])
async def get_quotations(status: Optional[str] = None, customer_id: Optional[int] = None):
    db, _, _ = _require_services()
    items = await db.get_quotations(status=status, customer_id=customer_id)
    return QuotationListResponse(status="success", items=items, count=len(items))


@app.get("/installed-base", response_model=InstalledBaseListResponse, tags=["Service"])
async def get_installed_base(customer_id: Optional[int] = None):
    db, _, _ = _require_services()
    items = await db.get_installed_base(customer_id=customer_id)
    return InstalledBaseListResponse(status="success", items=items, count=len(items))


@app.get("/service-tickets", response_model=ServiceTicketListResponse, tags=["Service"])
async def get_service_tickets(
    status: Optional[str] = None,
    customer_id: Optional[int] = None,
    installed_base_id: Optional[int] = None,
):
    db, _, _ = _require_services()
    items = await db.get_service_tickets(
        status=status,
        customer_id=customer_id,
        installed_base_id=installed_base_id,
    )
    return ServiceTicketListResponse(status="success", items=items, count=len(items))


@app.get("/spare-parts", response_model=SparePartListResponse, tags=["Spare Parts"])
async def get_spare_parts(category: Optional[str] = None, low_stock_only: bool = False):
    db, _, _ = _require_services()
    items = await db.get_spare_parts(category=category, low_stock_only=low_stock_only)
    return SparePartListResponse(status="success", items=items, count=len(items))


@app.get("/alerts", response_model=AlertListResponse, tags=["Digital Twin"])
async def get_alerts(machine_id: Optional[int] = None):
    db, _, _ = _require_services()
    items = await db.get_active_alerts(machine_id=machine_id)
    return AlertListResponse(status="success", items=items, count=len(items))


@app.post("/query", response_model=DatabaseQueryResponse, tags=["Database"])
async def natural_language_query(question: str):
    db, _, _ = _require_services()
    result = await db.nl_to_sql(question, LLMClient())
    return DatabaseQueryResponse(
        status="success" if result.get("success") else "error",
        question=question,
        generated_sql=result.get("generated_sql"),
        results=result.get("results"),
        row_count=result.get("row_count", 0),
        error=result.get("error"),
    )


@app.get("/sessions/{session_id}", response_model=SessionResponse, tags=["Sessions"])
async def get_session(session_id: str):
    _, _, orch = _require_services()
    info = orch.get_session_info(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(
        status="success",
        session=SessionInfo(
            session_id=info["session_id"],
            message_count=info["message_count"],
            preferences=info.get("preferences", {}),
        ),
    )


@app.delete("/sessions/{session_id}", tags=["Sessions"])
async def delete_session(session_id: str):
    _, _, orch = _require_services()
    if not orch.clear_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success", "message": "Session cleared"}


async def _ingest_openai_attachments(
    attachments: Optional[List[Dict[str, Any]]],
    collection_name: Optional[str],
) -> List[Dict[str, Any]]:
    if not attachments:
        return []

    results = []
    for attachment in attachments:
        try:
            payload = attachment if isinstance(attachment, dict) else attachment.model_dump()
            raw_bytes = await _decode_attachment_bytes(payload)
            original_filename = payload.get("name") or payload.get("file", {}).get("filename") or "attachment.bin"
            saved_path = _build_upload_path(original_filename)
            with open(saved_path, "wb") as handle:
                handle.write(raw_bytes)

            result = await _ingest_saved_file(
                file_path=saved_path,
                original_filename=original_filename,
                mime_type=None,
                file_size=len(raw_bytes),
                collection_name=collection_name,
                title=payload.get("metadata", {}).get("title"),
                doc_type=payload.get("metadata", {}).get("doc_type"),
                language=payload.get("metadata", {}).get("language", "en"),
                metadata=payload.get("metadata", {}),
                upload_source="openai_chat",
            )
            results.append({"file_name": original_filename, **result})
        except Exception as exc:
            logger.warning("Failed to process OpenAI attachment", error=str(exc))
    return results


def _build_openai_response(content: str, model_name: str, created_at: int) -> OpenAIChatResponse:
    return OpenAIChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=created_at,
        model=model_name,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    )


def _openwebui_utility_response(user_message: str) -> Optional[str]:
    """
    Open WebUI sends internal prompts for titles, tags, and follow-up questions.
    They should not consume the industrial agent pipeline or local LLM budget.
    """
    normalized = _extract_actual_user_question(user_message).lower()
    raw = (user_message or "").lower()
    if "### task:" not in raw:
        return None

    if "generate a concise" in raw and "title" in raw:
        return "Coiltech Platform"
    if "generate 1-3 broad tags" in raw or "categorizing the main themes" in raw:
        return json.dumps({"tags": ["coiltech", "industrial-ai", "automation"]})
    if "suggest 3-5 relevant follow-up questions" in raw or '"follow_ups"' in raw:
        return json.dumps({"follow_ups": []})
    return None


@app.post("/v1/chat/completions", tags=["OpenAI Compatibility"])
async def openai_chat_completions(request: OpenAIChatRequest, http_request: Request):
    _, _, orch = _require_services()
    session_id = getattr(http_request.state, "session_id", None)
    user_message = _get_last_user_message(request.messages)
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found in request.messages")

    utility_content = _openwebui_utility_response(user_message)
    if utility_content is not None:
        return _build_openai_response(
            content=utility_content,
            model_name=request.model,
            created_at=int(datetime.now(timezone.utc).timestamp()),
        )

    attachments_payload = [item.model_dump() for item in request.files] if request.files else None
    attachment_results = await _ingest_openai_attachments(attachments_payload, request.collection_name)
    collection_hint = request.collection_name
    if attachment_results and not collection_hint:
        collection_hint = attachment_results[-1]["collection_name"]

    if request.stream:
        async def stream_generator():
            async for chunk in orch.route_and_execute_stream(
                user_message=user_message,
                session_id=session_id,
                collection_hint=collection_hint,
            ):
                if chunk.get("type") == "content":
                    payload = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                        "object": "chat.completion.chunk",
                        "created": int(datetime.now(timezone.utc).timestamp()),
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {"content": chunk["content"]}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
            done_payload = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion.chunk",
                "created": int(datetime.now(timezone.utc).timestamp()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done_payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    result = await orch.route_and_execute(
        user_message=user_message,
        session_id=session_id,
        collection_hint=collection_hint,
    )
    response_content = result.content
    if attachment_results:
        response_content += (
            "\n\nProcessed attachments: "
            + ", ".join(
                f"{item['file_name']} -> {item['collection_name']}" for item in attachment_results
            )
        )

    return _build_openai_response(
        content=response_content,
        model_name=request.model,
        created_at=int(datetime.now(timezone.utc).timestamp()),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
