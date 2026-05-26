import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import structlog

from backend.rag.ingestion.pdf_ingest import ingest_file
from backend.rag.retrieval.query_engine import run_query

logger = structlog.get_logger(__name__)

_KEYWORD_MAP = {
    "products_compact_drives": ["compact drive", "press feeding", "compact"],
    "products_loop_drives": ["loop drive", "loop type"],
    "products_blank_lines": ["blank line", "blanking"],
    "products_destacker": ["destacker"],
    "products_press_transfer_robots": ["transfer robot", "press transfer robot", "robot"],
    "products_press_transfer": ["press transfer", "transfer system"],
    "products_servo_drives": ["servo drive", "triggered", "zigzag"],
    "products_decoilers": ["decoiler", "uncoiler", "mandrel"],
    "products_ctl_lines": ["cut to length", "ctl", "shear", "stacker"],
    "special_projects": ["special project", "custom", "layout", "rfq"],
    "installation_commissioning": ["commissioning", "installation", "startup"],
    "service_maintenance": ["service", "maintenance", "troubleshooting", "repair", "alarm"],
    "spare_parts": ["spare", "part", "replacement"],
    "quality_documents": ["ce", "quality", "certificate", "compliance"],
    "sales_and_quotations": ["quotation", "quote", "commercial", "proposal"],
}


class RAGPipeline:
    def __init__(self, pool, llm, settings):
        self.pool = pool
        self.llm = llm
        self.settings = settings

    def resolve_collection(
        self,
        collection_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        file_path: Optional[str] = None,
    ) -> str:
        if collection_name:
            return collection_name
        metadata = metadata or {}
        explicit = metadata.get("rag_collection") or metadata.get("collection_name")
        if explicit:
            return str(explicit)
        search_text = " ".join(
            str(v).lower()
            for v in [
                metadata.get("title"), metadata.get("doc_type"),
                metadata.get("document_role"),
                Path(file_path).stem if file_path else "",
            ]
            if v
        )
        for col_key, keywords in _KEYWORD_MAP.items():
            if any(kw in search_text for kw in keywords):
                return col_key
        return self.settings.OPENWEBUI_DEFAULT_COLLECTION

    async def ingest_async(
        self,
        file_path: str,
        department: str,
        metadata: Optional[Dict[str, Any]] = None,
        tenant_id: int = 1,
        user_id: int = 1,
    ) -> Dict[str, Any]:
        collection = self.resolve_collection(department, metadata, file_path)
        result = await ingest_file(
            self.pool, self.llm,
            file_path=file_path,
            collection=collection,
            tenant_id=tenant_id,
            user_id=user_id,
            metadata=metadata,
        )
        if self.settings.RAG_USE_GRAPH:
            try:
                from backend.rag.graph.client import get_driver
                driver = get_driver(
                    self.settings.NEO4J_URI,
                    self.settings.NEO4J_USER,
                    self.settings.NEO4J_PASSWORD,
                )
            except Exception as e:
                logger.warning("Graph ingest skipped", error=str(e))
        await self.maybe_create_agent(collection, tenant_id, user_id)
        return result

    async def maybe_create_agent(
        self,
        collection: str,
        tenant_id: int = 1,
        user_id: int = 1,
    ) -> None:
        try:
            async with self.pool.acquire() as conn:
                existing = await conn.fetchval(
                    "SELECT id FROM agents WHERE collection=$1 AND tenant_id=$2 AND user_id=$3 LIMIT 1",
                    collection, tenant_id, user_id,
                )
                if existing:
                    return
                name = collection.replace("_", " ").title() + "Bot"
                system_prompt = (
                    f"Sen Coiltech {name} uzmanısın. "
                    f"Yalnızca {collection} koleksiyonundaki belgelerden cevap ver. "
                    f"Cevaplarını Türkçe ver."
                )
                row = await conn.fetchrow(
                    """INSERT INTO agents (tenant_id, user_id, name, system_prompt, model, collection)
                       VALUES ($1,$2,$3,$4,$5,$6) RETURNING id""",
                    tenant_id, user_id, name, system_prompt,
                    self.settings.DEFAULT_MODEL, collection,
                )
                agent_id = row["id"]
                emb = await self.llm.get_embedding(f"{name}. {system_prompt[:200]}")
                emb_str = "[" + ",".join(str(v) for v in emb) + "]"
                await conn.execute(
                    "UPDATE agents SET embedding=$1::vector WHERE id=$2", emb_str, agent_id
                )
                try:
                    from backend.deps import get_router
                    get_router().invalidate_cache()
                except RuntimeError:
                    pass
            logger.info("Auto-created agent for collection", collection=collection, name=name)
        except Exception as e:
            logger.warning("Auto-agent creation failed", collection=collection, error=str(e))

    async def query(
        self,
        question: str,
        collection: str,
        tenant_id: int = 1,
        user_id: int = 1,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        graph_retriever = None
        if self.settings.RAG_USE_GRAPH:
            try:
                from backend.rag.graph.client import get_driver
                from backend.rag.graph import retriever as graph_mod
                driver = get_driver(
                    self.settings.NEO4J_URI,
                    self.settings.NEO4J_USER,
                    self.settings.NEO4J_PASSWORD,
                )
                class _GR:
                    async def get_context(self, q, srcs):
                        return await graph_mod.get_context(q, srcs, driver)
                graph_retriever = _GR()
            except Exception as e:
                logger.warning("Graph retriever not available", error=str(e))

        return await run_query(
            self.pool, self.llm,
            question=question,
            collection=collection,
            tenant_id=tenant_id,
            user_id=user_id,
            k=self.settings.RAG_TOP_K,
            alpha=self.settings.RAG_ALPHA,
            use_expansion=self.settings.RAG_MULTI_QUERY,
            reranker_model=self.settings.RERANKER_MODEL,
            score_threshold=self.settings.RAG_SCORE_THRESHOLD,
            language=self.settings.DEFAULT_ANSWER_LANGUAGE,
            graph_retriever=graph_retriever,
            system_prompt=system_prompt,
        )

    async def _retrieve(
        self,
        question: str,
        collection: str,
        tenant_id: int = 1,
        user_id: int = 1,
        system_prompt: Optional[str] = None,
    ):
        from backend.rag.retrieval.query_engine import retrieve_context
        return await retrieve_context(
            self.pool, self.llm,
            question=question,
            collection=collection,
            tenant_id=tenant_id,
            user_id=user_id,
            k=self.settings.RAG_TOP_K,
            alpha=self.settings.RAG_ALPHA,
            use_expansion=self.settings.RAG_MULTI_QUERY,
            reranker_model=self.settings.RERANKER_MODEL,
            score_threshold=self.settings.RAG_SCORE_THRESHOLD,
            language=self.settings.DEFAULT_ANSWER_LANGUAGE,
            system_prompt=system_prompt,
        )

    def list_all_collections(self) -> List[Dict[str, Any]]:
        return []
