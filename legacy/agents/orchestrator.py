"""
Orchestrator Module for the Coiltech Industrial Automation Platform.

Main entry point for query handling. Routes queries through the RouterAgent
and dispatches to appropriate domain-specific agents.
"""

from typing import Optional, Dict, Any, AsyncIterator, List
from dataclasses import dataclass, field
from datetime import datetime, timezone
import asyncio
import re
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from agents.base_agent import AgentResponse, AgentContext, GeneralAgent
from agents.router_agent import RouterAgent, IntentResult

logger = structlog.get_logger(__name__)


@dataclass
class FinalResponse:
    content: str
    intent: Dict[str, Any]
    agent_used: str
    data: Optional[Dict[str, Any]] = None
    sources: Optional[List[Dict[str, Any]]] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "intent": self.intent,
            "agent_used": self.agent_used,
            "data": self.data,
            "sources": self.sources,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SessionContext:
    session_id: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    user_metadata: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    rag_context: Optional[str] = None

    def add_message(self, role: str, content: str) -> None:
        self.conversation_history.append(
            {"role": role, "content": content, "timestamp": datetime.now(timezone.utc).isoformat()}
        )
        max_history = 20
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]

    def get_recent_history(self, n: int = 5) -> List[Dict[str, str]]:
        return self.conversation_history[-n:]


class Orchestrator:
    """
    Main orchestrator for the Coiltech Industrial Automation Platform.

    Routes queries through the router agent and dispatches to
    domain-specific agents based on intent classification AND RAG results.
    """

    def __init__(
        self,
        llm_client: Any = None,
        db_manager: Any = None,
        rag_pipeline: Any = None,
        cag_cache: Any = None,
    ):
        self._llm_client = llm_client
        self._db_manager = db_manager
        self._rag_pipeline = rag_pipeline
        self._cag_cache = cag_cache

        self.router = RouterAgent(llm_client=self._llm_client)

        # Lazy-init agents dict — import here to avoid circular imports
        self.agents: Dict[str, Any] = {}
        self._init_agents()

        self.sessions: Dict[str, SessionContext] = {}

        logger.info("Orchestrator initialized", agents=list(self.agents.keys()))

    _DIRECT_SQL_PATTERNS = [
        r"\b(list|show|get|find|search|filter|count|how many|total|sum|average|avg|mean|min|max|top)\b",
        r"\b(status|stock|inventory|available|availability|open tickets?|active alerts?)\b",
        r"\b(listele|goster|göster|kac|kaç|toplam|ortalama|filtre|stok|stokta|durum|adet|sayisi|sayısı|mevcut|envanter)\b",
        r"\b(var\s*(mi|mı|mu|mü)|mevcut\s*(mi|mı|mu|mü))\b",
    ]

    _DIRECT_SQL_EXCLUDED_INTENTS = {
        "application_engineering",
        "configuration_request",
        "commissioning_support",
        "general_knowledge",
    }

    _DOCUMENT_QUERY_PATTERNS = [
        r"\b(pdf|document|file|attachment|uploaded|source|report)\b",
        r"\b(bu|this)\s+(pdf|document|file|dosya|belge|rapor)\b",
        r"\b(dosya|belge|kaynak|rapor|yuklenen|yüklenen)\b",
    ]

    _PART_CODE_PATTERN = r"\b[A-Z]{1,8}[-_][A-Z0-9]{2,}(?:[-_][A-Z0-9]+)*\b"

    def _init_agents(self) -> None:
        """Initialize all domain-specific agents."""
        from agents.product_catalog_agent import ProductCatalogAgent
        from agents.application_engineering_agent import ApplicationEngineeringAgent
        from agents.configuration_agent import ConfigurationAgent
        from agents.quotation_agent import QuotationAgent
        from agents.project_execution_agent import ProjectExecutionAgent
        from agents.service_maintenance_agent import ServiceMaintenanceAgent
        from agents.spare_parts_agent import SparePartsAgent
        from agents.installation_commissioning_agent import InstallationCommissioningAgent
        from agents.quality_compliance_agent import QualityComplianceAgent
        from agents.digital_twin_agent import DigitalTwinAgent
        from agents.management_reporting_agent import ManagementReportingAgent

        common = dict(
            llm_client=self._llm_client,
            db_manager=self._db_manager,
            rag_pipeline=self._rag_pipeline,
        )

        self.agents = {
            "ProductCatalogAgent": ProductCatalogAgent(**common),
            "ApplicationEngineeringAgent": ApplicationEngineeringAgent(**common),
            "ConfigurationAgent": ConfigurationAgent(**common),
            "QuotationAgent": QuotationAgent(**common),
            "ProjectExecutionAgent": ProjectExecutionAgent(**common),
            "ServiceMaintenanceAgent": ServiceMaintenanceAgent(**common),
            "SparePartsAgent": SparePartsAgent(**common),
            "InstallationCommissioningAgent": InstallationCommissioningAgent(**common),
            "QualityComplianceAgent": QualityComplianceAgent(**common),
            "DigitalTwinAgent": DigitalTwinAgent(**common),
            "ManagementReportingAgent": ManagementReportingAgent(**common),
            "GeneralAgent": GeneralAgent(**common),
        }

    def get_or_create_session(self, session_id: str) -> SessionContext:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionContext(session_id=session_id)
        return self.sessions[session_id]

    # ------------------------------------------------------------------
    # RAG search across targeted collections
    # ------------------------------------------------------------------
    async def _search_rag_collections(
        self, query: str, collections: List[str], k: int = 5
    ) -> Dict[str, Any]:
        if not self._rag_pipeline or not collections:
            return {"sources": [], "context": ""}

        # CAG: check cache before hitting ChromaDB
        if self._cag_cache is not None:
            cached = self._cag_cache.get(collections, query)
            if cached is not None:
                return {"sources": cached["sources"], "context": cached["context"], "from_cache": True}

        all_sources: List[Dict[str, Any]] = []
        for coll_name in collections:
            try:
                results = self._rag_pipeline.vector_store.similarity_search(
                    query=query, department=coll_name, k=k, score_threshold=0.3
                )
                if results:
                    for r in results:
                        r["collection"] = coll_name
                        all_sources.append(r)
            except Exception as e:
                logger.debug(f"No results from {coll_name}: {e}")

        all_sources.sort(key=lambda x: x.get("score", 0), reverse=True)
        top = all_sources[:k]

        context_parts = []
        for i, src in enumerate(top, 1):
            content = src.get("content", "")[:500]
            coll = src.get("collection", "unknown")
            score = src.get("score", 0)
            context_parts.append(f"[Source {i} - {coll}] (Relevance: {score:.2f})\n{content}")

        context = "\n\n---\n\n".join(context_parts) if context_parts else ""

        # CAG: populate cache after fresh retrieval
        if self._cag_cache is not None:
            self._cag_cache.set(collections, query, top, context)

        return {"sources": top, "context": context, "from_cache": False}

    # ------------------------------------------------------------------
    # Core route-and-execute
    # ------------------------------------------------------------------
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def _route_query(
        self, user_message: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return await self.router.route(user_message, context)

    async def _execute_agent(
        self, agent_name: str, query: str, context: AgentContext
    ) -> AgentResponse:
        agent = self.agents.get(agent_name)
        if agent is None:
            logger.warning(f"Agent not found: {agent_name}, using GeneralAgent")
            agent = self.agents["GeneralAgent"]
        try:
            return await agent.handle(query, context)
        except Exception as e:
            logger.error("Agent execution failed", agent=agent_name, error=str(e))
            return AgentResponse(
                content=f"I encountered an issue processing your request. Error: {str(e)[:200]}",
                agent_name=agent_name,
                department="general",
                confidence=0.0,
                metadata={"error": str(e)},
            )

    def _should_use_direct_sql(self, user_message: str, intent: Dict[str, Any], route_result: Dict[str, Any]) -> bool:
        if not self._db_manager:
            return False
        if not route_result.get("should_query_db"):
            return False

        category = intent.get("category", "general_knowledge")
        if category in self._DIRECT_SQL_EXCLUDED_INTENTS:
            return False

        normalized = user_message.lower()
        return any(re.search(pattern, normalized) for pattern in self._DIRECT_SQL_PATTERNS)

    def _get_active_collection_hint(
        self,
        session_context: Optional[SessionContext],
        collection_hint: Optional[str],
    ) -> Optional[str]:
        if collection_hint and session_context:
            session_context.user_metadata["active_collection_hint"] = collection_hint
        if collection_hint:
            return collection_hint
        if session_context:
            return session_context.user_metadata.get("active_collection_hint")
        return None

    def _is_document_query(self, user_message: str) -> bool:
        normalized = user_message.lower()
        return any(re.search(pattern, normalized) for pattern in self._DOCUMENT_QUERY_PATTERNS)

    def _should_apply_collection_hint(
        self,
        user_message: str,
        intent: Dict[str, Any],
        routed_collections: List[str],
        collection_hint: Optional[str],
    ) -> bool:
        if not collection_hint:
            return False

        category = intent.get("category", "general_knowledge")
        if category == "general_knowledge":
            return True
        if self._is_document_query(user_message):
            return True
        if collection_hint in routed_collections:
            return True
        return False

    async def _try_direct_sql_response(
        self,
        user_message: str,
        intent: Dict[str, Any],
        agent_name: str,
        route_result: Dict[str, Any],
        session_context: Optional[SessionContext],
    ) -> Optional[FinalResponse]:
        if not self._should_use_direct_sql(user_message, intent, route_result):
            return None

        stock_response = await self._try_part_stock_response(
            user_message=user_message,
            intent=intent,
            agent_name=agent_name,
            route_result=route_result,
            session_context=session_context,
        )
        return stock_response

    async def _try_part_stock_response(
        self,
        user_message: str,
        intent: Dict[str, Any],
        agent_name: str,
        route_result: Dict[str, Any],
        session_context: Optional[SessionContext],
    ) -> Optional[FinalResponse]:
        if intent.get("category") != "spare_parts_lookup":
            return None

        match = re.search(self._PART_CODE_PATTERN, user_message, flags=re.IGNORECASE)
        if not match:
            return await self._try_general_stock_listing(
                user_message=user_message,
                intent=intent,
                agent_name=agent_name,
                route_result=route_result,
                session_context=session_context,
            )

        part_number = match.group(0).upper().replace("_", "-")
        sql = """
            SELECT
                sp.part_number, sp.name, sp.category, sp.applicable_models,
                sp.unit, sp.unit_cost, sp.lead_time_days, sp.supplier,
                sp.is_critical, sp.notes,
                COALESCE(sps.quantity, 0) AS quantity,
                sps.min_stock, sps.max_stock, sps.location, sps.last_restock_date
            FROM spare_parts sp
            LEFT JOIN spare_part_stock sps ON sp.id = sps.part_id
            WHERE UPPER(REPLACE(sp.part_number, '_', '-')) = ?
            ORDER BY sps.quantity DESC
            LIMIT 10
        """

        try:
            rows = await self._db_manager.async_execute_query(sql, (part_number,))
        except Exception as exc:
            logger.warning("Deterministic stock lookup failed before fallback", error=str(exc))
            return None

        content = self._format_part_stock_content(part_number, rows, user_message)
        if session_context:
            session_context.add_message("assistant", content)

        return FinalResponse(
            content=content,
            intent=intent,
            agent_used=agent_name,
            data={
                "direct_sql": True,
                "deterministic_stock_lookup": True,
                "question": user_message,
                "generated_sql": sql.strip(),
                "results": rows,
                "row_count": len(rows),
            },
            sources=None,
            confidence=0.94,
            metadata={
                "routing": route_result,
                "direct_sql": True,
                "deterministic_stock_lookup": True,
                "session_id": session_context.session_id if session_context else None,
            },
        )

    async def _try_general_stock_listing(
        self,
        user_message: str,
        intent: Dict[str, Any],
        agent_name: str,
        route_result: Dict[str, Any],
        session_context: Optional[SessionContext],
    ) -> Optional[FinalResponse]:
        sql = """
            SELECT sp.part_number, sp.name, sp.category, sp.unit,
                   COALESCE(sps.quantity, 0) AS quantity,
                   sps.min_stock, sps.max_stock, sps.location
            FROM spare_parts sp
            LEFT JOIN spare_part_stock sps ON sp.id = sps.part_id
            ORDER BY quantity DESC
            LIMIT 20
        """
        try:
            rows = await self._db_manager.async_execute_query(sql, ())
        except Exception as exc:
            logger.warning("General stock listing failed", error=str(exc))
            return None

        wants_turkish = bool(re.search(r"[çğıöşüÇĞİÖŞÜ]|\b(stok|listele|göster|goster)\b", user_message.lower()))
        if rows:
            header = "Stok listesi:" if wants_turkish else "Current stock listing:"
            lines = [header]
            for r in rows:
                lines.append(f"  {r['part_number']} — {r['name']}: {r['quantity']} {r.get('unit','pcs')}")
            content = "\n".join(lines)
        else:
            content = "Stok kaydı bulunamadı." if wants_turkish else "No stock records found."

        if session_context:
            session_context.add_message("assistant", content)

        return FinalResponse(
            content=content,
            intent=intent,
            agent_used=agent_name,
            data={
                "direct_sql": True,
                "question": user_message,
                "generated_sql": sql.strip(),
                "results": rows,
                "row_count": len(rows),
            },
            sources=None,
            confidence=0.90,
            metadata={
                "routing": route_result,
                "direct_sql": True,
                "session_id": session_context.session_id if session_context else None,
            },
        )

    def _format_part_stock_content(
        self,
        part_number: str,
        rows: List[Dict[str, Any]],
        user_message: str,
    ) -> str:
        wants_turkish = bool(re.search(r"[çğıöşüÇĞİÖŞÜ]|\b(var|stok|stokta|mevcut)\b", user_message.lower()))
        if not rows:
            if wants_turkish:
                return f"{part_number} için stok kaydı bulunamadı."
            return f"No stock record was found for {part_number}."

        row = rows[0]
        quantity = row.get("quantity") or 0
        unit = row.get("unit") or "pcs"
        min_stock = row.get("min_stock")
        max_stock = row.get("max_stock")
        location = row.get("location") or "unknown"
        lead_time = row.get("lead_time_days")
        supplier = row.get("supplier") or "unknown"
        critical = bool(row.get("is_critical"))
        low_stock = min_stock is not None and quantity <= min_stock

        if wants_turkish:
            availability = "stokta mevcut" if quantity > 0 else "stokta yok"
            lines = [
                f"{part_number} ({row.get('name')}) {availability}.",
                f"Mevcut stok: {quantity} {unit} | Minimum: {min_stock} | Maksimum: {max_stock} | Lokasyon: {location}.",
                f"Kritik parça: {'evet' if critical else 'hayır'} | Tedarikçi: {supplier} | Termin: {lead_time} gün.",
            ]
            if low_stock:
                lines.append("Stok minimum seviyede veya altında; yeniden sipariş önerilir.")
            return "\n".join(lines)

        availability = "is in stock" if quantity > 0 else "is out of stock"
        lines = [
            f"{part_number} ({row.get('name')}) {availability}.",
            f"Current stock: {quantity} {unit} | Minimum: {min_stock} | Maximum: {max_stock} | Location: {location}.",
            f"Critical part: {'yes' if critical else 'no'} | Supplier: {supplier} | Lead time: {lead_time} days.",
        ]
        if low_stock:
            lines.append("Stock is at or below the minimum level; reorder is recommended.")
        return "\n".join(lines)

    async def route_and_execute(
        self,
        user_message: str,
        session_context: Optional[SessionContext] = None,
        session_id: Optional[str] = None,
        collection_hint: Optional[str] = None,
        **kwargs,
    ) -> FinalResponse:
        if session_context is None and session_id:
            session_context = self.get_or_create_session(session_id)
        if session_context:
            session_context.add_message("user", user_message)

        logger.info("Processing query", query=user_message[:60])

        try:
            # Step 1: Route
            routing_context = {
                "history": session_context.get_recent_history() if session_context else [],
                "collection_hint": collection_hint,
            }
            route_result = await self._route_query(user_message, routing_context)
            intent = route_result["intent"]
            agent_name = route_result["target_agent"]
            routed_collections = route_result.get("rag_collections", [])
            active_collection_hint = self._get_active_collection_hint(session_context, collection_hint)
            hint_applied = self._should_apply_collection_hint(
                user_message=user_message,
                intent=intent,
                routed_collections=routed_collections,
                collection_hint=active_collection_hint,
            )
            collections = [active_collection_hint] if hint_applied and active_collection_hint else routed_collections

            direct_response = await self._try_direct_sql_response(
                user_message=user_message,
                intent=intent,
                agent_name=agent_name,
                route_result=route_result,
                session_context=session_context,
            )
            if direct_response:
                return direct_response

            # Step 2: RAG search
            rag_results = await self._search_rag_collections(user_message, collections, k=5)
            rag_sources = rag_results.get("sources", [])
            rag_context = rag_results.get("context", "")

            if session_context:
                session_context.rag_context = rag_context

            # Step 3: Build agent context
            agent_context = AgentContext(
                user_query=user_message,
                department=intent.get("category", "general_knowledge"),
                session_id=session_context.session_id if session_context else None,
                conversation_history=session_context.conversation_history if session_context else [],
                user_metadata={
                    "rag_context": rag_context,
                    "rag_sources": rag_sources,
                    "collection_hint": active_collection_hint,
                    "collection_hint_applied": hint_applied,
                },
            )

            # Step 4: Execute agent
            agent_response = await self._execute_agent(agent_name, user_message, agent_context)

            all_sources = rag_sources + (agent_response.sources or [])

            if session_context:
                session_context.add_message("assistant", agent_response.content)

            return FinalResponse(
                content=agent_response.content,
                intent=intent,
                agent_used=agent_name,
                data=agent_response.data,
                sources=all_sources if all_sources else None,
                confidence=agent_response.confidence,
                metadata={
                    "routing": route_result,
                    "agent_metadata": agent_response.metadata,
                    "session_id": session_context.session_id if session_context else None,
                    "rag_sources_count": len(rag_sources),
                    "collection_hint": active_collection_hint,
                    "collection_hint_applied": hint_applied,
                },
            )

        except Exception as e:
            logger.error("Query processing failed", error=str(e))
            return FinalResponse(
                content="I encountered an unexpected error. Please try again.",
                intent={"category": "error", "confidence": 0},
                agent_used="ErrorHandler",
                confidence=0.0,
                metadata={"error": str(e)},
            )

    async def route_and_execute_stream(
        self,
        user_message: str,
        session_context: Optional[SessionContext] = None,
        session_id: Optional[str] = None,
        collection_hint: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        if session_context is None and session_id:
            session_context = self.get_or_create_session(session_id)
        if session_context:
            session_context.add_message("user", user_message)

        try:
            active_collection_hint = self._get_active_collection_hint(session_context, collection_hint)
            route_result = await self._route_query(user_message, {"collection_hint": active_collection_hint})
            intent = route_result["intent"]
            agent_name = route_result["target_agent"]
            routed_collections = route_result.get("rag_collections", [])
            hint_applied = self._should_apply_collection_hint(
                user_message=user_message,
                intent=intent,
                routed_collections=routed_collections,
                collection_hint=active_collection_hint,
            )

            yield {"type": "routing", "intent": intent, "agent": agent_name}

            direct_response = await self._try_direct_sql_response(
                user_message=user_message,
                intent=intent,
                agent_name=agent_name,
                route_result=route_result,
                session_context=session_context,
            )
            if direct_response:
                yield {"type": "content", "content": direct_response.content}
                yield {
                    "type": "done",
                    "agent_used": agent_name,
                    "intent": intent,
                    "metadata": {"direct_sql": True},
                }
                return

            agent_context = AgentContext(
                user_query=user_message,
                department=intent.get("category", "general_knowledge"),
                session_id=session_context.session_id if session_context else None,
                conversation_history=session_context.conversation_history if session_context else [],
                user_metadata={
                    "collection_hint": active_collection_hint,
                    "collection_hint_applied": hint_applied,
                },
            )

            agent = self.agents.get(agent_name, self.agents["GeneralAgent"])
            full_content = ""
            async for chunk in agent.handle_stream(user_message, agent_context):
                full_content += chunk
                yield {"type": "content", "content": chunk}

            if session_context:
                session_context.add_message("assistant", full_content)

            yield {"type": "done", "agent_used": agent_name, "intent": intent}

        except Exception as e:
            logger.error("Streaming failed", error=str(e))
            yield {"type": "error", "error": str(e)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def list_agents(self) -> List[Dict[str, str]]:
        return [
            {"name": name, "department": agent.department, "description": agent.description}
            for name, agent in self.agents.items()
        ]

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        session = self.sessions.get(session_id)
        if session:
            return {
                "session_id": session.session_id,
                "message_count": len(session.conversation_history),
                "preferences": session.preferences,
            }
        return None

    def clear_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
