"""
Router Agent for the Coiltech Industrial Automation Platform.

Handles intent classification and routing of user queries to the
appropriate domain-specific agent using LLM-based classification
with keyword fallback.
"""

import json
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import structlog

from config.settings import INTENT_CATEGORIES, INTENT_AGENT_MAP, ROUTING_SYSTEM_PROMPT

logger = structlog.get_logger(__name__)


class IntentCategory(str, Enum):
    PRODUCT_LOOKUP = "product_lookup"
    APPLICATION_ENGINEERING = "application_engineering"
    CONFIGURATION_REQUEST = "configuration_request"
    QUOTATION_REQUEST = "quotation_request"
    PROJECT_STATUS = "project_status"
    COMMISSIONING_SUPPORT = "commissioning_support"
    MAINTENANCE_REQUEST = "maintenance_request"
    SPARE_PARTS_LOOKUP = "spare_parts_lookup"
    INSTALLED_BASE_QUERY = "installed_base_query"
    ANOMALY_DETECTION = "anomaly_detection"
    QUALITY_COMPLIANCE = "quality_compliance"
    EXECUTIVE_SUMMARY = "executive_summary"
    GENERAL_KNOWLEDGE = "general_knowledge"


@dataclass
class IntentResult:
    category: str
    confidence: float
    reasoning: str
    original_query: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "original_query": self.original_query,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], query: str) -> "IntentResult":
        return cls(
            category=data.get("category", "general_knowledge"),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
            original_query=query,
            metadata=data.get("metadata"),
        )


class RouterAgent:
    """Intent classification and routing agent for the Coiltech domain."""

    def __init__(self, llm_client: Any = None, confidence_threshold: float = 0.55):
        self._llm_client = llm_client
        self.confidence_threshold = confidence_threshold
        self.categories = INTENT_CATEGORIES

        # Domain-specific keyword patterns for fast pre-classification
        self._keyword_patterns = {
            "product_lookup": [
                "product", "catalog", "model", "family", "spec", "specification",
                "datasheet", "compact drive", "loop drive", "decoiler", "uncoiler",
                "straightener", "feeder", "servo drive", "transfer", "destacker",
                "cut-to-length", "ctl", "blank line", "press feeding",
                "ürün", "katalog",
            ],
            "application_engineering": [
                "application", "sizing", "feasibility", "which model", "recommend",
                "material", "strip width", "thickness", "speed requirement",
                "tonnage", "press type", "line layout", "engineering",
                "uygulama", "mühendislik",
            ],
            "configuration_request": [
                "configure", "configuration", "bom", "bill of material",
                "line setup", "components", "layout",
                "konfigürasyon",
            ],
            "quotation_request": [
                "quote", "quotation", "price", "pricing", "rfq", "proposal",
                "offer", "commercial", "discount", "payment terms",
                "teklif", "fiyat",
            ],
            "project_status": [
                "project", "milestone", "delivery", "schedule", "progress",
                "timeline", "shipping", "on time",
                "proje", "teslimat",
            ],
            "commissioning_support": [
                "commissioning", "installation", "startup", "handover",
                "acceptance test", "site", "erection",
                "devreye alma", "kurulum",
            ],
            "maintenance_request": [
                "maintenance", "service", "repair", "breakdown", "preventive",
                "pm schedule", "service ticket", "technician", "field service",
                "bakım", "arıza", "servis",
            ],
            "spare_parts_lookup": [
                "spare part", "part number", "replacement", "stock", "inventory",
                "order part", "lead time", "bearing", "belt", "motor",
                "yedek parça",
            ],
            "installed_base_query": [
                "installed", "machine list", "warranty", "serial number",
                "customer machine", "installed base",
                "kurulu makine",
            ],
            "anomaly_detection": [
                "alarm", "alert", "anomaly", "sensor", "overload", "vibration",
                "temperature", "deviation", "monitoring", "real-time", "twin",
                "alarm", "sapma",
            ],
            "quality_compliance": [
                "quality", "ce", "compliance", "ncr", "non-conformance",
                "audit", "certificate", "acceptance",
                "kalite", "uygunluk",
            ],
            "executive_summary": [
                "summary", "report", "kpi", "dashboard", "executive",
                "pipeline", "revenue", "overview", "management",
                "özet", "rapor", "yönetim",
            ],
        }
        self._keyword_priority = [
            "configuration_request",
            "application_engineering",
            "quotation_request",
            "project_status",
            "commissioning_support",
            "maintenance_request",
            "spare_parts_lookup",
            "installed_base_query",
            "anomaly_detection",
            "quality_compliance",
            "executive_summary",
            "product_lookup",
        ]

        logger.info("RouterAgent initialized (Coiltech domain)")

    @property
    def llm_client(self):
        if self._llm_client is None:
            from llm.llm_client import LLMClient
            self._llm_client = LLMClient()
        return self._llm_client

    def _quick_classify(self, query: str) -> Optional[str]:
        query_lower = query.lower()
        if self._looks_like_stock_lookup(query_lower):
            return "spare_parts_lookup"

        matches: Dict[str, int] = {}
        for category, keywords in self._keyword_patterns.items():
            match_count = sum(1 for kw in keywords if kw in query_lower)
            if match_count > 0:
                matches[category] = match_count

        if not matches:
            return None

        priority = {category: index for index, category in enumerate(self._keyword_priority)}
        best = sorted(matches, key=lambda category: (-matches[category], priority.get(category, 999)))[0]
        if matches[best] >= 2:
            return best
        return None

    def _looks_like_stock_lookup(self, query: str) -> bool:
        availability_terms = (
            "stock",
            "inventory",
            "available",
            "availability",
            "do we have",
            "stok",
            "stokta",
            "mevcut",
            "elde var",
            "var mi",
            "var mı",
        )
        spare_context_terms = (
            "spare",
            "part",
            "yedek",
            "parca",
            "parça",
            "bearing",
            "belt",
            "motor",
            "encoder",
        )
        part_code_pattern = r"\b[A-Z]{1,8}[-_][A-Z0-9]{2,}(?:[-_][A-Z0-9]+)*\b"
        has_availability_term = any(term in query for term in availability_terms)
        has_part_code = bool(re.search(part_code_pattern, query, flags=re.IGNORECASE))
        has_spare_context = any(term in query for term in spare_context_terms)
        return has_availability_term and (has_part_code or has_spare_context)

    async def classify_intent(
        self, user_message: str, context: Optional[Dict[str, Any]] = None
    ) -> IntentResult:
        logger.debug("Classifying intent", query=user_message[:60])

        quick_result = self._quick_classify(user_message)
        if quick_result:
            logger.info("Intent classified by keyword fast path", category=quick_result)
            return IntentResult(
                category=quick_result,
                confidence=0.82,
                reasoning="Keyword fast-path classification",
                original_query=user_message,
                metadata={"quick_classification": quick_result, "fast_path": True, "context_provided": bool(context)},
            )

        prompt = f"""Analyze this user message and classify its intent.

USER MESSAGE: {user_message}

{f"CONTEXT: {json.dumps(context)}" if context else ""}

Respond with a JSON object:
{{"category": "one of {self.categories}", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}

JSON Response:"""

        try:
            response = await self.llm_client.generate_json(
                prompt=prompt,
                system_prompt=ROUTING_SYSTEM_PROMPT,
                model_type="routing",
            )

            category = response.get("category", "general_knowledge")
            if category not in self.categories:
                category = "general_knowledge"

            result = IntentResult(
                category=category,
                confidence=min(max(float(response.get("confidence", 0.5)), 0), 1),
                reasoning=response.get("reasoning", ""),
                original_query=user_message,
                metadata={"quick_classification": quick_result, "context_provided": bool(context)},
            )

            logger.info("Intent classified", category=result.category, confidence=result.confidence)
            return result

        except Exception as e:
            logger.warning("LLM classification failed, using fallback", error=str(e))
            category = quick_result or "general_knowledge"
            return IntentResult(
                category=category,
                confidence=0.5,
                reasoning=f"Fallback classification: {str(e)}",
                original_query=user_message,
                metadata={"fallback": True, "error": str(e)},
            )

    def get_agent_for_intent(self, intent: IntentResult) -> str:
        if intent.confidence < self.confidence_threshold:
            logger.debug("Low confidence, routing to general", confidence=intent.confidence)
            return "GeneralAgent"
        return INTENT_AGENT_MAP.get(intent.category, "GeneralAgent")

    async def route(
        self, user_message: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        intent = await self.classify_intent(user_message, context)
        agent_name = self.get_agent_for_intent(intent)

        # Determine which RAG collections to search
        from config.settings import INTENT_COLLECTION_MAP
        collections_to_search = INTENT_COLLECTION_MAP.get(intent.category, [])

        return {
            "intent": intent.to_dict(),
            "target_agent": agent_name,
            "rag_collections": collections_to_search,
            "should_query_db": intent.category not in ["general_knowledge"],
        }


# =========================================================================
# Test queries for validation
# =========================================================================
TEST_QUERIES = [
    {"query": "What compact drive models do you offer for 300mm strip width?",
     "expected": "product_lookup"},
    {"query": "We need a feeder for 2mm steel, 60 strokes per minute, which model fits?",
     "expected": "application_engineering"},
    {"query": "Configure a line with decoiler, straightener and servo feeder for 400-ton press",
     "expected": "configuration_request"},
    {"query": "Can you send me a quotation for the CTL-3000 line?",
     "expected": "quotation_request"},
    {"query": "What is the status of project PRJ-2024-005?",
     "expected": "project_status"},
    {"query": "What is the commissioning checklist for a loop drive installation?",
     "expected": "commissioning_support"},
    {"query": "The servo feeder at Tofaş plant is showing error E-101, we need service",
     "expected": "maintenance_request"},
    {"query": "Do we have the servo motor SM-400 in stock?",
     "expected": "spare_parts_lookup"},
    {"query": "List all machines installed at Arçelik Eskişehir plant",
     "expected": "installed_base_query"},
    {"query": "Servo load alarm on machine IB-0015, what should we do?",
     "expected": "anomaly_detection"},
    {"query": "Where is the CE certificate for the CTL-3000?",
     "expected": "quality_compliance"},
    {"query": "Give me an executive summary of the current sales pipeline",
     "expected": "executive_summary"},
    {"query": "Hello, what can this platform do?",
     "expected": "general_knowledge"},
]
