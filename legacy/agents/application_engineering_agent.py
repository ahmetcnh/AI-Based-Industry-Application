"""
Application Engineering Agent for Coiltech Industrial Automation Platform.

Handles technical feasibility analysis, model selection recommendations,
sizing calculations, and line layout guidance for press feeding systems.
"""

from typing import Optional, Dict, Any, List
import re
import structlog
from agents.base_agent import BaseAgent, AgentResponse, AgentContext

logger = structlog.get_logger(__name__)


class ApplicationEngineeringAgent(BaseAgent):
    """Provides technical selection, sizing, and feasibility analysis."""

    def __init__(self, **kwargs):
        super().__init__(
            name="ApplicationEngineeringAgent",
            department="application_engineering",
            description=(
                "Assists with technical model selection, feasibility studies, "
                "sizing for strip width/thickness/speed requirements, material "
                "compatibility, and line layout recommendations for Coiltech systems."
            ),
            **kwargs,
        )

        self._queries = {
            "models_with_capabilities": """
                SELECT pm.*, pf.name AS family_name, pf.category,
                       GROUP_CONCAT(mc.capability_name || '=' || COALESCE(mc.capability_value,'yes')) AS capabilities
                FROM product_models pm
                JOIN product_families pf ON pm.family_id = pf.id
                LEFT JOIN machine_capabilities mc ON pm.id = mc.model_id
                WHERE pm.is_active = 1
                GROUP BY pm.id
                ORDER BY pf.category, pm.max_strip_width_mm
            """,
            "families": """
                SELECT pf.id, pf.code, pf.name, pf.category, pf.description
                FROM product_families pf WHERE pf.is_active = 1
                ORDER BY pf.category, pf.name
            """,
            "line_configs": """
                SELECT lc.*, p.title AS project_title, c.name AS customer_name
                FROM line_configurations lc
                LEFT JOIN projects p ON lc.project_id = p.id
                LEFT JOIN project_opportunities po ON lc.opportunity_id = po.id
                LEFT JOIN customers c ON COALESCE(p.customer_id, po.customer_id) = c.id
                ORDER BY lc.created_at DESC LIMIT 10
            """,
        }

    async def handle(self, query: str, context: Optional[AgentContext] = None) -> AgentResponse:
        logger.info("ApplicationEngineeringAgent handling query", query=query[:60])

        models = await self.query_database(self._queries["models_with_capabilities"])
        families = await self.query_database(self._queries["families"])
        candidate_models = self._rank_candidate_models(query, models)

        rag_context = ""
        sources = []
        if context and context.user_metadata:
            rag_context = context.user_metadata.get("rag_context", "")
            sources = context.user_metadata.get("rag_sources", [])

        sql_result = None
        if self.should_attempt_nl_to_sql(query):
            try:
                sql_result = await self.nl_to_sql(query)
            except Exception:
                pass

        data_context = f"""
APPLICATION REQUIREMENT EXTRACTION:
{self._format_requirements(query)}

CANDIDATE MODELS FOR THIS QUERY:
{self._format_table(candidate_models)}

COILTECH PRODUCT MODELS WITH CAPABILITIES:
{self._format_table(models[:25])}

PRODUCT FAMILIES:
{self._format_table(families)}
"""
        if sql_result and sql_result.get("success"):
            data_context += f"\nSPECIFIC QUERY RESULTS:\nSQL: {sql_result.get('generated_sql')}\n{self._format_table(sql_result.get('results', [])[:15])}\n"

        if rag_context:
            data_context += f"\nTECHNICAL DOCUMENTATION:\n{rag_context}\n"

        prompt = self.build_prompt(query=query, context=context, additional_context=data_context)

        system_prompt = self._system_prompt + """
When answering application engineering questions:
1. Match customer requirements (strip width, thickness, speed, material) to suitable models
2. Explain WHY a particular model fits or doesn't fit the application
3. Provide key specs: max strip width, max thickness, feed length, speed, coil weight capacity
4. Warn about edge cases: material hardness, high-strength steel, thin gauge flutter
5. Suggest line layout components (decoiler → straightener → feeder → press)
6. For ambiguous requirements, ask clarifying questions about press type, SPM, progression
7. Reference similar past configurations when available
8. Always recommend contacting Coiltech application engineering for final validation
9. If CANDIDATE MODELS contains rows, start from those candidates and compare them against the extracted requirements
10. Do not recommend a model that fails a stated width or thickness requirement unless you clearly label it as unsuitable
11. Respond in the same language as the user"""

        response = await self.call_llm(prompt, system_prompt)

        return self.format_response(
            content=response,
            data={
                "models": models[:10],
                "candidate_models": candidate_models,
                "families": families,
                "sql_results": sql_result.get("results") if sql_result else None,
            },
            sources=sources,
            confidence=0.85 if models else 0.6,
        )

    def _extract_requirement_number(self, query: str, labels: List[str]) -> Optional[float]:
        normalized = self._normalize_text(query)
        for label in labels:
            label_norm = self._normalize_text(label)
            match = re.search(rf"(\d+(?:[.,]\d+)?)\s*(?:mm|mpm|m/min)?\s*(?:{label_norm})", normalized)
            if match:
                return float(match.group(1).replace(",", "."))
            match = re.search(rf"(?:{label_norm})\s*(?:of|is|=|:)?\s*(\d+(?:[.,]\d+)?)", normalized)
            if match:
                return float(match.group(1).replace(",", "."))
        return None

    def _extract_requirements(self, query: str) -> Dict[str, Optional[float]]:
        return {
            "width_mm": self._extract_requirement_number(query, ["width", "genislik", "genişlik"]),
            "thickness_mm": self._extract_requirement_number(query, ["thickness", "kalinlik", "kalınlık"]),
            "speed_mpm": self._extract_requirement_number(query, ["speed", "hiz", "hız"]),
        }

    def _format_requirements(self, query: str) -> str:
        req = self._extract_requirements(query)
        extracted = {key: value for key, value in req.items() if value is not None}
        return str(extracted) if extracted else "No numeric width/thickness/speed requirement detected."

    def _rank_candidate_models(self, query: str, models: List[Dict]) -> List[Dict]:
        req = self._extract_requirements(query)
        candidates = []
        for model in models:
            width_ok = req["width_mm"] is None or (model.get("max_strip_width_mm") or 0) >= req["width_mm"]
            thickness_ok = req["thickness_mm"] is None or (model.get("max_strip_thickness_mm") or 0) >= req["thickness_mm"]
            speed_ok = req["speed_mpm"] is None or (model.get("max_speed_mpm") or 0) >= req["speed_mpm"]
            if width_ok and thickness_ok and speed_ok:
                candidates.append(model)

        if candidates and any(value is not None for value in req.values()):
            return sorted(
                candidates,
                key=lambda item: (
                    item.get("max_strip_width_mm") or 999999,
                    item.get("max_strip_thickness_mm") or 999999,
                    item.get("max_speed_mpm") or 999999,
                ),
            )[:8]

        return self._focus_records_by_query(
            query,
            models,
            ["model_code", "name", "family_name", "category", "description", "capabilities"],
            max_items=8,
        )

    def _format_table(self, data: List[Dict]) -> str:
        if not data:
            return "No data available"
        headers = list(data[0].keys())
        rows = [" | ".join(str(item.get(h, ""))[:30] for h in headers) for item in data]
        header_row = " | ".join(h[:30] for h in headers)
        return f"{header_row}\n{'-' * len(header_row)}\n" + "\n".join(rows)
