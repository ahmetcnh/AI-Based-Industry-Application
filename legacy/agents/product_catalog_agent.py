"""
Product Catalog Agent for Coiltech Industrial Automation Platform.

Handles product family information, model specifications, and capabilities
for press feeding systems, decoilers, CTL lines, and transfer systems.
"""

from typing import Optional, Dict, Any, List
import structlog
from agents.base_agent import BaseAgent, AgentResponse, AgentContext

logger = structlog.get_logger(__name__)


class ProductCatalogAgent(BaseAgent):
    """Handles product lookup, specification queries, and catalog browsing."""

    def __init__(self, **kwargs):
        super().__init__(
            name="ProductCatalogAgent",
            department="product_catalog",
            description=(
                "Provides information about Coiltech product families and models "
                "including compact drives, loop drives, servo drives, decoilers, "
                "cut-to-length lines, press transfer systems, and blank lines."
            ),
            **kwargs,
        )

        self._queries = {
            "families": """
                SELECT pf.id, pf.code, pf.name, pf.category, pf.description,
                       COUNT(pm.id) AS model_count
                FROM product_families pf
                LEFT JOIN product_models pm ON pf.id = pm.family_id AND pm.is_active = 1
                WHERE pf.is_active = 1
                GROUP BY pf.id ORDER BY pf.category, pf.name
            """,
            "models_by_family": """
                SELECT pm.*, pf.name AS family_name, pf.category
                FROM product_models pm
                JOIN product_families pf ON pm.family_id = pf.id
                WHERE pf.is_active = 1 AND pm.is_active = 1
                ORDER BY pf.category, pf.name, pm.name
            """,
            "capabilities": """
                SELECT mc.*, pm.model_code, pm.name AS model_name
                FROM machine_capabilities mc
                JOIN product_models pm ON mc.model_id = pm.id
                ORDER BY pm.model_code
            """,
        }

    async def handle(self, query: str, context: Optional[AgentContext] = None) -> AgentResponse:
        logger.info("ProductCatalogAgent handling query", query=query[:60])

        families = await self.query_database(self._queries["families"])
        models = await self.query_database(self._queries["models_by_family"])
        focused_models = self._filter_models_for_query(query, models)

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
COILTECH PRODUCT CATALOG:

INSTRUCTION FOR THIS QUERY:
{self._build_query_instruction(query, focused_models)}

MATCHING PRODUCT MODELS FOR THIS QUERY:
{self._format_model_list(focused_models)}

PRODUCT FAMILIES:
{self._format_table(families)}

PRODUCT MODELS (all families):
{self._format_table(models[:20])}
"""
        if sql_result and sql_result.get("success"):
            data_context += f"\nSPECIFIC QUERY RESULTS:\nSQL: {sql_result.get('generated_sql')}\n{self._format_table(sql_result.get('results', [])[:15])}\n"

        if rag_context:
            data_context += f"\nPRODUCT DOCUMENTATION:\n{rag_context}\n"

        prompt = self.build_prompt(query=query, context=context, additional_context=data_context)

        system_prompt = self._system_prompt + """
When answering product questions:
1. Be specific about model numbers and specifications
2. Compare models when the user is choosing between options
3. Mention key specs: strip width, thickness, speed, coil weight
4. Reference technical documents when available
5. Suggest contacting application engineering for complex selections
6. If the user asks to list models or asks which models are available, answer from MATCHING PRODUCT MODELS first
7. Do not ask the user for a model number when the requested product family is already clear
8. If MATCHING PRODUCT MODELS contains rows, name each model code and summarize its key specs
9. Respond in the same language as the user"""

        response = await self.call_llm(prompt, system_prompt)

        return self.format_response(
            content=response,
            data={
                "families": families,
                "matching_models": focused_models,
                "sql_results": sql_result.get("results") if sql_result else None,
            },
            sources=sources,
            confidence=0.85 if families else 0.6,
        )

    def _build_query_instruction(self, query: str, focused_models: List[Dict]) -> str:
        normalized = query.lower()
        asks_to_list = any(
            term in normalized
            for term in (
                "list",
                "show",
                "which",
                "available",
                "hangi",
                "liste",
                "listeler",
                "listeler misin",
                "modeller",
                "models",
            )
        )
        if focused_models and asks_to_list:
            codes = ", ".join(str(item.get("model_code", "")) for item in focused_models)
            return (
                "The user is asking for available product models. Answer directly "
                f"from MATCHING PRODUCT MODELS and include these model codes: {codes}. "
                "Do not ask the user to provide a model number."
            )
        if focused_models:
            return "Use MATCHING PRODUCT MODELS as the primary source for this product answer."
        return "Use PRODUCT FAMILIES and PRODUCT MODELS as the primary source for this product answer."

    def _filter_models_for_query(self, query: str, models: List[Dict]) -> List[Dict]:
        normalized = query.lower()
        family_aliases = [
            (("compact drive", "compact drives", "compact", "kompakt"), "Compact Drives"),
            (("loop drive", "loop type", "loop"), "Loop Type Drives"),
            (("decoiler", "uncoiler", "rulo acici", "rulo açıcı"), "Decoilers"),
            (("cut-to-length", "ctl"), "Cut-to-Length Lines"),
            (("destacker",), "Destacker Systems"),
            (("press transfer robot", "transfer robot"), "Press Transfer Robots"),
            (("press transfer",), "Press Transfer Systems"),
            (("servo drive", "servo"), "Servo Drives"),
        ]

        for aliases, family_name in family_aliases:
            if any(alias in normalized for alias in aliases):
                matched = [
                    model for model in models
                    if family_name.lower() in str(model.get("family_name", "")).lower()
                ]
                if matched:
                    return matched

        code_matches = [
            model for model in models
            if str(model.get("model_code", "")).lower() in normalized
        ]
        if code_matches:
            return code_matches

        return models[:12]

    def _format_model_list(self, data: List[Dict]) -> str:
        if not data:
            return "No matching models found in the product catalog."

        lines = []
        for item in data:
            lines.append(
                "- {code}: {name} | family={family} | max_width={width} mm | "
                "max_thickness={thickness} mm | max_speed={speed} m/min | "
                "coil_weight={coil_weight} kg | servo_power={servo_power} kW | {description}".format(
                    code=item.get("model_code", ""),
                    name=item.get("name", ""),
                    family=item.get("family_name", ""),
                    width=item.get("max_strip_width_mm", ""),
                    thickness=item.get("max_strip_thickness_mm", ""),
                    speed=item.get("max_speed_mpm", ""),
                    coil_weight=item.get("max_coil_weight_kg", ""),
                    servo_power=item.get("servo_power_kw", ""),
                    description=item.get("description", ""),
                )
            )
        return "\n".join(lines)

    def _format_table(self, data: List[Dict]) -> str:
        if not data:
            return "No data available"
        headers = list(data[0].keys())
        rows = [" | ".join(str(item.get(h, ""))[:25] for h in headers) for item in data]
        header_row = " | ".join(h[:25] for h in headers)
        return f"{header_row}\n{'-' * len(header_row)}\n" + "\n".join(rows)
