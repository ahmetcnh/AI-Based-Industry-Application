"""
Configuration Agent for Coiltech Industrial Automation Platform.

Handles line configuration assembly, BOM generation, component selection,
and layout recommendations for press feeding and CTL systems.
"""

from typing import Optional, Dict, Any, List
import structlog
from agents.base_agent import BaseAgent, AgentResponse, AgentContext

logger = structlog.get_logger(__name__)


class ConfigurationAgent(BaseAgent):
    """Handles line configuration, BOM assembly, and component selection."""

    def __init__(self, **kwargs):
        super().__init__(
            name="ConfigurationAgent",
            department="configuration",
            description=(
                "Assembles line configurations and bills of material for "
                "press feeding lines, CTL lines, blank lines, and transfer systems. "
                "Selects components based on material specs and production requirements."
            ),
            **kwargs,
        )

        self._queries = {
            "line_configs": """
                SELECT lc.*, p.title AS project_title, c.name AS customer_name
                FROM line_configurations lc
                LEFT JOIN projects p ON lc.project_id = p.id
                LEFT JOIN project_opportunities po ON lc.opportunity_id = po.id
                LEFT JOIN customers c ON COALESCE(p.customer_id, po.customer_id) = c.id
                ORDER BY lc.created_at DESC LIMIT 15
            """,
            "bom_components": """
                SELECT bc.*, lc.name AS config_name, lc.line_type,
                       pm.model_code, pm.name AS model_name
                FROM bom_components bc
                JOIN line_configurations lc ON bc.line_config_id = lc.id
                LEFT JOIN product_models pm ON bc.model_id = pm.id
                ORDER BY lc.name, bc.id
            """,
            "models": """
                SELECT pm.model_code, pm.name, pf.name AS family_name, pf.category,
                       pm.max_strip_width_mm, pm.max_strip_thickness_mm,
                       pm.max_speed_mpm, pm.max_coil_weight_kg
                FROM product_models pm
                JOIN product_families pf ON pm.family_id = pf.id
                WHERE pm.is_active = 1
                ORDER BY pf.category, pm.name
            """,
        }

    async def handle(self, query: str, context: Optional[AgentContext] = None) -> AgentResponse:
        logger.info("ConfigurationAgent handling query", query=query[:60])

        configs = await self.query_database(self._queries["line_configs"])
        bom = await self.query_database(self._queries["bom_components"])
        models = await self.query_database(self._queries["models"])
        focused_configs = self._focus_records_by_query(
            query,
            configs,
            ["code", "name", "title", "project_title", "customer_name", "line_type", "material_type", "notes"],
            max_items=6,
        )
        focused_bom = self._focus_records_by_query(
            query,
            bom,
            ["config_name", "line_type", "component_name", "component_type", "model_code", "model_name", "notes"],
            max_items=10,
        )
        focused_models = self._focus_records_by_query(
            query,
            models,
            ["model_code", "name", "family_name", "category"],
            max_items=8,
        )

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
FOCUSED CONFIGURATIONS FOR THIS QUERY:
{self._format_table(focused_configs)}

FOCUSED BOM COMPONENTS FOR THIS QUERY:
{self._format_table(focused_bom)}

FOCUSED PRODUCT MODELS FOR THIS QUERY:
{self._format_table(focused_models)}

EXISTING LINE CONFIGURATIONS:
{self._format_table(configs)}

BOM COMPONENTS (recent configs):
{self._format_table(bom[:20])}

AVAILABLE PRODUCT MODELS:
{self._format_table(models[:20])}
"""
        if sql_result and sql_result.get("success"):
            data_context += f"\nSPECIFIC QUERY RESULTS:\nSQL: {sql_result.get('generated_sql')}\n{self._format_table(sql_result.get('results', [])[:15])}\n"

        if rag_context:
            data_context += f"\nTECHNICAL DOCUMENTATION:\n{rag_context}\n"

        prompt = self.build_prompt(query=query, context=context, additional_context=data_context)

        system_prompt = self._system_prompt + """
When answering configuration questions:
1. A typical press feeding line includes: Decoiler → Straightener → Servo Feeder → Press interface
2. A CTL line includes: Decoiler → Straightener → Measuring system → Shear → Stacker
3. Explain which components are required vs. optional for the given application
4. Match component capacities — decoiler coil weight must handle the feeder's strip width
5. Note interface requirements: press cam signals, PLC communication, safety interlocks
6. Reference similar past configurations when available
7. Provide a structured BOM list when assembling a new configuration
8. Mention lead times and any long-lead components
9. If focused records are present, use them before generic templates
10. Do not invent BOM items when existing BOM components are available
11. Respond in the same language as the user"""

        response = await self.call_llm(prompt, system_prompt)

        return self.format_response(
            content=response,
            data={
                "configurations": configs,
                "focused_configurations": focused_configs,
                "focused_bom": focused_bom,
                "focused_models": focused_models,
                "bom": bom[:10],
                "sql_results": sql_result.get("results") if sql_result else None,
            },
            sources=sources,
            confidence=0.85 if configs else 0.65,
        )

    def _format_table(self, data: List[Dict]) -> str:
        if not data:
            return "No data available"
        headers = list(data[0].keys())
        rows = [" | ".join(str(item.get(h, ""))[:30] for h in headers) for item in data]
        header_row = " | ".join(h[:30] for h in headers)
        return f"{header_row}\n{'-' * len(header_row)}\n" + "\n".join(rows)
