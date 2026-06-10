"""
Spare Parts Agent for Coiltech Industrial Automation Platform.

Handles spare parts lookup, stock availability, part identification,
reorder recommendations, and parts ordering support.
"""

from typing import Optional, Dict, Any, List
import structlog
from agents.base_agent import BaseAgent, AgentResponse, AgentContext

logger = structlog.get_logger(__name__)


class SparePartsAgent(BaseAgent):
    """Handles spare parts queries, stock levels, and ordering support."""

    def __init__(self, **kwargs):
        super().__init__(
            name="SparePartsAgent",
            department="spare_parts",
            description=(
                "Provides spare parts information including stock availability, "
                "part numbers, pricing, lead times, applicable machine models, "
                "and reorder recommendations for Coiltech systems."
            ),
            **kwargs,
        )

        self._queries = {
            "low_stock": "SELECT * FROM v_spare_parts_low_stock",
            "all_parts": """
                SELECT sp.part_number, sp.name, sp.category,
                       sp.applicable_models, sp.unit, sp.unit_cost,
                       sp.lead_time_days, sp.supplier, sp.is_critical,
                       sps.quantity, sps.min_stock, sps.max_stock, sps.location
                FROM spare_parts sp
                LEFT JOIN spare_part_stock sps ON sp.id = sps.part_id
                ORDER BY sp.category, sp.name
            """,
            "critical_parts": """
                SELECT sp.part_number, sp.name, sp.category,
                       sp.unit_cost, sp.lead_time_days, sp.supplier,
                       sps.quantity, sps.min_stock
                FROM spare_parts sp
                LEFT JOIN spare_part_stock sps ON sp.id = sps.part_id
                WHERE sp.is_critical = 1
                ORDER BY sps.quantity ASC
            """,
        }

    async def handle(self, query: str, context: Optional[AgentContext] = None) -> AgentResponse:
        logger.info("SparePartsAgent handling query", query=query[:60])

        low_stock = await self.query_database(self._queries["low_stock"])
        all_parts = await self.query_database(self._queries["all_parts"])
        critical = await self.query_database(self._queries["critical_parts"])
        focused_parts = self._focus_records_by_query(
            query,
            all_parts,
            ["part_number", "name", "category", "applicable_models", "supplier", "location"],
            max_items=8,
        )
        focused_low_stock = self._focus_records_by_query(
            query,
            low_stock,
            ["part_number", "name", "category", "supplier"],
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
FOCUSED SPARE PARTS FOR THIS QUERY:
{self._format_table(focused_parts)}

FOCUSED LOW-STOCK PARTS FOR THIS QUERY:
{self._format_table(focused_low_stock)}

LOW STOCK PARTS (at or below reorder point):
{self._format_table(low_stock)}

CRITICAL SPARE PARTS:
{self._format_table(critical)}

FULL PARTS CATALOG:
{self._format_table(all_parts[:20])}
"""
        if sql_result and sql_result.get("success"):
            data_context += f"\nSPECIFIC QUERY RESULTS:\nSQL: {sql_result.get('generated_sql')}\n{self._format_table(sql_result.get('results', [])[:15])}\n"

        if rag_context:
            data_context += f"\nSPARE PARTS DOCUMENTATION:\n{rag_context}\n"

        prompt = self.build_prompt(query=query, context=context, additional_context=data_context)

        system_prompt = self._system_prompt + """
When answering spare parts questions:
1. Always reference actual part numbers from the data (for example SM-400 or ENC-CTL-01) for unambiguous identification
2. Show current stock levels and minimum stock thresholds
3. Highlight critical parts (is_critical=1) that could cause line stoppage
4. Include lead times and supplier information for reordering
5. Note which machine models each part applies to (applicable_models)
6. Flag parts below minimum stock that need immediate reorder
7. Provide unit cost information when available
8. For part identification questions, cross-reference with machine model and category
9. Suggest alternative or compatible parts when the exact part is out of stock
10. If focused spare part rows are present, answer from those rows first
11. Do not invent voltage, frequency, supplier, or compatibility details not present in the data
12. Respond in the same language as the user"""

        response = await self.call_llm(prompt, system_prompt)

        return self.format_response(
            content=response,
            data={
                "low_stock": low_stock,
                "focused_parts": focused_parts,
                "focused_low_stock": focused_low_stock,
                "critical_parts": critical,
                "sql_results": sql_result.get("results") if sql_result else None,
            },
            sources=sources,
            confidence=0.85 if all_parts else 0.6,
        )

    def _format_table(self, data: List[Dict]) -> str:
        if not data:
            return "No data available"
        headers = list(data[0].keys())
        rows = [" | ".join(str(item.get(h, ""))[:30] for h in headers) for item in data]
        header_row = " | ".join(h[:30] for h in headers)
        return f"{header_row}\n{'-' * len(header_row)}\n" + "\n".join(rows)
