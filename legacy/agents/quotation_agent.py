"""
Quotation Agent for Coiltech Industrial Automation Platform.

Handles quotation creation, pricing lookup, commercial terms,
and sales pipeline queries.
"""

from typing import Optional, Dict, Any, List
import structlog
from agents.base_agent import BaseAgent, AgentResponse, AgentContext

logger = structlog.get_logger(__name__)


class QuotationAgent(BaseAgent):
    """Handles quotation queries, pricing, and commercial information."""

    def __init__(self, **kwargs):
        super().__init__(
            name="QuotationAgent",
            department="quotation",
            description=(
                "Manages quotation information including pricing, commercial terms, "
                "payment conditions, delivery schedules, and sales pipeline status "
                "for Coiltech product lines."
            ),
            **kwargs,
        )

        self._queries = {
            "quotations": """
                SELECT q.code, q.revision, q.status, q.total_amount, q.currency,
                       q.delivery_weeks, q.payment_terms, q.incoterm, q.issued_date,
                       c.name AS customer_name, po.title AS opportunity_title
                FROM quotations q
                JOIN customers c ON q.customer_id = c.id
                LEFT JOIN project_opportunities po ON q.opportunity_id = po.id
                ORDER BY q.created_at DESC LIMIT 15
            """,
            "quotation_items": """
                SELECT qi.*, q.code AS quotation_code, pm.model_code, pm.name AS model_name
                FROM quotation_items qi
                JOIN quotations q ON qi.quotation_id = q.id
                LEFT JOIN product_models pm ON qi.model_id = pm.id
                ORDER BY q.code, qi.item_no
            """,
            "pipeline": """
                SELECT po.code, po.title, po.stage, po.estimated_value, po.currency,
                       po.probability_pct, po.expected_close_date, po.owner,
                       c.name AS customer_name
                FROM project_opportunities po
                JOIN customers c ON po.customer_id = c.id
                WHERE po.stage NOT IN ('lost','cancelled')
                ORDER BY po.stage, po.expected_close_date
            """,
            "pipeline_summary": "SELECT * FROM v_pipeline_summary",
        }

    async def handle(self, query: str, context: Optional[AgentContext] = None) -> AgentResponse:
        logger.info("QuotationAgent handling query", query=query[:60])

        quotations = await self.query_database(self._queries["quotations"])
        quotation_items = await self.query_database(self._queries["quotation_items"])
        pipeline = await self.query_database(self._queries["pipeline"])
        pipeline_summary = await self.query_database(self._queries["pipeline_summary"])
        focused_quotations = self._focus_records_by_query(
            query,
            quotations,
            ["code", "status", "customer_name", "opportunity_title", "prepared_by", "notes"],
            max_items=6,
        )
        focused_items = self._focus_records_by_query(
            query,
            quotation_items,
            ["quotation_code", "model_code", "model_name", "description", "notes"],
            max_items=10,
        )
        focused_pipeline = self._focus_records_by_query(
            query,
            pipeline,
            ["code", "title", "stage", "owner", "customer_name", "notes"],
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
FOCUSED QUOTATIONS FOR THIS QUERY:
{self._format_table(focused_quotations)}

FOCUSED QUOTATION ITEMS FOR THIS QUERY:
{self._format_table(focused_items)}

FOCUSED PIPELINE ITEMS FOR THIS QUERY:
{self._format_table(focused_pipeline)}

RECENT QUOTATIONS:
{self._format_table(quotations)}

SALES PIPELINE:
{self._format_table(pipeline[:15])}

PIPELINE SUMMARY:
{self._format_table(pipeline_summary)}
"""
        if sql_result and sql_result.get("success"):
            data_context += f"\nSPECIFIC QUERY RESULTS:\nSQL: {sql_result.get('generated_sql')}\n{self._format_table(sql_result.get('results', [])[:15])}\n"

        if rag_context:
            data_context += f"\nSALES DOCUMENTATION:\n{rag_context}\n"

        prompt = self.build_prompt(query=query, context=context, additional_context=data_context)

        system_prompt = self._system_prompt + """
When answering quotation and pricing questions:
1. Reference specific quotation codes (QUO-XXXX-XXX) and their status
2. Include commercial details: total amount, currency, delivery weeks, payment terms
3. Explain standard Coiltech terms: EXW, validity period, revision history
4. For pipeline questions, summarize by stage with value and probability
5. Note which quotations are linked to opportunities
6. Flag expired or at-risk quotations
7. Do NOT disclose customer-specific pricing to other customers
8. Suggest follow-up actions for draft or under-review quotations
9. If the user names a quotation code or customer, answer from focused quotations/items first
10. Do not ask for a quotation code if a matching quotation is already in the focused records
11. Respond in the same language as the user"""

        response = await self.call_llm(prompt, system_prompt)

        return self.format_response(
            content=response,
            data={
                "quotations": quotations,
                "focused_quotations": focused_quotations,
                "focused_items": focused_items,
                "focused_pipeline": focused_pipeline,
                "pipeline_summary": pipeline_summary,
                "sql_results": sql_result.get("results") if sql_result else None,
            },
            sources=sources,
            confidence=0.85 if quotations else 0.6,
        )

    def _format_table(self, data: List[Dict]) -> str:
        if not data:
            return "No data available"
        headers = list(data[0].keys())
        rows = [" | ".join(str(item.get(h, ""))[:30] for h in headers) for item in data]
        header_row = " | ".join(h[:30] for h in headers)
        return f"{header_row}\n{'-' * len(header_row)}\n" + "\n".join(rows)
