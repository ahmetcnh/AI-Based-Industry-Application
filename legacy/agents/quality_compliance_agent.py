"""
Quality & Compliance Agent for Coiltech Industrial Automation Platform.

Handles quality events, NCRs, CE compliance, acceptance tests,
audit findings, and quality documentation queries.
"""

from typing import Optional, Dict, Any, List
import structlog
from agents.base_agent import BaseAgent, AgentResponse, AgentContext

logger = structlog.get_logger(__name__)


class QualityComplianceAgent(BaseAgent):
    """Handles quality events, compliance documentation, and audit tracking."""

    def __init__(self, **kwargs):
        super().__init__(
            name="QualityComplianceAgent",
            department="quality_compliance",
            description=(
                "Manages quality events including NCRs, customer complaints, "
                "CE compliance reviews, acceptance test results, supplier issues, "
                "and warranty claims for Coiltech products."
            ),
            **kwargs,
        )

        self._queries = {
            "quality_events": """
                SELECT qe.code, qe.event_type, qe.severity, qe.title,
                       qe.status, qe.assigned_to, qe.created_at, qe.closed_at,
                       qe.related_entity_type, qe.related_entity_id
                FROM quality_events qe
                ORDER BY
                    CASE qe.severity
                        WHEN 'critical' THEN 0
                        WHEN 'major' THEN 1
                        ELSE 2
                    END,
                    qe.created_at DESC
            """,
            "open_events": """
                SELECT qe.code, qe.event_type, qe.severity, qe.title,
                       qe.status, qe.assigned_to, qe.description,
                       qe.root_cause, qe.corrective_action
                FROM quality_events qe
                WHERE qe.status NOT IN ('closed')
                ORDER BY qe.severity, qe.created_at DESC
            """,
            "event_summary": """
                SELECT event_type, severity, status, COUNT(*) AS count
                FROM quality_events
                GROUP BY event_type, severity, status
                ORDER BY event_type, severity
            """,
            "certificates": """
                SELECT td.title, td.doc_type, td.version, td.language, td.uploaded_at,
                       pm.model_code, pf.name AS family_name
                FROM technical_documents td
                LEFT JOIN product_models pm ON td.model_id = pm.id
                LEFT JOIN product_families pf ON td.family_id = pf.id
                WHERE td.doc_type IN ('certificate','sop')
                ORDER BY td.uploaded_at DESC
            """,
        }

    async def handle(self, query: str, context: Optional[AgentContext] = None) -> AgentResponse:
        logger.info("QualityComplianceAgent handling query", query=query[:60])

        all_events = await self.query_database(self._queries["quality_events"])
        open_events = await self.query_database(self._queries["open_events"])
        summary = await self.query_database(self._queries["event_summary"])
        certificates = await self.query_database(self._queries["certificates"])
        focused_events = self._focus_records_by_query(
            query,
            all_events,
            ["code", "event_type", "severity", "title", "status", "assigned_to", "related_entity_type"],
            max_items=8,
        )
        focused_open_events = self._focus_records_by_query(
            query,
            open_events,
            ["code", "event_type", "severity", "title", "status", "assigned_to", "description", "root_cause", "corrective_action"],
            max_items=8,
        )
        focused_documents = self._focus_records_by_query(
            query,
            certificates,
            ["title", "doc_type", "version", "language", "model_code", "family_name"],
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
FOCUSED QUALITY EVENTS FOR THIS QUERY:
{self._format_table(focused_events)}

FOCUSED OPEN QUALITY EVENTS FOR THIS QUERY:
{self._format_table(focused_open_events)}

FOCUSED CERTIFICATES & QUALITY DOCUMENTS FOR THIS QUERY:
{self._format_table(focused_documents)}

OPEN QUALITY EVENTS:
{self._format_table(open_events[:15])}

QUALITY EVENT SUMMARY:
{self._format_table(summary)}

ALL QUALITY EVENTS:
{self._format_table(all_events[:15])}

CERTIFICATES & QUALITY DOCUMENTS:
{self._format_table(certificates[:10])}
"""
        if sql_result and sql_result.get("success"):
            data_context += f"\nSPECIFIC QUERY RESULTS:\nSQL: {sql_result.get('generated_sql')}\n{self._format_table(sql_result.get('results', [])[:15])}\n"

        if rag_context:
            data_context += f"\nQUALITY & COMPLIANCE DOCUMENTATION:\n{rag_context}\n"

        prompt = self.build_prompt(query=query, context=context, additional_context=data_context)

        system_prompt = self._system_prompt + """
When answering quality and compliance questions:
1. Reference quality event codes (QE-XXXX-XXX) and their severity
2. Distinguish event types: NCR, customer complaint, internal audit, supplier issue, CE review, acceptance test, warranty claim
3. For open events, emphasize root cause analysis and corrective actions
4. CE compliance: reference Machinery Directive 2006/42/EC and relevant EN standards
5. For acceptance tests, reference commissioning report data
6. Track closure rates and recurring issues
7. Note which events are linked to specific projects or installed machines
8. Provide structured CAPA (Corrective and Preventive Action) guidance
9. Flag critical severity events that need immediate attention
10. If focused documents/events are present, answer from those rows first
11. Do not claim a certificate or CE document exists unless it is in focused documents, certificates, or retrieved sources
12. Respond in the same language as the user"""

        response = await self.call_llm(prompt, system_prompt)

        return self.format_response(
            content=response,
            data={
                "open_events": open_events,
                "focused_events": focused_events,
                "focused_open_events": focused_open_events,
                "focused_documents": focused_documents,
                "summary": summary,
                "certificates": certificates,
                "sql_results": sql_result.get("results") if sql_result else None,
            },
            sources=sources,
            confidence=0.85 if all_events else 0.6,
        )

    def _format_table(self, data: List[Dict]) -> str:
        if not data:
            return "No data available"
        headers = list(data[0].keys())
        rows = [" | ".join(str(item.get(h, ""))[:30] for h in headers) for item in data]
        header_row = " | ".join(h[:30] for h in headers)
        return f"{header_row}\n{'-' * len(header_row)}\n" + "\n".join(rows)
