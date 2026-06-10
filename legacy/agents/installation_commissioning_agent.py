"""
Installation & Commissioning Agent for Coiltech Industrial Automation Platform.

Handles installation procedures, commissioning checklists, startup support,
acceptance testing, and handover documentation.
"""

from typing import Optional, Dict, Any, List
import structlog
from agents.base_agent import BaseAgent, AgentResponse, AgentContext

logger = structlog.get_logger(__name__)


class InstallationCommissioningAgent(BaseAgent):
    """Handles installation guidance, commissioning reports, and acceptance tests."""

    def __init__(self, **kwargs):
        super().__init__(
            name="InstallationCommissioningAgent",
            department="installation_commissioning",
            description=(
                "Provides installation procedures, commissioning checklists, "
                "startup support, acceptance test guidance, and handover "
                "documentation for Coiltech machine installations."
            ),
            **kwargs,
        )

        self._queries = {
            "commissioning_reports": """
                SELECT cr.report_date, cr.engineer, cr.status,
                       cr.feed_accuracy_mm, cr.speed_achieved_mpm,
                       cr.vibration_level, cr.noise_level_db,
                       cr.safety_check_passed, cr.customer_signoff,
                       cr.punch_list,
                       ib.code AS machine_code, ib.machine_name,
                       c.name AS customer_name
                FROM commissioning_reports cr
                JOIN installed_base ib ON cr.installed_base_id = ib.id
                JOIN customers c ON ib.customer_id = c.id
                ORDER BY cr.report_date DESC
            """,
            "pending_installations": """
                SELECT p.code AS project_code, p.title, p.status,
                       c.name AS customer_name,
                       pmach.description AS machine_description,
                       pmach.status AS machine_status,
                       pmod.model_code
                FROM projects p
                JOIN customers c ON p.customer_id = c.id
                JOIN project_machines pmach ON p.id = pmach.project_id
                LEFT JOIN product_models pmod ON pmach.model_id = pmod.id
                WHERE p.status IN ('shipping','installation','commissioning')
                ORDER BY p.status, p.target_completion
            """,
            "recent_installations": """
                SELECT ib.code, ib.machine_name, ib.serial_number,
                       ib.installation_date, ib.warranty_expiry, ib.status,
                       c.name AS customer_name, ib.location_plant,
                       pm.model_code
                FROM installed_base ib
                JOIN customers c ON ib.customer_id = c.id
                LEFT JOIN product_models pm ON ib.model_id = pm.id
                WHERE ib.installation_date IS NOT NULL
                ORDER BY ib.installation_date DESC LIMIT 15
            """,
        }

    async def handle(self, query: str, context: Optional[AgentContext] = None) -> AgentResponse:
        logger.info("InstallationCommissioningAgent handling query", query=query[:60])

        reports = await self.query_database(self._queries["commissioning_reports"])
        pending = await self.query_database(self._queries["pending_installations"])
        recent = await self.query_database(self._queries["recent_installations"])
        focused_reports = self._focus_records_by_query(
            query,
            reports,
            ["machine_code", "machine_name", "customer_name", "engineer", "status", "punch_list"],
            max_items=6,
        )
        focused_pending = self._focus_records_by_query(
            query,
            pending,
            ["project_code", "title", "status", "customer_name", "machine_description", "machine_status", "model_code"],
            max_items=8,
        )
        focused_recent = self._focus_records_by_query(
            query,
            recent,
            ["code", "machine_name", "serial_number", "customer_name", "location_plant", "model_code", "status"],
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
FOCUSED COMMISSIONING REPORTS FOR THIS QUERY:
{self._format_table(focused_reports)}

FOCUSED PENDING INSTALLATIONS FOR THIS QUERY:
{self._format_table(focused_pending)}

FOCUSED RECENT INSTALLATIONS FOR THIS QUERY:
{self._format_table(focused_recent)}

COMMISSIONING REPORTS:
{self._format_table(reports[:10])}

PENDING INSTALLATIONS (shipping/installation/commissioning):
{self._format_table(pending)}

RECENT INSTALLATIONS:
{self._format_table(recent[:10])}
"""
        if sql_result and sql_result.get("success"):
            data_context += f"\nSPECIFIC QUERY RESULTS:\nSQL: {sql_result.get('generated_sql')}\n{self._format_table(sql_result.get('results', [])[:15])}\n"

        if rag_context:
            data_context += f"\nINSTALLATION & COMMISSIONING DOCUMENTATION:\n{rag_context}\n"

        prompt = self.build_prompt(query=query, context=context, additional_context=data_context)

        system_prompt = self._system_prompt + """
When answering installation and commissioning questions:
1. Provide structured checklists for installation and commissioning procedures
2. Reference commissioning report data: feed accuracy, speed achieved, vibration, noise
3. Standard commissioning sequence: mechanical install → electrical → pneumatic → safety → dry run → material test → acceptance
4. Highlight safety checks that MUST pass before operation
5. Note punch list items from commissioning reports
6. Track customer sign-off status
7. For startup issues, reference similar past commissioning reports
8. Include foundation requirements, utility connections, and safety interlock checks
9. Remind about warranty start date (typically from commissioning sign-off)
10. If focused reports/installations are present, answer from those rows first
11. Respond in the same language as the user"""

        response = await self.call_llm(prompt, system_prompt)

        return self.format_response(
            content=response,
            data={
                "commissioning_reports": reports[:5],
                "focused_reports": focused_reports,
                "focused_pending": focused_pending,
                "focused_recent": focused_recent,
                "pending_installations": pending,
                "sql_results": sql_result.get("results") if sql_result else None,
            },
            sources=sources,
            confidence=0.85 if reports or pending else 0.6,
        )

    def _format_table(self, data: List[Dict]) -> str:
        if not data:
            return "No data available"
        headers = list(data[0].keys())
        rows = [" | ".join(str(item.get(h, ""))[:30] for h in headers) for item in data]
        header_row = " | ".join(h[:30] for h in headers)
        return f"{header_row}\n{'-' * len(header_row)}\n" + "\n".join(rows)
