"""
Service & Maintenance Agent for Coiltech Industrial Automation Platform.

Handles service tickets, preventive maintenance schedules, repair history,
installed base queries, and field service coordination.
"""

from typing import Optional, Dict, Any, List
import structlog
from agents.base_agent import BaseAgent, AgentResponse, AgentContext

logger = structlog.get_logger(__name__)


class ServiceMaintenanceAgent(BaseAgent):
    """Handles service tickets, maintenance plans, installed base, and repair history."""

    def __init__(self, **kwargs):
        super().__init__(
            name="ServiceMaintenanceAgent",
            department="service_maintenance",
            description=(
                "Manages service tickets, preventive and corrective maintenance, "
                "installed base registry, warranty tracking, and field service "
                "coordination for Coiltech machines in the field."
            ),
            **kwargs,
        )

        self._queries = {
            "open_tickets": "SELECT * FROM v_open_service_tickets",
            "installed_base": "SELECT * FROM v_installed_base_overview",
            "maintenance_plans": """
                SELECT mp.task_name, mp.interval_type, mp.interval_value,
                       mp.last_performed, mp.next_due, mp.assigned_to,
                       ib.code AS machine_code, ib.machine_name,
                       c.name AS customer_name
                FROM maintenance_plans mp
                JOIN installed_base ib ON mp.installed_base_id = ib.id
                JOIN customers c ON ib.customer_id = c.id
                ORDER BY mp.next_due
            """,
            "recent_maintenance": """
                SELECT ml.performed_date, ml.performed_by, ml.type,
                       ml.description, ml.duration_hours, ml.cost,
                       ib.code AS machine_code, ib.machine_name
                FROM maintenance_logs ml
                JOIN installed_base ib ON ml.installed_base_id = ib.id
                ORDER BY ml.performed_date DESC LIMIT 20
            """,
            "ticket_summary": """
                SELECT st.type, st.priority, st.status, COUNT(*) AS count
                FROM service_tickets st
                GROUP BY st.type, st.priority, st.status
                ORDER BY st.priority, st.status
            """,
        }

    async def handle(self, query: str, context: Optional[AgentContext] = None) -> AgentResponse:
        logger.info("ServiceMaintenanceAgent handling query", query=query[:60])

        open_tickets = await self.query_database(self._queries["open_tickets"])
        installed_base = await self.query_database(self._queries["installed_base"])
        plans = await self.query_database(self._queries["maintenance_plans"])
        recent_logs = await self.query_database(self._queries["recent_maintenance"])
        ticket_summary = await self.query_database(self._queries["ticket_summary"])
        focused_tickets = self._focus_records_by_query(
            query,
            open_tickets,
            ["code", "type", "priority", "status", "customer_name", "machine_code", "machine_name", "title", "description"],
            max_items=8,
        )
        focused_machines = self._focus_records_by_query(
            query,
            installed_base,
            ["code", "machine_name", "serial_number", "customer_name", "location_plant", "line_name", "status", "model_code"],
            max_items=8,
        )
        focused_plans = self._focus_records_by_query(
            query,
            plans,
            ["task_name", "assigned_to", "machine_code", "machine_name", "customer_name"],
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
FOCUSED SERVICE TICKETS FOR THIS QUERY:
{self._format_table(focused_tickets)}

FOCUSED INSTALLED MACHINES FOR THIS QUERY:
{self._format_table(focused_machines)}

FOCUSED MAINTENANCE PLANS FOR THIS QUERY:
{self._format_table(focused_plans)}

OPEN SERVICE TICKETS:
{self._format_table(open_tickets[:15])}

TICKET SUMMARY (by type/priority/status):
{self._format_table(ticket_summary)}

INSTALLED BASE:
{self._format_table(installed_base[:15])}

UPCOMING MAINTENANCE:
{self._format_table(plans[:10])}

RECENT MAINTENANCE LOGS:
{self._format_table(recent_logs[:10])}
"""
        if sql_result and sql_result.get("success"):
            data_context += f"\nSPECIFIC QUERY RESULTS:\nSQL: {sql_result.get('generated_sql')}\n{self._format_table(sql_result.get('results', [])[:15])}\n"

        if rag_context:
            data_context += f"\nSERVICE DOCUMENTATION:\n{rag_context}\n"

        prompt = self.build_prompt(query=query, context=context, additional_context=data_context)

        system_prompt = self._system_prompt + """
When answering service and maintenance questions:
1. Reference ticket codes (SRV-XXXX-XXX) and machine codes (IB-XXXX)
2. Prioritize critical and breakdown tickets in responses
3. Show warranty status when discussing installed machines
4. Highlight overdue maintenance plans
5. For error codes, suggest initial troubleshooting steps from documentation
6. Track field service assignments and response times
7. Note parts used in repairs (for spare parts correlation)
8. Distinguish between preventive, corrective, predictive, and emergency maintenance
9. For installed base queries, include serial numbers and installation dates
10. If focused tickets or machines are present, answer from those rows first
11. Do not ask for customer/machine information when it is already in focused records
12. Respond in the same language as the user"""

        response = await self.call_llm(prompt, system_prompt)

        return self.format_response(
            content=response,
            data={
                "open_tickets": open_tickets,
                "focused_tickets": focused_tickets,
                "focused_machines": focused_machines,
                "focused_plans": focused_plans,
                "installed_base": installed_base[:10],
                "ticket_summary": ticket_summary,
                "sql_results": sql_result.get("results") if sql_result else None,
            },
            sources=sources,
            confidence=0.85 if open_tickets or installed_base else 0.6,
        )

    def _format_table(self, data: List[Dict]) -> str:
        if not data:
            return "No data available"
        headers = list(data[0].keys())
        rows = [" | ".join(str(item.get(h, ""))[:30] for h in headers) for item in data]
        header_row = " | ".join(h[:30] for h in headers)
        return f"{header_row}\n{'-' * len(header_row)}\n" + "\n".join(rows)
