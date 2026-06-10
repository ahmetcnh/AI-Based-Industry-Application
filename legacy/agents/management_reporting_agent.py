"""
Management Reporting Agent for Coiltech Industrial Automation Platform.

Handles executive summaries, KPI aggregation, pipeline reporting,
cross-domain dashboards, and management-level analytics.
"""

from typing import Optional, Dict, Any, List
import structlog
from agents.base_agent import BaseAgent, AgentResponse, AgentContext

logger = structlog.get_logger(__name__)


class ManagementReportingAgent(BaseAgent):
    """Handles executive summaries, KPIs, and cross-domain reporting."""

    def __init__(self, **kwargs):
        super().__init__(
            name="ManagementReportingAgent",
            department="management_reporting",
            description=(
                "Generates executive summaries, KPI dashboards, pipeline reports, "
                "and cross-domain analytics for Coiltech management. Aggregates "
                "data from sales, projects, service, and quality domains."
            ),
            **kwargs,
        )

        self._queries = {
            "pipeline_summary": "SELECT * FROM v_pipeline_summary",
            "project_overview": """
                SELECT status, COUNT(*) AS count,
                       SUM(contract_value) AS total_value,
                       AVG(contract_value) AS avg_value
                FROM projects
                GROUP BY status
                ORDER BY count DESC
            """,
            "service_summary": """
                SELECT type, priority, status, COUNT(*) AS count
                FROM service_tickets
                GROUP BY type, priority, status
                ORDER BY
                    CASE priority
                        WHEN 'critical' THEN 0
                        WHEN 'high' THEN 1
                        WHEN 'medium' THEN 2
                        ELSE 3
                    END
            """,
            "installed_base_stats": """
                SELECT status, COUNT(*) AS machine_count
                FROM installed_base
                GROUP BY status
            """,
            "quality_summary": """
                SELECT event_type, severity, status, COUNT(*) AS count
                FROM quality_events
                GROUP BY event_type, severity, status
            """,
            "customer_stats": """
                SELECT c.tier, COUNT(DISTINCT c.id) AS customer_count,
                       COUNT(DISTINCT p.id) AS project_count,
                       SUM(p.contract_value) AS total_project_value
                FROM customers c
                LEFT JOIN projects p ON c.id = p.customer_id
                GROUP BY c.tier
                ORDER BY total_project_value DESC
            """,
            "quotation_stats": """
                SELECT q.status, COUNT(*) AS count,
                       SUM(q.total_amount) AS total_value,
                       AVG(q.total_amount) AS avg_value
                FROM quotations q
                GROUP BY q.status
            """,
            "active_alerts_count": """
                SELECT severity, COUNT(*) AS count
                FROM machine_alerts
                WHERE resolved = 0
                GROUP BY severity
            """,
            "spare_parts_risk": """
                SELECT COUNT(*) AS low_stock_count
                FROM v_spare_parts_low_stock
            """,
        }

    async def handle(self, query: str, context: Optional[AgentContext] = None) -> AgentResponse:
        logger.info("ManagementReportingAgent handling query", query=query[:60])

        pipeline = await self.query_database(self._queries["pipeline_summary"])
        projects = await self.query_database(self._queries["project_overview"])
        service = await self.query_database(self._queries["service_summary"])
        ib_stats = await self.query_database(self._queries["installed_base_stats"])
        quality = await self.query_database(self._queries["quality_summary"])
        customers = await self.query_database(self._queries["customer_stats"])
        quotations = await self.query_database(self._queries["quotation_stats"])
        alerts = await self.query_database(self._queries["active_alerts_count"])
        spare_risk = await self.query_database(self._queries["spare_parts_risk"])
        risk_snapshot = self._build_risk_snapshot(projects, service, quality, alerts, spare_risk)

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
=== EXECUTIVE DASHBOARD ===

MANAGEMENT RISK SNAPSHOT:
{risk_snapshot}

SALES PIPELINE:
{self._format_table(pipeline)}

QUOTATION STATUS:
{self._format_table(quotations)}

PROJECT OVERVIEW:
{self._format_table(projects)}

CUSTOMER BREAKDOWN:
{self._format_table(customers)}

INSTALLED BASE:
{self._format_table(ib_stats)}

SERVICE TICKETS:
{self._format_table(service[:15])}

ACTIVE ALERTS:
{self._format_table(alerts)}

QUALITY EVENTS:
{self._format_table(quality[:15])}

SPARE PARTS AT RISK:
{self._format_table(spare_risk)}
"""
        if sql_result and sql_result.get("success"):
            data_context += f"\nSPECIFIC QUERY RESULTS:\nSQL: {sql_result.get('generated_sql')}\n{self._format_table(sql_result.get('results', [])[:15])}\n"

        if rag_context:
            data_context += f"\nADDITIONAL DOCUMENTATION:\n{rag_context}\n"

        prompt = self.build_prompt(query=query, context=context, additional_context=data_context)

        system_prompt = self._system_prompt + """
When generating executive summaries and management reports:
1. Lead with key metrics: pipeline value, active projects, open tickets
2. Highlight risks: critical alerts, overdue milestones, low stock parts
3. Show trends when data supports it (growing/declining pipeline, etc.)
4. Structure reports with clear sections: Sales, Projects, Service, Quality
5. Use concrete numbers — avoid vague language
6. Flag items needing management attention or decision
7. Compare current state to targets when available
8. Provide actionable recommendations based on the data
9. Keep executive summaries concise — details available on request
10. Use MANAGEMENT RISK SNAPSHOT as the opening frame when the user asks for an executive summary
11. Respond in the same language as the user"""

        response = await self.call_llm(prompt, system_prompt)

        return self.format_response(
            content=response,
            data={
                "pipeline": pipeline,
                "projects": projects,
                "service": service,
                "quotations": quotations,
                "installed_base": ib_stats,
                "quality": quality,
                "alerts": alerts,
                "risk_snapshot": risk_snapshot,
                "sql_results": sql_result.get("results") if sql_result else None,
            },
            sources=sources,
            confidence=0.9 if pipeline else 0.65,
        )

    def _build_risk_snapshot(
        self,
        projects: List[Dict],
        service: List[Dict],
        quality: List[Dict],
        alerts: List[Dict],
        spare_risk: List[Dict],
    ) -> str:
        active_project_groups = sum(item.get("count", 0) or 0 for item in projects)
        critical_service = sum(
            item.get("count", 0) or 0
            for item in service
            if str(item.get("priority", "")).lower() == "critical"
            and str(item.get("status", "")).lower() not in {"closed", "cancelled"}
        )
        critical_quality = sum(
            item.get("count", 0) or 0
            for item in quality
            if str(item.get("severity", "")).lower() == "critical"
            and str(item.get("status", "")).lower() != "closed"
        )
        critical_alerts = sum(
            item.get("count", 0) or 0
            for item in alerts
            if str(item.get("severity", "")).lower() == "critical"
        )
        low_stock = spare_risk[0].get("low_stock_count", 0) if spare_risk else 0

        return (
            f"Active project status groups total: {active_project_groups}\n"
            f"Critical open service tickets: {critical_service}\n"
            f"Critical open quality events: {critical_quality}\n"
            f"Critical active machine alerts: {critical_alerts}\n"
            f"Low-stock spare part count: {low_stock}"
        )

    def _format_table(self, data: List[Dict]) -> str:
        if not data:
            return "No data available"
        headers = list(data[0].keys())
        rows = [" | ".join(str(item.get(h, ""))[:30] for h in headers) for item in data]
        header_row = " | ".join(h[:30] for h in headers)
        return f"{header_row}\n{'-' * len(header_row)}\n" + "\n".join(rows)
