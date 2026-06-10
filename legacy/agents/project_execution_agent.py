"""
Project Execution Agent for Coiltech Industrial Automation Platform.

Handles project tracking, milestone monitoring, delivery schedules,
and project machine status for ongoing Coiltech installations.
"""

from typing import Optional, Dict, Any, List
import structlog
from agents.base_agent import BaseAgent, AgentResponse, AgentContext

logger = structlog.get_logger(__name__)


class ProjectExecutionAgent(BaseAgent):
    """Handles project status, milestones, and delivery tracking."""

    def __init__(self, **kwargs):
        super().__init__(
            name="ProjectExecutionAgent",
            department="project_execution",
            description=(
                "Tracks active Coiltech projects from planning through commissioning. "
                "Monitors milestones, delivery schedules, machine build status, "
                "and project-level issues."
            ),
            **kwargs,
        )

        self._queries = {
            "projects": """
                SELECT p.code, p.title, p.status, p.project_manager,
                       p.start_date, p.target_completion, p.contract_value, p.currency,
                       c.name AS customer_name
                FROM projects p
                JOIN customers c ON p.customer_id = c.id
                ORDER BY
                    CASE p.status
                        WHEN 'manufacturing' THEN 0
                        WHEN 'engineering' THEN 1
                        WHEN 'testing' THEN 2
                        WHEN 'shipping' THEN 3
                        WHEN 'installation' THEN 4
                        WHEN 'commissioning' THEN 5
                        WHEN 'planning' THEN 6
                        ELSE 7
                    END,
                    p.target_completion
            """,
            "milestones": """
                SELECT pm.name, pm.planned_date, pm.actual_date, pm.status, pm.notes,
                       p.code AS project_code, p.title AS project_title
                FROM project_milestones pm
                JOIN projects p ON pm.project_id = p.id
                WHERE p.status NOT IN ('completed','cancelled')
                ORDER BY pm.planned_date
            """,
            "project_machines": """
                SELECT pmach.serial_number, pmach.description, pmach.quantity,
                       pmach.status, pmod.model_code,
                       p.code AS project_code
                FROM project_machines pmach
                JOIN projects p ON pmach.project_id = p.id
                LEFT JOIN product_models pmod ON pmach.model_id = pmod.id
                WHERE p.status NOT IN ('completed','cancelled')
                ORDER BY p.code, pmach.id
            """,
            "project_status_view": "SELECT * FROM v_project_status",
        }

    async def handle(self, query: str, context: Optional[AgentContext] = None) -> AgentResponse:
        logger.info("ProjectExecutionAgent handling query", query=query[:60])

        projects = await self.query_database(self._queries["projects"])
        milestones = await self.query_database(self._queries["milestones"])
        machines = await self.query_database(self._queries["project_machines"])
        status_view = await self.query_database(self._queries["project_status_view"])
        focused_projects = self._focus_records_by_query(
            query,
            projects,
            ["code", "title", "status", "project_manager", "customer_name"],
            max_items=6,
        )
        focused_milestones = self._focus_records_by_query(
            query,
            milestones,
            ["project_code", "project_title", "name", "status", "notes"],
            max_items=10,
        )
        focused_machines = self._focus_records_by_query(
            query,
            machines,
            ["project_code", "model_code", "serial_number", "description", "status"],
            max_items=10,
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
FOCUSED PROJECTS FOR THIS QUERY:
{self._format_table(focused_projects)}

FOCUSED MILESTONES FOR THIS QUERY:
{self._format_table(focused_milestones)}

FOCUSED PROJECT MACHINES FOR THIS QUERY:
{self._format_table(focused_machines)}

ACTIVE PROJECTS:
{self._format_table(projects[:15])}

UPCOMING MILESTONES:
{self._format_table(milestones[:15])}

PROJECT MACHINES:
{self._format_table(machines[:15])}

PROJECT STATUS OVERVIEW:
{self._format_table(status_view)}
"""
        if sql_result and sql_result.get("success"):
            data_context += f"\nSPECIFIC QUERY RESULTS:\nSQL: {sql_result.get('generated_sql')}\n{self._format_table(sql_result.get('results', [])[:15])}\n"

        if rag_context:
            data_context += f"\nPROJECT DOCUMENTATION:\n{rag_context}\n"

        prompt = self.build_prompt(query=query, context=context, additional_context=data_context)

        system_prompt = self._system_prompt + """
When answering project status questions:
1. Reference project codes (PRJ-XXXX-XXX) and current status
2. Highlight delayed milestones and their impact on delivery
3. Show completion percentages (completed milestones / total milestones)
4. Track machine build status within each project
5. Note target vs actual completion dates
6. Flag projects at risk of delay
7. Include project manager name and customer for context
8. Suggest actions for overdue milestones or stalled projects
9. If a project code/customer appears in the query, answer from focused project/milestone/machine records first
10. Do not ask for a project code if a focused project is already present
11. Respond in the same language as the user"""

        response = await self.call_llm(prompt, system_prompt)

        return self.format_response(
            content=response,
            data={
                "projects": projects,
                "focused_projects": focused_projects,
                "focused_milestones": focused_milestones,
                "focused_machines": focused_machines,
                "milestones": milestones[:10],
                "status_view": status_view,
                "sql_results": sql_result.get("results") if sql_result else None,
            },
            sources=sources,
            confidence=0.85 if projects else 0.6,
        )

    def _format_table(self, data: List[Dict]) -> str:
        if not data:
            return "No data available"
        headers = list(data[0].keys())
        rows = [" | ".join(str(item.get(h, ""))[:30] for h in headers) for item in data]
        header_row = " | ".join(h[:30] for h in headers)
        return f"{header_row}\n{'-' * len(header_row)}\n" + "\n".join(rows)
