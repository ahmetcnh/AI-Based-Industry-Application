# TODO(legacy): Orphaned agent from original industrial-platform design.
# Not registered in agents/__init__.py or orchestrator.py — never invoked.
# The active replacement is ServiceMaintenanceAgent (agents/service_maintenance_agent.py).
# Decision pending: remove or integrate.
"""
Maintenance Agent Module for AI Industrial Platform.

Handles maintenance-related queries including equipment maintenance records,
preventive maintenance schedules, and predictive maintenance hints.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import structlog

from agents.base_agent import BaseAgent, AgentResponse, AgentContext

logger = structlog.get_logger(__name__)


class MaintenanceAgent(BaseAgent):
    """
    Specialized agent for maintenance operations.
    
    Handles queries about:
    - Equipment maintenance records
    - Preventive maintenance schedules
    - Repair history and costs
    - Predictive maintenance recommendations
    - Technician assignments
    - Parts and inventory for maintenance
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="MaintenanceAgent",
            department="maintenance",
            description="Manages maintenance records, schedules preventive maintenance, "
                       "analyzes equipment history, and provides predictive maintenance insights.",
            **kwargs
        )
        
        self._queries = {
            "upcoming_maintenance": """
                SELECT 
                    equipment_id,
                    equipment_name,
                    type,
                    next_service_date,
                    priority,
                    description
                FROM maintenance_records
                WHERE status = 'scheduled'
                AND next_service_date >= DATE('now')
                ORDER BY next_service_date ASC
                LIMIT 15
            """,
            "maintenance_history": """
                SELECT 
                    equipment_id,
                    equipment_name,
                    date,
                    type,
                    description,
                    technician,
                    cost,
                    duration_hours,
                    status
                FROM maintenance_records
                WHERE status = 'completed'
                ORDER BY date DESC
                LIMIT 20
            """,
            "equipment_summary": """
                SELECT 
                    equipment_id,
                    equipment_name,
                    COUNT(*) as maintenance_count,
                    SUM(cost) as total_cost,
                    SUM(duration_hours) as total_hours,
                    MAX(date) as last_maintenance,
                    MIN(next_service_date) as next_due
                FROM maintenance_records
                GROUP BY equipment_id, equipment_name
                ORDER BY total_cost DESC
            """,
            "critical_equipment": """
                SELECT 
                    e.equipment_id,
                    e.name,
                    e.type,
                    e.status,
                    e.criticality,
                    e.location,
                    COUNT(m.id) as maintenance_count,
                    SUM(m.cost) as total_maintenance_cost
                FROM equipment e
                LEFT JOIN maintenance_records m ON e.equipment_id = m.equipment_id
                WHERE e.criticality IN ('high', 'critical')
                GROUP BY e.equipment_id
                ORDER BY e.criticality DESC, maintenance_count DESC
            """,
            "overdue_maintenance": """
                SELECT 
                    equipment_id,
                    equipment_name,
                    type,
                    next_service_date,
                    priority,
                    JULIANDAY('now') - JULIANDAY(next_service_date) as days_overdue
                FROM maintenance_records
                WHERE status = 'scheduled'
                AND next_service_date < DATE('now')
                ORDER BY days_overdue DESC
            """
        }
    
    async def _get_maintenance_data(self) -> Dict[str, Any]:
        """Fetch current maintenance data from database."""
        try:
            upcoming = await self.query_database(self._queries["upcoming_maintenance"])
            history = await self.query_database(self._queries["maintenance_history"])
            equipment = await self.query_database(self._queries["equipment_summary"])
            critical = await self.query_database(self._queries["critical_equipment"])
            overdue = await self.query_database(self._queries["overdue_maintenance"])
            
            return {
                "upcoming_maintenance": upcoming,
                "maintenance_history": history,
                "equipment_summary": equipment,
                "critical_equipment": critical,
                "overdue_maintenance": overdue
            }
        except Exception as e:
            logger.error("Failed to fetch maintenance data", error=str(e))
            return {}
    
    def _predict_maintenance(self, equipment_data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Generate predictive maintenance hints based on history.
        
        Args:
            equipment_data: Equipment maintenance history.
        
        Returns:
            List of predictive maintenance recommendations.
        """
        predictions = []
        
        for equip in equipment_data:
            maintenance_count = equip.get("maintenance_count", 0)
            total_cost = equip.get("total_cost", 0) or 0
            
            # Simple prediction logic based on maintenance frequency
            if maintenance_count >= 3:
                avg_cost = total_cost / maintenance_count
                predictions.append({
                    "equipment_id": equip.get("equipment_id"),
                    "equipment_name": equip.get("equipment_name"),
                    "prediction": "Consider reviewing maintenance intervals",
                    "reason": f"High maintenance frequency ({maintenance_count} records)",
                    "estimated_next_cost": round(avg_cost, 2),
                    "risk_level": "medium" if maintenance_count < 5 else "high"
                })
            
            # Cost-based prediction
            if total_cost > 5000:
                predictions.append({
                    "equipment_id": equip.get("equipment_id"),
                    "equipment_name": equip.get("equipment_name"),
                    "prediction": "Evaluate equipment replacement",
                    "reason": f"High cumulative maintenance cost (${total_cost:.2f})",
                    "risk_level": "high"
                })
        
        return predictions
    
    async def handle(
        self,
        query: str,
        context: Optional[AgentContext] = None
    ) -> AgentResponse:
        """
        Handle maintenance-related queries.
        
        Args:
            query: User's query.
            context: Agent context.
        
        Returns:
            AgentResponse with maintenance insights.
        """
        logger.info("MaintenanceAgent handling query", query=query[:50])
        
        # Fetch maintenance data
        maint_data = await self._get_maintenance_data()
        
        # Try RAG for document-based queries
        rag_context = ""
        sources = []
        try:
            rag_response = await self.query_rag(query, "maintenance", k=4)
            if rag_response.get("sources"):
                sources = rag_response["sources"]
                rag_context = f"\nMAINTENANCE DOCUMENTS:\n{rag_response.get('answer', '')}"
        except Exception as e:
            logger.debug("RAG query failed", error=str(e))
        
        # Try NL-to-SQL for specific queries
        sql_result = None
        if self.should_attempt_nl_to_sql(query):
            try:
                sql_result = await self.nl_to_sql(query)
            except Exception as e:
                logger.debug("NL-to-SQL failed", error=str(e))
        
        # Generate predictions
        predictions = []
        if maint_data.get("equipment_summary"):
            predictions = self._predict_maintenance(maint_data["equipment_summary"])
        
        # Build context
        data_context = f"""
MAINTENANCE DATABASE SUMMARY:

UPCOMING SCHEDULED MAINTENANCE:
{self._format_table(maint_data.get('upcoming_maintenance', [])[:10])}

OVERDUE MAINTENANCE:
{self._format_table(maint_data.get('overdue_maintenance', []))}

RECENT MAINTENANCE HISTORY:
{self._format_table(maint_data.get('maintenance_history', [])[:8])}

CRITICAL EQUIPMENT:
{self._format_table(maint_data.get('critical_equipment', [])[:5])}

EQUIPMENT MAINTENANCE SUMMARY:
{self._format_table(maint_data.get('equipment_summary', [])[:8])}

PREDICTIVE MAINTENANCE INSIGHTS:
{self._format_predictions(predictions)}
"""
        
        if sql_result and sql_result.get("success"):
            data_context += f"""
SPECIFIC QUERY RESULTS:
SQL: {sql_result.get('generated_sql', 'N/A')}
Results: {self._format_table(sql_result.get('results', [])[:10])}
"""
        
        # Build prompt
        prompt = self.build_prompt(
            query=query,
            context=context,
            additional_context=data_context + rag_context
        )
        
        # Generate response
        system_prompt = self._system_prompt + """

When answering maintenance questions:
1. Prioritize safety-related maintenance
2. Highlight any overdue or critical maintenance
3. Provide predictive insights when relevant
4. Reference maintenance documentation when available
5. Suggest preventive actions to avoid breakdowns"""
        
        response_text = await self.call_llm(prompt, system_prompt)
        
        confidence = 0.85
        if not maint_data:
            confidence -= 0.2
        if not sources:
            confidence -= 0.1
        
        return self.format_response(
            content=response_text,
            data={
                "upcoming_maintenance": maint_data.get("upcoming_maintenance", [])[:5],
                "overdue_count": len(maint_data.get("overdue_maintenance", [])),
                "predictions": predictions[:3],
                "sql_results": sql_result.get("results") if sql_result else None
            },
            sources=sources,
            confidence=max(confidence, 0.5)
        )
    
    def _format_table(self, data: List[Dict]) -> str:
        """Format data as readable table."""
        if not data:
            return "No data available"
        
        headers = list(data[0].keys())
        rows = []
        for item in data:
            row = " | ".join(str(item.get(h, ""))[:25] for h in headers)
            rows.append(row)
        
        header_row = " | ".join(h[:25] for h in headers)
        return f"{header_row}\n{'-' * len(header_row)}\n" + "\n".join(rows)
    
    def _format_predictions(self, predictions: List[Dict]) -> str:
        """Format predictions for display."""
        if not predictions:
            return "No predictions available"
        
        lines = []
        for pred in predictions[:5]:
            lines.append(
                f"- {pred.get('equipment_name', 'Unknown')}: "
                f"{pred.get('prediction')} ({pred.get('risk_level', 'unknown')} risk)"
            )
        return "\n".join(lines)


if __name__ == "__main__":
    import asyncio
    
    async def test_maintenance_agent():
        print("Testing MaintenanceAgent...")
        
        agent = MaintenanceAgent()
        
        test_queries = [
            "When is the next scheduled maintenance for the CNC machine?",
            "Show me equipment that is overdue for service",
            "What are the maintenance costs for the main assembly robot?",
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            response = await agent.handle(
                query=query,
                context=AgentContext(user_query=query, department="maintenance")
            )
            print(f"Response: {response.content[:300]}...")
            print(f"Confidence: {response.confidence}")
        
        print("\nMaintenanceAgent test completed!")
    
    asyncio.run(test_maintenance_agent())
