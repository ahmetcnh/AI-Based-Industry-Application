# TODO(legacy): Orphaned agent from original industrial-platform design.
# Not registered in agents/__init__.py or orchestrator.py — never invoked.
# The active replacement is ManagementReportingAgent (agents/management_reporting_agent.py).
# Decision pending: remove or integrate.
"""
Management Agent Module for AI Industrial Platform.

Handles cross-department queries, executive summaries, KPI aggregation,
and management reporting.
"""

from typing import Optional, Dict, Any, List
import structlog

from agents.base_agent import BaseAgent, AgentResponse, AgentContext

logger = structlog.get_logger(__name__)


class ManagementAgent(BaseAgent):
    """
    Specialized agent for management and executive operations.
    
    Handles queries about:
    - Cross-department summaries
    - Executive dashboards
    - KPI aggregation
    - Performance reports
    - Strategic insights
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="ManagementAgent",
            department="management",
            description="Aggregates data across departments, generates executive summaries, "
                       "tracks KPIs, and provides strategic insights for management decisions.",
            **kwargs
        )
        
        self._queries = {
            "production_kpis": """
                SELECT 
                    COUNT(DISTINCT line_id) as active_lines,
                    SUM(output_qty) as total_output,
                    SUM(target_qty) as total_target,
                    ROUND((SUM(output_qty) * 100.0) / NULLIF(SUM(target_qty), 0), 2) as efficiency_pct,
                    ROUND(AVG(defect_rate), 2) as avg_defect_rate,
                    SUM(downtime_min) as total_downtime_min
                FROM production_logs
                WHERE DATE(timestamp) >= DATE('now', '-7 days')
            """,
            "maintenance_kpis": """
                SELECT 
                    COUNT(*) as total_maintenance,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'scheduled' THEN 1 ELSE 0 END) as scheduled,
                    SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as in_progress,
                    ROUND(SUM(cost), 2) as total_cost,
                    ROUND(AVG(duration_hours), 1) as avg_duration
                FROM maintenance_records
                WHERE date >= DATE('now', '-30 days')
            """,
            "quality_kpis": """
                SELECT 
                    COUNT(*) as batches_inspected,
                    SUM(total_inspected) as units_inspected,
                    SUM(total_passed) as units_passed,
                    ROUND((SUM(total_passed) * 100.0) / NULLIF(SUM(total_inspected), 0), 2) as overall_pass_rate,
                    SUM(CASE WHEN disposition = 'rejected' THEN 1 ELSE 0 END) as rejected_batches,
                    SUM(CASE WHEN disposition = 'rework' THEN 1 ELSE 0 END) as rework_batches
                FROM quality_reports
                WHERE inspection_date >= DATE('now', '-7 days')
            """,
            "inventory_kpis": """
                SELECT 
                    COUNT(*) as total_items,
                    SUM(quantity * unit_cost) as inventory_value,
                    SUM(CASE WHEN quantity < min_stock_level THEN 1 ELSE 0 END) as critical_items,
                    SUM(CASE WHEN quantity <= min_stock_level * 1.2 THEN 1 ELSE 0 END) as low_stock_items
                FROM inventory
            """,
            "sensor_alerts": """
                SELECT 
                    COUNT(*) as total_alerts,
                    COUNT(DISTINCT equipment_id) as equipment_with_alerts
                FROM sensor_data
                WHERE alert_flag = 1
                AND timestamp >= DATETIME('now', '-24 hours')
            """,
            "equipment_status": """
                SELECT 
                    status,
                    COUNT(*) as count
                FROM equipment
                GROUP BY status
            """
        }
    
    async def _get_all_kpis(self) -> Dict[str, Any]:
        """Fetch KPIs from all departments."""
        try:
            production = await self.query_database(self._queries["production_kpis"])
            maintenance = await self.query_database(self._queries["maintenance_kpis"])
            quality = await self.query_database(self._queries["quality_kpis"])
            inventory = await self.query_database(self._queries["inventory_kpis"])
            alerts = await self.query_database(self._queries["sensor_alerts"])
            equipment = await self.query_database(self._queries["equipment_status"])
            
            return {
                "production": production[0] if production else {},
                "maintenance": maintenance[0] if maintenance else {},
                "quality": quality[0] if quality else {},
                "inventory": inventory[0] if inventory else {},
                "alerts": alerts[0] if alerts else {},
                "equipment_status": equipment
            }
        except Exception as e:
            logger.error("Failed to fetch KPIs", error=str(e))
            return {}
    
    def _generate_executive_summary(self, kpis: Dict[str, Any]) -> str:
        """
        Generate an executive summary from KPIs.
        
        Args:
            kpis: Dictionary of KPI data.
        
        Returns:
            Formatted executive summary.
        """
        prod = kpis.get("production", {}) or {}
        maint = kpis.get("maintenance", {}) or {}
        qual = kpis.get("quality", {}) or {}
        inv = kpis.get("inventory", {}) or {}
        alerts = kpis.get("alerts", {}) or {}
        
        summary_parts = ["EXECUTIVE SUMMARY (Last 7 Days)"]
        summary_parts.append("=" * 50)
        
        # Production - handle None values safely
        efficiency = prod.get("efficiency_pct") or 0
        total_output = prod.get("total_output") or 0
        avg_defect_rate = prod.get("avg_defect_rate") or 0
        total_downtime = prod.get("total_downtime_min") or 0
        
        summary_parts.append(f"\n📊 PRODUCTION")
        summary_parts.append(f"   Output: {total_output:,} units")
        summary_parts.append(f"   Efficiency: {efficiency:.1f}%")
        summary_parts.append(f"   Defect Rate: {avg_defect_rate:.1f}%")
        summary_parts.append(f"   Downtime: {total_downtime:,} minutes")
        
        # Quality - handle None values safely
        pass_rate = qual.get("overall_pass_rate") or 0
        batches_inspected = qual.get("batches_inspected") or 0
        rejected_batches = qual.get("rejected_batches") or 0
        
        summary_parts.append(f"\n✅ QUALITY")
        summary_parts.append(f"   Batches Inspected: {batches_inspected}")
        summary_parts.append(f"   Overall Pass Rate: {pass_rate:.1f}%")
        summary_parts.append(f"   Rejected Batches: {rejected_batches}")
        
        # Maintenance - handle None values safely
        completed = maint.get("completed") or 0
        scheduled = maint.get("scheduled") or 0
        total_cost = maint.get("total_cost") or 0
        
        summary_parts.append(f"\n🔧 MAINTENANCE")
        summary_parts.append(f"   Completed: {completed}")
        summary_parts.append(f"   Scheduled: {scheduled}")
        summary_parts.append(f"   Cost: ${total_cost:,.2f}")
        
        # Inventory - handle None values safely
        inv_value = inv.get("inventory_value") or 0
        critical_items = inv.get("critical_items") or 0
        low_stock_items = inv.get("low_stock_items") or 0
        
        summary_parts.append(f"\n📦 INVENTORY")
        summary_parts.append(f"   Total Value: ${inv_value:,.2f}")
        summary_parts.append(f"   Critical Items: {critical_items}")
        summary_parts.append(f"   Low Stock Items: {low_stock_items}")
        
        # Alerts - handle None values safely
        alert_count = alerts.get("total_alerts") or 0
        equipment_with_alerts = alerts.get("equipment_with_alerts") or 0
        
        if alert_count > 0:
            summary_parts.append(f"\n⚠️ ALERTS")
            summary_parts.append(f"   Active Alerts: {alert_count}")
            summary_parts.append(f"   Equipment Affected: {equipment_with_alerts}")
        
        # Overall Assessment
        summary_parts.append("\n" + "=" * 50)
        summary_parts.append("OVERALL ASSESSMENT:")
        
        issues = []
        if efficiency < 85:
            issues.append("Production efficiency below target")
        if pass_rate < 95:
            issues.append("Quality pass rate needs improvement")
        if critical_items > 0:
            issues.append(f"{critical_items} inventory items critical")
        if alert_count > 0:
            issues.append(f"{alert_count} sensor alerts require attention")
        
        if issues:
            summary_parts.append("⚠️ Issues requiring attention:")
            for issue in issues:
                summary_parts.append(f"   - {issue}")
        else:
            summary_parts.append("✅ All systems operating within normal parameters")
        
        return "\n".join(summary_parts)
    
    def _calculate_trends(self, kpis: Dict[str, Any]) -> Dict[str, str]:
        """Calculate trend indicators."""
        # In a real system, this would compare to previous periods
        trends = {
            "production": "stable",
            "quality": "stable",
            "maintenance": "stable",
            "inventory": "stable"
        }
        
        prod = kpis.get("production", {})
        if prod.get("efficiency_pct", 0) and prod.get("efficiency_pct", 0) >= 90:
            trends["production"] = "improving"
        elif prod.get("efficiency_pct", 0) and prod.get("efficiency_pct", 0) < 80:
            trends["production"] = "declining"
        
        return trends
    
    async def handle(
        self,
        query: str,
        context: Optional[AgentContext] = None
    ) -> AgentResponse:
        """
        Handle management and executive queries.
        
        Args:
            query: User's query.
            context: Agent context.
        
        Returns:
            AgentResponse with management insights.
        """
        logger.info("ManagementAgent handling query", query=query[:50])
        
        # Fetch all KPIs
        kpis = await self._get_all_kpis()
        
        # Generate executive summary
        exec_summary = self._generate_executive_summary(kpis)
        
        # Calculate trends
        trends = self._calculate_trends(kpis)
        
        # Try RAG for strategic documents
        rag_context = ""
        sources = []
        try:
            rag_response = await self.query_rag(query, "management", k=3)
            if rag_response.get("sources"):
                sources = rag_response["sources"]
                rag_context = f"\nRELEVANT DOCUMENTS:\n{rag_response.get('answer', '')}"
        except Exception as e:
            logger.debug("RAG query failed", error=str(e))
        
        # Try NL-to-SQL for specific queries
        sql_result = None
        if self.should_attempt_nl_to_sql(query):
            try:
                sql_result = await self.nl_to_sql(query)
            except Exception as e:
                logger.debug("NL-to-SQL failed", error=str(e))
        
        # Build context
        data_context = f"""
{exec_summary}

DEPARTMENT TRENDS:
- Production: {trends.get('production', 'unknown')}
- Quality: {trends.get('quality', 'unknown')}
- Maintenance: {trends.get('maintenance', 'unknown')}
- Inventory: {trends.get('inventory', 'unknown')}

EQUIPMENT STATUS:
{self._format_table(kpis.get('equipment_status', []))}
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

When providing management insights:
1. Lead with the most important findings
2. Quantify impacts where possible
3. Provide actionable recommendations
4. Consider cross-department implications
5. Highlight trends and their business impact
6. Be concise but comprehensive"""
        
        response_text = await self.call_llm(prompt, system_prompt)
        
        confidence = 0.85
        if not kpis:
            confidence -= 0.3
        
        return self.format_response(
            content=response_text,
            data={
                "kpis": {
                    "production": kpis.get("production", {}),
                    "quality": kpis.get("quality", {}),
                    "maintenance": kpis.get("maintenance", {}),
                    "inventory": kpis.get("inventory", {})
                },
                "trends": trends,
                "alerts_count": kpis.get("alerts", {}).get("total_alerts", 0),
                "executive_summary": exec_summary
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
            row = " | ".join(str(item.get(h, ""))[:20] for h in headers)
            rows.append(row)
        
        header_row = " | ".join(h[:20] for h in headers)
        return f"{header_row}\n{'-' * len(header_row)}\n" + "\n".join(rows)


if __name__ == "__main__":
    import asyncio
    
    async def test_management_agent():
        print("Testing ManagementAgent...")
        
        agent = ManagementAgent()
        
        test_queries = [
            "Give me an executive summary of this week's performance",
            "What are the key KPIs across all departments?",
            "Are there any issues that need immediate attention?",
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            response = await agent.handle(
                query=query,
                context=AgentContext(user_query=query, department="management")
            )
            print(f"Response: {response.content[:500]}...")
            print(f"Confidence: {response.confidence}")
        
        print("\nManagementAgent test completed!")
    
    asyncio.run(test_management_agent())
