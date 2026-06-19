# TODO(legacy): Orphaned agent from original industrial-platform design.
# Not registered in agents/__init__.py or orchestrator.py — never invoked.
# Decision pending: remove or integrate.
"""
Production Agent Module for AI Industrial Platform.

Handles production-related queries including output tracking, efficiency analysis,
OEE metrics, and production line performance.
"""

from typing import Optional, Dict, Any, List
import structlog

from agents.base_agent import BaseAgent, AgentResponse, AgentContext

logger = structlog.get_logger(__name__)


class ProductionAgent(BaseAgent):
    """
    Specialized agent for production operations.
    
    Handles queries about:
    - Production output and targets
    - Efficiency and OEE metrics
    - Defect rates and quality issues
    - Downtime analysis
    - Shift performance
    - Product types and lines
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="ProductionAgent",
            department="production",
            description="Analyzes production data, tracks output metrics, "
                       "calculates OEE, and provides insights on manufacturing performance.",
            **kwargs
        )
        
        # SQL queries for common production metrics
        self._queries = {
            "daily_summary": """
                SELECT 
                    line_id,
                    DATE(timestamp) as date,
                    SUM(output_qty) as total_output,
                    SUM(target_qty) as total_target,
                    ROUND(AVG(defect_rate), 2) as avg_defect_rate,
                    SUM(downtime_min) as total_downtime,
                    ROUND((SUM(output_qty) * 100.0) / NULLIF(SUM(target_qty), 0), 2) as efficiency
                FROM production_logs
                WHERE DATE(timestamp) >= DATE('now', '-7 days')
                GROUP BY line_id, DATE(timestamp)
                ORDER BY date DESC, line_id
            """,
            "line_performance": """
                SELECT 
                    line_id,
                    COUNT(*) as shifts,
                    SUM(output_qty) as total_output,
                    SUM(target_qty) as total_target,
                    ROUND(AVG(defect_rate), 2) as avg_defect_rate,
                    SUM(downtime_min) as total_downtime,
                    ROUND((SUM(output_qty) * 100.0) / NULLIF(SUM(target_qty), 0), 2) as efficiency
                FROM production_logs
                GROUP BY line_id
                ORDER BY efficiency DESC
            """,
            "shift_analysis": """
                SELECT 
                    shift,
                    COUNT(*) as shift_count,
                    ROUND(AVG(output_qty), 0) as avg_output,
                    ROUND(AVG(defect_rate), 2) as avg_defect_rate,
                    ROUND(AVG(downtime_min), 0) as avg_downtime
                FROM production_logs
                GROUP BY shift
                ORDER BY shift
            """,
            "recent_issues": """
                SELECT 
                    timestamp,
                    line_id,
                    product_type,
                    defect_rate,
                    downtime_min,
                    notes
                FROM production_logs
                WHERE defect_rate > 3.0 OR downtime_min > 30
                ORDER BY timestamp DESC
                LIMIT 10
            """
        }
    
    def _calculate_oee(
        self,
        availability: float,
        performance: float,
        quality: float
    ) -> Dict[str, float]:
        """
        Calculate OEE (Overall Equipment Effectiveness).
        
        Args:
            availability: Availability rate (0-100).
            performance: Performance rate (0-100).
            quality: Quality rate (0-100).
        
        Returns:
            Dict with OEE components and total.
        """
        oee = (availability / 100) * (performance / 100) * (quality / 100) * 100
        
        return {
            "availability": round(availability, 2),
            "performance": round(performance, 2),
            "quality": round(quality, 2),
            "oee": round(oee, 2),
            "world_class": oee >= 85,
            "assessment": self._assess_oee(oee)
        }
    
    def _assess_oee(self, oee: float) -> str:
        """Assess OEE score."""
        if oee >= 85:
            return "World Class - Excellent performance"
        elif oee >= 75:
            return "Good - Above average, room for improvement"
        elif oee >= 65:
            return "Average - Typical for manufacturing"
        elif oee >= 50:
            return "Below Average - Significant improvement needed"
        else:
            return "Poor - Immediate attention required"
    
    async def _get_production_data(self) -> Dict[str, Any]:
        """Fetch current production data from database."""
        try:
            daily = await self.query_database(self._queries["daily_summary"])
            lines = await self.query_database(self._queries["line_performance"])
            shifts = await self.query_database(self._queries["shift_analysis"])
            issues = await self.query_database(self._queries["recent_issues"])
            
            return {
                "daily_summary": daily,
                "line_performance": lines,
                "shift_analysis": shifts,
                "recent_issues": issues
            }
        except Exception as e:
            logger.error("Failed to fetch production data", error=str(e))
            return {}
    
    async def handle(
        self,
        query: str,
        context: Optional[AgentContext] = None
    ) -> AgentResponse:
        """
        Handle production-related queries.
        
        Args:
            query: User's query.
            context: Agent context.
        
        Returns:
            AgentResponse with production insights.
        """
        logger.info("ProductionAgent handling query", query=query[:50])
        
        # Fetch production data
        prod_data = await self._get_production_data()
        
        # Try RAG for document-based queries
        rag_context = ""
        sources = []
        try:
            rag_response = await self.query_rag(query, "production", k=3)
            if rag_response.get("sources"):
                sources = rag_response["sources"]
                rag_context = f"\nRELEVANT DOCUMENTS:\n{rag_response.get('answer', '')}"
        except Exception as e:
            logger.debug("RAG query failed", error=str(e))
        
        # Try NL-to-SQL for specific data queries
        sql_result = None
        if self.should_attempt_nl_to_sql(query):
            try:
                sql_result = await self.nl_to_sql(query)
                if sql_result.get("success") and sql_result.get("results"):
                    logger.debug("NL-to-SQL successful", rows=sql_result.get("row_count"))
            except Exception as e:
                logger.debug("NL-to-SQL failed", error=str(e))
        
        # Build context for LLM
        data_context = f"""
PRODUCTION DATABASE SUMMARY:

DAILY PRODUCTION (Last 7 days):
{self._format_table(prod_data.get('daily_summary', [])[:10])}

LINE PERFORMANCE:
{self._format_table(prod_data.get('line_performance', []))}

SHIFT ANALYSIS:
{self._format_table(prod_data.get('shift_analysis', []))}

RECENT ISSUES (High defect/downtime):
{self._format_table(prod_data.get('recent_issues', [])[:5])}
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
        response_text = await self.call_llm(prompt)
        
        # Calculate confidence based on data availability
        confidence = 0.85
        if not prod_data:
            confidence -= 0.2
        if not sources:
            confidence -= 0.1
        
        return self.format_response(
            content=response_text,
            data={
                "production_summary": prod_data.get("line_performance", []),
                "sql_query": sql_result.get("generated_sql") if sql_result else None,
                "sql_results": sql_result.get("results") if sql_result else None
            },
            sources=sources,
            confidence=max(confidence, 0.5)
        )
    
    def _format_table(self, data: List[Dict]) -> str:
        """Format data as a readable table."""
        if not data:
            return "No data available"
        
        # Get column headers
        headers = list(data[0].keys())
        
        # Format rows
        rows = []
        for item in data:
            row = " | ".join(str(item.get(h, ""))[:20] for h in headers)
            rows.append(row)
        
        header_row = " | ".join(h[:20] for h in headers)
        separator = "-" * len(header_row)
        
        return f"{header_row}\n{separator}\n" + "\n".join(rows)


if __name__ == "__main__":
    import asyncio
    
    async def test_production_agent():
        print("Testing ProductionAgent...")
        
        agent = ProductionAgent()
        
        test_queries = [
            "What was the production output for Line A this week?",
            "Show me the efficiency metrics for all production lines",
            "Which shift has the highest defect rate?",
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            response = await agent.handle(
                query=query,
                context=AgentContext(user_query=query, department="production")
            )
            print(f"Response: {response.content[:300]}...")
            print(f"Confidence: {response.confidence}")
        
        print("\nProductionAgent test completed!")
    
    asyncio.run(test_production_agent())
