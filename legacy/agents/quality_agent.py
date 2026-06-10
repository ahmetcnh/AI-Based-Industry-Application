# TODO(legacy): Orphaned agent from original industrial-platform design.
# Not registered in agents/__init__.py or orchestrator.py — never invoked.
# The active replacement is QualityComplianceAgent (agents/quality_compliance_agent.py).
# Decision pending: remove or integrate.
"""
Quality Agent Module for AI Industrial Platform.

Handles quality-related queries including QA reports, inspection results,
defect analysis, and compliance tracking.
"""

from typing import Optional, Dict, Any, List
import structlog

from agents.base_agent import BaseAgent, AgentResponse, AgentContext

logger = structlog.get_logger(__name__)


class QualityAgent(BaseAgent):
    """
    Specialized agent for quality assurance operations.
    
    Handles queries about:
    - Quality inspection reports
    - Batch pass/fail rates
    - Defect type analysis
    - Inspector performance
    - Quality trends
    - Compliance and standards
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="QualityAgent",
            department="quality",
            description="Analyzes quality data, tracks inspection results, "
                       "identifies defect patterns, and ensures compliance with standards.",
            **kwargs
        )
        
        self._queries = {
            "quality_summary": """
                SELECT 
                    product,
                    COUNT(*) as batches,
                    SUM(total_inspected) as total_inspected,
                    SUM(total_passed) as total_passed,
                    ROUND(AVG(pass_rate), 2) as avg_pass_rate,
                    SUM(CASE WHEN disposition = 'rejected' THEN 1 ELSE 0 END) as rejected,
                    SUM(CASE WHEN disposition = 'rework' THEN 1 ELSE 0 END) as rework
                FROM quality_reports
                GROUP BY product
                ORDER BY avg_pass_rate ASC
            """,
            "recent_reports": """
                SELECT 
                    batch_id,
                    product,
                    inspection_date,
                    total_inspected,
                    total_passed,
                    pass_rate,
                    defect_types,
                    inspector,
                    disposition
                FROM quality_reports
                ORDER BY inspection_date DESC
                LIMIT 15
            """,
            "defect_analysis": """
                SELECT 
                    defect_types,
                    COUNT(*) as occurrences,
                    AVG(100 - pass_rate) as avg_defect_rate
                FROM quality_reports
                WHERE defect_types IS NOT NULL AND defect_types != ''
                GROUP BY defect_types
                ORDER BY occurrences DESC
            """,
            "inspector_performance": """
                SELECT 
                    inspector,
                    COUNT(*) as inspections,
                    ROUND(AVG(pass_rate), 2) as avg_pass_rate,
                    SUM(total_inspected) as total_inspected
                FROM quality_reports
                GROUP BY inspector
                ORDER BY inspections DESC
            """,
            "problem_batches": """
                SELECT 
                    batch_id,
                    product,
                    inspection_date,
                    pass_rate,
                    defect_types,
                    disposition,
                    notes
                FROM quality_reports
                WHERE pass_rate < 95 OR disposition IN ('rejected', 'rework')
                ORDER BY inspection_date DESC
                LIMIT 10
            """,
            "daily_quality": """
                SELECT 
                    DATE(inspection_date) as date,
                    COUNT(*) as batches_inspected,
                    ROUND(AVG(pass_rate), 2) as avg_pass_rate,
                    SUM(total_inspected) as units_inspected,
                    SUM(total_passed) as units_passed
                FROM quality_reports
                GROUP BY DATE(inspection_date)
                ORDER BY date DESC
                LIMIT 7
            """
        }
    
    async def _get_quality_data(self) -> Dict[str, Any]:
        """Fetch quality data from database."""
        try:
            summary = await self.query_database(self._queries["quality_summary"])
            recent = await self.query_database(self._queries["recent_reports"])
            defects = await self.query_database(self._queries["defect_analysis"])
            inspectors = await self.query_database(self._queries["inspector_performance"])
            problems = await self.query_database(self._queries["problem_batches"])
            daily = await self.query_database(self._queries["daily_quality"])
            
            return {
                "quality_summary": summary,
                "recent_reports": recent,
                "defect_analysis": defects,
                "inspector_performance": inspectors,
                "problem_batches": problems,
                "daily_quality": daily
            }
        except Exception as e:
            logger.error("Failed to fetch quality data", error=str(e))
            return {}
    
    def _analyze_defect_patterns(self, defect_data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze defect patterns and generate insights.
        
        Args:
            defect_data: Defect analysis data.
        
        Returns:
            Dict with defect insights.
        """
        if not defect_data:
            return {"insights": [], "most_common": None}
        
        # Parse defect types (they're stored as JSON strings)
        defect_counts = {}
        for record in defect_data:
            types = record.get("defect_types", "")
            if types:
                # Handle JSON array format
                import json
                try:
                    if types.startswith("["):
                        defect_list = json.loads(types)
                    else:
                        defect_list = [types]
                except:
                    defect_list = [types]
                
                for defect in defect_list:
                    defect_counts[defect] = defect_counts.get(defect, 0) + 1
        
        # Sort by frequency
        sorted_defects = sorted(
            defect_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        insights = []
        if sorted_defects:
            most_common = sorted_defects[0]
            insights.append(
                f"Most common defect: {most_common[0]} ({most_common[1]} occurrences)"
            )
            
            if len(sorted_defects) > 1:
                total = sum(d[1] for d in sorted_defects)
                top_3_pct = sum(d[1] for d in sorted_defects[:3]) / total * 100
                insights.append(
                    f"Top 3 defect types account for {top_3_pct:.1f}% of all defects"
                )
        
        return {
            "insights": insights,
            "most_common": sorted_defects[0][0] if sorted_defects else None,
            "defect_breakdown": dict(sorted_defects[:5])
        }
    
    async def handle(
        self,
        query: str,
        context: Optional[AgentContext] = None
    ) -> AgentResponse:
        """
        Handle quality-related queries.
        
        Args:
            query: User's query.
            context: Agent context.
        
        Returns:
            AgentResponse with quality insights.
        """
        logger.info("QualityAgent handling query", query=query[:50])
        
        # Fetch quality data
        quality_data = await self._get_quality_data()
        
        # Try RAG for document-based queries
        rag_context = ""
        sources = []
        try:
            rag_response = await self.query_rag(query, "quality", k=4)
            if rag_response.get("sources"):
                sources = rag_response["sources"]
                rag_context = f"\nQUALITY DOCUMENTS:\n{rag_response.get('answer', '')}"
        except Exception as e:
            logger.debug("RAG query failed", error=str(e))
        
        # Try NL-to-SQL for specific queries
        sql_result = None
        if self.should_attempt_nl_to_sql(query):
            try:
                sql_result = await self.nl_to_sql(query)
            except Exception as e:
                logger.debug("NL-to-SQL failed", error=str(e))
        
        # Analyze defect patterns
        defect_insights = self._analyze_defect_patterns(
            quality_data.get("defect_analysis", [])
        )
        
        # Build context
        data_context = f"""
QUALITY DATABASE SUMMARY:

PRODUCT QUALITY SUMMARY:
{self._format_table(quality_data.get('quality_summary', []))}

DAILY QUALITY METRICS (Last 7 days):
{self._format_table(quality_data.get('daily_quality', []))}

RECENT QUALITY REPORTS:
{self._format_table(quality_data.get('recent_reports', [])[:8])}

PROBLEM BATCHES (Low pass rate or rejected):
{self._format_table(quality_data.get('problem_batches', [])[:5])}

DEFECT ANALYSIS:
{self._format_table(quality_data.get('defect_analysis', [])[:5])}

DEFECT INSIGHTS:
{chr(10).join(defect_insights.get('insights', ['No insights available']))}

INSPECTOR PERFORMANCE:
{self._format_table(quality_data.get('inspector_performance', []))}
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

When answering quality questions:
1. Focus on actionable insights and root cause analysis
2. Highlight quality trends (improving or declining)
3. Reference quality standards and compliance requirements
4. Suggest corrective actions for quality issues
5. Compare against industry benchmarks when relevant"""
        
        response_text = await self.call_llm(prompt, system_prompt)
        
        confidence = 0.85
        if not quality_data:
            confidence -= 0.2
        if not sources:
            confidence -= 0.1
        
        return self.format_response(
            content=response_text,
            data={
                "quality_summary": quality_data.get("quality_summary", []),
                "defect_insights": defect_insights,
                "problem_batches": quality_data.get("problem_batches", [])[:5],
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
            row = " | ".join(str(item.get(h, ""))[:20] for h in headers)
            rows.append(row)
        
        header_row = " | ".join(h[:20] for h in headers)
        return f"{header_row}\n{'-' * len(header_row)}\n" + "\n".join(rows)


if __name__ == "__main__":
    import asyncio
    
    async def test_quality_agent():
        print("Testing QualityAgent...")
        
        agent = QualityAgent()
        
        test_queries = [
            "What is the pass rate for Widget-A?",
            "Show me batches that were rejected this week",
            "What are the most common defect types?",
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            response = await agent.handle(
                query=query,
                context=AgentContext(user_query=query, department="quality")
            )
            print(f"Response: {response.content[:300]}...")
            print(f"Confidence: {response.confidence}")
        
        print("\nQualityAgent test completed!")
    
    asyncio.run(test_quality_agent())
