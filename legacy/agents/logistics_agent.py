# TODO(legacy): This file is an orphaned agent from the original industrial-platform design.
# It is NOT registered in agents/__init__.py or agents/orchestrator.py and is never invoked.
# Decision pending: either remove this file or integrate it into the Coiltech domain routing.
"""
Logistics Agent Module for AI Industrial Platform.

Handles logistics-related queries including inventory management,
stock levels, shipping, and warehouse operations.
"""

from typing import Optional, Dict, Any, List
import structlog

from agents.base_agent import BaseAgent, AgentResponse, AgentContext

logger = structlog.get_logger(__name__)


class LogisticsAgent(BaseAgent):
    """
    Specialized agent for logistics operations.
    
    Handles queries about:
    - Inventory levels and stock status
    - Low stock alerts
    - Warehouse locations
    - Supplier information
    - Stock valuation
    - Reorder recommendations
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="LogisticsAgent",
            department="logistics",
            description="Manages inventory data, tracks stock levels, "
                       "monitors supply chain, and provides logistics insights.",
            **kwargs
        )
        
        self._queries = {
            "inventory_summary": """
                SELECT 
                    category,
                    COUNT(*) as item_count,
                    SUM(quantity) as total_quantity,
                    SUM(quantity * unit_cost) as total_value
                FROM inventory
                GROUP BY category
                ORDER BY total_value DESC
            """,
            "all_inventory": """
                SELECT 
                    item_code,
                    item_name,
                    category,
                    quantity,
                    unit,
                    min_stock_level,
                    location,
                    unit_cost,
                    (quantity * unit_cost) as stock_value
                FROM inventory
                ORDER BY category, item_name
            """,
            "low_stock": """
                SELECT 
                    item_code,
                    item_name,
                    category,
                    quantity,
                    min_stock_level,
                    (quantity - min_stock_level) as margin,
                    supplier,
                    unit_cost
                FROM inventory
                WHERE quantity <= min_stock_level * 1.2
                ORDER BY margin ASC
            """,
            "critical_stock": """
                SELECT 
                    item_code,
                    item_name,
                    quantity,
                    min_stock_level,
                    supplier,
                    location
                FROM inventory
                WHERE quantity < min_stock_level
                ORDER BY quantity ASC
            """,
            "stock_by_location": """
                SELECT 
                    location,
                    COUNT(*) as items,
                    SUM(quantity * unit_cost) as total_value
                FROM inventory
                GROUP BY location
                ORDER BY total_value DESC
            """,
            "spare_parts": """
                SELECT 
                    item_code,
                    item_name,
                    quantity,
                    min_stock_level,
                    unit_cost,
                    supplier
                FROM inventory
                WHERE category = 'Spare Part'
                ORDER BY quantity ASC
            """,
            "supplier_summary": """
                SELECT 
                    supplier,
                    COUNT(*) as items_supplied,
                    SUM(quantity * unit_cost) as inventory_value
                FROM inventory
                WHERE supplier IS NOT NULL
                GROUP BY supplier
                ORDER BY inventory_value DESC
            """
        }
    
    async def _get_logistics_data(self) -> Dict[str, Any]:
        """Fetch logistics data from database."""
        try:
            summary = await self.query_database(self._queries["inventory_summary"])
            all_items = await self.query_database(self._queries["all_inventory"])
            low_stock = await self.query_database(self._queries["low_stock"])
            critical = await self.query_database(self._queries["critical_stock"])
            locations = await self.query_database(self._queries["stock_by_location"])
            spare_parts = await self.query_database(self._queries["spare_parts"])
            suppliers = await self.query_database(self._queries["supplier_summary"])
            
            return {
                "inventory_summary": summary,
                "all_inventory": all_items,
                "low_stock": low_stock,
                "critical_stock": critical,
                "stock_by_location": locations,
                "spare_parts": spare_parts,
                "supplier_summary": suppliers
            }
        except Exception as e:
            logger.error("Failed to fetch logistics data", error=str(e))
            return {}
    
    def _calculate_reorder_recommendations(
        self,
        low_stock_items: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Generate reorder recommendations.
        
        Args:
            low_stock_items: Items with low stock.
        
        Returns:
            List of reorder recommendations.
        """
        recommendations = []
        
        for item in low_stock_items:
            quantity = item.get("quantity", 0)
            min_level = item.get("min_stock_level", 0)
            margin = quantity - min_level
            
            # Calculate urgency
            if margin < 0:
                urgency = "CRITICAL"
                reorder_qty = min_level * 2  # Order double to build buffer
            elif margin <= min_level * 0.1:
                urgency = "HIGH"
                reorder_qty = min_level * 1.5
            else:
                urgency = "MEDIUM"
                reorder_qty = min_level
            
            recommendations.append({
                "item_code": item.get("item_code"),
                "item_name": item.get("item_name"),
                "current_qty": quantity,
                "min_level": min_level,
                "recommended_order": int(reorder_qty),
                "urgency": urgency,
                "supplier": item.get("supplier"),
                "estimated_cost": round(
                    reorder_qty * item.get("unit_cost", 0), 2
                )
            })
        
        return sorted(
            recommendations,
            key=lambda x: (
                {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}.get(x["urgency"], 3),
                x["current_qty"]
            )
        )
    
    async def handle(
        self,
        query: str,
        context: Optional[AgentContext] = None
    ) -> AgentResponse:
        """
        Handle logistics-related queries.
        
        Args:
            query: User's query.
            context: Agent context.
        
        Returns:
            AgentResponse with logistics insights.
        """
        logger.info("LogisticsAgent handling query", query=query[:50])
        
        # Fetch logistics data
        logistics_data = await self._get_logistics_data()
        
        # Try RAG for document-based queries
        rag_context = ""
        sources = []
        try:
            rag_response = await self.query_rag(query, "logistics", k=3)
            if rag_response.get("sources"):
                sources = rag_response["sources"]
                rag_context = f"\nLOGISTICS DOCUMENTS:\n{rag_response.get('answer', '')}"
        except Exception as e:
            logger.debug("RAG query failed", error=str(e))
        
        # Try NL-to-SQL for specific queries
        sql_result = None
        if self.should_attempt_nl_to_sql(query):
            try:
                sql_result = await self.nl_to_sql(query)
            except Exception as e:
                logger.debug("NL-to-SQL failed", error=str(e))
        
        # Generate reorder recommendations
        recommendations = []
        if logistics_data.get("low_stock"):
            recommendations = self._calculate_reorder_recommendations(
                logistics_data["low_stock"]
            )
        
        # Calculate totals
        total_value = sum(
            cat.get("total_value", 0) or 0 
            for cat in logistics_data.get("inventory_summary", [])
        )
        
        critical_count = len(logistics_data.get("critical_stock", []))
        low_stock_count = len(logistics_data.get("low_stock", []))
        
        # Build context
        data_context = f"""
LOGISTICS DATABASE SUMMARY:

INVENTORY VALUE: ${total_value:,.2f}
CRITICAL STOCK ITEMS: {critical_count}
LOW STOCK ITEMS: {low_stock_count}

INVENTORY BY CATEGORY:
{self._format_table(logistics_data.get('inventory_summary', []))}

LOW STOCK ALERT:
{self._format_table(logistics_data.get('low_stock', [])[:8])}

CRITICAL STOCK (Below minimum):
{self._format_table(logistics_data.get('critical_stock', []))}

STOCK BY LOCATION:
{self._format_table(logistics_data.get('stock_by_location', []))}

SUPPLIER SUMMARY:
{self._format_table(logistics_data.get('supplier_summary', []))}

REORDER RECOMMENDATIONS:
{self._format_recommendations(recommendations[:5])}
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

When answering logistics questions:
1. Prioritize critical and low stock alerts
2. Provide reorder recommendations with urgency levels
3. Consider lead times and supplier reliability
4. Include cost implications for inventory decisions
5. Suggest inventory optimization strategies"""
        
        response_text = await self.call_llm(prompt, system_prompt)
        
        confidence = 0.85
        if not logistics_data:
            confidence -= 0.2
        if not sources:
            confidence -= 0.05
        
        return self.format_response(
            content=response_text,
            data={
                "inventory_summary": logistics_data.get("inventory_summary", []),
                "low_stock_count": low_stock_count,
                "critical_count": critical_count,
                "total_inventory_value": total_value,
                "reorder_recommendations": recommendations[:5],
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
    
    def _format_recommendations(self, recommendations: List[Dict]) -> str:
        """Format reorder recommendations."""
        if not recommendations:
            return "No recommendations at this time"
        
        lines = []
        for rec in recommendations:
            lines.append(
                f"- [{rec['urgency']}] {rec['item_name']}: "
                f"Order {rec['recommended_order']} units "
                f"(current: {rec['current_qty']}, min: {rec['min_level']}) "
                f"Est. cost: ${rec['estimated_cost']:,.2f}"
            )
        return "\n".join(lines)


if __name__ == "__main__":
    import asyncio
    
    async def test_logistics_agent():
        print("Testing LogisticsAgent...")
        
        agent = LogisticsAgent()
        
        test_queries = [
            "What's the current inventory level for steel rods?",
            "Show me items that need to be reordered",
            "What's the total value of our spare parts inventory?",
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            response = await agent.handle(
                query=query,
                context=AgentContext(user_query=query, department="logistics")
            )
            print(f"Response: {response.content[:300]}...")
            print(f"Confidence: {response.confidence}")
        
        print("\nLogisticsAgent test completed!")
    
    asyncio.run(test_logistics_agent())
