"""
Base Agent Module for AI Industrial Platform.

Provides abstract base class for all specialized agents with common
functionality for LLM interaction, prompt building, and response formatting.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
import unicodedata
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class AgentResponse:
    """
    Standardized response from an agent.
    
    Attributes:
        content: Main response content.
        agent_name: Name of the responding agent.
        department: Department context.
        data: Structured data from the agent.
        sources: Source documents/references used.
        confidence: Confidence score (0-1).
        metadata: Additional response metadata.
        timestamp: Response generation time.
    """
    content: str
    agent_name: str
    department: str
    data: Optional[Dict[str, Any]] = None
    sources: Optional[List[Dict[str, Any]]] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "content": self.content,
            "agent_name": self.agent_name,
            "department": self.department,
            "data": self.data,
            "sources": self.sources,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class AgentContext:
    """
    Context information passed to agents.
    
    Attributes:
        user_query: Original user query.
        department: Department context.
        session_id: User session identifier.
        conversation_history: Previous messages in conversation.
        user_metadata: Additional user information.
    """
    user_query: str
    department: str
    session_id: Optional[str] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    user_metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all specialized agents.
    
    Provides common functionality for prompt building, LLM interaction,
    database queries, and RAG integration.
    
    Attributes:
        name: Agent name identifier.
        department: Primary department this agent serves.
        description: Human-readable agent description.
    """
    
    def __init__(
        self,
        name: str,
        department: str,
        description: str,
        llm_client: Any = None,
        db_manager: Any = None,
        rag_pipeline: Any = None
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Agent identifier.
            department: Primary department.
            description: Agent description.
            llm_client: LLM client instance.
            db_manager: Database manager instance.
            rag_pipeline: RAG pipeline instance.
        """
        self.name = name
        self.department = department
        self.description = description
        
        self._llm_client = llm_client
        self._db_manager = db_manager
        self._rag_pipeline = rag_pipeline
        
        # System prompt for this agent
        self._system_prompt = self._build_system_prompt()
        
        logger.info(
            "Agent initialized",
            name=name,
            department=department
        )
    
    @property
    def llm_client(self):
        """Lazy load LLM client."""
        if self._llm_client is None:
            from llm.llm_client import LLMClient
            self._llm_client = LLMClient()
        return self._llm_client
    
    @property
    def db_manager(self):
        """Lazy load database manager."""
        if self._db_manager is None:
            from database.db_manager import SQLiteManager
            self._db_manager = SQLiteManager()
        return self._db_manager
    
    @property
    def rag_pipeline(self):
        """Lazy load RAG pipeline."""
        if self._rag_pipeline is None:
            from rag.rag_pipeline import RAGPipeline
            self._rag_pipeline = RAGPipeline()
        return self._rag_pipeline
    
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt for this agent.
        
        Returns:
            System prompt string.
        """
        return f"""You are {self.name}, an AI assistant specializing in {self.department} operations 
for an industrial platform. {self.description}

Your responsibilities:
1. Provide accurate, helpful responses based on available data and documents
2. Cite sources when using retrieved information
3. Be safety-conscious and flag potential hazards
4. Ask for clarification when queries are ambiguous
5. Provide structured data when applicable
6. Treat provided database/RAG context as the source of truth
7. Do not ask the user for information that is already present in the context
8. Do not invent product codes, project codes, prices, dates, stock levels, or customer names

Always maintain a professional tone, respond in the same language as the user's query,
and focus on actionable insights."""
    
    def build_prompt(
        self,
        query: str,
        context: Optional[AgentContext] = None,
        additional_context: Optional[str] = None
    ) -> str:
        """
        Build the user prompt with context.
        
        Args:
            query: User's query.
            context: Agent context information.
            additional_context: Additional context to include.
        
        Returns:
            Formatted prompt string.
        """
        parts = []
        
        # Add conversation history if available
        if context and context.conversation_history:
            history = context.conversation_history[-5:]  # Last 5 messages
            history_text = "\n".join(
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in history
            )
            parts.append(f"CONVERSATION HISTORY:\n{history_text}\n")
        
        # Add additional context
        if additional_context:
            parts.append(f"RELEVANT INFORMATION:\n{additional_context}\n")

        parts.append(
            "ANSWERING RULES:\n"
            "- Use RELEVANT INFORMATION as the primary source of truth.\n"
            "- If focused/matching records are present, answer from those records first.\n"
            "- If exact data is present, do not ask the user to provide it again.\n"
            "- If data is missing, say what is missing and what can still be inferred.\n"
            "- Keep the answer in the same language as the user query.\n"
        )
        
        # Add the query
        parts.append(f"USER QUERY: {query}")
        
        return "\n".join(parts)

    def _normalize_text(self, value: Any) -> str:
        text = "" if value is None else str(value)
        text = text.replace("İ", "I").replace("ı", "i")
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return text.lower()

    def _query_tokens(self, query: str) -> List[str]:
        stop_words = {
            "the", "and", "for", "with", "show", "list", "what", "which", "give", "me",
            "bir", "bu", "icin", "için", "ve", "ile", "mi", "mı", "mu", "mü", "nedir",
            "neler", "hangi", "goster", "göster", "listele", "listeler", "misin",
        }
        tokens = re.findall(r"[a-zA-Z0-9_-]{2,}", self._normalize_text(query))
        return [token for token in tokens if token not in stop_words]

    def _focus_records_by_query(
        self,
        query: str,
        records: List[Dict[str, Any]],
        fields: List[str],
        max_items: int = 8,
    ) -> List[Dict[str, Any]]:
        if not records:
            return []

        normalized_query = self._normalize_text(query)
        tokens = self._query_tokens(query)
        scored: List[tuple[int, int, Dict[str, Any]]] = []

        for index, record in enumerate(records):
            score = 0
            for field in fields:
                raw_value = record.get(field)
                if raw_value is None:
                    continue
                value = self._normalize_text(raw_value)
                if value and value in normalized_query:
                    score += 8
                if normalized_query and normalized_query in value:
                    score += 4
                for token in tokens:
                    if token and token in value:
                        score += 1
            if score > 0:
                scored.append((score, index, record))

        if not scored:
            return records[:max_items]

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [record for _, _, record in scored[:max_items]]
    
    async def call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_type: str = "generation",
        **kwargs
    ) -> str:
        """
        Call the LLM with the given prompt.
        
        Args:
            prompt: User prompt.
            system_prompt: Override system prompt.
            model_type: "routing" or "generation".
            **kwargs: Additional LLM parameters.
        
        Returns:
            LLM response text.
        """
        return await self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt or self._system_prompt,
            model_type=model_type,
            **kwargs
        )
    
    async def call_llm_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_type: str = "generation",
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Call the LLM with streaming response.
        
        Args:
            prompt: User prompt.
            system_prompt: Override system prompt.
            model_type: "routing" or "generation".
            **kwargs: Additional LLM parameters.
        
        Yields:
            Response text chunks.
        """
        async for chunk in self.llm_client.generate_stream(
            prompt=prompt,
            system_prompt=system_prompt or self._system_prompt,
            model_type=model_type,
            **kwargs
        ):
            yield chunk
    
    async def query_database(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a database query.
        
        Args:
            query: SQL query string.
            params: Query parameters.
        
        Returns:
            Query results.
        """
        return await self.db_manager.async_execute_query(query, params)
    
    async def query_rag(
        self,
        question: str,
        department: Optional[str] = None,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Query the RAG pipeline.
        
        Args:
            question: Query question.
            department: Department to search.
            k: Number of documents to retrieve.
        
        Returns:
            RAG response with answer and sources.
        """
        dept = department or self.department
        response = await self.rag_pipeline.query(
            question=question,
            department=dept,
            k=k
        )
        return response.to_dict()
    
    async def nl_to_sql(self, question: str) -> Dict[str, Any]:
        """
        Convert natural language to SQL and execute.
        
        Args:
            question: Natural language question.
        
        Returns:
            Query results with generated SQL.
        """
        return await self.db_manager.nl_to_sql(question, self.llm_client)

    def should_attempt_nl_to_sql(self, question: str) -> bool:
        """
        Agent-level NL-to-SQL is a fallback for explicit tabular operations.
        Routine domain answers already receive curated SQL context, so calling
        NL-to-SQL for every message wastes local LLM capacity and can stall UI.
        """
        normalized = self._normalize_text(question)
        patterns = [
            r"\b(sql|query|database|table|row|rows|record|records)\b",
            r"\b(count|how many|total|sum|average|avg|mean|min|max|filter|where|group by)\b",
            r"\b(kac|kaç|toplam|ortalama|adet|sayisi|sayısı|filtre|grupl[a-z]*)\b",
        ]
        return any(re.search(pattern, normalized) for pattern in patterns)
    
    def format_response(
        self,
        content: str,
        data: Optional[Dict[str, Any]] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        confidence: float = 0.8
    ) -> AgentResponse:
        """
        Format the agent response.
        
        Args:
            content: Response content.
            data: Structured data.
            sources: Source references.
            confidence: Confidence score.
        
        Returns:
            Formatted AgentResponse.
        """
        return AgentResponse(
            content=content,
            agent_name=self.name,
            department=self.department,
            data=data,
            sources=sources,
            confidence=confidence
        )
    
    @abstractmethod
    async def handle(
        self,
        query: str,
        context: Optional[AgentContext] = None
    ) -> AgentResponse:
        """
        Handle a user query.
        
        Args:
            query: User's query.
            context: Agent context information.
        
        Returns:
            AgentResponse with the result.
        """
        pass
    
    async def handle_stream(
        self,
        query: str,
        context: Optional[AgentContext] = None
    ) -> AsyncIterator[str]:
        """
        Handle a query with streaming response.
        
        Default implementation calls handle() and yields the full response.
        Override for true streaming support.
        
        Args:
            query: User's query.
            context: Agent context.
        
        Yields:
            Response chunks.
        """
        response = await self.handle(query, context)
        yield response.content
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', department='{self.department}')"


class GeneralAgent(BaseAgent):
    """
    General-purpose agent for queries that don't fit specific departments.
    
    Handles general questions, greetings, and clarification requests.
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="GeneralAgent",
            department="general",
            description=(
                "Handles general Coiltech platform inquiries and fallback requests "
                "that do not map cleanly to a specific presales, project, service, "
                "quality, or digital twin workflow."
            ),
            **kwargs
        )
    
    async def handle(
        self,
        query: str,
        context: Optional[AgentContext] = None
    ) -> AgentResponse:
        """Handle general queries with RAG context if available."""
        # Check if RAG context was passed from orchestrator
        rag_context = None
        rag_sources = []
        
        if context and context.user_metadata:
            rag_context = context.user_metadata.get("rag_context", "")
            rag_sources = context.user_metadata.get("rag_sources", [])
        
        # Build prompt with RAG context if available
        if rag_context:
            additional_context = f"""
YÜKLENEN BELGELERDEN BULUNAN BİLGİLER:
{rag_context}

Lütfen yukarıdaki kaynaklara dayanarak cevap ver. Eğer kaynaklar yeterli bilgi içermiyorsa, 
genel bilgini de kullanabilirsin ama bunu belirt.
"""
            prompt = self.build_prompt(query, context, additional_context)
        else:
            prompt = self.build_prompt(query, context)
        
        system_prompt = """You are a helpful assistant for the Coiltech Industrial Automation Platform.
You help users navigate the system and answer general questions about products, projects,
service, quotations, quality, and digital twin monitoring.

If documents have been provided in the context, USE THEM to answer the question.
Always cite your sources when using document information.

If the query seems to be about a specific workflow, suggest one of the platform domains
for more accurate assistance.

Available capabilities:
- Product Catalog / Ürün Kataloğu: Product family and model lookup
- Application Engineering / Uygulama Mühendisliği: Technical selection and sizing
- Configuration / Konfigürasyon: Line configuration and BOM guidance
- Quotations / Teklifler: RFQs, pricing, and commercial status
- Project Execution / Proje Yönetimi: Milestones, delivery, and commissioning status
- Service & Maintenance / Servis ve Bakım: Installed base, tickets, and PM history
- Spare Parts / Yedek Parça: Part lookup, stock, and lead time
- Quality & Compliance / Kalite ve Uygunluk: CE, NCR, and acceptance-test records
- Digital Twin / Dijital İkiz: Sensor data, alerts, and anomaly detection
- Management Reporting / Yönetim Raporlama: Pipeline, KPIs, and executive summaries

Be helpful, concise, and guide users to the right resources.
Use the exact Turkish domain labels above; do not translate "Spare Parts" as "Ersatz".
ALWAYS respond in the same language as the user's query."""
        
        response = await self.call_llm(prompt, system_prompt)
        
        # Include RAG sources if available
        confidence = 0.85 if rag_sources else 0.7
        
        return self.format_response(
            content=response,
            sources=rag_sources if rag_sources else None,
            confidence=confidence
        )


if __name__ == "__main__":
    import asyncio
    
    async def test_base_agent():
        print("Testing GeneralAgent...")
        
        agent = GeneralAgent()
        
        # Test basic query
        response = await agent.handle(
            query="Hello, what can you help me with?",
            context=AgentContext(
                user_query="Hello, what can you help me with?",
                department="general"
            )
        )
        
        print(f"\nAgent: {response.agent_name}")
        print(f"Department: {response.department}")
        print(f"Response: {response.content}")
        print(f"Confidence: {response.confidence}")
        
        print("\nBase agent test completed!")
    
    asyncio.run(test_base_agent())
