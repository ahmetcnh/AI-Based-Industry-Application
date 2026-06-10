"""
RAG Pipeline Module for the Coiltech Industrial Automation Platform.

Provides domain-aware document ingestion, retrieval, and answer generation
for product families, service content, quotations, and special projects.
"""

import os
import asyncio
from typing import List, Optional, Dict, Any, AsyncIterator
from dataclasses import dataclass
from pathlib import Path
import structlog

from config.settings import RAG_COLLECTION_REGISTRY
from rag.document_loader import DocumentLoader
from rag.vector_store import ChromaVectorStore

logger = structlog.get_logger(__name__)


@dataclass
class RAGResponse:
    """
    Response from the RAG pipeline.
    
    Attributes:
        answer: Generated answer text.
        sources: List of source documents used.
        confidence: Confidence score for the answer.
        query: Original query.
        department: Department context.
        metadata: Additional response metadata.
    """
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    query: str
    department: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
            "query": self.query,
            "department": self.department,
            "metadata": self.metadata
        }


class RAGPipeline:
    """
    Complete RAG pipeline for document ingestion and querying.
    
    Orchestrates document loading, chunking, embedding, storage,
    retrieval, and answer generation with LLM.
    
    Attributes:
        document_loader: Document loader instance.
        vector_store: Vector store instance.
        llm_client: LLM client for answer generation.
    """
    
    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        document_loader: Optional[DocumentLoader] = None,
        llm_client: Any = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store: ChromaDB vector store instance.
            document_loader: Document loader instance.
            llm_client: LLM client for generation.
        """
        from config.settings import get_settings
        self.settings = get_settings()
        
        self.document_loader = document_loader or DocumentLoader()
        self.vector_store = vector_store or ChromaVectorStore()
        self._llm_client = llm_client
    
    @property
    def llm_client(self):
        """Lazy load LLM client."""
        if self._llm_client is None:
            from llm.llm_client import LLMClient
            self._llm_client = LLMClient()
        return self._llm_client
    
    def resolve_collection(
        self,
        collection_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        file_path: Optional[str] = None,
    ) -> str:
        """Resolve the target RAG collection using explicit input and metadata heuristics."""
        metadata = metadata or {}

        if collection_name:
            return collection_name

        explicit = metadata.get("rag_collection") or metadata.get("collection_name")
        if explicit:
            return str(explicit)

        search_text = " ".join(
            str(value).lower()
            for value in [
                metadata.get("title"),
                metadata.get("document_role"),
                metadata.get("doc_type"),
                metadata.get("family_code"),
                metadata.get("family_name"),
                metadata.get("product_family"),
                Path(file_path).stem if file_path else "",
            ]
            if value
        )

        keyword_map = {
            "products_compact_drives": ["compact drive", "press feeding", "compact"],
            "products_loop_drives": ["loop drive", "loop type"],
            "products_blank_lines": ["blank line", "blanking"],
            "products_destacker": ["destacker"],
            "products_press_transfer_robots": ["transfer robot", "press transfer robot", "robot"],
            "products_press_transfer": ["press transfer", "transfer system"],
            "products_servo_drives": ["servo drive", "triggered", "zigzag", "straightener servo"],
            "products_decoilers": ["decoiler", "uncoiler", "mandrel"],
            "products_ctl_lines": ["cut to length", "ctl", "shear", "stacker"],
            "special_projects": ["special project", "custom project", "layout", "rfq"],
            "installation_commissioning": ["commissioning", "installation", "startup", "sat", "fat"],
            "service_maintenance": ["service", "maintenance", "troubleshooting", "repair", "alarm"],
            "spare_parts": ["spare", "part", "parts list", "replacement"],
            "quality_documents": ["ce", "quality", "certificate", "audit", "compliance"],
            "sales_and_quotations": ["quotation", "quote", "commercial", "proposal", "sales"],
        }

        for collection_key, keywords in keyword_map.items():
            if any(keyword in search_text for keyword in keywords):
                return collection_key

        return self.settings.OPENWEBUI_DEFAULT_COLLECTION

    def ingest(
        self,
        file_path: str,
        department: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest a document into the RAG system.
        
        Args:
            file_path: Path to the document file.
            department: Target collection key for the document.
            metadata: Additional metadata to attach.
        
        Returns:
            Dict with ingestion results.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        collection_name = self.resolve_collection(department, metadata, file_path)

        logger.info(
            "Starting document ingestion",
            file=file_path,
            department=collection_name
        )
        
        # Load document
        documents = self.document_loader.load(file_path)
        
        # Add custom metadata
        if metadata:
            for doc in documents:
                doc.metadata.update(metadata)
        for doc in documents:
            doc.metadata["rag_collection"] = collection_name
        
        # Chunk documents
        chunks = self.document_loader.chunk_documents(documents)
        
        # Add collection metadata to chunks
        for chunk in chunks:
            chunk.metadata["department"] = collection_name
            chunk.metadata["rag_collection"] = collection_name

        # Store in vector store
        result = self.vector_store.add_documents(chunks, collection_name)

        ingestion_result = {
            "file": file_path,
            "department": collection_name,
            "collection": collection_name,
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "chunks_stored": result["added"],
            "collection_total": result["collection_total"]
        }

        logger.info("Document ingestion complete", **ingestion_result)

        return ingestion_result
    
    async def ingest_async(
        self,
        file_path: str,
        department: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Asynchronously ingest a document.
        
        Args:
            file_path: Path to the document file.
            department: Target department.
            metadata: Additional metadata.
        
        Returns:
            Dict with ingestion results.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.ingest(file_path, department, metadata)
        )
    
    def _build_context(self, sources: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved sources.
        
        Args:
            sources: List of retrieved documents.
        
        Returns:
            Formatted context string.
        """
        if not sources:
            return "No relevant documents found."
        
        context_parts = []
        for i, source in enumerate(sources, 1):
            source_info = source.get("metadata", {}).get("source", "Unknown")
            content = source.get("content", "")
            score = source.get("score", 0)
            
            context_parts.append(
                f"[Source {i}] (Relevance: {score:.2f}) {Path(source_info).name}\n"
                f"{content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        department: str
    ) -> str:
        """
        Build the generation prompt with context.
        
        Args:
            query: User's question.
            context: Retrieved context.
            department: Department context.
        
        Returns:
            Formatted prompt string.
        """
        label = RAG_COLLECTION_REGISTRY.get(department, {}).get("label", department)
        return f"""You are an AI assistant for the Coiltech industrial automation platform.
You are answering using the "{label}" knowledge collection.

Use the following context to answer the question. If the context doesn't contain 
relevant information, say so but try to provide general guidance based on your knowledge.

CONTEXT:
{context}

QUESTION: {query}

Provide a clear, concise answer. If you use information from the context,
mention which source(s) you are referencing. If the question requires information
not in the context, clearly state that and provide practical engineering guidance.

ANSWER:"""
    
    def _calculate_confidence(self, sources: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score based on retrieval results.
        
        Args:
            sources: Retrieved source documents.
        
        Returns:
            Confidence score between 0 and 1.
        """
        if not sources:
            return 0.0
        
        # Average of top source scores
        scores = [s.get("score", 0) for s in sources[:3]]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Boost if multiple relevant sources
        source_count_boost = min(len(sources) / 5, 0.2)
        
        confidence = min(avg_score + source_count_boost, 1.0)
        return round(confidence, 3)
    
    async def query(
        self,
        question: str,
        department: str,
        k: int = None,
        use_mmr: bool = True
    ) -> RAGResponse:
        """
        Query the RAG system and generate an answer.
        
        Args:
            question: User's question.
            department: Department to search in.
            k: Number of documents to retrieve.
            use_mmr: Whether to use MMR for diverse results.
        
        Returns:
            RAGResponse with answer and sources.
        """
        k = k or self.settings.RAG_TOP_K

        logger.info("RAG query started", question=question[:50], department=department)

        # Retrieve relevant documents
        if use_mmr:
            sources = self.vector_store.mmr_search(
                query=question,
                department=department,
                k=k,
                fetch_k=k * 4
            )
        else:
            sources = self.vector_store.similarity_search(
                query=question,
                department=department,
                k=k
            )

        # Build context and prompt
        context = self._build_context(sources)
        prompt = self._build_prompt(question, context, department)

        # Generate answer
        system_prompt = (
            "You are an expert AI assistant for Coiltech, a manufacturer and integrator "
            "of press feeding systems, servo lines, decoilers, transfer systems, and CTL lines. "
            "Provide accurate, practical, and safety-conscious responses. "
            "When citing sources, be specific about which document the information comes from."
        )

        answer = await self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model_type="generation"
        )

        # Calculate confidence
        confidence = self._calculate_confidence(sources)

        # Build response
        response = RAGResponse(
            answer=answer.strip(),
            sources=sources,
            confidence=confidence,
            query=question,
            department=department,
            metadata={
                "retrieved_count": len(sources),
                "used_mmr": use_mmr,
                "k": k
            }
        )

        logger.info(
            "RAG query completed",
            department=department,
            sources_found=len(sources),
            confidence=confidence
        )

        return response
    
    async def query_stream(
        self,
        question: str,
        department: str,
        k: int = None,
        use_mmr: bool = True
    ) -> AsyncIterator[str]:
        """
        Query the RAG system with streaming response.
        
        Args:
            question: User's question.
            department: Department to search in.
            k: Number of documents to retrieve.
            use_mmr: Whether to use MMR.
        
        Yields:
            Chunks of the generated answer.
        """
        k = k or self.settings.RAG_TOP_K
        
        # Retrieve documents
        if use_mmr:
            sources = self.vector_store.mmr_search(
                query=question,
                department=department,
                k=k,
                fetch_k=k * 4
            )
        else:
            sources = self.vector_store.similarity_search(
                query=question,
                department=department,
                k=k
            )
        
        # Build context and prompt
        context = self._build_context(sources)
        prompt = self._build_prompt(question, context, department)
        
        system_prompt = (
            "You are an expert AI assistant for Coiltech industrial automation. "
            "Provide accurate, practical, and safety-conscious responses."
        )
        
        # Stream the response
        async for chunk in self.llm_client.generate_stream(
            prompt=prompt,
            system_prompt=system_prompt,
            model_type="generation"
        ):
            yield chunk
    
    async def query_multi_department(
        self,
        question: str,
        departments: Optional[List[str]] = None,
        k_per_dept: int = 3
    ) -> Dict[str, RAGResponse]:
        """
        Query multiple departments and aggregate results.
        
        Args:
            question: User's question.
            departments: List of departments to search. Defaults to all.
            k_per_dept: Documents to retrieve per department.
        
        Returns:
            Dict mapping department names to their RAG responses.
        """
        if departments is None:
            departments = self.settings.rag_collections_list
        
        results = {}
        
        # Query each department
        for dept in departments:
            try:
                response = await self.query(
                    question=question,
                    department=dept,
                    k=k_per_dept
                )
                if response.sources:  # Only include if sources found
                    results[dept] = response
            except Exception as e:
                logger.warning(
                    "Department query failed",
                    department=dept,
                    error=str(e)
                )
        
        return results
    
    def get_collection_info(self, department: str) -> Dict[str, Any]:
        """
        Get information about a department's document collection.
        
        Args:
            department: Department name.
        
        Returns:
            Collection statistics.
        """
        return self.vector_store.get_collection_stats(department)
    
    def list_all_collections(self) -> List[Dict[str, Any]]:
        """
        List all document collections.
        
        Returns:
            List of collection information.
        """
        return self.vector_store.list_collections()
    
    def delete_department_documents(self, department: str) -> bool:
        """
        Delete all documents for a department.
        
        Args:
            department: Department whose documents to delete.
        
        Returns:
            True if successful.
        """
        return self.vector_store.delete_collection(department)


if __name__ == "__main__":
    import asyncio
    
    async def test_rag():
        print("Initializing RAG pipeline...")
        rag = RAGPipeline()
        
        # Create test documents
        test_dir = "data/test_docs"
        os.makedirs(test_dir, exist_ok=True)
        
        # Maintenance document
        maintenance_doc = """
        Industrial Robot Maintenance Guide
        
        Chapter 1: Daily Inspections
        
        Before operating the robot each day, perform the following checks:
        1. Verify all emergency stops are functioning
        2. Check lubricant levels in all joints
        3. Inspect cables for wear or damage
        4. Test the collision detection system
        
        Chapter 2: Weekly Maintenance
        
        Every week, the following tasks should be performed:
        1. Clean all sensors and cameras
        2. Check and tighten all bolts
        3. Verify backup battery status
        4. Update operation logs
        
        Chapter 3: Troubleshooting
        
        If the robot shows error code E-101:
        - This indicates a servo motor overload
        - Reduce payload or check for mechanical binding
        - If problem persists, contact maintenance supervisor
        """
        
        maintenance_path = os.path.join(test_dir, "robot_maintenance.txt")
        with open(maintenance_path, "w") as f:
            f.write(maintenance_doc)
        
        # Ingest document
        print("\nIngesting maintenance document...")
        result = rag.ingest(maintenance_path, "maintenance")
        print(f"Ingested: {result}")
        
        # Query the system
        print("\nQuerying RAG system...")
        response = await rag.query(
            question="What should I check during daily robot inspections?",
            department="maintenance"
        )
        
        print(f"\nQuestion: {response.query}")
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence}")
        print(f"Sources: {len(response.sources)}")
        
        for i, source in enumerate(response.sources, 1):
            print(f"\n  Source {i}:")
            print(f"    Score: {source['score']}")
            print(f"    Preview: {source['content'][:100]}...")
        
        # Cleanup
        rag.delete_department_documents("maintenance")
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
        
        print("\nRAG pipeline test completed!")
    
    asyncio.run(test_rag())
