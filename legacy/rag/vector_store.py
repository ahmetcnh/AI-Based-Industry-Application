"""
Vector Store Module for the Coiltech Industrial Automation Platform.

Provides ChromaDB-based vector storage with domain collection namespacing
for product families, service content, quotations, and special projects.
"""

import os
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
import structlog

from rag.document_loader import Document

logger = structlog.get_logger(__name__)


class ChromaVectorStore:
    """
    ChromaDB-based vector store with domain collection namespacing.
    
    Manages document embeddings and similarity search using ChromaDB
    with persistent storage and support for multiple department collections.
    
    Attributes:
        persist_dir: Directory for ChromaDB persistence.
        collection_prefix: Prefix for collection names.
        embedding_model: Name of the sentence-transformers model.
    """
    
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_prefix: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_dir: Directory for persistence. Defaults to config value.
            collection_prefix: Collection name prefix. Defaults to config value.
            embedding_model: Embedding model name. Defaults to config value.
        """
        from config.settings import get_settings
        settings = get_settings()
        
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.collection_prefix = collection_prefix or settings.CHROMA_COLLECTION_PREFIX
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        
        # Ensure persist directory exists
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding function
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )
        
        # Cache for collections
        self._collections: Dict[str, chromadb.Collection] = {}
        
        logger.info(
            "Vector store initialized",
            persist_dir=self.persist_dir,
            embedding_model=self.embedding_model
        )
    
    def _get_collection_name(self, department: str) -> str:
        """
        Get the full collection name for a collection key.
        
        Args:
            department: Collection key or logical department name.
        
        Returns:
            Full collection name with prefix.
        """
        safe_dept = department.lower().replace(" ", "_").replace("-", "_")
        return f"{self.collection_prefix}_{safe_dept}"
    
    def init_collection(
        self,
        department: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> chromadb.Collection:
        """
        Initialize or get a collection for a department.
        
        Args:
            department: Department name.
            metadata: Optional collection metadata.
        
        Returns:
            ChromaDB Collection object.
        """
        collection_name = self._get_collection_name(department)
        
        if collection_name in self._collections:
            return self._collections[collection_name]
        
        from config.settings import RAG_COLLECTION_REGISTRY

        collection_metadata = metadata or {}
        collection_metadata["department"] = department
        collection_metadata.update(
            {
                key: value
                for key, value in RAG_COLLECTION_REGISTRY.get(department, {}).items()
                if key not in collection_metadata and isinstance(value, (str, int, float, bool))
            }
        )
        
        collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata=collection_metadata
        )
        
        self._collections[collection_name] = collection
        
        logger.info(
            "Collection initialized",
            collection=collection_name,
            department=department,
            count=collection.count()
        )
        
        return collection
    
    def add_documents(
        self,
        documents: List[Document],
        department: str
    ) -> Dict[str, Any]:
        """
        Add documents to a department's collection.
        
        Args:
            documents: List of Document objects to add.
            department: Target department.
        
        Returns:
            Dict with operation results.
        """
        if not documents:
            return {"added": 0, "department": department}
        
        collection = self.init_collection(department)
        
        # Prepare data for ChromaDB
        ids = []
        contents = []
        metadatas = []
        
        for doc in documents:
            doc_id = doc.doc_id or f"doc_{len(ids)}"
            ids.append(doc_id)
            contents.append(doc.content)
            
            # Ensure metadata values are ChromaDB-compatible
            metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                elif isinstance(value, list):
                    metadata[key] = str(value)
                else:
                    metadata[key] = str(value)
            metadata["department"] = department
            metadatas.append(metadata)
        
        try:
            collection.upsert(
                ids=ids,
                documents=contents,
                metadatas=metadatas
            )
            
            logger.info(
                "Documents added",
                department=department,
                count=len(documents),
                collection_total=collection.count()
            )
            
            return {
                "added": len(documents),
                "department": department,
                "collection_total": collection.count()
            }
        
        except Exception as e:
            logger.error(
                "Failed to add documents",
                department=department,
                error=str(e)
            )
            raise
    
    def similarity_search(
        self,
        query: str,
        department: str,
        k: int = 5,
        score_threshold: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents in a domain collection.
        
        Args:
            query: Search query text.
            department: Collection key to search in.
            k: Number of results to return.
            score_threshold: Minimum similarity score (0-1, higher is more similar).
            where: Optional metadata filter.
            include_embeddings: Whether to include embeddings in results.
        
        Returns:
            List of search results with content, metadata, and scores.
        """
        from config.settings import get_settings
        settings = get_settings()
        
        collection = self.init_collection(department)
        
        if collection.count() == 0:
            logger.warning("Empty collection", department=department)
            return []
        
        score_threshold = score_threshold or settings.RAG_SCORE_THRESHOLD
        
        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")
        
        try:
            results = collection.query(
                query_texts=[query],
                n_results=min(k, collection.count()),
                where=where,
                include=include
            )
            
            # Process results
            search_results = []
            
            if results and results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    # ChromaDB returns distances, convert to similarity score
                    # For L2 distance, smaller is better, so we invert
                    distance = results["distances"][0][i] if results["distances"] else 0
                    # Convert L2 distance to similarity score (0-1 range)
                    similarity = 1 / (1 + distance)
                    
                    if similarity < score_threshold:
                        continue
                    
                    result = {
                        "id": doc_id,
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "score": round(similarity, 4),
                        "distance": round(distance, 4)
                    }
                    
                    if include_embeddings and results.get("embeddings"):
                        result["embedding"] = results["embeddings"][0][i]
                    
                    search_results.append(result)
            
            logger.debug(
                "Search completed",
                department=department,
                query_preview=query[:50],
                results=len(search_results)
            )
            
            return search_results
        
        except Exception as e:
            logger.error(
                "Search failed",
                department=department,
                error=str(e)
            )
            raise
    
    def search_all_departments(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across all domain collections.
        
        Args:
            query: Search query text.
            k: Number of results per department.
            score_threshold: Minimum similarity score.
        
        Returns:
            Dict mapping collection keys to their search results.
        """
        results = {}
        
        for collection in self._client.list_collections():
            if collection.name.startswith(self.collection_prefix):
                dept = collection.name.replace(f"{self.collection_prefix}_", "")
                dept_results = self.similarity_search(
                    query=query,
                    department=dept,
                    k=k,
                    score_threshold=score_threshold
                )
                if dept_results:
                    results[dept] = dept_results
        
        return results
    
    def mmr_search(
        self,
        query: str,
        department: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Maximal Marginal Relevance search for diverse results.
        
        Args:
            query: Search query text.
            department: Department to search in.
            k: Number of final results.
            fetch_k: Number of initial candidates.
            lambda_mult: Diversity factor (0=max diversity, 1=max relevance).
        
        Returns:
            List of diverse search results.
        """
        # First, get more candidates
        candidates = self.similarity_search(
            query=query,
            department=department,
            k=fetch_k,
            include_embeddings=True
        )
        
        if not candidates or len(candidates) <= k:
            return [
                {k: v for k, v in c.items() if k != "embedding"}
                for c in candidates
            ]
        
        # Get query embedding
        query_embedding = self._embedding_fn([query])[0]
        
        # MMR selection
        import numpy as np
        
        selected = []
        candidate_embeddings = [c.get("embedding", []) for c in candidates]
        
        while len(selected) < k and candidates:
            # Calculate MMR scores
            mmr_scores = []
            
            for i, candidate in enumerate(candidates):
                if i in [s["idx"] for s in selected]:
                    continue
                
                # Relevance to query
                query_sim = candidates[i]["score"]
                
                # Max similarity to already selected
                max_selected_sim = 0
                if selected:
                    for sel in selected:
                        if candidate_embeddings[i] and candidate_embeddings[sel["idx"]]:
                            sim = np.dot(
                                candidate_embeddings[i],
                                candidate_embeddings[sel["idx"]]
                            )
                            max_selected_sim = max(max_selected_sim, sim)
                
                # MMR score
                mmr = lambda_mult * query_sim - (1 - lambda_mult) * max_selected_sim
                mmr_scores.append((i, mmr))
            
            if not mmr_scores:
                break
            
            # Select highest MMR
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected.append({"idx": best_idx, "candidate": candidates[best_idx]})
        
        # Return results without embeddings
        return [
            {k: v for k, v in s["candidate"].items() if k != "embedding"}
            for s in selected
        ]
    
    def delete_collection(self, department: str) -> bool:
        """
        Delete a department's collection.
        
        Args:
            department: Department whose collection to delete.
        
        Returns:
            True if deleted, False if not found.
        """
        collection_name = self._get_collection_name(department)
        
        try:
            self._client.delete_collection(collection_name)
            
            if collection_name in self._collections:
                del self._collections[collection_name]
            
            logger.info("Collection deleted", collection=collection_name)
            return True
        
        except Exception as e:
            logger.warning(
                "Collection deletion failed",
                collection=collection_name,
                error=str(e)
            )
            return False
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all collections with their metadata.
        
        Returns:
            List of collection information dicts.
        """
        collections = []
        
        for collection in self._client.list_collections():
            coll = self._client.get_collection(
                collection.name,
                embedding_function=self._embedding_fn
            )
            collections.append({
                "name": collection.name,
                "collection_key": collection.name.replace(f"{self.collection_prefix}_", ""),
                "count": coll.count(),
                "metadata": collection.metadata
            })
        
        return collections
    
    def get_collection_stats(self, department: str) -> Dict[str, Any]:
        """
        Get statistics for a department's collection.
        
        Args:
            department: Department name.
        
        Returns:
            Dict with collection statistics.
        """
        collection = self.init_collection(department)
        
        return {
            "department": department,
            "collection_name": collection.name,
            "document_count": collection.count(),
            "metadata": collection.metadata
        }
    
    def clear_all(self) -> None:
        """Delete all collections (use with caution)."""
        for collection in self._client.list_collections():
            self._client.delete_collection(collection.name)
        
        self._collections.clear()
        logger.warning("All collections cleared")


if __name__ == "__main__":
    # Test vector store
    from rag.document_loader import Document
    
    print("Initializing vector store...")
    store = ChromaVectorStore(persist_dir="data/test_chroma")
    
    # Create test documents
    docs = [
        Document(
            content="The assembly robot requires quarterly lubrication of all joints. "
                   "Use industrial-grade grease meeting ISO VG 100 specifications.",
            metadata={"source": "maintenance_manual.txt", "page": 1}
        ),
        Document(
            content="When temperature exceeds 85 degrees Celsius, the safety system "
                   "automatically reduces robot speed by 50% to prevent overheating.",
            metadata={"source": "safety_guide.txt", "page": 5}
        ),
        Document(
            content="Production line A achieved 98.5% efficiency last quarter. "
                   "The main bottleneck was identified at the packaging station.",
            metadata={"source": "production_report.txt", "page": 2}
        ),
    ]
    
    # Add documents
    print("\nAdding documents to 'maintenance' collection...")
    result = store.add_documents(docs[:2], "maintenance")
    print(f"Added {result['added']} documents")
    
    print("\nAdding documents to 'production' collection...")
    result = store.add_documents(docs[2:], "production")
    print(f"Added {result['added']} documents")
    
    # Search
    print("\nSearching for 'robot maintenance'...")
    results = store.similarity_search(
        query="robot maintenance lubrication",
        department="maintenance",
        k=3
    )
    
    for i, r in enumerate(results):
        print(f"\nResult {i + 1}:")
        print(f"  Score: {r['score']}")
        print(f"  Content: {r['content'][:100]}...")
    
    # List collections
    print("\nCollections:")
    for coll in store.list_collections():
        print(f"  {coll['name']}: {coll['count']} documents")
    
    # Cleanup
    store.clear_all()
    print("\nTest completed!")
