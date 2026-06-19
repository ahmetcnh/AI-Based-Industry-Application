import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional
import asyncpg
import structlog

from backend.rag.retrieval.hybrid import hybrid_search
from backend.rag.retrieval.reranker import rerank
from backend.rag.retrieval.supplement import supplementary_search

logger = structlog.get_logger(__name__)

_DEFAULT_SYSTEM_ROLE = (
    "You are an expert assistant for the Coiltech industrial automation platform."
)

_ANSWER_PROMPT = """{system_role}
Use the context below to answer the question. If the context is insufficient, say so clearly.
Answer in {language}.
In your answer, reuse the exact technical terms, component names and signal names
that appear in the question and in the context (e.g. sensör, kablo, soket, valf,
modül, adres) — do not replace them with generic paraphrases.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def _build_context(sources: List[Dict[str, Any]]) -> str:
    if not sources:
        return "No relevant documents found."
    parts = []
    for i, s in enumerate(sources, 1):
        meta = s.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        src = meta.get("source", "unknown")
        parts.append(f"[{i}] {src}\n{s['content']}")
    return "\n\n---\n\n".join(parts)


async def run_query(
    pool: asyncpg.Pool,
    llm,
    *,
    question: str,
    collection: str,
    tenant_id: int = 1,
    user_id: int = 1,
    k: int = 10,
    alpha: float = 0.3,
    use_expansion: bool = False,
    reranker_model: Optional[str] = None,
    score_threshold: float = 0.3,
    language: str = "tr",
    graph_retriever=None,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    t0 = time.time()

    queries = [question]
    if use_expansion:
        from backend.rag.retrieval.query_expansion import expand_query
        queries = await expand_query(question, llm)

    embedding = await llm.get_embedding(queries[0])

    candidates: Dict[int, Dict[str, Any]] = {}
    for q in queries:
        q_emb = await llm.get_embedding(q) if q != queries[0] else embedding
        results = await hybrid_search(
            pool, query=q, query_embedding=q_emb,
            collection=collection, tenant_id=tenant_id, user_id=user_id,
            k=k, alpha=alpha,
        )
        for r in results:
            if r["id"] not in candidates:
                candidates[r["id"]] = r

    merged = sorted(candidates.values(), key=lambda x: x.get("final_score", 0), reverse=True)

    if reranker_model and merged:
        import asyncio
        loop = asyncio.get_event_loop()
        merged = await loop.run_in_executor(
            None, lambda: rerank(question, merged, reranker_model, k)
        )

    top_score = merged[0].get("rerank_score", merged[0].get("final_score", 0)) if merged else 0.0

    async with pool.acquire() as conn:
        supplementary = await supplementary_search(
            conn, query=question, collection=collection,
            tenant_id=tenant_id, user_id=user_id,
            top_score=top_score, threshold=score_threshold,
        )
    for s in supplementary:
        if s["id"] not in {r["id"] for r in merged}:
            merged.append(s)

    sources = merged[:k]

    graph_context = ""
    if graph_retriever:
        try:
            graph_context = await graph_retriever.get_context(question, sources)
        except Exception as e:
            logger.warning("Graph retrieval failed", error=str(e))

    context = _build_context(sources)
    if graph_context:
        context = f"GRAPH CONTEXT:\n{graph_context}\n\n---\n\n{context}"

    prompt = _ANSWER_PROMPT.format(
        system_role=system_prompt or _DEFAULT_SYSTEM_ROLE,
        context=context,
        question=question,
        language=language,
    )
    answer = await llm.generate(prompt, model=model)

    scores = [s.get("rerank_score", s.get("final_score", 0.0)) for s in sources[:3]]
    confidence = round(sum(scores) / len(scores), 3) if scores else 0.0

    duration_ms = int((time.time() - t0) * 1000)

    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO query_logs (tenant_id, user_id, query, collection, sources_count, confidence, duration_ms)
               VALUES ($1, $2, $3, $4, $5, $6, $7)""",
            tenant_id, user_id, question, collection, len(sources), confidence, duration_ms,
        )

    return {
        "answer": answer.strip(),
        "sources": sources,
        "confidence": confidence,
        "query": question,
        "collection": collection,
        "duration_ms": duration_ms,
    }


async def retrieve_context(
    pool: asyncpg.Pool,
    llm,
    *,
    question: str,
    collection: str,
    tenant_id: int = 1,
    user_id: int = 1,
    k: int = 5,
    alpha: float = 0.3,
    use_expansion: bool = False,
    reranker_model: Optional[str] = None,
    score_threshold: float = 0.3,
    language: str = "tr",
    graph_retriever=None,
    system_prompt: Optional[str] = None,
) -> tuple:
    """Returns (prompt, sources) without calling LLM."""
    queries = [question]
    if use_expansion:
        from backend.rag.retrieval.query_expansion import expand_query
        queries = await expand_query(question, llm)

    embedding = await llm.get_embedding(queries[0])

    candidates: Dict[int, Dict[str, Any]] = {}
    for q in queries:
        q_emb = await llm.get_embedding(q) if q != queries[0] else embedding
        results = await hybrid_search(
            pool, query=q, query_embedding=q_emb,
            collection=collection, tenant_id=tenant_id, user_id=user_id,
            k=k, alpha=alpha,
        )
        for r in results:
            if r["id"] not in candidates:
                candidates[r["id"]] = r

    merged = sorted(candidates.values(), key=lambda x: x.get("final_score", 0), reverse=True)

    if reranker_model and merged:
        import asyncio
        loop = asyncio.get_event_loop()
        merged = await loop.run_in_executor(
            None, lambda: rerank(question, merged, reranker_model, k)
        )

    top_score = merged[0].get("rerank_score", merged[0].get("final_score", 0)) if merged else 0.0

    async with pool.acquire() as conn:
        supplementary = await supplementary_search(
            conn, query=question, collection=collection,
            tenant_id=tenant_id, user_id=user_id,
            top_score=top_score, threshold=score_threshold,
        )
    for s in supplementary:
        if s["id"] not in {r["id"] for r in merged}:
            merged.append(s)

    sources = merged[:k]

    graph_context = ""
    if graph_retriever:
        try:
            graph_context = await graph_retriever.get_context(question, sources)
        except Exception as e:
            logger.warning("Graph retrieval failed", error=str(e))

    context = _build_context(sources)
    if graph_context:
        context = f"GRAPH CONTEXT:\n{graph_context}\n\n---\n\n{context}"

    prompt = _ANSWER_PROMPT.format(
        system_role=system_prompt or _DEFAULT_SYSTEM_ROLE,
        context=context,
        question=question,
        language=language,
    )
    return prompt, sources
