from typing import Any, Dict, List
import structlog

logger = structlog.get_logger(__name__)

_encoder = None


def _get_encoder(model_name: str):
    global _encoder
    if _encoder is None:
        from sentence_transformers import CrossEncoder
        _encoder = CrossEncoder(model_name)
        logger.info("CrossEncoder loaded", model=model_name)
    return _encoder


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    model_name: str,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    if not candidates:
        return candidates
    try:
        encoder = _get_encoder(model_name)
        pairs = [(query, c["content"]) for c in candidates]
        scores = encoder.predict(pairs)
        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)
        return sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)[:top_k]
    except Exception as e:
        logger.warning("Reranking failed, returning original order", error=str(e))
        return candidates[:top_k]
