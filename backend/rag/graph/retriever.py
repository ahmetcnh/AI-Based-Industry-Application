from typing import Any, Dict, List
import structlog

logger = structlog.get_logger(__name__)


async def get_context(
    question: str,
    sources: List[Dict[str, Any]],
    driver,
) -> str:
    if not driver or not sources:
        return ""
    chunk_ids = [s["id"] for s in sources[:5]]
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE c.chunk_id IN $ids
                WITH e, count(*) as freq
                ORDER BY freq DESC LIMIT 20
                RETURN e.name as name, e.type as type, freq
                """,
                ids=chunk_ids,
            )
            rows = result.data()
            if not rows:
                return ""
            lines = [f"{r['name']} ({r['type']}, mentioned {r['freq']}x)" for r in rows]
            return "Related entities:\n" + "\n".join(lines)
    except Exception as e:
        logger.warning("Graph context retrieval failed", error=str(e))
        return ""
