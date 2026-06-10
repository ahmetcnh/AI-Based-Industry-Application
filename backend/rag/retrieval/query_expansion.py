import json
from typing import List
import structlog

logger = structlog.get_logger(__name__)

_EXPANSION_PROMPT = """Given the user query below, generate 2 alternative phrasings that mean the same thing but use different words. Respond with a JSON array of strings only.

Query: {query}

Example response: ["alternative phrasing 1", "alternative phrasing 2"]"""


async def expand_query(query: str, llm) -> List[str]:
    try:
        raw = await llm.generate(
            _EXPANSION_PROMPT.format(query=query),
            system_prompt="You expand search queries. Reply only with a JSON array.",
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = "\n".join(l for l in raw.splitlines() if not l.startswith("```"))
        variants = json.loads(raw)
        if isinstance(variants, list):
            return [query] + [v for v in variants if isinstance(v, str)]
    except Exception as e:
        logger.warning("Query expansion failed, using original", error=str(e))
    return [query]
