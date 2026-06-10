import json
from typing import Any, Dict, List
import structlog

logger = structlog.get_logger(__name__)

_ENTITY_PROMPT = """Extract named entities from this industrial text. Return a JSON object with key "entities" containing a list of objects with fields: "name" (string), "type" (one of: machine, component, error_code, alarm, procedure, part).

TEXT: {text}

JSON:"""


async def extract_entities(text: str, llm) -> List[Dict[str, Any]]:
    try:
        raw = await llm.generate(
            _ENTITY_PROMPT.format(text=text[:1000]),
            system_prompt="Extract entities. Return only JSON.",
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = "\n".join(l for l in raw.splitlines() if not l.startswith("```"))
        data = json.loads(raw)
        return data.get("entities", [])
    except Exception as e:
        logger.warning("Entity extraction failed", error=str(e))
        return []


async def ingest_chunk_to_graph(
    chunk_content: str,
    chunk_id: int,
    driver,
    llm,
) -> None:
    entities = await extract_entities(chunk_content, llm)
    if not entities:
        return
    with driver.session() as session:
        for entity in entities:
            name = entity.get("name", "").strip()
            etype = entity.get("type", "unknown")
            if not name:
                continue
            session.run(
                """
                MERGE (e:Entity {name: $name, type: $type})
                MERGE (c:Chunk {chunk_id: $chunk_id})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                name=name, type=etype, chunk_id=chunk_id,
            )
