import asyncpg
import structlog
import pandas as pd
from backend.rag.store import content_hash, chunk_exists, insert_chunk

logger = structlog.get_logger(__name__)

COLLECTION = "service_maintenance"
TENANT_ID = 1
USER_ID = 1


async def seed_excel(
    pool: asyncpg.Pool,
    llm,
    excel_path: str,
    document_id: int,
) -> int:
    df = pd.read_excel(excel_path, sheet_name="Sheet1")
    df.columns = [c.strip() for c in df.columns]

    issue_col = df.columns[0]
    solution_col = df.columns[1]

    # Collect all rows first
    all_texts = []
    for _, row in df.iterrows():
        issue = str(row[issue_col] or "").strip()
        solution = str(row[solution_col] or "").strip()
        if issue and solution:
            all_texts.append(f"Sorun: {issue}\nÇözüm: {solution}")

    inserted = 0
    async with pool.acquire() as conn:
        # Filter already-stored
        new_texts = [t for t in all_texts if not await chunk_exists(conn, content_hash(t))]

        # Batch embed
        embeddings = await llm.get_embeddings(new_texts)

        for text, embedding in zip(new_texts, embeddings):
            await insert_chunk(
                conn,
                document_id=document_id,
                tenant_id=TENANT_ID,
                user_id=USER_ID,
                collection=COLLECTION,
                content=text,
                embedding=embedding,
                metadata={"source": excel_path, "type": "issue_solution"},
            )
            inserted += 1

    logger.info("Excel seed complete", inserted=inserted)
    return inserted


async def run_seed_if_needed(pool: asyncpg.Pool, llm, excel_path: str) -> None:
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM chunks WHERE collection = $1 AND tenant_id = $2",
            COLLECTION, TENANT_ID,
        )
        if count and count > 0:
            logger.info("Excel seed already done, skipping", existing_chunks=count)
            return

        row = await conn.fetchrow(
            """
            INSERT INTO documents (tenant_id, user_id, original_filename, rag_collection)
            VALUES ($1, $2, $3, $4) RETURNING id
            """,
            TENANT_ID, USER_ID, excel_path, COLLECTION,
        )
        document_id = row["id"]

    await seed_excel(pool, llm, excel_path, document_id)
