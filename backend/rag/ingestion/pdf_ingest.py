import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import asyncpg
import structlog

logger = structlog.get_logger(__name__)


def _load_text(file_path: str) -> List[Dict[str, Any]]:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        return [
            {"content": (p.extract_text() or "").strip(), "page": i + 1}
            for i, p in enumerate(reader.pages)
            if (p.extract_text() or "").strip()
        ]
    if ext == ".docx":
        from docx import Document
        doc = Document(file_path)
        text = "\n\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
        return [{"content": text, "page": 1}]
    if ext in (".txt", ".md"):
        text = Path(file_path).read_text(encoding="utf-8", errors="ignore").strip()
        return [{"content": text, "page": 1}]
    if ext == ".xlsx":
        import pandas as pd
        xl = pd.ExcelFile(file_path)
        pages = []
        for sheet in xl.sheet_names:
            df = pd.read_excel(xl, sheet_name=sheet)
            rows = [
                " | ".join(f"{c}: {v}" for c, v in row.items())
                for _, row in df.iterrows()
            ]
            pages.append({"content": "\n".join(rows), "page": sheet})
        return pages
    raise ValueError(f"Unsupported extension: {ext}")


def _chunk_text(text: str, chunk_size: int = 2048, overlap: int = 256) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


async def ingest_file(
    pool: asyncpg.Pool,
    llm,
    *,
    file_path: str,
    collection: str,
    tenant_id: int = 1,
    user_id: int = 1,
    metadata: Optional[Dict[str, Any]] = None,
    mime_type: Optional[str] = None,
    file_size: int = 0,
) -> Dict[str, Any]:
    from backend.rag.store import content_hash, chunk_exists, insert_chunk

    metadata = metadata or {}
    pages = _load_text(file_path)

    async with pool.acquire() as conn:
        doc_row = await conn.fetchrow(
            """
            INSERT INTO documents
                (tenant_id, user_id, original_filename, rag_collection, metadata)
            VALUES ($1, $2, $3, $4, $5) RETURNING id
            """,
            tenant_id, user_id, os.path.basename(file_path),
            collection, json.dumps(metadata),
        )
        document_id = doc_row["id"]

        # Collect all new chunks first, then batch embed
        all_chunks = []
        for page in pages:
            for chunk_text in _chunk_text(page["content"]):
                all_chunks.append({"text": chunk_text, "page": page.get("page", 1)})

        chunks_created = len(all_chunks)
        # Filter already-stored chunks
        new_chunks = []
        for c in all_chunks:
            if not await chunk_exists(conn, content_hash(c["text"])):
                new_chunks.append(c)

        # Batch embed all new chunks in one call
        embeddings = await llm.get_embeddings([c["text"] for c in new_chunks])

        chunks_stored = 0
        for c, embedding in zip(new_chunks, embeddings):
            chunk_meta = {**metadata, "page": c["page"], "source": file_path}
            await insert_chunk(
                conn,
                document_id=document_id,
                tenant_id=tenant_id,
                user_id=user_id,
                collection=collection,
                content=c["text"],
                embedding=embedding,
                metadata=chunk_meta,
            )
            chunks_stored += 1

    return {
        "chunks_created": chunks_created,
        "chunks_stored": chunks_stored,
        "document_id": document_id,
        "collection": collection,
    }
