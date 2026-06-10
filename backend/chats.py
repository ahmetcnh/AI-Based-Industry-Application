from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import structlog

from backend.db import get_pool
from backend.tenancy import get_tenant

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/chats", tags=["Chats"])


class ChatCreate(BaseModel):
    title: Optional[str] = None
    agent_id: Optional[int] = None


@router.get("", response_model=List[Dict[str, Any]])
async def list_chats(request: Request):
    tenant_id, user_id = get_tenant(request)
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, title, agent_id, created_at FROM chats "
            "WHERE tenant_id=$1 AND user_id=$2 ORDER BY created_at DESC",
            tenant_id, user_id,
        )
    return [dict(r) for r in rows]


@router.post("", response_model=Dict[str, Any])
async def create_chat(body: ChatCreate, request: Request):
    tenant_id, user_id = get_tenant(request)
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO chats (tenant_id, user_id, title, agent_id) "
            "VALUES ($1,$2,$3,$4) RETURNING id, title, agent_id, created_at",
            tenant_id, user_id, body.title, body.agent_id,
        )
    return dict(row)


@router.patch("/{chat_id}", response_model=Dict[str, Any])
async def update_chat_title(chat_id: int, title: str, request: Request):
    tenant_id, user_id = get_tenant(request)
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "UPDATE chats SET title=$1 WHERE id=$2 AND tenant_id=$3 AND user_id=$4 RETURNING *",
            title, chat_id, tenant_id, user_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Chat not found")
    return dict(row)


@router.delete("/{chat_id}")
async def delete_chat(chat_id: int, request: Request):
    tenant_id, user_id = get_tenant(request)
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM chats WHERE id=$1 AND tenant_id=$2 AND user_id=$3",
            chat_id, tenant_id, user_id,
        )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"status": "deleted"}


@router.get("/{chat_id}/messages", response_model=List[Dict[str, Any]])
async def get_messages(chat_id: int, request: Request):
    tenant_id, user_id = get_tenant(request)
    pool = await get_pool()
    async with pool.acquire() as conn:
        chat = await conn.fetchval(
            "SELECT id FROM chats WHERE id=$1 AND tenant_id=$2 AND user_id=$3",
            chat_id, tenant_id, user_id,
        )
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        rows = await conn.fetch(
            "SELECT id, role, content, created_at FROM messages "
            "WHERE chat_id=$1 ORDER BY created_at",
            chat_id,
        )
    return [dict(r) for r in rows]
