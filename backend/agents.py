from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import structlog

from backend.db import get_pool
from backend.tenancy import get_tenant

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/agents", tags=["Agents"])


class AgentCreate(BaseModel):
    name: str
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    collection: Optional[str] = None
    type: str = "chatbot"
    flow_config: Optional[Dict[str, Any]] = None


class AgentUpdate(BaseModel):
    name: Optional[str] = None
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    collection: Optional[str] = None
    type: Optional[str] = None
    flow_config: Optional[Dict[str, Any]] = None


async def _embed_and_store(conn, agent_id: int, name: str, system_prompt: Optional[str]) -> None:
    try:
        from backend.deps import get_llm, get_router
        llm = get_llm()
        text = f"{name}. {(system_prompt or '')[:200]}"
        emb = await llm.get_embedding(text)
        emb_str = "[" + ",".join(str(v) for v in emb) + "]"
        await conn.execute(
            "UPDATE agents SET embedding=$1::vector WHERE id=$2", emb_str, agent_id
        )
        try:
            get_router().invalidate_cache()
        except RuntimeError:
            pass  # router not yet initialized (startup)
    except Exception as e:
        logger.warning("Agent embedding failed", error=str(e))


@router.get("", response_model=List[Dict[str, Any]])
async def list_agents(request: Request):
    tenant_id, user_id = get_tenant(request)
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, name, system_prompt, model, collection, type, created_at "
            "FROM agents WHERE tenant_id=$1 AND user_id=$2 ORDER BY id",
            tenant_id, user_id,
        )
    return [dict(r) for r in rows]


@router.post("", response_model=Dict[str, Any])
async def create_agent(body: AgentCreate, request: Request):
    import json
    tenant_id, user_id = get_tenant(request)
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO agents (tenant_id, user_id, name, system_prompt, model, collection, type, flow_config)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8) RETURNING id, name, system_prompt, model, collection, type, created_at""",
            tenant_id, user_id, body.name, body.system_prompt, body.model,
            body.collection, body.type,
            json.dumps(body.flow_config) if body.flow_config else None,
        )
        agent_id = row["id"]
        await _embed_and_store(conn, agent_id, body.name, body.system_prompt)
    return dict(row)


@router.get("/{agent_id}", response_model=Dict[str, Any])
async def get_agent(agent_id: int, request: Request):
    tenant_id, user_id = get_tenant(request)
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM agents WHERE id=$1 AND tenant_id=$2 AND user_id=$3",
            agent_id, tenant_id, user_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Agent not found")
    return dict(row)


@router.put("/{agent_id}", response_model=Dict[str, Any])
async def update_agent(agent_id: int, body: AgentUpdate, request: Request):
    tenant_id, user_id = get_tenant(request)
    pool = await get_pool()
    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT id FROM agents WHERE id=$1 AND tenant_id=$2 AND user_id=$3",
            agent_id, tenant_id, user_id,
        )
        if not existing:
            raise HTTPException(status_code=404, detail="Agent not found")
        updates = {k: v for k, v in body.model_dump().items() if v is not None}
        if not updates:
            return await get_agent(agent_id, request)
        set_clause = ", ".join(f"{k}=${i+2}" for i, k in enumerate(updates))
        values = list(updates.values())
        row = await conn.fetchrow(
            f"UPDATE agents SET {set_clause} WHERE id=$1 RETURNING *",
            agent_id, *values,
        )
        new_name = row["name"]
        new_prompt = row["system_prompt"]
        await _embed_and_store(conn, agent_id, new_name, new_prompt)
    return dict(row)


@router.delete("/{agent_id}")
async def delete_agent(agent_id: int, request: Request):
    tenant_id, user_id = get_tenant(request)
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM agents WHERE id=$1 AND tenant_id=$2 AND user_id=$3",
            agent_id, tenant_id, user_id,
        )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"status": "deleted", "id": agent_id}
