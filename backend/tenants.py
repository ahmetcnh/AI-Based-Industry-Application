from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import structlog

from backend.db import get_pool

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/tenants", tags=["Tenants"])


class TenantCreate(BaseModel):
    name: str


class UserCreate(BaseModel):
    name: str


@router.get("", response_model=List[Dict[str, Any]])
async def list_tenants():
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT id, name, created_at FROM tenants ORDER BY id")
    return [dict(r) for r in rows]


@router.post("", response_model=Dict[str, Any])
async def create_tenant(body: TenantCreate):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO tenants (name) VALUES ($1) RETURNING id, name, created_at",
            body.name,
        )
    return dict(row)


@router.get("/{tenant_id}/users", response_model=List[Dict[str, Any]])
async def list_users(tenant_id: int):
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, tenant_id, name, created_at FROM users WHERE tenant_id = $1 ORDER BY id",
            tenant_id,
        )
    return [dict(r) for r in rows]


@router.post("/{tenant_id}/users", response_model=Dict[str, Any])
async def create_user(tenant_id: int, body: UserCreate):
    pool = await get_pool()
    async with pool.acquire() as conn:
        tenant = await conn.fetchval("SELECT id FROM tenants WHERE id = $1", tenant_id)
        if not tenant:
            raise HTTPException(status_code=404, detail="Tenant not found")
        row = await conn.fetchrow(
            "INSERT INTO users (tenant_id, name) VALUES ($1, $2) RETURNING id, tenant_id, name, created_at",
            tenant_id, body.name,
        )
    return dict(row)


@router.get("/me", tags=["Tenants"])
async def me(request: Request):
    from backend.tenancy import get_tenant
    tenant_id, user_id = get_tenant(request)
    return {"tenant_id": tenant_id, "user_id": user_id}
