from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

logger = structlog.get_logger(__name__)


class TenancyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            tenant_id = int(request.headers.get("X-Tenant-ID", "1"))
        except ValueError:
            tenant_id = 1
        try:
            user_id = int(request.headers.get("X-User-ID", "1"))
        except ValueError:
            user_id = 1
        request.state.tenant_id = tenant_id
        request.state.user_id = user_id
        return await call_next(request)


def get_tenant(request: Request) -> tuple[int, int]:
    return getattr(request.state, "tenant_id", 1), getattr(request.state, "user_id", 1)
