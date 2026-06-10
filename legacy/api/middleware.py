"""
Middleware Module for AI Industrial Platform API.

Provides CORS, request logging, session management, and error handling.
"""

import time
import uuid
from typing import Callable, Dict, Any
from datetime import datetime
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging all HTTP requests and responses.
    
    Logs request method, path, status code, and duration.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and log details."""
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        
        # Store request ID in state for access in handlers
        request.state.request_id = request_id
        
        # Get client info
        client_ip = request.client.host if request.client else "unknown"
        
        # Log request start
        start_time = time.time()
        
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=client_ip
        )
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log request completion
            logger.info(
                "Request completed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2)
            )
            
            # Add custom headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration * 1000:.2f}ms"
            
            return response
        
        except Exception as e:
            duration = time.time() - start_time
            
            logger.error(
                "Request failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                error=str(e),
                duration_ms=round(duration * 1000, 2)
            )
            raise


class SessionMiddleware(BaseHTTPMiddleware):
    """
    Middleware for session management.
    
    Extracts or generates session ID from headers/cookies.
    """
    
    SESSION_HEADER = "X-Session-ID"
    SESSION_COOKIE = "session_id"
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and manage session."""
        # Try to get session ID from header or cookie
        session_id = (
            request.headers.get(self.SESSION_HEADER) or
            request.cookies.get(self.SESSION_COOKIE)
        )
        
        # Generate new session ID if not provided
        if not session_id:
            session_id = f"session_{uuid.uuid4().hex[:16]}"
        
        # Store in request state
        request.state.session_id = session_id
        
        # Process request
        response = await call_next(request)
        
        # Set session ID in response header
        response.headers[self.SESSION_HEADER] = session_id
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple rate limiting middleware.
    
    Limits requests per client IP.
    """
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check rate limit before processing request."""
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old entries
        if client_ip in self.requests:
            self.requests[client_ip] = [
                t for t in self.requests[client_ip]
                if current_time - t < 60
            ]
        else:
            self.requests[client_ip] = []
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            logger.warning(
                "Rate limit exceeded",
                client_ip=client_ip,
                requests=len(self.requests[client_ip])
            )
            
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={
                    "status": "error",
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {self.requests_per_minute} requests per minute"
                }
            )
        
        # Record request
        self.requests[client_ip].append(current_time)
        
        return await call_next(request)


def setup_cors(app, origins: list) -> None:
    """
    Configure CORS middleware.
    
    Args:
        app: FastAPI application instance.
        origins: List of allowed origins.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time", "X-Session-ID"]
    )
    
    logger.info("CORS configured", origins=origins)


def setup_middleware(app, settings) -> None:
    """
    Set up all middleware for the application.
    
    Args:
        app: FastAPI application instance.
        settings: Application settings.
    """
    # CORS (must be first)
    setup_cors(app, settings.cors_origins_list)
    
    # Request logging
    app.add_middleware(RequestLoggingMiddleware)
    
    # Session management
    app.add_middleware(SessionMiddleware)
    
    # Rate limiting (optional, can be disabled in settings)
    if not settings.DEBUG:
        app.add_middleware(RateLimitMiddleware, requests_per_minute=120)
    
    logger.info("Middleware configured")


# Exception handlers
async def validation_exception_handler(request: Request, exc: Exception):
    """Handle validation errors."""
    from fastapi.responses import JSONResponse
    
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "error": "ValidationError",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    from fastapi.responses import JSONResponse
    
    logger.error(
        "Unhandled exception",
        error=str(exc),
        path=request.url.path,
        method=request.method
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": "InternalServerError",
            "detail": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )
