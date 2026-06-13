"""
Auth middleware — Bearer token / X-API-Key verification on /api/* routes.
"""

from loguru import logger
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import get_settings

PUBLIC_PREFIXES = ("/health", "/docs", "/openapi.json", "/redoc", "/favicon.ico")


class AuthMiddleware(BaseHTTPMiddleware):
    """Verify auth on protected routes. Dev mode (auth_mode=none) skips all checks."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if any(path == p or path.startswith(p + "/") for p in PUBLIC_PREFIXES if p != "/health"):
            if path.startswith("/docs") or path.startswith("/redoc"):
                return await call_next(request)

        if path in ("/health", "/favicon.ico"):
            return await call_next(request)

        settings = get_settings()

        if settings.auth_mode == "none":
            return await call_next(request)

        # Only enforce on /api/* and internal AI routes when auth is enabled
        protected = path.startswith("/api/") or path.startswith("/chat/")
        if not protected:
            return await call_next(request)

        token = self._extract_token(request)
        if not token:
            logger.warning(f"auth_missing | {request.method} {path}")
            return JSONResponse(status_code=401, content={"error": "Missing Authorization header or X-API-Key"})

        if settings.auth_mode == "api_key":
            if token != settings.auth_secret_key:
                logger.warning(f"auth_invalid | {request.method} {path}")
                return JSONResponse(status_code=403, content={"error": "Invalid API key"})
        elif settings.auth_mode == "jwt":
            try:
                import jwt
                jwt.decode(token, settings.auth_secret_key, algorithms=["HS256"])
            except Exception as exc:
                logger.warning(f"auth_jwt_invalid | {path} | {exc}")
                return JSONResponse(status_code=403, content={"error": "Invalid or expired token"})

        return await call_next(request)

    @staticmethod
    def _extract_token(request: Request) -> str | None:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:].strip()
        api_key = request.headers.get("X-API-Key", "").strip()
        return api_key or None
