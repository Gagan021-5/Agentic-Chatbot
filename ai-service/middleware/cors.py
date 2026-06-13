"""
CORS middleware configuration for FastAPI.
"""

from fastapi.middleware.cors import CORSMiddleware

from config import get_settings


def add_cors_middleware(app) -> None:
    """Register CORSMiddleware with origins from REACT_ORIGIN / CORS_ORIGINS env."""
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
