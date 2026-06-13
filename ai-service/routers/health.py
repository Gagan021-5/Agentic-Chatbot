"""
Health Router — GET /health
"""

from loguru import logger
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from config import get_settings

router = APIRouter()
settings = get_settings()


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    redis: str = "unknown"
    cms: str = "unknown"
    services: dict[str, str] = Field(default_factory=dict)


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Ping Redis, Payload CMS, ChromaDB, and embeddings."""
    services: dict[str, str] = {}

    try:
        await request.app.state.redis.get_session("__health_check__")
        redis_status = "healthy"
    except Exception as exc:
        redis_status = f"unhealthy: {exc}"
    services["redis"] = redis_status

    try:
        cms_ok = await request.app.state.cms.health_check()
        cms_status = "healthy" if cms_ok else "unreachable"
    except Exception as exc:
        cms_status = f"unhealthy: {exc}"
    services["cms"] = cms_status

    try:
        vs = request.app.state.vector_store
        stats = vs.get_collection_stats()
        services["chromadb"] = f"healthy ({sum(stats.values())} docs)"
    except Exception as exc:
        services["chromadb"] = f"unhealthy: {exc}"

    try:
        from services.embedding_service import get_embedding_service

        emb = get_embedding_service()
        vec = emb.embed_text("health check")
        services["embeddings"] = f"healthy (dim={len(vec)})"
    except Exception as exc:
        services["embeddings"] = f"unhealthy: {exc}"

    core_ok = redis_status == "healthy" and cms_status == "healthy"
    overall = "healthy" if core_ok and all("healthy" in v for v in services.values()) else "degraded"
    if not core_ok and all("unhealthy" in v for v in (redis_status, cms_status)):
        overall = "unhealthy"

    logger.debug(f"health_check | status={overall} | services={services}")

    return HealthResponse(
        status=overall,
        version=settings.app_version,
        redis=redis_status,
        cms=cms_status,
        services=services,
    )
