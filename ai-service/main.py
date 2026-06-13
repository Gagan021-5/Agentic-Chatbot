"""
RentPrompts Backend - FastAPI sole entry point.
"""

import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from config import get_settings
from middleware.auth import AuthMiddleware
from middleware.cors import add_cors_middleware
from middleware.logging import RequestLoggingMiddleware
from routers import agent, chat, health, preview, prompt_optimizer, retrieval, web_research
from services.llm import LLMService
from services.payload_cms import PayloadCMSClient
from services.redis_client import RedisClient
from services.session import SessionService
from rag.vector_store import VectorStoreManager

settings = get_settings()

logger.remove()
logger.add(sys.stderr, level=settings.log_level.upper())


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting RentPrompts backend | env={settings.environment} | port={settings.port}")

    app.state.session = SessionService()
    app.state.cms = PayloadCMSClient()
    app.state.llm = LLMService()
    app.state.redis = RedisClient(
        url=settings.upstash_redis_rest_url,
        token=settings.upstash_redis_rest_token,
    )
    app.state.vector_store = VectorStoreManager(settings)
    await app.state.vector_store.initialize()

    logger.info("All services initialized")
    yield

    await app.state.session.close()
    await app.state.cms.close()
    await app.state.llm.close()
    await app.state.redis.close()
    logger.info("Shutdown complete")


app = FastAPI(
    title="RentPrompts Backend",
    description="FastAPI backend — agent chat, previews, RAG, CMS integration",
    version=settings.app_version,
    lifespan=lifespan,
)

add_cors_middleware(app)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(AuthMiddleware)

# React-facing routes
app.include_router(health.router, tags=["Health"])
app.include_router(agent.router, prefix="/api/agent", tags=["Agent"])
app.include_router(preview.router, prefix="/api", tags=["Preview"])

# Internal AI / LangGraph routes
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(retrieval.router, prefix="/retrieve-context", tags=["Retrieval"])
app.include_router(prompt_optimizer.router, prefix="/optimize-prompt", tags=["Prompt Optimization"])
app.include_router(web_research.router, prefix="/web-research", tags=["Web Research"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=not settings.is_production,
        log_level=settings.log_level.lower(),
    )
