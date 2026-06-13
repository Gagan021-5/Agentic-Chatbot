"""
═══════════════════════════════════════════════════════════════
Web Research Router — /web-research endpoint
═══════════════════════════════════════════════════════════════
Agent-driven web search with summarization.
Called async (fire-and-forget) for model research, or sync for explicit requests.
"""

import time
import structlog
from fastapi import APIRouter, Request, HTTPException

from schemas.api_schemas import WebResearchRequest, WebResearchResponse
from tools.web_search import get_web_search_tool

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/", response_model=WebResearchResponse)
async def web_research(request: Request, body: WebResearchRequest):
    """Perform web research and return summarized results.

    The agent decides when web search is required. Common triggers:
    - Latest Flux documentation
    - Latest Imagen capabilities
    - Provider updates
    - Prompt engineering best practices

    Results are cached in Redis for 1 hour.
    """
    start = time.time()

    try:
        redis = request.app.state.redis

        # Check cache
        cache_key = redis._cache_key("web_research", body.query)
        cached = await redis.get_cache(cache_key)
        if cached:
            logger.info("web_research_cache_hit", query=body.query[:50])
            return WebResearchResponse(
                query=body.query,
                summary=cached["summary"],
                sources=cached.get("sources", []),
                raw_results=cached.get("raw_results", []),
                cached=True,
                processing_time_ms=round((time.time() - start) * 1000, 2),
            )

        # Perform search
        tool = get_web_search_tool()
        result = await tool.search_and_summarize(
            query=body.query,
            context=body.context,
            max_results=body.max_results,
        )

        # Cache results
        from config import get_settings
        settings = get_settings()
        await redis.set_cache(cache_key, result, settings.cache_ttl_seconds)

        processing_time = round((time.time() - start) * 1000, 2)
        logger.info(
            "web_research_complete",
            query=body.query[:50],
            sources=len(result.get("sources", [])),
            time_ms=processing_time,
        )

        return WebResearchResponse(
            query=body.query,
            summary=result["summary"],
            sources=result.get("sources", []),
            raw_results=result.get("raw_results", []),
            cached=False,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error("web_research_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Web research failed: {str(e)}")
