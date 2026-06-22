"""
═══════════════════════════════════════════════════════════════
Retrieval Router — /retrieve-context endpoint
═══════════════════════════════════════════════════════════════
RAG context retrieval for FastAPI workflows.
Called at Step 0 (TRIAGE) to enrich session with knowledge base context.
"""

import time
import structlog
from fastapi import APIRouter, Request, HTTPException

from schemas.api_schemas import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievedDocument,
)

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/", response_model=RetrievalResponse)
async def retrieve_context(request: Request, body: RetrievalRequest):
    """Retrieve relevant documents from the knowledge base.

    Used by FastAPI workflows at:
    - Step 0 (TRIAGE): Enrich extraction with domain knowledge
    - Step 2 (PREVIEW): Get model-specific guidance before prompt generation
    """
    start = time.time()

    try:
        vector_store = request.app.state.vector_store
        redis = request.app.state.redis

        # Check cache first
        cache_key = redis._cache_key("retrieval", f"{body.query}:{','.join(body.categories)}")
        cached = await redis.get_cache(cache_key)
        if cached:
            logger.info("retrieval_cache_hit", query=body.query[:50])
            return RetrievalResponse(
                documents=[RetrievedDocument(**d) for d in cached["documents"]],
                query=body.query,
                total_found=cached["total_found"],
                processing_time_ms=round((time.time() - start) * 1000, 2),
            )

        # Build search query with context enrichment
        search_query = body.query
        if body.app_type:
            search_query = f"{body.app_type} app: {search_query}"
        if body.model_id:
            search_query = f"{search_query} (model: {body.model_id})"

        # Search across specified categories
        results = await vector_store.search(
            query=search_query,
            categories=body.categories,
            top_k=body.top_k,
            boost_gold_standards=body.boost_gold_standards,
        )

        # If model_id specified, also search model-specific docs
        if body.model_id:
            model_results = await vector_store.search_with_filter(
                query=search_query,
                category="models",
                model_id=body.model_id,
                top_k=3,
            )
            # Merge and deduplicate
            seen = {r["content"][:100] for r in results}
            for r in model_results:
                if r["content"][:100] not in seen:
                    results.append(r)
                    seen.add(r["content"][:100])

        # Filter by score threshold
        from config import get_settings
        settings = get_settings()
        results = [r for r in results if r["relevance_score"] >= settings.rag_score_threshold]

        documents = [
            RetrievedDocument(
                content=r["content"],
                source=r["source"],
                category=r["category"],
                relevance_score=r["relevance_score"],
                metadata=r.get("metadata", {}),
            )
            for r in results
        ]

        # Cache results
        cache_data = {
            "documents": [d.model_dump() for d in documents],
            "total_found": len(documents),
        }
        await redis.set_cache(cache_key, cache_data, settings.cache_ttl_seconds)

        processing_time = round((time.time() - start) * 1000, 2)
        logger.info(
            "retrieval_complete",
            query=body.query[:50],
            total=len(documents),
            time_ms=processing_time,
        )

        return RetrievalResponse(
            documents=documents,
            query=body.query,
            total_found=len(documents),
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error("retrieval_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")
