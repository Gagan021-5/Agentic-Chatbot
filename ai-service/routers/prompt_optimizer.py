"""
═══════════════════════════════════════════════════════════════
Prompt Optimizer Router — /optimize-prompt endpoint
═══════════════════════════════════════════════════════════════
Runs synchronously when prompt optimization is requested before prompt generation.
Runs the full LangGraph pipeline to enhance prompts with RAG context.
"""

import time
import structlog
from fastapi import APIRouter, Request, HTTPException

from schemas.api_schemas import (
    PromptOptimizeRequest,
    PromptOptimizeResponse,
    VariableSchema,
)
from graphs.pipeline import build_pipeline_graph, PipelineState

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/", response_model=PromptOptimizeResponse)
async def optimize_prompt(request: Request, body: PromptOptimizeRequest):
    """Run the LangGraph pipeline to enhance prompt generation with RAG context.

    Used at prompt-generation time before final prompt rendering.
    Callers should await this response (sync, blocking, <5s budget).

    Flow:
    1. RequirementAnalysis → validates inputs
    2. ModelSelection → loads model-specific hints
    3. Retrieval → RAG from ChromaDB (model docs, prompting guides, examples)
    4. PromptEngineering → merges context into optimization notes
    5. VariableExtraction → extracts $$variables
    6. Output → returns enhanced context
    """
    start = time.time()

    try:
        vector_store = request.app.state.vector_store
        redis = request.app.state.redis

        # Check cache
        cache_key = redis._cache_key(
            "prompt_opt",
            f"{body.app_type}:{body.app_purpose}:{body.model_id or 'none'}"
        )
        cached = await redis.get_cache(cache_key)
        if cached and not body.edit_instruction:
            logger.info("prompt_optimize_cache_hit", session_id=body.session_id)
            cached["processing_time_ms"] = round((time.time() - start) * 1000, 2)
            return PromptOptimizeResponse(**cached)

        # Build initial state
        initial_state: PipelineState = {
            "session_id": body.session_id,
            "message": body.app_purpose,
            "app_type": body.app_type,
            "app_purpose": body.app_purpose,
            "model_id": body.model_id,
            "extraction": {},
            "deep_answers": body.deep_answers,
            "history": [],
            "requirements_complete": True,
            "enhanced_system_prompt": body.existing_system_prompt,
            "enhanced_user_prompt": body.existing_user_prompt,
        }

        # Run LangGraph pipeline
        graph = build_pipeline_graph(vector_store)
        final_state = await graph.ainvoke(initial_state)

        # Build response
        variables = [
            VariableSchema(**v) for v in final_state.get("extracted_variables", [])
        ]

        response = PromptOptimizeResponse(
            enhanced_system_prompt=final_state.get("enhanced_system_prompt"),
            enhanced_user_prompt=final_state.get("enhanced_user_prompt"),
            rag_context_injected=final_state.get("rag_context_injected", ""),
            model_guidance=final_state.get("model_guidance"),
            similar_apps=final_state.get("similar_apps", []),
            optimization_notes=final_state.get("optimization_notes", []),
            variables=variables,
            processing_time_ms=round((time.time() - start) * 1000, 2),
        )

        # Cache (skip if edit instruction — those are unique)
        if not body.edit_instruction:
            from config import get_settings
            settings = get_settings()
            await redis.set_cache(
                cache_key,
                response.model_dump(),
                settings.cache_ttl_seconds,
            )

        logger.info(
            "prompt_optimize_complete",
            session_id=body.session_id,
            rag_docs=len(final_state.get("rag_documents", [])),
            time_ms=response.processing_time_ms,
        )

        return response

    except Exception as e:
        logger.error("prompt_optimize_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Prompt optimization failed: {str(e)}")
