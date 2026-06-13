"""
Chat Router — LangGraph pipeline + SSE stream endpoint.
"""

import asyncio
import json
import time

import structlog
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from schemas.api_schemas import (
    ChatRequest,
    ChatResponse,
    RetrievedDocument,
    VariableSchema,
)
from graphs.pipeline import build_pipeline_graph, PipelineState

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/stream/{session_id}")
async def chat_stream(session_id: str, request: Request):
    """SSE stream — React EventSource connects directly to FastAPI.

    Emits session status events and keep-alive pings. Clients receive:
    - connected: initial handshake
    - session: current session snapshot (if exists)
    - ping: keep-alive every 15s
    """
    session_svc = request.app.state.session

    async def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'connected', 'session_id': session_id})}\n\n"

            session = await session_svc.get_session(session_id)
            if session:
                yield f"data: {json.dumps({'type': 'session', 'step': session.get('step', 0), 'appType': session.get('appType')})}\n\n"

            while True:
                if await request.is_disconnected():
                    break
                yield f"data: {json.dumps({'type': 'ping', 'ts': int(time.time())})}\n\n"
                await asyncio.sleep(15)
        except asyncio.CancelledError:
            logger.info("chat_stream_cancelled", session_id=session_id)
        except Exception as exc:
            logger.error("chat_stream_error", session_id=session_id, error=str(exc))
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    """Full LangGraph pipeline execution.

    This endpoint runs the complete workflow:
    START → RequirementAnalysis → ModelSelection → Retrieval → PromptEngineering → VariableExtraction → Output → END

    Use this when:
    - Complex multi-step reasoning is needed
    - Full RAG + web research should be combined
    - the caller wants the complete enhanced context in one call
    """
    start = time.time()

    try:
        vector_store = request.app.state.vector_store

        # Build initial state from request
        initial_state: PipelineState = {
            "session_id": body.session_id,
            "message": body.message,
            "app_type": body.app_type,
            "app_purpose": body.app_purpose,
            "model_id": body.model_id,
            "extraction": body.extraction or {},
            "deep_answers": body.deep_answers or {},
            "history": body.history,
            "requirements_complete": bool(body.app_purpose and body.app_type),
        }

        # Run LangGraph pipeline
        graph = build_pipeline_graph(vector_store)
        final_state = await graph.ainvoke(initial_state)

        # Build response
        documents = [
            RetrievedDocument(
                content=d["content"],
                source=d["source"],
                category=d["category"],
                relevance_score=d["relevance_score"],
                metadata=d.get("metadata", {}),
            )
            for d in final_state.get("rag_documents", [])
        ]

        variables = [
            VariableSchema(**v) for v in final_state.get("extracted_variables", [])
        ]

        enhanced_context = {
            "rag_context_injected": final_state.get("rag_context_injected", ""),
            "model_guidance": final_state.get("model_guidance", ""),
            "optimization_notes": final_state.get("optimization_notes", []),
            "similar_apps": final_state.get("similar_apps", []),
        }

        optimized_prompt = None
        if final_state.get("enhanced_system_prompt") or final_state.get("enhanced_user_prompt"):
            optimized_prompt = {
                "system_prompt": final_state.get("enhanced_system_prompt"),
                "user_prompt": final_state.get("enhanced_user_prompt"),
            }

        processing_time = round((time.time() - start) * 1000, 2)
        logger.info(
            "chat_pipeline_complete",
            session_id=body.session_id,
            rag_docs=len(documents),
            time_ms=processing_time,
        )

        return ChatResponse(
            session_id=body.session_id,
            enhanced_context=enhanced_context,
            retrieved_documents=documents,
            optimized_prompt=optimized_prompt,
            variables=variables,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error("chat_pipeline_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Chat pipeline failed: {str(e)}")
