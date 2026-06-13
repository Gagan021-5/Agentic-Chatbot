"""
═══════════════════════════════════════════════════════════════
Chat Router — LangGraph pipeline + SSE stream endpoint.
═══════════════════════════════════════════════════════════════
"""

import asyncio
import json
import time
from typing import Optional

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
    """Conversational LangGraph pipeline execution entrypoint.

    Handles message-based turns, intent routing, dynamic state gating,
    and persistency logic with the Redis session cache.
    """
    start = time.time()

    try:
        session_svc = request.app.state.session

        # 1. Retrieve or initialize current session
        session = await session_svc.get_or_create_session(body.session_id)

        # 2. Append new user message to history
        if "history" not in session or not isinstance(session["history"], list):
            session["history"] = []
        session["history"].append({"role": "user", "content": body.message})

        # 3. Synchronize incoming state payload override overrides if supplied
        if body.app_type:
            session["appType"] = body.app_type
        if body.app_purpose:
            session["appPurpose"] = body.app_purpose
        if body.model_id:
            session["modelId"] = body.model_id

        # 4. Formulate graph state from session & request
        hist = []
        for m in session.get("history", []):
            role = m.get("role", "user")
            content = m.get("content", "")
            norm_role = "assistant" if role in ("agent", "assistant") else "user"
            hist.append({"role": norm_role, "content": content})

        initial_state: PipelineState = {
            "session_id": body.session_id,
            "message": body.message,
            "history": hist,
            "app_type": session.get("appType"),
            "app_purpose": session.get("appPurpose"),
            "model_id": session.get("modelId"),
            "extraction": session.get("extraction") or {},
            "deep_answers": session.get("deepAnswers") or {},
            "current_step": session.get("step") or 1,
            "requirements_complete": session.get("requirements_complete") or False,
            "preview_approved": session.get("preview_approved") or False,
            "cms_registered": session.get("cms_registered") or False,
            "enhanced_system_prompt": session.get("enhanced_system_prompt") or session.get("promptData", {}).get("systemPrompt"),
            "enhanced_user_prompt": session.get("enhanced_user_prompt") or session.get("promptData", {}).get("userPrompt"),
            "extracted_variables": session.get("extracted_variables") or [],
            "rag_documents": session.get("rag_documents") or [],
            "rag_context_injected": session.get("rag_context_injected") or "",
            "model_guidance": session.get("model_guidance") or "",
            "optimization_notes": session.get("optimization_notes") or [],
            "similar_apps": session.get("similar_apps") or [],
        }

        # 5. Invoke conversational state pipeline
        graph = build_pipeline_graph()
        final_state = await graph.ainvoke(
            initial_state,
            config={"configurable": {"app_state": request.app.state}}
        )

        # 6. Synchronize execution outcome back to persistence layer
        reply = final_state.get("reply", "")
        session["appType"] = final_state.get("app_type")
        session["appPurpose"] = final_state.get("app_purpose")
        session["modelId"] = final_state.get("model_id")
        session["extraction"] = final_state.get("extraction")
        session["deepAnswers"] = final_state.get("deep_answers")
        session["step"] = final_state.get("current_step") or 1
        session["requirements_complete"] = final_state.get("requirements_complete") or False
        session["preview_approved"] = final_state.get("preview_approved") or False
        session["cms_registered"] = final_state.get("cms_registered") or False
        session["enhanced_system_prompt"] = final_state.get("enhanced_system_prompt")
        session["enhanced_user_prompt"] = final_state.get("enhanced_user_prompt")
        session["extracted_variables"] = final_state.get("extracted_variables")
        session["rag_documents"] = final_state.get("rag_documents")
        session["rag_context_injected"] = final_state.get("rag_context_injected")
        session["model_guidance"] = final_state.get("model_guidance")
        session["optimization_notes"] = final_state.get("optimization_notes")
        session["similar_apps"] = final_state.get("similar_apps")

        session["history"].append({"role": "agent", "content": reply})
        await session_svc.save_session(session)

        # 7. Package response models
        documents = [
            RetrievedDocument(
                content=d["content"],
                source=d["source"],
                category=d["category"],
                relevance_score=d.get("relevance_score", 0.0),
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
            reply=reply,
            history=session["history"],
            enhanced_context=enhanced_context,
            retrieved_documents=documents,
            optimized_prompt=optimized_prompt,
            variables=variables,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error("chat_pipeline_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Chat pipeline failed: {str(e)}")
