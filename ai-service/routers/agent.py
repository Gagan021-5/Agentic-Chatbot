
import time
import json
from loguru import logger
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import Any

router = APIRouter()



class AgentChatRequest(BaseModel):
    sessionId: str
    message: str


class AgentChatResponse(BaseModel):
    reply: str
    uiType: str | None = None
    uiData: dict[str, Any] | None = None
    step: int = 0
    coins: float | None = None
    confirm: dict[str, Any] | None = None


class HistoryResponse(BaseModel):
    history: list[dict[str, Any]] = Field(default_factory=list)



def format_user_display(message: str) -> str | None:
    """Format special payloads for display in chat history."""
    if message.lower().startswith("multi_select_form::"):
        return "✅ Confirmed app configuration"
    if message.lower().startswith("seo_publish::"):
        return "🚀 Published to marketplace"
    if message.lower().startswith("seo_draft::"):
        return "📋 Saved as draft"
    if message.lower().startswith("select "):
        return f"Selected: {message[7:]}"
    return None



@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(request: Request, body: AgentChatRequest):
    """Main chat endpoint for POST /api/agent/chat.

    This is the primary entry point React calls. It:
    1. Gets or creates a session from Redis
    2. Pushes user message to history
    3. Runs the orchestration pipeline (route)
    4. Pushes agent response to history
    5. Saves session back to Redis
    6. Returns the response in React's expected format
    """
    start = time.time()
    session_svc = request.app.state.session

    try:
        # 1. Get or create session
        session = await session_svc.get_or_create_session(body.sessionId)

        # Ensure history is a list
        if not isinstance(session.get("history"), list):
            session["history"] = []

        # 2. Push user message to history
        user_display = format_user_display(body.message)
        session["history"].append({
            "role": "user",
            "content": body.message,
            **({"displayContent": user_display} if user_display else {}),
        })

        # 3. Run the orchestration pipeline
        from graphs.orchestrator import route
        result = await route(session, body.message, request.app.state)

        # 4. Update session state
        session["step"] = result.get("nextStep", session.get("step", 0))

        # Push agent response to history
        session["history"].append({
            "role": "agent",
            "content": result.get("reply", ""),
            "uiType": result.get("uiType"),
            "uiData": result.get("uiData"),
        })

        # 5. Save or clear session
        if result.get("clearSession"):
            await session_svc.delete_session(body.sessionId)
        else:
            await session_svc.save_session(session)

        duration_ms = round((time.time() - start) * 1000, 2)
        logger.info(
            f"agent_chat | session={body.sessionId[:8]} | step={result.get('nextStep')} "
            f"| uiType={result.get('uiType')} | {duration_ms}ms"
        )

        # 6. Return in React's expected format
        return AgentChatResponse(
            reply=result.get("reply", "Something went wrong."),
            uiType=result.get("uiType"),
            uiData=result.get("uiData"),
            step=result.get("nextStep", 0),
            coins=result.get("coins"),
            confirm=result.get("confirm"),
        )

    except Exception as e:
        logger.exception("agent_chat error")
        raise HTTPException(status_code=500, detail=str(e))


# ─── GET /api/agent/history ────────────────────────────────

@router.get("/history", response_model=HistoryResponse)
async def agent_history(request: Request, sessionId: str):
    """Fetch chat history for GET /api/agent/history."""
    session_svc = request.app.state.session

    try:
        session = await session_svc.get_session(sessionId)
        if not session or not session.get("history"):
            return HistoryResponse(history=[])
        return HistoryResponse(history=session["history"])
    except Exception as e:
        logger.error(f"agent_history error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
