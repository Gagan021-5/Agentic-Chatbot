"""
Conversational State Machine using LangGraph.
Refactored from the legacy procedural step router.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, Literal, Optional, TypedDict

from fastapi import APIRouter, HTTPException, Request
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from loguru import logger
from pydantic import BaseModel

from data.models import MODELS
from services.intent_engine import get_agentic_decision
from services.requirement_router import OFF_TOPIC_RESPONSE
from services.step_router import (
    BUDGET_CHIP_OPTIONS,
    _detect_language_mode,
    _is_no,
    _is_yes,
    _normalize,
    _lower,
    _parse_chip_app_type,
    _parse_multi_select_payload,
    _parse_selected_model_id,
    _parse_selected_plan,
    _extract_budget_tier,
    _find_model,
)

# ─── STATE DEFINITION ────────────────────────────────────────

class ConversationState(TypedDict, total=False):
    """Unified conversational properties state schema for LangGraph."""
    session_id: str
    message: str
    history: Annotated[list, add_messages]
    
    # Active Application Scope Attributes
    app_type: Optional[str]
    app_purpose: Optional[str]
    extraction: Dict[str, Any]
    dynamic_context: Optional[Dict[str, Any]]
    deep_answers: Dict[str, Any]
    
    # Conversation progress tracking
    current_step: int
    recommended_action: str
    response_payload: Dict[str, Any]
    
    # Interview and Form state flags
    awaiting_confirmation: bool
    awaiting_prompt_tweak: bool
    awaiting_deep_answer: bool
    current_deep_field: Optional[str]
    triage_rounds: int
    form_confirmed: bool
    
    # Selected engine options
    selected_model_id: Optional[str]
    model_cost: Optional[float]
    model_name: Optional[str]
    
    # Compiled blueprints & SEO cards
    prompt_data: Optional[Dict[str, Any]]
    seo_data: Optional[Dict[str, Any]]
    clear_session: bool
    language_mode: str
    enterprise_signals: Optional[bool]
    user_type: Optional[str]
    is_pivot: bool
    decision_payload: Dict[str, Any]


# Helper utilities for state/session translation
def _session_to_state(session: dict, message: str) -> ConversationState:
    hist = []
    for m in session.get("history", []):
        role = m.get("role", "user")
        content = m.get("content", "")
        norm_role = "assistant" if role in ("agent", "assistant") else "user"
        hist.append({"role": norm_role, "content": content})
    
    return {
        "session_id": session.get("sessionId") or session.get("session_id") or "",
        "message": message,
        "history": hist,
        "app_type": session.get("appType"),
        "app_purpose": (session.get("extraction") or {}).get("appPurpose"),
        "extraction": session.get("extraction") or {},
        "dynamic_context": session.get("dynamicContext"),
        "deep_answers": session.get("deepAnswers") or {},
        "current_step": session.get("step") or 0,
        "recommended_action": "",
        "response_payload": {},
        "awaiting_confirmation": session.get("awaitingConfirmation") or False,
        "awaiting_prompt_tweak": session.get("awaitingPromptTweak") or False,
        "awaiting_deep_answer": session.get("awaitingDeepAnswer") or False,
        "current_deep_field": session.get("currentDeepField"),
        "triage_rounds": session.get("triageRounds") or 0,
        "form_confirmed": session.get("formConfirmed") or False,
        "selected_model_id": session.get("modelId") or session.get("selectedModelId"),
        "model_cost": session.get("modelCost"),
        "model_name": session.get("modelName"),
        "prompt_data": session.get("promptData"),
        "seo_data": session.get("seoData"),
        "clear_session": False,
        "language_mode": session.get("languageMode") or "English",
        "enterprise_signals": session.get("enterpriseSignals"),
        "user_type": session.get("userType"),
        "is_pivot": session.get("isPivot") or False,
        "decision_payload": {},
    }


def _state_to_session(state: ConversationState, session: dict) -> None:
    session["step"] = state.get("current_step", 0)
    session["appType"] = state.get("app_type")
    session["extraction"] = state.get("extraction") or {}
    session["dynamicContext"] = state.get("dynamic_context")
    session["deepAnswers"] = state.get("deep_answers") or {}
    session["awaitingConfirmation"] = state.get("awaiting_confirmation") or False
    session["awaitingPromptTweak"] = state.get("awaiting_prompt_tweak") or False
    session["awaitingDeepAnswer"] = state.get("awaiting_deep_answer") or False
    session["currentDeepField"] = state.get("current_deep_field")
    session["triageRounds"] = state.get("triage_rounds", 0)
    session["formConfirmed"] = state.get("form_confirmed") or False
    session["modelId"] = state.get("selected_model_id")
    session["modelCost"] = state.get("model_cost")
    session["modelName"] = state.get("model_name")
    session["promptData"] = state.get("prompt_data")
    session["seoData"] = state.get("seo_data")
    session["languageMode"] = state.get("language_mode", "English")
    session["enterpriseSignals"] = state.get("enterprise_signals")
    session["userType"] = state.get("user_type")
    session["isPivot"] = state.get("is_pivot") or False
    
    session_history = []
    for m in state.get("history", []):
        if isinstance(m, dict):
            role = m.get("role")
            content = m.get("content")
        else:
            role = m.type
            content = m.content
        norm_role = "agent" if role in ("assistant", "agent", "ai") else "user"
        session_history.append({"role": norm_role, "content": content})
    session["history"] = session_history


# ─── ISOLATED, ASYNC NODES ───────────────────────────────────

async def intent_classifier_node(state: ConversationState, config: dict) -> dict:
    """Classifies user intent, intercepts form posts, and determines next action."""
    app_state = config["configurable"]["app_state"]
    message = state.get("message", "")
    text = _normalize(message)
    msg = _lower(text)
    
    # 1. Check UI/Action Form Interceptors
    if text.lower().startswith("multi_select_form::"):
        return {"recommended_action": "PROCESS_FORM"}
    if text.startswith("SEO_PUBLISH::"):
        return {"recommended_action": "PUBLISH_APP"}
    if text.startswith("SEO_DRAFT::"):
        return {"recommended_action": "SAVE_DRAFT"}
    if state.get("current_step") in (2, 3) and msg == "edit app":
        return {"recommended_action": "INITIATE_TWEAK"}
        
    # 2. Invoke Intent Engine API
    temp_session = {}
    _state_to_session(state, temp_session)
    decision = await get_agentic_decision(app_state.llm, text, temp_session)
    action = decision.get("recommended_action") or "GATHER_REQUIREMENTS"
    
    # Extract dynamic classification corrections
    app_type = state.get("app_type")
    extraction = state.get("extraction") or {}
    if decision.get("app_type") and decision["app_type"] != app_type:
        app_type = decision["app_type"]
        extraction = {**extraction, "appType": app_type}
        
    return {
        "recommended_action": action,
        "app_type": app_type,
        "extraction": extraction,
        "decision_payload": decision,
    }


async def gather_requirements_node(state: ConversationState, config: dict) -> dict:
    """Triage/Elicitation node — handles dialog flow, dynamic questioning, and requirement analysis."""
    app_state = config["configurable"]["app_state"]
    text = state.get("message", "")
    temp_session = {}
    _state_to_session(state, temp_session)
    
    from services.step_router import _exec_gather_requirements
    result = await _exec_gather_requirements(temp_session, text, app_state)
    
    new_state = _session_to_state(temp_session, text)
    new_state["response_payload"] = result
    return new_state


async def process_form_node(state: ConversationState, config: dict) -> dict:
    """Interprets and saves options & variables received from the configuration form."""
    app_state = config["configurable"]["app_state"]
    text = state.get("message", "")
    temp_session = {}
    _state_to_session(state, temp_session)
    
    payload = _parse_multi_select_payload(text)
    if payload:
        if not temp_session.get("dynamicContext"):
            temp_session["dynamicContext"] = {}
        temp_session["dynamicContext"]["options"] = payload.get("selectedOptions") or []
        temp_session["dynamicContext"]["variables"] = [
            {
                "name": v.get("name"),
                "placeholder": v.get("placeholder") or "Enter details...",
                "test_value": v.get("value") or "",
            }
            for v in (payload.get("variables") or [])
            if isinstance(v, dict)
        ]
        temp_session["formConfirmed"] = True
        if not temp_session.get("extraction"):
            temp_session["extraction"] = {}
        temp_session["extraction"]["keyFeatures"] = payload.get("selectedOptions") or []
        
        budget = (temp_session.get("deepAnswers") or {}).get("budgetPreference") or (
            (temp_session.get("extraction") or {}).get("budget")
        )
        from services.step_router import _show_models
        if budget:
            result = await _show_models(temp_session, app_state)
        else:
            temp_session["currentDeepField"] = "budgetPreference"
            temp_session["awaitingDeepAnswer"] = True
            result = {
                "reply": "One last thing — **what's your budget per generation?**",
                "uiType": "chips",
                "uiData": {"options": BUDGET_CHIP_OPTIONS},
                "nextStep": 0,
                "coins": None,
            }
    else:
        result = {"reply": "Invalid form configuration.", "uiType": "text"}
        
    new_state = _session_to_state(temp_session, text)
    new_state["response_payload"] = result
    return new_state


async def handle_budget_node(state: ConversationState, config: dict) -> dict:
    """Saves user budget selections and moves the flow forward to model selection cards."""
    app_state = config["configurable"]["app_state"]
    text = state.get("message", "")
    temp_session = {}
    _state_to_session(state, temp_session)
    
    decision = state.get("decision_payload") or {}
    from services.step_router import _exec_handle_budget
    result = await _exec_handle_budget(temp_session, text, decision, app_state)
    
    new_state = _session_to_state(temp_session, text)
    new_state["response_payload"] = result
    return new_state


async def model_selection_node(state: ConversationState, config: dict) -> dict:
    """Ranks and renders available models based on user category and budget preference."""
    app_state = config["configurable"]["app_state"]
    temp_session = {}
    _state_to_session(state, temp_session)
    
    from services.step_router import _show_models
    result = await _show_models(temp_session, app_state)
    
    new_state = _session_to_state(temp_session, state["message"])
    new_state["response_payload"] = result
    return new_state


async def change_model_node(state: ConversationState, config: dict) -> dict:
    """Updates active engine configuration when a model swap request is made."""
    app_state = config["configurable"]["app_state"]
    text = state.get("message", "")
    temp_session = {}
    _state_to_session(state, temp_session)
    
    decision = state.get("decision_payload") or {}
    from services.step_router import _exec_change_model
    result = await _exec_change_model(temp_session, text, decision, app_state)
    
    new_state = _session_to_state(temp_session, text)
    new_state["response_payload"] = result
    return new_state


async def generate_preview_node(state: ConversationState, config: dict) -> dict:
    """Compiles prompts and SEO descriptions, serving the interactive preview card."""
    app_state = config["configurable"]["app_state"]
    text = state.get("message", "")
    temp_session = {}
    _state_to_session(state, temp_session)
    
    from services.step_router import _exec_generate_preview
    result = await _exec_generate_preview(temp_session, text, app_state)
    
    new_state = _session_to_state(temp_session, text)
    new_state["response_payload"] = result
    return new_state


async def render_form_node(state: ConversationState, config: dict) -> dict:
    """Renders the configuration form options and fields based on extraction."""
    app_state = config["configurable"]["app_state"]
    text = state.get("message", "")
    temp_session = {}
    _state_to_session(state, temp_session)
    
    from services.extraction import extract_requirements, _merge_extraction
    from services.step_router import _exec_render_form
    
    if not temp_session.get("history"):
        temp_session["history"] = []
    extraction = temp_session.get("extraction") or {}
    if not extraction.get("appPurpose"):
        ext = await extract_requirements(app_state.llm, text, temp_session["history"])
        temp_session["extraction"] = _merge_extraction(temp_session.get("extraction"), ext, text)
        
    result = await _exec_render_form(temp_session, app_state)
    
    new_state = _session_to_state(temp_session, text)
    new_state["response_payload"] = result
    return new_state



async def edit_app_node(state: ConversationState, config: dict) -> dict:
    """Applies tweaks and modifications to prompt configurations based on creator instructions."""
    app_state = config["configurable"]["app_state"]
    text = state.get("message", "")
    temp_session = {}
    _state_to_session(state, temp_session)
    
    decision = state.get("decision_payload") or {}
    from services.step_router import _exec_edit_app
    result = await _exec_edit_app(temp_session, text, decision, app_state)
    
    new_state = _session_to_state(temp_session, text)
    new_state["response_payload"] = result
    return new_state


async def initiate_tweak_node(state: ConversationState, config: dict) -> dict:
    """Prompts creators for editing instructions to adjust their generated prompt blueprint."""
    result = {
        "reply": (
            "I'm listening! 📝\n\n"
            "- **Tweak the prompt** — tell me what to change\n"
            "- **Switch the AI model** — pick a different engine\n"
            "- **Start fresh** — describe a completely new app idea\n\n"
            "What would you like to adjust?"
        ),
        "uiType": "text",
        "uiData": None,
        "nextStep": state.get("current_step", 2),
        "coins": state.get("model_cost"),
    }
    return {
        "awaiting_prompt_tweak": True,
        "response_payload": result,
    }


async def review_seo_node(state: ConversationState, config: dict) -> dict:
    """Renders final SEO name, tags, and description for validation prior to listing."""
    app_state = config["configurable"]["app_state"]
    temp_session = {}
    _state_to_session(state, temp_session)
    
    from services.step_router import _exec_review_seo
    result = await _exec_review_seo(temp_session, app_state)
    
    new_state = _session_to_state(temp_session, state["message"])
    new_state["response_payload"] = result
    return new_state


async def publish_app_node(state: ConversationState, config: dict) -> dict:
    """Pushes final configuration payload directly to the central marketplace CMS."""
    app_state = config["configurable"]["app_state"]
    text = state.get("message", "")
    temp_session = {}
    _state_to_session(state, temp_session)
    
    card_data = {}
    if text.startswith("SEO_PUBLISH::"):
        try:
            json_str = text[len("SEO_PUBLISH::"):]
            card_data = json.loads(json_str)
        except Exception:
            pass
            
    from services.step_router import _handle_seo_publish
    result = await _handle_seo_publish(temp_session, card_data, app_state)
    
    new_state = _session_to_state(temp_session, text)
    new_state["response_payload"] = result
    new_state["clear_session"] = result.get("clearSession", False)
    return new_state


async def save_draft_node(state: ConversationState, config: dict) -> dict:
    """Saves the scope details to Redis as a persistent draft and clears the workspace session."""
    text = state.get("message", "")
    temp_session = {}
    _state_to_session(state, temp_session)
    
    card_data = {}
    if text.startswith("SEO_DRAFT::"):
        try:
            json_str = text[len("SEO_DRAFT::"):]
            card_data = json.loads(json_str)
        except Exception:
            pass
            
    temp_session["seoData"] = {**(temp_session.get("seoData") or {}), **card_data}
    temp_session["status"] = "draft"
    
    app_name = (temp_session.get("seoData") or {}).get("appName") or "Your App"
    result = {
        "reply": f'## 📋 Draft Saved\n\n**"{app_name}"** saved. Resume anytime from your dashboard.',
        "uiType": "success",
        "uiData": {"appName": app_name, "status": "Draft"},
        "nextStep": 0,
        "coins": None,
        "clearSession": True,
    }
    
    new_state = _session_to_state(temp_session, text)
    new_state["response_payload"] = result
    new_state["clear_session"] = True
    return new_state


async def pivot_app_node(state: ConversationState, config: dict) -> dict:
    """Resets conversational steps and purpose metadata for major pivots in creator ideas."""
    app_state = config["configurable"]["app_state"]
    text = state.get("message", "")
    temp_session = {}
    _state_to_session(state, temp_session)
    
    decision = state.get("decision_payload") or {}
    from services.step_router import _exec_pivot_app
    result = await _exec_pivot_app(temp_session, text, decision, app_state)
    
    new_state = _session_to_state(temp_session, text)
    new_state["response_payload"] = result
    return new_state


async def handle_greeting_node(state: ConversationState, config: dict) -> dict:
    """Welcomes the user or transitions straight to requirements scoping."""
    app_state = config["configurable"]["app_state"]
    text = state.get("message", "")
    temp_session = {}
    _state_to_session(state, temp_session)
    
    if len(temp_session.get("history") or []) < 3:
        result = {
            "reply": (
                "Hey there! 👋 I'm your **RentPrompts App Architect** — I help you design, "
                "configure, and publish AI-powered apps in minutes.\n\n"
                "**Just describe your app idea** and I'll handle the rest!\n\n"
                "What would you like to build today?"
            ),
            "uiType": None,
            "uiData": None,
            "nextStep": temp_session.get("step", 0),
            "coins": None,
        }
        new_state = _session_to_state(temp_session, text)
        new_state["response_payload"] = result
        return new_state
    
    from services.step_router import _exec_gather_requirements
    result = await _exec_gather_requirements(temp_session, text, app_state)
    new_state = _session_to_state(temp_session, text)
    new_state["response_payload"] = result
    return new_state


async def handle_off_topic_node(state: ConversationState, config: dict) -> dict:
    """Alerts about unrelated developer questions or fallbacks to triage."""
    app_state = config["configurable"]["app_state"]
    text = state.get("message", "")
    decision = state.get("decision_payload") or {}
    
    if decision.get("confidence") != "low":
        return {"response_payload": OFF_TOPIC_RESPONSE}
    
    temp_session = {}
    _state_to_session(state, temp_session)
    from services.step_router import _exec_gather_requirements
    result = await _exec_gather_requirements(temp_session, text, app_state)
    
    new_state = _session_to_state(temp_session, text)
    new_state["response_payload"] = result
    return new_state


async def handle_violation_node(state: ConversationState, config: dict) -> dict:
    """Issues safety guideline warning."""
    result = {
        "reply": (
            "I can only help build apps that comply with RentPrompts' safety and content "
            "guidelines. Please suggest a different idea."
        ),
        "uiType": "text",
        "uiData": None,
        "nextStep": state.get("current_step", 0),
        "coins": None,
    }
    return {"response_payload": result}


async def handle_gibberish_node(state: ConversationState, config: dict) -> dict:
    """Returns fallback options when input cannot be resolved."""
    result = {
        "reply": "Hmm, I didn't quite catch that! 🤔 What type of output should your AI app generate?",
        "uiType": "chips",
        "uiData": {"options": ["Text", "Image", "Audio", "Video", "Vision"]},
        "nextStep": state.get("current_step", 0),
        "coins": None,
    }
    return {"response_payload": result}


# ─── CONDITIONAL ROUTING EDGE ────────────────────────────────

def route_conditional_edge(state: ConversationState) -> str:
    """Determines the target execution node based on the classified action."""
    action = state.get("recommended_action")
    
    # Priority structural intercepts override standard positional workflows
    if action == "HANDLE_OFF_TOPIC":
        return "handle_off_topic"
    if action == "HANDLE_VIOLATION":
        return "handle_violation"
    if action == "HANDLE_GIBBERISH":
        return "handle_gibberish"
    if action == "HANDLE_GREETING":
        return "handle_greeting"
    if action == "HANDLE_BUDGET":
        return "handle_budget"
    if action == "CHANGE_MODEL":
        return "change_model"
    if action == "PIVOT_APP":
        return "pivot_app"
    if action == "EDIT_APP":
        return "edit_app"
    if action == "RENDER_FORM":
        return "render_form"
    if action == "SHOW_MODEL_CARDS":
        return "show_model_cards"
    if action == "GENERATE_PREVIEW":
        return "generate_preview"
    if action == "REVIEW_SEO":
        return "review_seo"
    if action == "PUBLISH_APP":
        return "publish_app"
    if action == "SAVE_DRAFT":
        return "save_draft"
    if action == "INITIATE_TWEAK":
        return "initiate_tweak"
    if action == "PROCESS_FORM":
        return "process_form"
        
    return "gather_requirements"


# ─── GRAPH ASSEMBLY & COMPILATION ────────────────────────────

def build_orchestrator_graph() -> StateGraph:
    """Compiles the LangGraph state machine with all execution routes."""
    graph = StateGraph(ConversationState)
    
    # Register Classifier & Exec Nodes
    graph.add_node("intent_classifier", intent_classifier_node)
    graph.add_node("gather_requirements", gather_requirements_node)
    graph.add_node("process_form", process_form_node)
    graph.add_node("handle_budget", handle_budget_node)
    graph.add_node("show_model_cards", model_selection_node)
    graph.add_node("change_model", change_model_node)
    graph.add_node("generate_preview", generate_preview_node)
    graph.add_node("edit_app", edit_app_node)
    graph.add_node("initiate_tweak", initiate_tweak_node)
    graph.add_node("review_seo", review_seo_node)
    graph.add_node("publish_app", publish_app_node)
    graph.add_node("save_draft", save_draft_node)
    graph.add_node("pivot_app", pivot_app_node)
    graph.add_node("handle_greeting", handle_greeting_node)
    graph.add_node("handle_off_topic", handle_off_topic_node)
    graph.add_node("handle_violation", handle_violation_node)
    graph.add_node("handle_gibberish", handle_gibberish_node)
    graph.add_node("render_form", render_form_node)
    
    # Establish Entry point
    graph.set_entry_point("intent_classifier")
    
    # Connect intent classifier outputs via the routing conditional edge
    graph.add_conditional_edges(
        "intent_classifier",
        route_conditional_edge,
        {
            "gather_requirements": "gather_requirements",
            "process_form": "process_form",
            "handle_budget": "handle_budget",
            "show_model_cards": "show_model_cards",
            "change_model": "change_model",
            "generate_preview": "generate_preview",
            "edit_app": "edit_app",
            "initiate_tweak": "initiate_tweak",
            "review_seo": "review_seo",
            "publish_app": "publish_app",
            "save_draft": "save_draft",
            "pivot_app": "pivot_app",
            "handle_greeting": "handle_greeting",
            "handle_off_topic": "handle_off_topic",
            "handle_violation": "handle_violation",
            "handle_gibberish": "handle_gibberish",
            "render_form": "render_form",
        }
    )
    
    # Link execution routes back to END state
    graph.add_edge("gather_requirements", END)
    graph.add_edge("process_form", END)
    graph.add_edge("handle_budget", END)
    graph.add_edge("show_model_cards", END)
    graph.add_edge("change_model", END)
    graph.add_edge("generate_preview", END)
    graph.add_edge("edit_app", END)
    graph.add_edge("initiate_tweak", END)
    graph.add_edge("review_seo", END)
    graph.add_edge("publish_app", END)
    graph.add_edge("save_draft", END)
    graph.add_edge("pivot_app", END)
    graph.add_edge("handle_greeting", END)
    graph.add_edge("handle_off_topic", END)
    graph.add_edge("handle_violation", END)
    graph.add_edge("handle_gibberish", END)
    graph.add_edge("render_form", END)
    
    return graph.compile()


# Instance compile
compiled_graph = build_orchestrator_graph()


# ─── CORE ORCHESTRATOR ENTRYPOINT ────────────────────────────

async def route(session: dict, message: str, app_state: Any) -> dict:
    """Main orchestration entry point replacing legacy procedural route logic.

    Invokes the event-driven LangGraph state machine workflow.
    """
    initial_state = _session_to_state(session, message)
    config = {"configurable": {"app_state": app_state}}
    
    # Run state machine
    final_state = await compiled_graph.ainvoke(initial_state, config=config)
    
    # Sync updated properties back to Redis session representation
    _state_to_session(final_state, session)
    
    # Return calculated response payload matching React contract
    return final_state.get("response_payload") or {}


# ─── FASTAPI EXECUTION ROUTER REFERENCE ───────────────────────

# Pydantic Schemas for reference FastAPI routing
class AgentChatRequest(BaseModel):
    sessionId: str
    message: str

class AgentChatResponse(BaseModel):
    reply: str
    uiType: Optional[str] = None
    uiData: Optional[dict] = None
    step: int = 0
    coins: Optional[float] = None
    confirm: Optional[dict] = None

# Reference Endpoint implementation showcasing invocation in a production system
# router = APIRouter()
# @router.post("/chat", response_model=AgentChatResponse)
# async def agent_chat(request: Request, body: AgentChatRequest):
#     session_svc = request.app.state.session
#     session = await session_svc.get_or_create_session(body.sessionId)
#     if not isinstance(session.get("history"), list):
#         session["history"] = []
#     
#     # Push user statement to persistence history
#     session["history"].append({"role": "user", "content": body.message})
#     
#     # Invoke orchestrator wrapper calling LangGraph.ainvoke()
#     result = await route(session, body.message, request.app.state)
#     
#     # Persist session changes back to the state store
#     if result.get("clearSession"):
#         await session_svc.delete_session(body.sessionId)
#     else:
#         await session["history"].append({"role": "agent", "content": result.get("reply", ""), "uiType": result.get("uiType")})
#         await session_svc.save_session(session)
#         
#     return AgentChatResponse(
#         reply=result.get("reply", ""),
#         uiType=result.get("uiType"),
#         uiData=result.get("uiData"),
#         step=result.get("nextStep", session.get("step", 0)),
#         coins=result.get("coins"),
#     )
