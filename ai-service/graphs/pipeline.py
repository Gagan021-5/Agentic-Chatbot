"""
═══════════════════════════════════════════════════════════════
LangGraph Workflow — RentPrompts AI Pipeline StateGraph
═══════════════════════════════════════════════════════════════

Multi-turn conversational agent pipeline for configuring,
optimizing, and registering Rapps.
"""

from __future__ import annotations
import re
import json
from datetime import datetime, timezone
from typing import TypedDict, Annotated, Any, Literal, Optional

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from loguru import logger

from data.models import MODELS
from services.extraction import (
    extract_requirements,
    triage_dynamic_context,
    generate_dynamic_context,
    build_dynamic_context_fallback,
)
from services.prompt_generation import generate_prompt_template, generate_seo

BUDGET_CHIP_OPTIONS = [
    "Free models only (0 coins)",
    "Low (< 5 coins)",
    "Medium (5 - 20 coins)",
    "Premium (> 20 coins)",
]


# ─── State Definition ──────────────────────────────────────

class PipelineState(TypedDict, total=False):
    """Full state flowing through the LangGraph pipeline."""
    session_id: str
    message: str
    history: Annotated[list, add_messages]

    # Active Application Scope Attributes
    app_type: Optional[str]
    app_purpose: Optional[str]
    extraction: dict[str, Any]
    deep_answers: dict[str, Any]
    dynamic_context: Optional[dict[str, Any]]

    # Conversation progress tracking
    current_step: int
    recommended_action: str
    reply: str
    response_payload: dict[str, Any]

    # Model attributes
    model_id: Optional[str]
    model_name: Optional[str]
    model_cost: Optional[float]
    model_guidance: Optional[str]

    # Retrieval and context
    rag_documents: list[dict[str, Any]]
    rag_context_injected: Optional[str]
    similar_apps: list[dict[str, Any]]
    optimization_notes: list[str]

    # Prompt outputs and registration status
    enhanced_system_prompt: Optional[str]
    enhanced_user_prompt: Optional[str]
    extracted_variables: list[dict[str, Any]]
    requirements_complete: bool
    preview_approved: bool
    cms_registered: bool
    form_confirmed: bool

    # Form states
    awaiting_deep_answer: bool
    current_deep_field: Optional[str]


# ─── Parser and Heuristics Helpers ──────────────────────────

def _normalize(msg: Any) -> str:
    return str(msg or "").strip()


def _lower(msg: Any) -> str:
    return str(msg or "").strip().lower()


def _parse_multi_select_payload(msg: Any) -> dict | None:
    text = _normalize(msg)
    if not text.lower().startswith("multi_select_form::"):
        return None
    try:
        payload = json.loads(text[len("multi_select_form::") :])
        if not payload or not isinstance(payload, dict):
            return None
        return payload
    except (json.JSONDecodeError, TypeError):
        return None


def _parse_selected_model_id(msg: Any, available_models: list | None) -> str | None:
    text = _normalize(msg).lower()
    if not text.startswith("select"):
        return None
    query = re.sub(r"^select\s+", "", text, flags=re.IGNORECASE).strip()
    if not query:
        return None
    models = available_models or []
    for model in models:
        model_id = str(model.get("id") or "").lower()
        name = str(model.get("name") or "").lower()
        if (model_id and model_id == query) or (name and name == query):
            return model.get("id")
    return query


def _parse_chip_app_type(msg: Any) -> str | None:
    v = _lower(msg)
    if v in ("text", "image", "audio", "video", "vision"):
        return v
    if v == "images":
        return "image"
    if any(s in v for s in ("image generator", "image app", "generate images or photos")):
        return "image"
    if any(s in v for s in ("video creator", "video app", "create videos or animations")):
        return "video"
    if any(s in v for s in ("text", "writing tool", "write text", "written", "content")):
        return "text"
    if any(s in v for s in ("audio generator", "audio app", "generate voice or music")):
        return "audio"
    if any(s in v for s in ("vision analyzer", "analyze image", "ocr app", "vision scan")):
        return "vision"
    return None


# ─── Node Implementations ──────────────────────────────────

def _get_clean_history(history: list) -> list[dict]:
    """Converts a mix of LangChain message objects and dictionaries into standard dictionaries,
    and prunes the history to only include the messages from the most recent app idea to prevent cross-contamination.
    """
    clean = []
    for h in (history or []):
        if not h:
            continue
        if hasattr(h, "type") and hasattr(h, "content"):
            role = "assistant" if h.type in ("ai", "assistant", "agent") else "user"
            content = h.content
        elif isinstance(h, dict):
            role = "assistant" if h.get("role") in ("agent", "assistant", "ai") else "user"
            content = h.get("content") or h.get("text") or ""
        else:
            role = "user"
            content = str(h)

        if not isinstance(content, str):
            try:
                content = json.dumps(content)
            except Exception:
                content = str(content)

        clean.append({"role": role, "content": content})

    # Prune history to only start from the last time the user described a new app idea
    start_idx = 0
    new_idea_patterns = (
        "i want to build", "i want to create", "i want to make", "i want to write", "i want to generate", "i want to design", "i want to develop",
        "let's build", "lets build", "let's create", "lets create", "let's make", "lets make",
        "create a", "create an", "build a", "build an", "make a", "make an", "generate a", "generate an", "write a", "write an", "design a", "design an",
        "i want a", "i want an", "i need a", "i need an", "make me a", "make me an", "create me a", "create me an"
    )
    for idx, msg in enumerate(clean):
        if msg["role"] == "user":
            content_lower = msg["content"].lower()
            if any(pat in content_lower for pat in new_idea_patterns):
                start_idx = idx

    return clean[start_idx:]


async def intent_classifier_node(state: PipelineState, config: dict) -> dict:
    """Classifies user intent and determines the next execution node."""
    app_state = config["configurable"]["app_state"]
    message = state.get("message", "")
    msg_clean = message.strip().lower()

    # 1. Quick regex intercepts for fast path
    if msg_clean in ("hi", "hello", "hey", "hola"):
        return {"recommended_action": "HANDLE_GREETING"}
    if msg_clean in ("approve", "publish", "approve app", "looks good", "confirm", "approve_app", "publish_app"):
        return {"recommended_action": "APPROVE"}
    if msg_clean.startswith("change:") or msg_clean.startswith("tweak"):
        return {"recommended_action": "EDIT_APP"}

    # Check model selection intent
    app_type = state.get("app_type", "text")
    candidates = MODELS.get(app_type, [])
    parsed_model = _parse_selected_model_id(message, candidates)
    if parsed_model:
        return {"recommended_action": "MODEL_SELECT", "model_id": parsed_model}

    # 2. Informational questions / off-topic heuristics
    question_prefixes = (
        "how does", "how do", "how is", "what is", "what are", "whats", "what's",
        "why does", "why do", "why is", "tell me about", "explain how", "explain what",
        "can you explain", "how can i", "how do i", "how to", "what model", "what models"
    )
    is_informational_question = msg_clean.startswith(question_prefixes) or (
        ("?" in msg_clean or msg_clean.startswith(("what", "how", "why", "explain")))
        and not any(phrase in msg_clean for phrase in (
            "i want to build", "i want to create", "i want to make", "i want to write", "i want to generate", "i want to design", "i want to develop", "i want to start",
            "let's build", "lets build", "let's create", "lets create", "let's make", "lets make",
            "create a", "create an", "build a", "build an", "make a", "make an", "generate a", "generate an", "write a", "write an", "design a", "design an",
            "i want a", "i want an", "i need a", "i need an", "make me a", "make me an", "create me a", "create me an"
        ))
    )
    if is_informational_question:
        return {"recommended_action": "HANDLE_OFF_TOPIC"}

    history = state.get("history", []) or []
    clean_hist = _get_clean_history(history)
    history_slice = [{"role": m["role"], "content": m["content"][:400]} for m in clean_hist[-8:]]

    system_prompt = """You are an intent classifier for a conversational AI App Builder.
Classify the user's message into one of these actions:
- "HANDLE_OFF_TOPIC": General question about AI, technology, programming, or general chit-chat.
- "HANDLE_GREETING": Hello, hi, greetings.
- "MODEL_SELECT": Selecting or switching an AI model (e.g. "let's use gpt-4o", "change to flux").
- "APPROVE": Approving the prompt preview or asking to publish.
- "EDIT_APP": User wants to edit or tweak the generated prompt template (e.g., "change the output to be shorter", "make it more creative").
- "BUILD": Providing information about the app they want to build (describing goals, type, inputs, audience).

Return JSON only:
{
  "action": "HANDLE_OFF_TOPIC|HANDLE_GREETING|MODEL_SELECT|APPROVE|EDIT_APP|BUILD",
  "model_id": "extracted model id if model select, else null"
}"""
    try:
        res = await app_state.llm.groq_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                *history_slice,
                {"role": "user", "content": message}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"}
        )
        if not res or not isinstance(res, dict):
            raise ValueError("Invalid LLM response for intent classification")
        content = res.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        parsed = json.loads(content)
        action = (parsed.get("action") or "BUILD").upper()
        # Ensure enums/actions are clean
        if action not in ("HANDLE_OFF_TOPIC", "HANDLE_GREETING", "MODEL_SELECT", "APPROVE", "EDIT_APP", "BUILD"):
            action = "BUILD"
        m_id = parsed.get("model_id")
        return {
            "recommended_action": action,
            "model_id": m_id if m_id else state.get("model_id")
        }
    except Exception as e:
        logger.warning(f"LLM intent classification failed, falling back to BUILD: {e}")
        return {"recommended_action": "BUILD"}


async def off_topic_responder_node(state: PipelineState, config: dict) -> PipelineState:
    """Answers informational queries using the RAG knowledge base."""
    app_state = config["configurable"]["app_state"]
    query = state.get("message", "")

    # 1. RAG retrieval
    rag_docs = []
    if hasattr(app_state, "vector_store") and app_state.vector_store:
        try:
            rag_results = await app_state.vector_store.search(
                query=query,
                categories=["models", "prompting", "marketplace", "examples", "seo"],
                top_k=3,
            )
            rag_docs = rag_results
        except Exception as e:
            logger.warning(f"RAG search failed in off-topic responder: {e}")

    context = "\n\n".join([d["content"] for d in rag_docs])

    # 2. LLM response
    system_prompt = f"""You are the RentPrompts Help Desk Assistant.
Answer the user's question friendly and naturally. Use the following retrieved knowledge base context to provide accurate answers about the platform, models, or prompt engineering where relevant:
{context}

Keep your answer relatively concise (1-3 paragraphs) and helpful.
Do not ask or prompt the user about their app setup in this message."""

    history = state.get("history", []) or []
    clean_hist = _get_clean_history(history)
    history_slice = [{"role": m["role"], "content": m["content"][:400]} for m in clean_hist[-8:]]

    try:
        res = await app_state.llm.groq_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                *history_slice,
                {"role": "user", "content": query}
            ],
            model="llama-3.1-8b-instant"
        )
        if not res or not isinstance(res, dict):
            raise ValueError("Invalid LLM response for off topic responder")
        reply = res.get("choices", [{}])[0].get("message", {}).get("content", "I am here to help you learn about AI models and our platform.")
    except Exception as e:
        reply = "I'm sorry, I'm having trouble answering your question right now. How can I help you build your app?"

    state["reply"] = reply
    state["response_payload"] = {
        "reply": reply,
        "uiType": None,
        "uiData": None,
    }
    state["rag_documents"] = rag_docs
    return state


async def greeting_node(state: PipelineState) -> PipelineState:
    """Welcomes the user and asks for app specifications."""
    reply = (
        "Hello! 👋 I'm your App Creator Assistant. I will guide you through designing your application.\n\n"
        "To get started, **what type of AI app would you like to build today?**"
    )
    state["reply"] = reply
    state["response_payload"] = {
        "reply": reply,
        "uiType": None,
        "uiData": None,
    }
    return state


async def ideation_node(state: PipelineState, config: dict) -> PipelineState:
    """Step 1: Scopes app type/purpose conversational elicitation.
    Gates progression until form variables are confirmed and budget is set.
    """
    app_state = config["configurable"]["app_state"]
    message = state.get("message", "")
    history = state.get("history", [])

    app_type = state.get("app_type")
    app_purpose = state.get("app_purpose")
    extraction = state.get("extraction") or {}
    deep_answers = state.get("deep_answers") or {}
    dynamic_context = state.get("dynamic_context") or {}
    form_confirmed = state.get("form_confirmed") or False

    # Check for direct format correction or chip type selection
    chip_type = _parse_chip_app_type(message)
    if chip_type:
        app_type = chip_type
        extraction["appType"] = chip_type

    # Awaiting a budget deep answer
    awaiting_deep = state.get("awaiting_deep_answer", False)
    current_deep_field = state.get("current_deep_field")
    if awaiting_deep and current_deep_field == "budgetPreference":
        if message in BUDGET_CHIP_OPTIONS or any(b in message.lower() for b in ["free", "low", "medium", "premium"]):
            deep_answers["budgetPreference"] = message
            extraction["budget"] = message
            state["awaiting_deep_answer"] = False
            state["current_deep_field"] = None

    # If the user has responded to a triage question, capture their answer in deep_answers
    # to prevent the triage LLM from asking the same question again.
    has_agent_turned = any(
        h.get("role") in ("agent", "assistant", "ai") if isinstance(h, dict)
        else (h.type in ("ai", "assistant", "agent") if hasattr(h, "type") else False)
        for h in history
    )
    if not form_confirmed and not awaiting_deep and has_agent_turned and message:
        if not message.lower().startswith("multi_select_form::") and not message.lower().startswith("select"):
            key = f"clarification_{len(deep_answers) + 1}"
            deep_answers[key] = message

    # Call LLM extraction on user's message
    latest_ext = await extract_requirements(app_state.llm, message, _get_clean_history(history))
    for k, v in latest_ext.items():
        if v is not None and v != "null":
            extraction[k] = v

    if extraction.get("appType") and extraction["appType"] != "null":
        app_type = extraction["appType"]
    if extraction.get("appPurpose") and len(extraction["appPurpose"]) > 5:
        app_purpose = extraction["appPurpose"]

    # Detect adaptive intent / change of app idea
    new_purpose = latest_ext.get("appPurpose")
    if new_purpose and new_purpose != "null" and app_purpose:
        old_words = set(re.findall(r"\w+", app_purpose.lower()))
        new_words = set(re.findall(r"\w+", new_purpose.lower()))
        important_old = {w for w in old_words if len(w) > 3}
        important_new = {w for w in new_words if len(w) > 3}
        if important_old and important_new and not (important_old & important_new):
            logger.info(f"New app idea detected: '{new_purpose}' vs old: '{app_purpose}'. Resetting elicitation state.")
            deep_answers = {}
            extraction = latest_ext
            dynamic_context = {}
            state["deep_answers"] = {}
            state["extraction"] = latest_ext
            state["dynamic_context"] = {}
            app_purpose = new_purpose
            state["app_purpose"] = new_purpose
            state["form_confirmed"] = False
            state["requirements_complete"] = False
            app_type = latest_ext.get("appType") if latest_ext.get("appType") != "null" else None
            state["app_type"] = app_type
            # Reset history to start fresh from the user message that introduced the new app idea
            history = [{"role": "user", "content": message}]
            state["pruned_history"] = history

    state["app_type"] = app_type
    state["app_purpose"] = app_purpose
    state["extraction"] = extraction
    state["deep_answers"] = deep_answers

    # Check if the user confirmed the form options
    payload = _parse_multi_select_payload(message)
    if payload:
        form_confirmed = True
        dynamic_context["options"] = payload.get("selectedOptions") or []
        dynamic_context["variables"] = [
            {
                "name": v.get("name"),
                "placeholder": v.get("placeholder") or "Enter details...",
                "test_value": v.get("value") or "",
            }
            for v in (payload.get("variables") or [])
            if isinstance(v, dict)
        ]
        extraction["keyFeatures"] = payload.get("selectedOptions") or []
        state["dynamic_context"] = dynamic_context
        state["form_confirmed"] = True

    if not form_confirmed:
        # Triage loop using triage_dynamic_context
        triage_res = await triage_dynamic_context(
            app_state.llm,
            app_type,
            app_purpose or message,
            "English",
            _get_clean_history(history),
            deep_answers
        )

        if triage_res.get("corrected_app_type") and triage_res["corrected_app_type"] != app_type:
            app_type = triage_res["corrected_app_type"]
            state["app_type"] = app_type
            extraction["appType"] = app_type

        if triage_res.get("status") == "needs_context":
            question = triage_res.get("question")
            suggested = triage_res.get("suggested_options")
            has_chips = isinstance(suggested, list) and len(suggested) >= 2

            state["reply"] = question
            state["response_payload"] = {
                "reply": question,
                "uiType": "chips" if has_chips else None,
                "uiData": {"options": suggested} if has_chips else None,
                "nextStep": 0,
            }
            state["requirements_complete"] = False
            state["current_step"] = 0
            return state

        # If triage status is ready, render the configuration form
        if triage_res.get("form"):
            dynamic_context = triage_res["form"]
        else:
            dynamic_context = await generate_dynamic_context(
                app_state.llm,
                app_type or "text",
                app_purpose or message,
                "English"
            )
        state["dynamic_context"] = dynamic_context

        # Present the options form
        reply = (
            "## 📋 Customize Your App Configuration\n\n"
            "I've generated a draft of key features and input fields based on our conversation.\n\n"
            "Verify or adjust the options below, then click **Confirm options**!"
        )
        state["reply"] = reply
        state["response_payload"] = {
            "reply": reply,
            "uiType": "multi_select_form",
            "uiData": {
                "options": dynamic_context.get("options") or [],
                "variables": dynamic_context.get("variables") or [],
            },
            "nextStep": 0,
        }
        state["requirements_complete"] = False
        state["current_step"] = 0
        return state

    # Form is confirmed: check budget Preference
    budget = deep_answers.get("budgetPreference") or extraction.get("budget")
    if not budget:
        state["awaiting_deep_answer"] = True
        state["current_deep_field"] = "budgetPreference"
        reply = (
            "Got everything I need to build your app! One last thing — "
            "**what's your budget per generation?** This helps me pick the right AI model."
        )
        state["reply"] = reply
        state["response_payload"] = {
            "reply": reply,
            "uiType": "chips",
            "uiData": {"options": BUDGET_CHIP_OPTIONS},
            "nextStep": 0,
        }
        state["requirements_complete"] = False
        state["current_step"] = 0
        return state

    # Everything is complete for ideation
    state["requirements_complete"] = True
    state["current_step"] = 1
    return state


async def model_selection_node(state: PipelineState, config: dict) -> PipelineState:
    """Step 2: Renders available models and handles selection."""
    message = state.get("message", "")
    app_type = state.get("app_type", "text")
    model_id = state.get("model_id")

    candidates = MODELS.get(app_type, [])

    # Try parsing selected model from message
    selected = _parse_selected_model_id(message, candidates)
    if selected:
        model_id = selected
        state["model_id"] = model_id
        for m in candidates:
            if m["id"] == model_id:
                state["model_name"] = m["name"]
                state["model_cost"] = float(m["cost"])
                break

    if not model_id:
        card_texts = []
        for m in candidates:
            card_texts.append(f"- **{m['name']}** (Cost: {m['cost']} coins) - *{m['desc']}*")

        options = "\n".join(card_texts)
        reply = (
            f"Requirements verified! For **{app_type}** apps, here are our recommended models:\n\n"
            f"{options}\n\n"
            "Please select a model by typing its name or choosing one to proceed."
        )

        state["reply"] = reply
        state["response_payload"] = {
            "reply": reply,
            "uiType": "models" if candidates else None,
            "uiData": {"models": candidates} if candidates else None,
            "nextStep": 1,
        }
        state["current_step"] = 1
    else:
        state["current_step"] = 2

    return state


async def app_preview_node(state: PipelineState, config: dict) -> PipelineState:
    """Step 3 & 4: Prompt template generation & live preview with [Variable] mappings."""
    app_state = config["configurable"]["app_state"]
    message = state.get("message", "")
    app_type = state.get("app_type", "text")
    app_purpose = state.get("app_purpose", "")
    model_id = state.get("model_id")
    deep_answers = state.get("deep_answers") or {}

    action = state.get("recommended_action")
    if action == "EDIT_APP" or message.lower().startswith("change:"):
        instruction = re.sub(r"^change:\s*", "", message, flags=re.I).strip()
        deep_answers["lastEditInstruction"] = instruction
        state["deep_answers"] = deep_answers
        state["preview_approved"] = False

    # Package a temporary session for prompt generation
    temp_session = {
        "appType": app_type,
        "modelId": model_id,
        "deepAnswers": deep_answers,
        "extraction": state.get("extraction") or {},
        "history": _get_clean_history(state.get("history") or []),
        "dynamicContext": state.get("dynamic_context") or {},
    }

    # Generate optimized system and user prompts using the [Variable] convention
    prompt_data = await generate_prompt_template(app_state.llm, temp_session)
    seo_data = await generate_seo(app_state.llm, temp_session)

    enhanced_system_prompt = prompt_data.get("systemPrompt") or f"You are a specialized AI assistant for {app_purpose}."
    enhanced_user_prompt = prompt_data.get("userPrompt") or f"Process input: [main_input]."

    state["enhanced_system_prompt"] = enhanced_system_prompt
    state["enhanced_user_prompt"] = enhanced_user_prompt
    state["prompt_data"] = prompt_data
    state["seo_data"] = seo_data

    # Extract variables from brackets (supporting letters, digits, underscores, dashes, spaces)
    combined_prompts = f"{enhanced_system_prompt}\n{enhanced_user_prompt}"
    var_pattern = re.compile(r"\[([a-zA-Z0-9_\s-]+)\]")
    found_vars = list(set(var_name.strip() for var_name in var_pattern.findall(combined_prompts)))

    # Get user-confirmed variables from dynamic_context to populate test_value
    confirmed_vars = (state.get("dynamic_context") or {}).get("variables") or []
    
    # Helper to normalize variable names for matching (ignoring spaces, underscores, dashes, case)
    def normalize_var_name(n: str) -> str:
        return re.sub(r"[\s_-]", "", str(n)).lower()

    confirmed_lookup = {}
    for cv in confirmed_vars:
        if isinstance(cv, dict) and cv.get("name"):
            confirmed_lookup[normalize_var_name(cv["name"])] = cv
        elif isinstance(cv, str):
            confirmed_lookup[normalize_var_name(cv)] = {"name": cv, "test_value": "", "placeholder": ""}

    variables = []
    for var_name in found_vars:
        if not var_name:
            continue
        norm = normalize_var_name(var_name)
        matched_cv = confirmed_lookup.get(norm)
        
        placeholder = ""
        test_value = ""
        display_name = var_name
        
        if matched_cv:
            placeholder = matched_cv.get("placeholder") or matched_cv.get("value") or ""
            test_value = matched_cv.get("test_value") or matched_cv.get("value") or ""
            display_name = matched_cv.get("name") or var_name
        
        if not placeholder:
            placeholder = _generate_placeholder(display_name, app_type)

        variables.append({
            "name": display_name,
            "placeholder": placeholder,
            "test_value": test_value,
        })

    state["extracted_variables"] = variables

    # Build the live preview payload
    reply = "Here is the generated configuration for your Rapp:"
    state["reply"] = reply

    ui_data = {
        "appName": seo_data.get("appName") or f"{app_purpose.title()[:30]} Creator",
        "appType": app_type,
        "appDescription": seo_data.get("appDescription") or f"AI app for {app_purpose}",
        "cost": state.get("model_cost") or 2.0,
        "systemPrompt": enhanced_system_prompt,
        "userPrompt": enhanced_user_prompt,
        "variables": variables,
        "variablesUsed": [v["name"] for v in variables],
        "acceptImageInput": bool(prompt_data.get("acceptImageInput")),
        "options": ["Approve App", "Edit App"],
    }

    state["response_payload"] = {
        "reply": reply,
        "uiType": "app_preview",
        "uiData": ui_data,
        "nextStep": 2,
        "coins": state.get("model_cost"),
    }

    state["current_step"] = 2
    return state


async def preview_and_registration_node(state: PipelineState, config: dict) -> PipelineState:
    """Step 5: Catalog registration on Payload CMS."""
    app_state = config["configurable"]["app_state"]
    action = state.get("recommended_action")
    app_type = state.get("app_type", "text")
    app_purpose = state.get("app_purpose", "")

    if action == "APPROVE" or state.get("preview_approved"):
        payload = {
            "appType": app_type,
            "modelId": state.get("model_id"),
            "costPerRun": state.get("model_cost") or 2.0,
            "systemPrompt": state.get("enhanced_system_prompt"),
            "userPrompt": state.get("enhanced_user_prompt"),
            "appName": (state.get("seo_data") or {}).get("appName") or f"{app_purpose.title()[:30]} Creator",
            "appDescription": (state.get("seo_data") or {}).get("appDescription") or f"AI app for {app_purpose}",
            "tags": (state.get("seo_data") or {}).get("tags") or [app_type, "automated"],
            "category": (state.get("seo_data") or {}).get("category") or "creative",
            "publishedAt": datetime.now(timezone.utc).isoformat(),
        }

        try:
            res = await app_state.cms.create_rapp(payload)
            state["cms_registered"] = True
            reply = (
                f"## 🎉 RAPP Registered Successfully!\n\n"
                f"Your application **\"{payload['appName']}\"** has been registered in the CMS catalog.\n\n"
                f"- **Model:** {state.get('model_id')}\n"
                f"- **Rapp ID:** {res.get('id', 'N/A')}\n"
                f"- **Status:** Live & Ready! 🚀"
            )
            state["reply"] = reply
            state["response_payload"] = {
                "reply": reply,
                "uiType": "text",
                "uiData": None,
                "nextStep": 3,
                "clearSession": True,
            }
        except Exception as e:
            logger.error(f"CMS registration failed: {e}")
            reply = "Rapp registration failed due to CMS connection error. Please try again."
            state["reply"] = reply
            state["response_payload"] = {
                "reply": reply,
                "uiType": "text",
                "uiData": None,
                "nextStep": 2,
            }
    return state


# ─── Routing Logic ──────────────────────────────────────────

def route_conditional_edge(state: PipelineState) -> str:
    """Routes initial intent classifier decisions."""
    action = state.get("recommended_action")
    if action == "HANDLE_OFF_TOPIC":
        return "off_topic_responder"
    if action == "HANDLE_GREETING":
        return "greeting"
    if action == "APPROVE":
        return "preview_and_registration"

    # Route by step
    step = state.get("current_step", 0)
    if step == 0:
        return "ideation"
    if step == 1:
        return "model_selection"
    if step == 2:
        return "app_preview"
    return "preview_and_registration"


def should_continue_from_ideation(state: PipelineState) -> Literal["model_selection", "end"]:
    """Gates continuation based on requirement completeness."""
    # Strict single-turn response gate: if payload was generated, stop graph execution immediately
    if state.get("response_payload"):
        return "end"
    if state.get("requirements_complete"):
        return "model_selection"
    return "end"


def should_continue_from_model(state: PipelineState) -> Literal["app_preview", "end"]:
    """Gates model choice requirements."""
    # Strict single-turn response gate: if payload was generated, stop graph execution immediately
    if state.get("response_payload"):
        return "end"
    if state.get("model_id"):
        return "app_preview"
    return "end"


# ─── Helper Functions ───────────────────────────────────────

def _generate_placeholder(name: str, app_type: str) -> str:
    name_lower = name.lower()
    placeholders = {
        "topic": "e.g., The future of AI in healthcare",
        "style": "e.g., photorealistic, anime, watercolor",
        "tone": "e.g., professional, casual, humorous",
        "language": "e.g., English, Hindi, Spanish",
        "audience": "e.g., college students, business executives",
    }
    for key, placeholder in placeholders.items():
        if key in name_lower:
            return placeholder
    return f"Enter {name.replace('_', ' ').title().lower()}"


# ─── Graph Builder ──────────────────────────────────────────

def build_pipeline_graph(vector_store=None) -> StateGraph:
    """Build the conversational StateGraph workflow."""
    graph = StateGraph(PipelineState)

    # Register Nodes
    graph.add_node("intent_classifier", intent_classifier_node)
    graph.add_node("off_topic_responder", off_topic_responder_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("ideation", ideation_node)
    graph.add_node("model_selection", model_selection_node)
    graph.add_node("app_preview", app_preview_node)
    graph.add_node("preview_and_registration", preview_and_registration_node)

    # Entry point
    graph.set_entry_point("intent_classifier")

    graph.add_conditional_edges(
        "intent_classifier",
        route_conditional_edge,
        {
            "off_topic_responder": "off_topic_responder",
            "greeting": "greeting",
            "ideation": "ideation",
            "model_selection": "model_selection",
            "app_preview": "app_preview",
            "preview_and_registration": "preview_and_registration",
        }
    )

    graph.add_conditional_edges(
        "ideation",
        should_continue_from_ideation,
        {
            "model_selection": "model_selection",
            "end": END,
        }
    )

    graph.add_conditional_edges(
        "model_selection",
        should_continue_from_model,
        {
            "app_preview": "app_preview",
            "end": END,
        }
    )

    graph.add_edge("app_preview", END)
    graph.add_edge("preview_and_registration", END)
    graph.add_edge("off_topic_responder", END)
    graph.add_edge("greeting", END)

    return graph.compile()


# Compile default graph
compiled_graph = build_pipeline_graph()


# ─── Unified Interface Endpoint route() ─────────────────────

async def route(session: dict, message: str, app_state: Any) -> dict:
    """Main entrypoint invoking the event-driven LangGraph pipeline.
    This serves as the drop-in replacement for step_router.route().
    """
    # Map raw session history (which has role='agent') to standard role='assistant' so LangGraph coercion doesn't fail
    raw_hist = session.get("history", []) or []
    cleaned_hist = []
    for h in raw_hist:
        if isinstance(h, dict):
            role = h.get("role", "user")
            if role == "agent":
                role = "assistant"
            cleaned_hist.append({**h, "role": role})
        else:
            cleaned_hist.append(h)

    # Prune history to only start from the last time the user described a new app idea
    start_idx = 0
    new_idea_patterns = (
        "i want to build", "i want to create", "i want to make", "i want to write", "i want to generate", "i want to design", "i want to develop",
        "let's build", "lets build", "let's create", "lets create", "let's make", "lets make",
        "create a", "create an", "build a", "build an", "make a", "make an", "generate a", "generate an", "write a", "write an", "design a", "design an",
        "i want a", "i want an", "i need a", "i need an", "make me a", "make me an", "create me a", "create me an"
    )
    for idx, msg in enumerate(cleaned_hist):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content_lower = str(msg.get("content") or "").lower()
            if any(pat in content_lower for pat in new_idea_patterns):
                start_idx = idx
        elif hasattr(msg, "type") and msg.type == "user":
            content_lower = str(getattr(msg, "content", "")).lower()
            if any(pat in content_lower for pat in new_idea_patterns):
                start_idx = idx

    if start_idx > 0:
        pruned = cleaned_hist[start_idx:]
        logger.info(f"Pruned stale conversation history (from {len(cleaned_hist)} to {len(pruned)} messages) based on new app intent pattern.")
        session["history"] = pruned
        cleaned_hist = pruned

    # Convert session to pipeline state structure
    initial_state: PipelineState = {
        "session_id": session.get("sessionId") or session.get("session_id") or "",
        "message": message,
        "history": cleaned_hist,
        "app_type": session.get("appType"),
        "app_purpose": session.get("appPurpose") or (session.get("extraction") or {}).get("appPurpose"),
        "extraction": session.get("extraction") or {},
        "deep_answers": session.get("deepAnswers") or {},
        "dynamic_context": session.get("dynamicContext"),
        "model_id": session.get("modelId"),
        "model_name": session.get("modelName"),
        "model_cost": session.get("modelCost"),
        "prompt_data": session.get("promptData") or {},
        "seo_data": session.get("seoData") or {},
        "current_step": session.get("step") or 0,
        "requirements_complete": session.get("requirements_complete") or False,
        "preview_approved": session.get("preview_approved") or False,
        "cms_registered": session.get("cms_registered") or False,
        "form_confirmed": session.get("formConfirmed") or False,
        "enhanced_system_prompt": session.get("enhanced_system_prompt") or (session.get("promptData") or {}).get("systemPrompt"),
        "enhanced_user_prompt": session.get("enhanced_user_prompt") or (session.get("promptData") or {}).get("userPrompt"),
        "extracted_variables": session.get("extracted_variables") or [],
        "rag_documents": session.get("rag_documents") or [],
        "rag_context_injected": session.get("rag_context_injected") or "",
        "model_guidance": session.get("model_guidance") or "",
        "optimization_notes": session.get("optimization_notes") or [],
        "similar_apps": session.get("similar_apps") or [],
        "awaiting_deep_answer": session.get("awaitingDeepAnswer") or False,
        "current_deep_field": session.get("currentDeepField"),
    }

    config = {"configurable": {"app_state": app_state}}
    # We compile with the active vector store if initialized
    graph = build_pipeline_graph(app_state.vector_store)
    final_state = await graph.ainvoke(initial_state, config=config)

    # Sync back the mutations to session map
    session["step"] = final_state.get("current_step", session.get("step", 0))
    session["appType"] = final_state.get("app_type")
    session["appPurpose"] = final_state.get("app_purpose")
    session["extraction"] = final_state.get("extraction") or {}
    session["deepAnswers"] = final_state.get("deep_answers") or {}
    session["dynamicContext"] = final_state.get("dynamic_context")
    if "pruned_history" in final_state:
        session["history"] = final_state["pruned_history"]
    session["modelId"] = final_state.get("model_id")
    session["modelName"] = final_state.get("model_name")
    session["modelCost"] = final_state.get("model_cost")
    session["promptData"] = final_state.get("prompt_data") or {}
    session["seoData"] = final_state.get("seo_data") or {}
    session["requirements_complete"] = final_state.get("requirements_complete") or False
    session["preview_approved"] = final_state.get("preview_approved") or False
    session["cms_registered"] = final_state.get("cms_registered") or False
    session["formConfirmed"] = final_state.get("form_confirmed") or False
    session["enhanced_system_prompt"] = final_state.get("enhanced_system_prompt")
    session["enhanced_user_prompt"] = final_state.get("enhanced_user_prompt")
    session["extracted_variables"] = final_state.get("extracted_variables")
    session["rag_documents"] = final_state.get("rag_documents")
    session["rag_context_injected"] = final_state.get("rag_context_injected")
    session["model_guidance"] = final_state.get("model_guidance")
    session["optimization_notes"] = final_state.get("optimization_notes")
    session["similar_apps"] = final_state.get("similar_apps")
    session["awaitingDeepAnswer"] = final_state.get("awaiting_deep_answer") or False
    session["currentDeepField"] = final_state.get("current_deep_field")

    # If the graph outputs systemPrompt and userPrompt, sync them to promptData
    if final_state.get("enhanced_system_prompt") or final_state.get("enhanced_user_prompt"):
        session["promptData"] = {
            "systemPrompt": final_state.get("enhanced_system_prompt"),
            "userPrompt": final_state.get("enhanced_user_prompt"),
            "acceptImageInput": bool((final_state.get("prompt_data") or {}).get("acceptImageInput")),
        }

    return final_state.get("response_payload") or {}
