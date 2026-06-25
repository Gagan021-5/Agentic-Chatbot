"""
Step router for the RentPrompts chat lifecycle.
Async route(session, message, app_state) -> dict matching the React response contract.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from datetime import datetime, timezone
from typing import Any, TypedDict, Annotated, Dict, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from loguru import logger

from data.mock_data import MOCK_DATA
from data.models import MODELS
from services.cost_calculator import build_budget_tiers
from services.extraction import (
    build_dynamic_context_fallback,
    extract_requirements,
    generate_dynamic_context,
    is_personal_boilerplate,
    _slot_is_captured,
)
from services.clarification_planner import (
    recover_app_purpose,
    build_known_information,
    plan_clarification,
    log_clarification_trace,
    clarification_is_complete,
    MAX_CLARIFICATION_TURNS,
    get_max_clarification_turns,
)
from services.intent_engine import get_agentic_decision
from services.artifact_utils import requires_input_artifact, is_creation_workflow
from services.prompt_generation import (
    apply_prompt_instruction,
    generate_prompt_template,
    generate_seo,
)
from services.requirement_router import OFF_TOPIC_RESPONSE
from tools.web_search import get_web_search_tool



COST_WARNING_THRESHOLD = 100

_PREVIEW_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

BUDGET_CHIP_OPTIONS = [
    "Free models only (0 coins)",
    "Low (< 5 coins)",
    "Medium (5 - 20 coins)",
    "Premium (> 20 coins)",
]


def _lower(msg: Any) -> str:
    return str(msg or "").strip().lower()


def _normalize(msg: Any) -> str:
    return str(msg or "").strip()


def _extract_budget_tier(text: str, decision: dict) -> str | None:
    t = str(text or "").lower()
    
    if "free" in t:
        return "free"
    if "low" in t or "< 5" in t or "under 5" in t:
        return "low"
    if "medium" in t or "5-20" in t or "5 - 20" in t:
        return "medium"
    if "premium" in t or "> 20" in t or "over 20" in t:
        return "premium"
    if "ultra" in t:
        return "ultra"
        
    dec_tier = decision.get("budget_tier") or (decision.get("extracted_variables") or {}).get("budgetPreference")
    if dec_tier:
        dt = str(dec_tier).lower()
        for tier in ("free", "low", "medium", "premium", "ultra"):
            if tier in dt:
                return tier
                
    return None


def _has_active_context(state: ConversationState) -> bool:
    extraction = state.get("extraction") or {}
    deep_answers = state.get("deep_answers") or {}
    return bool(
        state.get("app_type")
        and (
            state.get("form_confirmed")
            or (state.get("current_step") or 0) > 0
            or state.get("dynamic_context")
            or extraction.get("appPurpose")
            or deep_answers
        )
    )


def _session_has_budget(extraction: dict | None, deep_answers: dict | None) -> bool:
    ext = extraction or {}
    deep = deep_answers or {}
    return bool(
        ext.get("budget")
        or ext.get("budgetTier")
        or deep.get("budgetPreference")
        or deep.get("budgetTier")
    )


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


def _parse_selected_plan(msg: Any) -> str | None:
    match = re.match(r"^select\s+(lean|recommended|full)$", _normalize(msg), re.IGNORECASE)
    return match.group(1).lower() if match else None


def _parse_chip_app_type(msg: Any) -> str | None:
    if not msg:
        return None
    v = _lower(msg).strip()
    if v in ("text", "image", "audio", "video", "vision"):
        return "image" if v == "images" else v
    if re.search(r"\b(image|images|poster|posters|photo|photos|graphic|art)\b", v):
        return "image"
    if re.search(r"\b(image generator|image app|generate images or photos)\b", v):
        return "image"
    if re.search(r"\b(video creator|video app|create videos or animations)\b", v):
        return "video"
    if re.search(r"\b(text|writing tool|write text|written|content)\b", v):
        return "text"
    if re.search(r"\b(audio generator|audio app|generate voice or music)\b", v):
        return "audio"
    if re.search(r"\b(vision|image analyzer|analyze or understand images)\b", v):
        return "vision"
    if re.search(r"\bvideo\b", v):
        return "video"
    return None


def _should_rerun_triage_after_format_correction(session: dict, user_message: str) -> bool:
    if not session.get("dynamicContext") or session.get("step") != 0:
        return False
    t = _normalize(user_message)
    if not t or t.lower().startswith("multi_select_form::"):
        return False
    v = t.lower()
    correction_cue = bool(
        re.search(
            r"\b(no|nope|not that|wrong|actually|instead|change it|make it|switch to|"
            r"i want|i meant|correction|not a text|not text)\b",
            v,
            re.IGNORECASE,
        )
    )
    if not correction_cue:
        return False
    has_format_signal = _parse_chip_app_type(t) is not None or bool(
        re.search(
            r"\b(text|image|images|picture|pictures|visual|audio|sound|voice|video|vision|tiktok|reel|clip)\b",
            v,
            re.IGNORECASE,
        )
    )
    return has_format_signal


def _is_yes(msg: Any) -> bool:
    v = _lower(msg)
    return (
        v == "yes"
        or v == "yes, proceed"
        or "looks good" in v
        or "proceed" in v
        or "yes," in v
        or v == "confirm"
        or "sahi hai" in v
        or "haan" in v
        or "ha " in v
    )


def _is_no(msg: Any) -> bool:
    v = _lower(msg)
    return (
        v.startswith("no")
        or v.startswith("change:")
        or v == "nahi"
        or "let me change" in v
    )


def _infer_slot_key_from_question(question: str) -> str | None:
    q = question.lower()
    if any(w in q for w in ("type of image", "kind of image", "what images", "image type", "photos")):
        return "image_type"
    if any(w in q for w in ("background", "replace", "output style", "transparent", "white bg")):
        return "output_style"
    if any(w in q for w in ("platform", "where will", "used on", "deployed")):
        return "target_platform"
    if any(w in q for w in ("audience", "who will", "target user", "who is this for", "users be")):
        return "target_users"
    if any(w in q for w in ("resolution", "quality level", "high res", "image size")):
        return "quality_level"
    if any(w in q for w in ("batch", "multiple images", "bulk", "many images")):
        return "batch_support"
    if any(w in q for w in ("style", "aesthetic", "look and feel", "visual tone")):
        return "visual_style"
    if any(w in q for w in ("room", "space", "interior", "area", "which rooms")):
        return "room_type"
    if any(w in q for w in ("square", "footage", "size of", "dimensions", "how big")):
        return "room_dimensions"
    if any(w in q for w in ("feature", "functionality", "should it", "capabilities")):
        return "key_features"
    if any(w in q for w in ("purpose", "goal", "use case", "what will", "main thing")):
        return "use_case"
    words = re.findall(r'\b[a-z]{4,}\b', q)
    meaningful = [w for w in words if w not in (
        "what", "would", "like", "your", "want", "should", "will", "have", 
        "that", "this", "with", "from", "when", "where", "which", "about",
        "more", "just", "also", "only", "some", "them", "they", "does"
    )]
    return meaningful[0] if meaningful else None


def _detect_language_mode(session: dict) -> str:
    extraction = session.get("extraction") or {}
    lang = str(extraction.get("detectedLanguage") or session.get("languageMode") or "English").lower()
    if "hinglish" in lang:
        return "Hinglish"
    if "hindi" in lang:
        return "Hindi"
    return "English"


def _localized_text(session: dict, english: str, hindi: str = "", hinglish: str = "") -> str:
    mode = _detect_language_mode(session)
    if mode == "Hindi":
        return hindi or english
    if mode == "Hinglish":
        return hinglish or english
    return english


def _sanitize_accept_image_input(raw_value: Any, app_type: Any) -> bool:
    app_type_str = str(app_type or "").lower()
    if app_type_str in ("image", "vision"):
        return bool(raw_value)
    return False


def _find_model(app_type: str | None, model_id: str) -> dict | None:
    for model in MODELS.get(app_type or "", []):
        if model.get("id") == model_id:
            return model
    return None


async def _find_artistic_models_from_catalog(app_state: Any) -> list[str]:
    vector_store = getattr(app_state, "vector_store", None)
    if not vector_store:
        return []
    try:
        matches = await vector_store.search(
            query="artistic pro-edit creative cinematic editing drawing style",
            categories=["models"],
            top_k=5
        )
        artistic_models = []
        for match in matches:
            content = match.get("content", "")
            lines = content.split("\n")
            for line in lines:
                if line.startswith("## "):
                    model_name = line.replace("## ", "").strip()
                    if model_name not in artistic_models:
                        artistic_models.append(model_name)
        return artistic_models
    except Exception as e:
        logger.warning(f"Failed to find artistic models from RAG: {e}")
        return []


def _rank_models(
    available_models: list | None, 
    user_input: str, 
    budget_str: str, 
    artistic_priority: bool = False, 
    artistic_model_names: list[str] | None = None
) -> list:
    if not available_models:
        return []

    filtered = list(available_models)
    b = (budget_str or "").lower()
    
    if not artistic_priority and b:
        if any(x in b for x in ("medium", "5-20", "5 - 20")):
            filtered = [m for m in filtered if 5 <= m.get("cost", 0) <= 20]
        elif any(x in b for x in ("low", "under 5", "< 5")):
            filtered = [m for m in filtered if 0 < m.get("cost", 0) < 5]
        elif any(x in b for x in ("premium", "best", "> 20")):
            filtered = [m for m in filtered if m.get("cost", 0) >= 20]
        elif any(x in b for x in ("free", "0 coins")):
            filtered = [m for m in filtered if m.get("cost", 0) == 0]
        else:
            number_match = re.search(r"\d+(?:\.\d+)?", b)
            if number_match:
                limit = float(number_match.group(0))
                filtered = [m for m in filtered if m.get("cost", 0) <= limit]

    if not filtered:
        is_premium_intent = any(x in b for x in ("premium", "best", "> 20")) if not artistic_priority else True
        is_medium_intent = any(x in b for x in ("medium", "5-20", "5 - 20")) if not artistic_priority else False
        if is_premium_intent or is_medium_intent:
            return sorted(available_models, key=lambda m: m.get("cost", 0), reverse=True)[:3]
        return sorted(available_models, key=lambda m: m.get("cost", 0))[:3]

    input_lower = (user_input or "").lower()

    def score_model(model: dict) -> dict:
        score = 0
        model_id = str(model.get("id") or "").lower()
        model_name = str(model.get("name") or "").lower()
        
        if artistic_priority and artistic_model_names:
            for name in artistic_model_names:
                name_lower = name.lower()
                if name_lower in model_id or name_lower in model_name or model_id in name_lower or model_name in name_lower:
                    score += 50
                    
        model_tags = [str(t).lower() for t in (model.get("tags") or [])]
        if any(t in model_tags for t in ("artistic", "pro-edit", "creative", "detailed", "editing", "cinematic", "design")):
            if artistic_priority:
                score += 30
            else:
                score += 5
        
        for tag in model.get("tags") or []:
            if str(tag).lower() in input_lower:
                score += 5
                
        if any(w in input_lower for w in ("fast", "quick", "speed")):
            if model.get("tier") == "fast":
                score += 5
        if any(w in input_lower for w in ("quality", "best", "advanced")):
            if model.get("tier") in ("premium", "ultra"):
                score += 5
                
        return {**model, "_score": score}

    scored = [score_model(m) for m in filtered]
    scored.sort(key=lambda m: (-m["_score"], m.get("cost", 0)))
    return [{k: v for k, v in m.items() if k != "_score"} for m in scored[:3]]




def _apply_edit_to_session(session: dict, edit_instruction: str) -> None:
    instr = str(edit_instruction or "").strip().lower()
    if not instr:
        return

    extraction = session.get("extraction")
    if extraction:
        old_purpose = str(extraction.get("appPurpose") or "")
        if instr[:20] not in old_purpose.lower():
            extraction["appPurpose"] = f"{old_purpose} (updated: {edit_instruction.strip()})"
        extraction["oneLineUnderstanding"] = extraction.get("appPurpose")

    deep_answers = session.get("deepAnswers")
    if isinstance(deep_answers, dict):
        wants_transparent = bool(re.search(r"transparent|no background|remove background|no bg", edit_instruction, re.IGNORECASE))
        wants_new_background = bool(re.search(r"background|backdrop|scene|environment", edit_instruction, re.IGNORECASE))
        no_x_match = re.search(r"no\s+(\w+)", edit_instruction, re.IGNORECASE)

        keys_to_remove: list[str] = []
        for key, value in deep_answers.items():
            kl = key.lower()
            vl = str(value or "").lower()
            if wants_transparent and re.search(r"scene|background|backdrop|environment|location|setting", kl, re.IGNORECASE):
                keys_to_remove.append(key)
            elif wants_new_background and re.search(r"scene|background|backdrop|environment", kl, re.IGNORECASE):
                deep_answers[key] = edit_instruction.strip()
            elif no_x_match:
                rejected = no_x_match.group(1).lower()
                if rejected in vl or rejected in kl:
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            deep_answers.pop(key, None)

        if wants_transparent:
            deep_answers["outputType"] = "transparent PNG"
            deep_answers["backgroundType"] = "transparent"

    dynamic_context = session.get("dynamicContext") or {}
    variables = dynamic_context.get("variables")
    if variables and re.search(r"transparent|no background|remove background|no bg", edit_instruction, re.IGNORECASE):
        dynamic_context["variables"] = [
            v for v in variables if not re.search(r"scene|backdrop|environment|location|background_scene", str(v.get("name") if isinstance(v, dict) else v), re.IGNORECASE)
        ]


def prefill_dynamic_context_variables(session: dict) -> None:
    if not session.get("dynamicContext") or not isinstance(session["dynamicContext"], dict):
        session["dynamicContext"] = {"variables": [], "options": []}
    
    dynamic_context = session["dynamicContext"]
    if "variables" not in dynamic_context or not isinstance(dynamic_context["variables"], list):
        dynamic_context["variables"] = []
        
    variables = dynamic_context["variables"]
    extraction = session.get("extraction") or {}
    deep_answers = session.get("deepAnswers") or {}
    
    lookup = {}
    for k, v in extraction.items():
        if isinstance(v, str) and v.strip():
            lookup[k.lower().replace("_", " ")] = v.strip()
    for k, v in deep_answers.items():
        if not k.startswith("_") and isinstance(v, str) and v.strip():
            lookup[k.lower().replace("_", " ")] = v.strip()
            
    universal_mappings = {
        "subject": extraction.get("PRIMARY_SUBJECT"),
        "primary subject": extraction.get("PRIMARY_SUBJECT"),
        "setting": extraction.get("ENVIRONMENT_SETTING"),
        "environment setting": extraction.get("ENVIRONMENT_SETTING"),
        "environment": extraction.get("ENVIRONMENT_SETTING"),
        "action": extraction.get("ACTION_DYNAMIC"),
        "action dynamic": extraction.get("ACTION_DYNAMIC"),
        "style": extraction.get("AESTHETIC_STYLE"),
        "aesthetic style": extraction.get("AESTHETIC_STYLE"),
    }
    for k, v in universal_mappings.items():
        if v and isinstance(v, str) and v.strip():
            lookup[k] = v.strip()
            
    prefilled = []
    existing_names = set()
    bloat_patterns = [
        r"date\s*of\s*creation", r"jump\s*scare", r"video\s*title", r"age\s*rating",
        r"creation\s*date", r"scare\s*frequency", r"frequency"
    ]
    
    for var in variables:
        if not isinstance(var, dict):
            continue
        var_name = str(var.get("name") or "").strip()
        var_name_lower = var_name.lower()
        if any(re.search(pat, var_name_lower) for pat in bloat_patterns):
            continue
        if is_personal_boilerplate(var_name, extraction.get("appPurpose") or ""):
            continue
            
        prefilled_value = None
        if var_name_lower in lookup:
            prefilled_value = lookup[var_name_lower]
        else:
            for lookup_key, lookup_val in lookup.items():
                if lookup_key in var_name_lower or var_name_lower in lookup_key:
                    prefilled_value = lookup_val
                    break
                    
        var_copy = dict(var)
        var_copy["test_value"] = prefilled_value or var.get("test_value") or var.get("placeholder") or ""
        prefilled.append(var_copy)
        existing_names.add(var_name_lower)
        
    for dim_key, dim_name in [
        ("PRIMARY_SUBJECT", "Primary Subject"),
        ("ENVIRONMENT_SETTING", "Environment Setting"),
        ("ACTION_DYNAMIC", "Action Dynamic"),
        ("AESTHETIC_STYLE", "Aesthetic Style"),
    ]:
        val = extraction.get(dim_key)
        if val and isinstance(val, str) and val.strip() and dim_name.lower() not in existing_names:
            if not is_personal_boilerplate(dim_name, extraction.get("appPurpose") or ""):
                prefilled.append({
                    "name": dim_name,
                    "placeholder": f"Enter {dim_name.lower()}...",
                    "test_value": val.strip()
                })
                existing_names.add(dim_name.lower())
            
    dynamic_context["variables"] = prefilled[:4]
    session["dynamicContext"] = dynamic_context


def _build_capability_summary_state(app_type: str, ranked_models: list) -> str:
    if app_type == "audio":
        model_tags_flat = " ".join(
            " ".join(str(t).lower() for t in (m.get("tags") or []))
            for m in ranked_models
        )
        has_speech = any(w in model_tags_flat for w in ["tts", "voice", "speech", "narration"])
        has_music  = any(w in model_tags_flat for w in ["music", "song", "melody", "creative"])
        if has_speech and has_music:
            return "🎙 Guided voice narration + 🎵 ambient background music generation"
        if has_music:
            return "🎵 AI music and song generation"
        return "🎙 AI voice narration and speech synthesis"
    if app_type == "image":
        return "🖼 High-quality AI image generation"
    if app_type == "video":
        return "🎬 AI video and animation generation"
    if app_type == "vision":
        return "👁 AI image analysis and understanding"
    return "✍️ AI text and content generation"


async def _show_models_state(state: ConversationState, app_state: Any) -> dict:
    v_meta = state.get("verification_metadata") or {}
    ing_status = v_meta.get("ingestion_vector", "missing")

    # Skip ingestion vector question for pure generation apps
    # These apps generate content — they don't analyze uploaded files
    _current_app_type = state.get("app_type") or (state.get("extraction") or {}).get("appType") or "text"
    _wants_image = (state.get("extraction") or {}).get("wantsImageInput") or False
    _is_pure_generation = (
        _current_app_type in ("text", "audio")
        and not _wants_image
        and ing_status not in ("explicit",)
    )
    if _is_pure_generation:
        ing_status = "not_required"
        v_meta["ingestion_vector"] = "not_required"
        state["ingestion_vector"] = "plain_text"
        state["verification_metadata"] = v_meta

    budget_status = v_meta.get("budget", "missing")
    
    if ing_status in ("missing", "inferred"):
        app_purpose = state.get("app_purpose") or state.get("extraction", {}).get("appPurpose") or ""
        question, options = _get_ingestion_vector_question_and_chips(app_purpose)
        state["last_slot_key"] = "ingestion_vector"
        state["current_deep_field"] = "ingestion_vector"
        state["awaiting_deep_answer"] = True
        state["reply"] = question
        return {
            "reply": question,
            "uiType": "chips",
            "uiData": {"options": options},
            "nextStep": 0,
            "coins": None,
        }
        
    if budget_status in ("missing", "inferred"):
        state["current_deep_field"] = "budgetPreference"
        state["awaiting_deep_answer"] = True
        state["reply"] = "What is your budget preference per run?"
        return {
            "reply": "What is your budget preference per run?",
            "uiType": "chips",
            "uiData": {"options": BUDGET_CHIP_OPTIONS},
            "nextStep": 0,
            "coins": None,
        }

    # Double validation step ensuring dynamic context doesn't bypass catalog formats
    user_history = [h for h in (state.get("history") or []) if (h.get("role") if isinstance(h, dict) else h.type) in ("user", "human")]
    user_history_str = " ".join([str(h.get("content") if isinstance(h, dict) else h.content).lower() for h in user_history])
    extraction = state.get("extraction") or {}
    purpose_str = str(extraction.get("appPurpose") or "").lower()
    
    last_user_msg = str(user_history[-1].get("content") if isinstance(user_history[-1], dict) else user_history[-1].content).lower() if user_history else ""
    is_text_correction = "text" in last_user_msg and any(w in last_user_msg for w in ("it is", "i want", "use", "change to", "make it", "it's", "app", "tool"))
    
    if not is_text_correction and any(s in user_history_str or s in purpose_str for s in ("image", "images", "poster", "posters", "photo")):
        if state.get("app_type") not in ("audio", "video", "vision", "text") or is_text_correction:
            old_val = state.get("app_type")
            if old_val != "image":
                print(f"[MUTATION] appType: {old_val} -> image | file: step_router.py | function: _show_models_state | msg: {last_user_msg}")
            state["app_type"] = "image"
            state["extraction"]["appType"] = "image"

    app_type_str = state.get("app_type") or "text"
    full_text = " ".join([
        str(state.get("extraction", {}).get("appPurpose") or ""),
        str(state.get("extraction", {}).get("oneLineUnderstanding") or ""),
        json.dumps(state.get("deep_answers") or {}),
    ])
    budget = state.get("deep_answers", {}).get("budgetPreference") or state.get("extraction", {}).get("budget")
    model_collection = MODELS.get(app_type_str, MODELS.get("text", []))
    
    models = _rank_models(model_collection, full_text, str(budget or ""), artistic_priority=False)

    state["current_step"] = 1
    
    capability_summary = _build_capability_summary_state(app_type_str, models)

    display_models = []
    for m in models:
        display_models.append({
            **m,
            "displayName": m.get("name"),
            "name": m.get("name"),
        })

    payload = {
        "reply": (
            f"## 🤖 AI Engine Ready\n\n"
            f"Based on your app requirements, I've selected the best AI engine for your "
            f"**{app_type_str.capitalize()}** app.\n\n"
            f"**What will be generated:** {capability_summary}\n\n"
            f"Each card below shows the engine's strengths and cost per run — "
            f"**click any card** to confirm your selection."
        ),
        "uiType": "models",
        "uiData": {
            "appType": app_type_str,
            "models": display_models,
            "capabilitySummary": capability_summary,
        },
        "nextStep": 1,
        "coins": None,
    }
    state["reply"] = payload["reply"]
    return payload



MAX_CLARIFICATION_ROUNDS = MAX_CLARIFICATION_TURNS


async def _build_step0_response_state(state: ConversationState, text: str, app_state: Any) -> dict:
    llm = app_state.llm
    ext = state.get("extraction") or {}

    history_dicts = []
    for m in state.get("history") or []:
        role = "agent" if (m.get("role") if isinstance(m, dict) else m.type) in ("assistant", "agent") else "user"
        content = m.get("content") if isinstance(m, dict) else m.content
        history_dicts.append({"role": role, "content": content})

    app_purpose, recovered = recover_app_purpose(ext, text, history_dicts)
    if recovered and app_purpose:
        logger.info(f"[WORKFLOW] appPurpose recovered: {app_purpose}")
        if "extraction" not in state or state["extraction"] is None:
            state["extraction"] = {}
        state["extraction"]["appPurpose"] = app_purpose
        ext = state["extraction"]

    resolved = state.get("app_type") or "text"
    state["app_type"] = resolved
    if "extraction" not in state or state["extraction"] is None:
        state["extraction"] = {}
    state["extraction"]["appType"] = resolved

    state["current_step"] = 0

    triage_rounds = state.get("triage_rounds") or 0
    app_type = state.get("app_type") or "text"
    max_turns = get_max_clarification_turns(app_type)

    # HARD CEILING — always exit after MAX turns regardless of LLM
    if triage_rounds >= max_turns:
        state["clarification_complete"] = True
        state["form_confirmed"] = True
        state["triage_rounds"] = 0
        
        # generate dynamic context and proceed to models
        _app_purpose = state.get("extraction", {}).get("appPurpose") or ""
        _rag_context = ""
        vector_store = getattr(app_state, "vector_store", None)
        if vector_store and hasattr(vector_store, "search"):
            try:
                matches = await vector_store.search(
                    query=f"{state.get('app_type')} app variables for {_app_purpose}",
                    categories=["examples", "blueprints"],
                    top_k=3,
                )
                _rag_context = "\n\n".join([
                    m.get("content", "") for m in matches
                    if m.get("content") and (m.get("relevance_score") or 0) >= 0.65
                ])
            except Exception as e:
                logger.warning(f"RAG lookup for generate_dynamic_context failed: {e}")

        state["dynamic_context"] = await generate_dynamic_context(
            llm,
            state.get("app_type") or "text",
            _app_purpose,
            state.get("language_mode") or "English",
            rag_context=_rag_context,
        )
        
        state["last_slot_key"] = None
        _prefill_dynamic_context_variables_state(state)
        return await _show_models_state(state, app_state)

    deep_answers = state.get("deep_answers") or {}
    known = build_known_information(ext, deep_answers, app_purpose)
    asked_keys = list(state.get("asked_clarification_keys") or [])
    asked_questions = list(state.get("asked_clarification_questions") or [])

    plan = await plan_clarification(
        llm=llm,
        app_purpose=app_purpose,
        app_type=app_type,
        known_information=known,
        conversation_history=history_dicts,
        asked_keys=asked_keys,
        asked_questions=asked_questions,
        triage_rounds=triage_rounds,
    )

    state["behavior_goal"] = plan.get("behavior_goal")
    state["clarification_plan"] = plan
    # Persist workflow identity extracted by the clarification planner so later
    # stages (preview, prompt generation) use workflow intent rather than
    # relying solely on `app_type` or modality.
    # Try to derive a canonical workflow_name from the app purpose using the planner's matcher
    try:
        from services.clarification_planner import _match_workflow, _get_generic_workflow
        app_purp = state.get("extraction", {}).get("appPurpose") or state.get("app_purpose") or ""
        matched = _match_workflow(app_purp) or _get_generic_workflow(app_purp, state.get("app_type") or "text")
        wf_name = matched.get("workflow_name") if isinstance(matched, dict) else None
    except Exception:
        wf_name = None

    workflow_identity = {
        "workflow_name": wf_name,
        "workflow_confidence": plan.get("confidence") or 0.0,
        "workflow_plan": plan,
    }
    state["workflow_identity"] = workflow_identity

    missing_items = plan.get("missing_information") or []
    known_items = plan.get("known_information") or []
    dynamic_slots = [
        {"key": m["key"], "question": m["question"]} for m in missing_items
    ] + [
        {"key": k["key"], "question": k.get("value", "")} for k in known_items
    ]
    state["dynamic_slots"] = dynamic_slots
    state["dynamic_workflow"] = {
        "behavior_goal": plan.get("behavior_goal") or f"build {app_purpose}",
        "required_fields": [s["key"] for s in dynamic_slots],
        "field_questions": {m["key"]: m["question"] for m in missing_items},
    }

    clarification_complete = clarification_is_complete(plan)
    log_clarification_trace(
        app_purpose=app_purpose,
        plan=plan,
        clarification_complete=clarification_complete,
    )

    v_meta = state.get("verification_metadata") or {}
    ing_status = v_meta.get("ingestion_vector") or "missing"

    needs_clarification = (
        not clarification_complete
        and plan.get("selected_question")
        and triage_rounds < max_turns
    )

    if needs_clarification:
        next_slot_key = plan["selected_key"]
        next_question = plan["selected_question"]

        asked_keys.append(next_slot_key)
        asked_questions.append(next_question)
        state["asked_clarification_keys"] = asked_keys
        state["asked_clarification_questions"] = asked_questions

        state["last_slot_key"] = next_slot_key
        state["current_deep_field"] = next_slot_key
        state["awaiting_deep_answer"] = True
        state["triage_rounds"] = triage_rounds + 1
        state["clarification_complete"] = False

        state["reply"] = next_question
        return {
            "reply": next_question,
            "uiType": None,
            "uiData": None,
            "nextStep": 0,
            "coins": None,
        }

    if plan.get("forced_complete"):
        logger.warning(
            f"[CLARIFICATION] Best-effort app generation after {max_turns} turns "
            f"for purpose: {app_purpose[:80]}"
        )

    state["clarification_complete"] = True

    # 5. Ready state satisfied or hard cap rounds hit
    # Generate dynamic context blueprint
    _app_purpose = state.get("extraction", {}).get("appPurpose") or ""
    _rag_context = ""
    vector_store = getattr(app_state, "vector_store", None)
    if vector_store and hasattr(vector_store, "search"):
        try:
            matches = await vector_store.search(
                query=f"{state.get('app_type')} app variables for {_app_purpose}",
                categories=["examples", "blueprints"],
                top_k=3,
            )
            _rag_context = "\n\n".join([
                m.get("content", "") for m in matches
                if m.get("content") and (m.get("relevance_score") or 0) >= 0.65
            ])
        except Exception as e:
            logger.warning(f"RAG lookup for generate_dynamic_context failed: {e}")

    state["dynamic_context"] = await generate_dynamic_context(
        llm,
        state.get("app_type") or "text",
        _app_purpose,
        state.get("language_mode") or "English",
        rag_context=_rag_context,
    )
    
    state["form_confirmed"] = True
    state["last_slot_key"] = None
    
    _prefill_dynamic_context_variables_state(state)

    v_meta = state.get("verification_metadata") or {}
    ing_status = v_meta.get("ingestion_vector", "missing")

    # Skip ingestion vector question for pure generation apps
    # These apps generate content — they don't analyze uploaded files
    _current_app_type = state.get("app_type") or (state.get("extraction") or {}).get("appType") or "text"
    _wants_image = (state.get("extraction") or {}).get("wantsImageInput") or False
    _is_pure_generation = (
        _current_app_type in ("text", "audio")
        and not _wants_image
        and ing_status not in ("explicit",)
    )
    if _is_pure_generation:
        ing_status = "not_required"
        v_meta["ingestion_vector"] = "not_required"
        state["ingestion_vector"] = "plain_text"
        state["verification_metadata"] = v_meta

    if ing_status in ("missing", "inferred"):
        app_purpose = state.get("extraction", {}).get("appPurpose") or ""
        # Only ask about ingestion if the workflow actually requires an uploaded artifact
        if requires_input_artifact(state.get("app_type") or state.get("extraction", {}).get("appType"), app_purpose):
            question, options = _get_ingestion_vector_question_and_chips(app_purpose)
            state["last_slot_key"] = "ingestion_vector"
            state["current_deep_field"] = "ingestion_vector"
            state["awaiting_deep_answer"] = True
            state["reply"] = question
            return {
                "reply": question,
                "uiType": "chips",
                "uiData": {"options": options},
                "nextStep": 0,
                "coins": None,
            }

    deep_answers = state.get("deep_answers") or {}
    if not _session_has_budget(state.get("extraction"), deep_answers):
        state["current_deep_field"] = "budgetPreference"
        state["awaiting_deep_answer"] = True
        msg_text = "Excellent choice! I have saved the blueprint for your app. One last parameter to proceed: **What is your budget preference per run?**"
        state["reply"] = msg_text
        return {
            "reply": msg_text,
            "uiType": "chips",
            "uiData": {"options": BUDGET_CHIP_OPTIONS},
            "nextStep": 0,
            "coins": None,
        }

    return await _show_models_state(state, app_state)


def _prefill_dynamic_context_variables_state(state: ConversationState) -> None:
    if not state.get("dynamic_context") or not isinstance(state["dynamic_context"], dict):
        state["dynamic_context"] = {"variables": [], "options": []}
    
    dynamic_context = state["dynamic_context"]
    if "variables" not in dynamic_context or not isinstance(dynamic_context["variables"], list):
        dynamic_context["variables"] = []
        
    variables = dynamic_context["variables"]
    extraction = state.get("extraction") or {}
    deep_answers = state.get("deep_answers") or {}
    
    lookup = {}
    for k, v in extraction.items():
        if isinstance(v, str) and v.strip():
            lookup[k.lower().replace("_", " ")] = v.strip()
    for k, v in deep_answers.items():
        if not k.startswith("_") and isinstance(v, str) and v.strip():
            lookup[k.lower().replace("_", " ")] = v.strip()
            
    universal_mappings = {
        "subject": extraction.get("PRIMARY_SUBJECT"),
        "primary subject": extraction.get("PRIMARY_SUBJECT"),
        "setting": extraction.get("ENVIRONMENT_SETTING"),
        "environment setting": extraction.get("ENVIRONMENT_SETTING"),
        "environment": extraction.get("ENVIRONMENT_SETTING"),
        "action": extraction.get("ACTION_DYNAMIC"),
        "action dynamic": extraction.get("ACTION_DYNAMIC"),
        "style": extraction.get("AESTHETIC_STYLE"),
        "aesthetic style": extraction.get("AESTHETIC_STYLE"),
    }
    for k, v in universal_mappings.items():
        if v and isinstance(v, str) and v.strip():
            lookup[k] = v.strip()
            
    prefilled = []
    existing_names = set()
    bloat_patterns = [
        r"date\s*of\s*creation", r"jump\s*scare", r"video\s*title", r"age\s*rating",
        r"creation\s*date", r"scare\s*frequency", r"frequency"
    ]
    
    for var in variables:
        if not isinstance(var, dict):
            continue
        var_name = str(var.get("name") or "").strip()
        var_name_lower = var_name.lower()
        if any(re.search(pat, var_name_lower) for pat in bloat_patterns):
            continue
        if is_personal_boilerplate(var_name, extraction.get("appPurpose") or ""):
            continue
            
        prefilled_value = None
        if var_name_lower in lookup:
            prefilled_value = lookup[var_name_lower]
        else:
            for lookup_key, lookup_val in lookup.items():
                if lookup_key in var_name_lower or var_name_lower in lookup_key:
                    prefilled_value = lookup_val
                    break
                    
        var_copy = dict(var)
        var_copy["test_value"] = prefilled_value or var.get("test_value") or var.get("placeholder") or ""
        prefilled.append(var_copy)
        existing_names.add(var_name_lower)
        
    for dim_key, dim_name in [
        ("PRIMARY_SUBJECT", "Primary Subject"),
        ("ENVIRONMENT_SETTING", "Environment Setting"),
        ("ACTION_DYNAMIC", "Action Dynamic"),
        ("AESTHETIC_STYLE", "Aesthetic Style"),
    ]:
        val = extraction.get(dim_key)
        if val and isinstance(val, str) and val.strip() and dim_name.lower() not in existing_names:
            if not is_personal_boilerplate(dim_name, extraction.get("appPurpose") or ""):
                prefilled.append({
                    "name": dim_name,
                    "placeholder": f"Enter {dim_name.lower()}...",
                    "test_value": val.strip()
                })
                existing_names.add(dim_name.lower())
            
    dynamic_context["variables"] = prefilled[:4]
    state["dynamic_context"] = dynamic_context


def _merge_extraction_state_helper(existing: dict | None, latest: dict | None, message: str, state: dict) -> dict:
    if not latest:
        return existing or {}
    if not existing:
        existing = {}

    is_control = bool(
        _parse_selected_model_id(message, None)
        or _parse_selected_plan(message)
        or _parse_chip_app_type(message)
        or _is_yes(message)
    )

    keep_purpose = is_control or not latest.get("appPurpose") or len(str(latest.get("appPurpose") or "")) < 8

    existing_conf = existing.get("confidence") or {}
    latest_conf = latest.get("confidence") or {}

    return {
        **existing,
        **latest,
        "PRIMARY_SUBJECT": latest.get("PRIMARY_SUBJECT") or existing.get("PRIMARY_SUBJECT"),
        "ENVIRONMENT_SETTING": latest.get("ENVIRONMENT_SETTING") or existing.get("ENVIRONMENT_SETTING"),
        "ACTION_DYNAMIC": latest.get("ACTION_DYNAMIC") or existing.get("ACTION_DYNAMIC"),
        "AESTHETIC_STYLE": latest.get("AESTHETIC_STYLE") or existing.get("AESTHETIC_STYLE"),
        "appType": state.get("app_type") or "text",
        "appPurpose": existing.get("appPurpose") if keep_purpose else latest.get("appPurpose"),
        "targetUsers": (
            existing.get("targetUsers")
            if is_control or not latest.get("targetUsers") or latest.get("targetUsers") == "general users"
            else latest.get("targetUsers")
        ),
        "budget": latest.get("budget") if latest.get("budget") else existing.get("budget"),
        "wantsImageInput": bool(existing.get("wantsImageInput") or latest.get("wantsImageInput")),
        "detectedLanguage": latest.get("detectedLanguage") or existing.get("detectedLanguage"),
        "userTone": latest.get("userTone") or existing.get("userTone"),
        "oneLineUnderstanding": latest.get("oneLineUnderstanding") or existing.get("oneLineUnderstanding"),
        "confidence": {
            "appType": existing_conf.get("appType") or latest_conf.get("appType") or "LOW",
            "budget": latest_conf.get("budget") or existing_conf.get("budget") or "LOW",
        },
        "keyFeatures": latest.get("keyFeatures") or existing.get("keyFeatures"),
        "missingFields": list(set((existing.get("missingFields") or []) + (latest.get("missingFields") or []))),
    }


def _detect_language_mode_state(state: ConversationState) -> str:
    extraction = state.get("extraction") or {}
    lang = str(extraction.get("detectedLanguage") or state.get("language_mode") or "English").lower()
    if "hinglish" in lang:
        return "Hinglish"
    if "hindi" in lang:
        return "Hindi"
    return "English"


async def _exec_gather_requirements_state(state: ConversationState, text: str, app_state: Any) -> dict:
    last_slot = state.get("last_slot_key")
    app_purpose = (state.get("extraction") or {}).get("appPurpose")
    if not last_slot and not app_purpose:
        if len(text.strip()) < 6 and not any(c.isalpha() for c in text[2:]):
            state["reply"] = "I am your RentPrompts Architect. Let me know what app you want to configure!"
            return {
                "reply": "I am your RentPrompts Architect. Let me know what app you want to configure!",
                "uiType": None,
                "uiData": None,
                "nextStep": 0,
            }

    # Grab the slot answers if we were waiting for them
    if last_slot and text:
        if "deep_answers" not in state or state["deep_answers"] is None:
            state["deep_answers"] = {}
        state["deep_answers"][str(last_slot).strip()] = text.strip()
        state["last_slot_key"] = None

    history_dicts = []
    for m in state.get("history") or []:
        role = "agent" if (m.get("role") if isinstance(m, dict) else m.type) in ("assistant", "agent") else "user"
        content = m.get("content") if isinstance(m, dict) else m.content
        history_dicts.append({"role": role, "content": content})

    latest_extraction = await extract_requirements(app_state.llm, text, history_dicts)
    
    if "extraction" not in state or state["extraction"] is None:
        state["extraction"] = {}

    state["extraction"] = _merge_extraction_state_helper(state.get("extraction"), latest_extraction, text, state)
    state["language_mode"] = _detect_language_mode_state(state)
    
    _update_verification_metadata_state(state, text)

    return await _build_step0_response_state(state, text, app_state)


def _apply_edit_to_state(state: dict, edit_instruction: str) -> None:
    instr = str(edit_instruction or "").strip().lower()
    if not instr:
        return

    extraction = state.get("extraction")
    if extraction:
        old_purpose = str(extraction.get("appPurpose") or "")
        if instr[:20] not in old_purpose.lower():
            extraction["appPurpose"] = f"{old_purpose} (updated: {edit_instruction.strip()})"
        extraction["oneLineUnderstanding"] = extraction.get("appPurpose")

    deep_answers = state.get("deep_answers")
    if isinstance(deep_answers, dict):
        wants_transparent = bool(re.search(r"transparent|no background|remove background|no bg", edit_instruction, re.IGNORECASE))
        wants_new_background = bool(re.search(r"background|backdrop|scene|environment", edit_instruction, re.IGNORECASE))
        no_x_match = re.search(r"no\s+(\w+)", edit_instruction, re.IGNORECASE)

        keys_to_remove: list[str] = []
        for key, value in deep_answers.items():
            kl = key.lower()
            vl = str(value or "").lower()
            if wants_transparent and re.search(r"scene|background|backdrop|environment|location|setting", kl, re.IGNORECASE):
                keys_to_remove.append(key)
            elif wants_new_background and re.search(r"scene|background|backdrop|environment", kl, re.IGNORECASE):
                deep_answers[key] = edit_instruction.strip()
            elif no_x_match:
                rejected = no_x_match.group(1).lower()
                if rejected in vl or rejected in kl:
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            deep_answers.pop(key, None)

        if wants_transparent:
            deep_answers["outputType"] = "transparent PNG"
            deep_answers["backgroundType"] = "transparent"

    dynamic_context = state.get("dynamic_context") or {}
    variables = dynamic_context.get("variables")
    if variables and re.search(r"transparent|no background|remove background|no bg", edit_instruction, re.IGNORECASE):
        dynamic_context["variables"] = [
            v for v in variables if not re.search(r"scene|backdrop|environment|location|background_scene", str(v.get("name") if isinstance(v, dict) else v), re.IGNORECASE)
        ]


async def _exec_generate_preview_state(state: ConversationState, text: str, app_state: Any) -> dict:
    llm = app_state.llm
    app_type = state.get("app_type") or "text"
    selected_model_id = _parse_selected_model_id(text, MODELS.get(app_type, []))
    if not selected_model_id:
        selected_model_id = state.get("model_id")
    if not selected_model_id:
        state["reply"] = "Please **click one of the model cards** above to select the AI engine. 👆"
        return {"reply": "Please **click one of the model cards** above to select the AI engine. 👆", "uiType": "text", "nextStep": 1}

    selected_model = _find_model(app_type, selected_model_id)
    if not selected_model:
        state["reply"] = "I couldn't match that model. Please click one of the options above."
        return {"reply": "I couldn't match that model. Please click one of the options above.", "uiType": "text", "nextStep": 1}

    state["model_id"] = selected_model["id"]
    state["model_cost"] = selected_model["cost"]
    state["model_name"] = selected_model["name"]
    try:
        if not state.get("dynamic_context"):
            # Only generate if truly missing (e.g. session restored without it)
            state["dynamic_context"] = await generate_dynamic_context(
                llm, app_type,
                state.get("extraction", {}).get("appPurpose") or "",
                state.get("language_mode") or "English",
            )
        _prefill_dynamic_context_variables_state(state)

        search_query = f"{app_type} model {state.get('model_name')} prompting guidelines parameters"
        try:
            search_tool = get_web_search_tool()
            search_result = await search_tool.search_and_summarize(search_query)
            state["webSearchContext"] = search_result
        except Exception as search_err:
            logger.warning(f"WebSearch grounding failed: {search_err}")

        _prompt_rag = ""
        vector_store = getattr(app_state, "vector_store", None)
        if vector_store and hasattr(vector_store, "search"):
            try:
                matches = await vector_store.search(
                    query=f"{app_type} app prompt template for {state.get('extraction', {}).get('appPurpose') or ''}",
                    categories=["examples", "prompting"],
                    top_k=3,
                )
                _prompt_rag = "\n\n".join([
                    m.get("content", "") for m in matches
                    if m.get("content") and (m.get("relevance_score") or 0) >= 0.6
                ])
                if _prompt_rag:
                    state["ragContext"] = _prompt_rag
            except Exception as e:
                logger.warning(f"RAG lookup for prompt generation failed: {e}")

        prompt_data, seo_data = await asyncio.gather(
            generate_prompt_template(
                llm,
                app_type=app_type,
                app_purpose=state.get("extraction", {}).get("appPurpose") or "Not specified",
                model_name=state.get("model_name"),
                model_id=selected_model["id"],
                target_users=(state.get("extraction", {}).get("targetAudience") or "General Public"),
                variables=(state.get("dynamic_context") or {}).get("variables") or [],
                deep_answers=state.get("deep_answers"),
                history=state.get("history"),
                language_mode=state.get("language_mode") or "English",
                web_search_context=state.get("webSearchContext"),
                rag_context=state.get("ragContext"),
            ),
            generate_seo(
                llm,
                app_type=app_type,
                app_purpose=state.get("extraction", {}).get("appPurpose") or "Not specified",
                deep_answers=state.get("deep_answers"),
                history=state.get("history"),
                language_mode=state.get("language_mode") or "English",
                model_id=selected_model["id"],
                vector_store=vector_store,
            ),
        )

        state["prompt_data"] = prompt_data
        state["seo_data"] = seo_data
        state["current_step"] = 2

        # Synchronize prompt optimizer output channels back to state top-level fields for /chat compatibility
        state["enhanced_system_prompt"] = prompt_data.get("systemPrompt")
        state["enhanced_user_prompt"] = prompt_data.get("userPrompt")
        state["extracted_variables"] = [{"identifier": v, "display_name": v.replace("_", " ").capitalize(), "type": "string", "placeholder": f"Enter {v}..."} for v in prompt_data.get("variablesUsed") or []]

        preview_image_url = None
        if app_type in ("image", "vision"):
            try:
                compiled_prompt = (prompt_data.get("userPrompt") or "")
                dc_vars = (state.get("dynamic_context") or {}).get("variables") or []
                test_inputs = {}
                for var in dc_vars:
                    if isinstance(var, dict):
                        name = var.get("name", "")
                        test_val = var.get("test_value") or var.get("placeholder") or name
                        test_inputs[name] = test_val
                    else:
                        test_inputs[str(var)] = str(var)

                for key, value in test_inputs.items():
                    val_str = str(value or "")
                    keys_to_try = [key]
                    for alt in [key.replace(" ", "_"), key.replace("_", " "), key.replace(" ", ""), key.lower(), key.upper()]:
                        if alt not in keys_to_try:
                            keys_to_try.append(alt)

                    for k in keys_to_try:
                        compiled_prompt = re.sub(re.escape(f"$${k}$$"), val_str, compiled_prompt, flags=re.I)
                        compiled_prompt = re.sub(re.escape(f"$${k}"), val_str, compiled_prompt, flags=re.I)
                        compiled_prompt = re.sub(re.escape(f"${k}$$"), val_str, compiled_prompt, flags=re.I)
                        compiled_prompt = re.sub(re.escape(f"${k}"), val_str, compiled_prompt, flags=re.I)
                        compiled_prompt = re.sub(re.escape(f"[{k}]"), val_str, compiled_prompt, flags=re.I)

                compiled_prompt = re.sub(r"\*\*+(?:\$\$|\[)[a-zA-Z0-9_'\s-]+?(?:\$\$|\])?\*\*+", "", compiled_prompt)
                compiled_prompt = re.sub(r"\[[a-zA-Z0-9_'\s-]+?\]", "", compiled_prompt)
                compiled_prompt = re.sub(r"\$$[a-zA-Z0-9_']+\b", "", compiled_prompt)
                compiled_prompt = re.sub(r"\*\*+\s*\*+", "", compiled_prompt)
                compiled_prompt = re.sub(r"\s+", " ", compiled_prompt).strip()

                model_api_url = getattr(app_state, "image_model_base_url", None)
                if model_api_url and compiled_prompt:
                    img_resp = await llm._openrouter_client.post(
                        "/chat/completions",
                        json={
                            "model": selected_model_id,
                            "messages": [{"role": "user", "content": compiled_prompt}],
                            "max_tokens": 1,
                        }
                    )
                    img_data = img_resp.json()
                    preview_image_url = (
                        img_data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        .strip() or None
                    )
            except Exception as img_err:
                logger.warning(f"Image preview generation failed, frontend will fallback: {img_err}")

        payload = {
            "reply": f"## App Preview Ready\n\nI've configured the full AI logic using **{selected_model['name']}**.\n\nTest it in the Live Preview below — click **Approve App** when ready!",
            "uiType": "app_preview",
            "uiData": {
                "appName": seo_data.get("appName"),
                "appType": app_type,
                "appDescription": seo_data.get("appDescription"),
                "cost": state.get("model_cost"),
                "systemPrompt": prompt_data.get("systemPrompt"),
                "userPrompt": prompt_data.get("userPrompt"),
                "variablesUsed": prompt_data.get("variablesUsed"),
                "variables": (state.get("dynamic_context") or {}).get("variables") or [],
                "acceptImageInput": _sanitize_accept_image_input(prompt_data.get("acceptImageInput"), app_type),
                "previewImageUrl": preview_image_url,
                "options": ["Approve App", "Edit App"],
                "step": 2,
            },
            "nextStep": 2,
            "coins": state.get("model_cost"),
        }
        state["reply"] = payload["reply"]
        return payload
    except Exception as err:
        logger.exception("Preview generation Crash")
        err_msg = f"Oops! hit a snag generating the config: {err}. Please retry."
        state["reply"] = err_msg
        return {"reply": err_msg, "uiType": "text", "nextStep": 1}


async def _handle_seo_publish_state(state: ConversationState, card_data: dict, app_state: Any) -> dict:
    prompt_data = state.get("prompt_data") or {}
    seo_data = {**(state.get("seo_data") or {}), **card_data}
    state["seo_data"] = seo_data

    vector_store = getattr(app_state, "vector_store", None)
    if vector_store and hasattr(vector_store, "search"):
        try:
            app_type = state.get("app_type") or "text"
            app_purpose = state.get("extraction", {}).get("appPurpose") or ""
            matches = await vector_store.search(
                query=f"{app_type} app: {app_purpose} publishing validation",
                categories=["examples", "marketplace"],
                top_k=2,
                boost_gold_standards=True,
            )
            logger.info("Publishing metadata verification against gold standards complete", 
                        num_matches=len(matches))
        except Exception as e:
            logger.warning(f"Gold standards retrieval failed during publishing: {e}")

    app_name = seo_data.get("appName") or "Your App"
    alt_text = str(seo_data.get("appDescription") or app_name)[:500]
    media_id = None

    try:
        safe_filename = re.sub(r"[^a-zA-Z0-9_-]", "_", app_name)[:40] or "app-preview"
        media_result = await app_state.cms.upload_media(
            _PREVIEW_PNG_BYTES,
            filename=f"{safe_filename}-preview.png",
            content_type="image/png",
            alt=alt_text,
        )
        media_id = media_result.get("id") or (media_result.get("doc") or {}).get("id")
    except Exception as media_err:
        logger.warning(f"Real media upload gateway error: {media_err}")

    is_private = bool(state.get("seo_data", {}).get("isPrivate"))
    variables_used = prompt_data.get("variablesUsed") or []
    var_descriptions = prompt_data.get("variableDescriptions") or {}
    prompt_variables = [
        {"name": v, "description": var_descriptions.get(v, f"Enter {str(v).replace('_', ' ')}")}
        for v in variables_used
    ]

    if is_private:
        payload = {
            "name": app_name,
            "description": seo_data.get("appDescription"),
            "modelType": state.get("app_type"),
            "model": state.get("model_id"),
            "systemprompt": prompt_data.get("systemPrompt"),
            "prompt": prompt_data.get("userPrompt"),
            "negativeprompt": prompt_data.get("negativePrompt"),
            "priceapplicable": True,
            "price": state.get("model_cost"),
            "promptVariables": prompt_variables,
        }
    else:
        payload = {
            "name": app_name,
            "description": seo_data.get("appDescription"),
            "modelType": state.get("app_type"),
            "model": state.get("model_id"),
            "systemprompt": prompt_data.get("systemPrompt"),
            "prompt": prompt_data.get("userPrompt"),
            "negativeprompt": prompt_data.get("negativePrompt"),
            "priceapplicable": False,
            "publishedAt": datetime.now(timezone.utc).isoformat(),
            "promptVariables": prompt_variables,
        }

    if media_id:
        payload["images"] = [{"image": media_id}]

    try:
        await app_state.cms.create_rapp(payload, private=is_private)
    except Exception as err:
        logger.error(f"CMS Publish backend failed: {err}")
        state["reply"] = "Publishing hit a snag on our end. Please try again."
        return {"reply": "Publishing hit a snag on our end. Please try again.", "uiType": "text"}

    session_id = state.get("session_id")
    if session_id:
        await app_state.session.delete_session(session_id)

    reply_msg = f'## 🎉 Published Successfully!\n\nYour app "{app_name}" is now live on {"your private registry" if is_private else "the marketplace"}!'
    state["reply"] = reply_msg
    return {
        "reply": reply_msg,
        "uiType": "success",
        "uiData": {"appName": app_name, "modelCost": state.get("model_cost")},
        "nextStep": 0,
        "clearSession": True,
    }


async def _exec_edit_app_state(state: ConversationState, text: str, decision: dict, app_state: Any) -> dict:
    instruction = text.strip()
    edit_scope = decision.get("edit_scope") or \
                 (decision.get("extracted_variables") or {}).get("edit_scope") or \
                 "PATCH_PROMPT"

    if edit_scope == "PATCH_VALUE":
        dc = state.get("dynamic_context") or {}
        variables = dc.get("variables") or []
        instr_lower = instruction.lower()
        patched = False
        for var in variables:
            var_name_lower = str(var.get("name") or "").lower().replace(" ", "_")
            if any(
                kw in instr_lower
                for kw in [var_name_lower, var_name_lower.replace("_", " ")]
            ):
                value_match = re.search(
                    r"(?:be|to|is|=|:)\s+(.+)$", instruction, re.IGNORECASE
                )
                if value_match:
                    var["test_value"] = value_match.group(1).strip()
                    patched = True
                    break
        if not patched and variables:
            for var in variables:
                if any(w in str(var.get("name") or "").lower() for w in
                       ["name", "character", "subject", "protagonist", "hero"]):
                    value_match = re.search(
                        r"(?:be|to|is|=|:)\s+(.+)$", instruction, re.IGNORECASE
                    )
                    if value_match:
                        var["test_value"] = value_match.group(1).strip()
                    break
        state["dynamic_context"] = dc
        state["current_step"] = 2
        return await _exec_generate_preview_state(state, text, app_state)

    if edit_scope == "DOMAIN_SHIFT":
        state["dynamic_context"] = None
        state["ragContext"] = None
        state["webSearchContext"] = None
        state["dynamic_slots"] = []
        state["form_confirmed"] = False
        if "extraction" not in state or state["extraction"] is None:
            state["extraction"] = {}
        state["extraction"]["appPurpose"] = instruction
        state["current_step"] = 0
        return await _build_step0_response_state(state, instruction, app_state)

    _apply_edit_to_state(state, instruction)
    state["ragContext"] = None
    state["webSearchContext"] = None
    state["current_step"] = 2
    return await _exec_generate_preview_state(state, text, app_state)


async def _exec_review_seo_state(state: ConversationState, app_state: Any) -> dict:
    state["current_step"] = 3
    seo_data = state.get("seo_data") or {}
    return {
        "reply": "## 🎉 App Configured — Final Review\n\nReview your app metadata profile details below.",
        "uiType": "seo_preview",
        "uiData": {
            "appName": seo_data.get("appName") or "Your App",
            "appDescription": seo_data.get("appDescription") or "",
            "tags": seo_data.get("tags") or [],
            "appType": state.get("app_type"),
            "modelId": state.get("model_id"),
            "costPerRun": state.get("model_cost"),
        },
        "nextStep": 3,
    }


async def _rebuild_current_step_response_state(state: ConversationState, app_state: Any, step: int) -> dict:
    if step >= 3: return await _exec_review_seo_state(state, app_state)
    if step == 2: return await _exec_generate_preview_state(state, "", app_state)
    if step == 1: return await _show_models_state(state, app_state)
    return {"reply": "Let's continue shaping your application.", "uiType": None}


# ─── LANGGRAPH CORE CONTEXT INTERFACE LIFECYCLE ───

# ─── LANGGRAPH CORE CONTEXT INTERFACE LIFECYCLE ───

class ConversationState(TypedDict, total=False):
    session_id: str
    message: str
    reply: Optional[str]
    history: Annotated[list, add_messages]
    app_type: Optional[str]
    app_purpose: Optional[str]
    extraction: Dict[str, Any]
    dynamic_context: Optional[Dict[str, Any]]
    deep_answers: Dict[str, Any]
    current_step: int
    recommended_action: str
    confidence: str
    reasoning: str
    response_payload: Dict[str, Any]
    form_confirmed: bool
    model_id: Optional[str]
    model_name: Optional[str]
    model_cost: Optional[float]
    prompt_data: Dict[str, Any]
    seo_data: Dict[str, Any]
    clear_session: bool
    language_mode: str
    triage_rounds: int
    last_slot_key: Optional[str]
    dynamic_slots: list[dict]
    awaiting_deep_answer: Optional[bool]
    current_deep_field: Optional[str]
    ingestion_vector: Optional[str]
    verification_metadata: Dict[str, str]
    requirements_complete: bool
    preview_approved: bool
    cms_registered: bool
    enhanced_system_prompt: Optional[str]
    enhanced_user_prompt: Optional[str]
    extracted_variables: list[dict]
    rag_documents: list[dict]
    rag_context_injected: str
    model_guidance: Optional[str]
    optimization_notes: list[str]
    similar_apps: list[dict]
    pivot_transition: Optional[str]
    dynamic_workflow: Optional[Dict[str, Any]]
    behavior_goal: Optional[str]
    clarification_plan: Optional[Dict[str, Any]]
    clarification_complete: bool
    asked_clarification_keys: list[str]
    asked_clarification_questions: list[str]


def _update_verification_metadata_state(state: ConversationState, message: str, decision: dict | None = None) -> None:
    v_meta = state.get("verification_metadata")
    if not isinstance(v_meta, dict):
        v_meta = {
            "app_type": "inferred" if state.get("app_type") else "missing",
            "ingestion_vector": "missing",
            "budget": "missing"
        }
        state["verification_metadata"] = v_meta

    lower = message.lower()
    
    # 1. Update Ingestion Vector & Status
    ing_vec = None
    ing_status = None
    if re.search(r"\b(url|website url|link|links|webpage|webpages|website|websites)\b", lower):
        ing_vec = "url"
        ing_status = "explicit"
    elif re.search(r"\b(screenshot|screenshots|upload screenshots?|upload photos?|upload images?)\b", lower):
        ing_vec = "screenshots"
        ing_status = "explicit"
    elif re.search(r"\b(source code|source_code|github|code files?|repository)\b", lower):
        ing_vec = "source_code"
        ing_status = "explicit"
    elif re.search(r"\b(figma|figma_files|figmafiles)\b", lower):
        ing_vec = "figma_files"
        ing_status = "explicit"

    # Get dynamic context / slot keys from state directly
    last_q = (state.get("dynamic_context") or {}).get("lastQuestion") or ""
    last_slot = str(state.get("last_slot_key") or "")
    current_deep = str(state.get("current_deep_field") or "")
    is_answering_vector = (
        "ingestion_vector" in last_slot 
        or "ingestion_vector" in current_deep 
        or ("how will" in last_q.lower() and ("provide" in last_q.lower() or "provided" in last_q.lower()))
    )
    
    if is_answering_vector:
        if "figma" in lower:
            ing_vec = "figma_files"
        elif "code" in lower or "source" in lower or "github" in lower:
            ing_vec = "source_code"
        elif "screenshot" in lower or "photo" in lower or "image" in lower or "upload" in lower or "pdf" in lower:
            ing_vec = "screenshots"
        elif "url" in lower or "website" in lower or "link" in lower:
            ing_vec = "url"
        else:
            ing_vec = "plain_text"
        ing_status = "explicit"

    if decision:
        dec_vec = decision.get("ingestion_vector")
        dec_status = decision.get("ingestion_vector_status")
        if dec_status == "explicit" or (dec_status == "inferred" and v_meta.get("ingestion_vector") == "missing"):
            ing_vec = dec_vec
            ing_status = dec_status

    if ing_vec and ing_vec != "missing":
        state["ingestion_vector"] = ing_vec
        v_meta["ingestion_vector"] = ing_status
    elif not state.get("ingestion_vector"):
        state["ingestion_vector"] = None
        v_meta["ingestion_vector"] = "missing"

    # 2. Update Budget & Status
    budget_map = {
        "free models only (0 coins)": "free",
        "low (< 5 coins)": "low",
        "medium (5 - 20 coins)": "medium",
        "premium (> 20 coins)": "premium",
    }

    is_budget_q = (state.get("last_slot_key") == "budgetPreference") or (state.get("current_deep_field") == "budgetPreference") or ("budget preference" in last_q.lower())
    if is_budget_q:
        v_meta["budget"] = "explicit"
    else:
        has_budget = False
        extraction = state.get("extraction") or {}
        deep_answers = state.get("deep_answers") or {}
        if extraction.get("budget") or deep_answers.get("budgetPreference"):
            has_budget = True

        if has_budget:
            is_chip = lower in budget_map
            if is_chip or v_meta.get("budget") == "explicit":
                v_meta["budget"] = "explicit"
            else:
                v_meta["budget"] = "inferred"
        else:
            v_meta["budget"] = "missing"

    # 3. Update App Type Status
    if state.get("app_type"):
        is_type_q = "what kind of app" in last_q.lower() or "app type" in last_q.lower()
        if is_type_q or v_meta.get("app_type") == "explicit":
            v_meta["app_type"] = "explicit"
        elif decision and decision.get("app_type_status") == "explicit":
            v_meta["app_type"] = "explicit"
        else:
            modality_pattern = r"\b(?:it is|i want|use|change to|make it|it\'s|want a|need a|not an?|instead of)\s+a?n?\s*(text|image|audio|video|vision)\s*(?:app|tool|generator|model|models)?\b"
            if re.search(modality_pattern, lower) or lower in [
                "text", "image", "audio", "video", "vision", "text app", "image app", "audio app", "video app", "vision app"
            ]:
                v_meta["app_type"] = "explicit"
            elif v_meta.get("app_type") != "explicit":
                v_meta["app_type"] = "inferred"
    else:
        v_meta["app_type"] = "missing"

    state["verification_metadata"] = v_meta
    
    # Synchronize budget preference between extraction and deep_answers if explicit chip selected
    if lower in budget_map:
        if "extraction" not in state or state["extraction"] is None:
            state["extraction"] = {}
        state["extraction"]["budget"] = budget_map[lower]
        if "deep_answers" not in state or state["deep_answers"] is None:
            state["deep_answers"] = {}
        state["deep_answers"]["budgetPreference"] = budget_map[lower]



def _get_ingestion_vector_question_and_chips(app_purpose: str) -> tuple[str, list[str]]:
    purpose = str(app_purpose or "").lower()
    
    # 1. Menu
    if "menu" in purpose:
        return (
            "How will restaurant menus be provided for review?",
            ["Uploaded photos", "PDF menus", "Website URLs", "Plain text menu descriptions"]
        )
    # 2. Pitch deck / Presentation
    if any(w in purpose for w in ("pitch deck", "deck", "presentation", "slide")):
        return (
            "How will the pitch deck be provided for review?",
            ["PDF upload", "PowerPoint file", "Web link", "Paste text"]
        )
    # 3. Resume / Portfolio / CV
    if any(w in purpose for w in ("resume", "cv", "portfolio", "profile")):
        return (
            "How will the resume or profile be provided for analysis?",
            ["PDF/Word upload", "LinkedIn link", "Paste text", "Screenshot image"]
        )
    # 4. Code / Source code
    if any(w in purpose for w in ("code", "repo", "software", "github", "source")):
        return (
            "How will the source code be provided for analysis?",
            ["GitHub repository link", "Zip file upload", "Paste code snippet", "Figma design link"]
        )
    # 5. Document / PDF / Report / Invoice / Receipt
    if any(w in purpose for w in ("document", "pdf", "report", "invoice", "receipt", "scan")):
        return (
            "How will the document be provided for analysis?",
            ["PDF upload", "Scan/Image upload", "Paste text", "Website URL"]
        )
    # 6. Website / Webpage
    if any(w in purpose for w in ("website", "webpage", "site", "figma")):
        return (
            "How will users provide their website for analysis?",
            ["Website URL", "Screenshots", "Figma design files", "Source code"]
        )
    # Default fallback
    return (
        "How will the main input be provided to the application?",
        ["Upload files (PDF/Image)", "Website URL", "Paste plain text", "Screenshots"]
    )


def _session_to_state(session: dict, message: str) -> ConversationState:
    hist = []
    for m in session.get("history", []):
        role = "assistant" if m.get("role") in ("agent", "assistant") else "user"
        hist.append({"role": role, "content": m.get("content", "")})
    
    a_type = session.get("appType") or "text"

    if "extraction" not in session or not isinstance(session["extraction"], dict):
        session["extraction"] = {}
    if "deepAnswers" not in session or not isinstance(session["deepAnswers"], dict):
        session["deepAnswers"] = {}
    if "promptData" not in session or not isinstance(session["promptData"], dict):
        session["promptData"] = {}
    if "seoData" not in session or not isinstance(session["seoData"], dict):
        session["seoData"] = {}

    state_extraction = {}
    state_extraction.update(session["extraction"])
    state_extraction["appType"] = a_type

    state_deep_answers = {}
    state_deep_answers.update(session["deepAnswers"])

    state_prompt_data = {}
    state_prompt_data.update(session["promptData"])

    state_seo_data = {}
    state_seo_data.update(session["seoData"])

    triage_rounds = session.get("triageRounds", 0)
    last_slot_key = session.get("lastSlotKey")
    dynamic_slots = session.get("dynamicSlots") or []

    # Parse model selection if present in the current message
    parsed_model_id = _parse_selected_model_id(message, MODELS.get(a_type, []))
    model_id = parsed_model_id or session.get("modelId")

    v_meta = session.get("verificationMetadata") or {}
    for key in ("app_type", "ingestion_vector", "budget"):
        if key not in v_meta:
            v_meta[key] = "missing"

    return {
        "session_id": session.get("sessionId") or "",
        "message": message,
        "history": hist,
        "app_type": a_type,
        "extraction": state_extraction,
        "dynamic_context": session.get("dynamicContext"),
        "deep_answers": state_deep_answers,
        "current_step": session.get("step") or 0,
        "recommended_action": "",
        "confidence": session.get("confidence") or "MEDIUM",
        "reasoning": session.get("reasoning") or "",
        "response_payload": {},
        "form_confirmed": session.get("formConfirmed") or False,
        "model_id": model_id,
        "model_cost": session.get("modelCost"),
        "model_name": session.get("modelName"),
        "prompt_data": state_prompt_data,
        "seo_data": state_seo_data,
        "clear_session": False,
        "language_mode": session.get("languageMode") or "English",
        "triage_rounds": triage_rounds,
        "last_slot_key": last_slot_key,
        "dynamic_slots": dynamic_slots,
        "awaiting_deep_answer": session.get("awaitingDeepAnswer") or False,
        "current_deep_field": session.get("currentDeepField"),
        "ingestion_vector": session.get("ingestionVector"),
        "verification_metadata": v_meta,
        "dynamic_workflow": session.get("dynamicWorkflow"),
        "behavior_goal": session.get("behaviorGoal"),
        "clarification_plan": session.get("clarificationPlan"),
        "clarification_complete": session.get("clarificationComplete") or False,
        "asked_clarification_keys": session.get("askedClarificationKeys") or [],
        "asked_clarification_questions": session.get("askedClarificationQuestions") or [],
    }


def _state_to_session(state: ConversationState, session: dict) -> None:
    session["step"] = state.get("current_step", 0)
    
    a_type = state.get("app_type") or "text"
    old_app_type = session.get("appType")
    if old_app_type != a_type:
        print(f"[MUTATION] appType: {old_app_type} -> {a_type} | file: step_router.py | function: _state_to_session | line: 1613 | msg: {state.get('message')}")
    session["appType"] = a_type
    
    if "extraction" not in session or not isinstance(session["extraction"], dict):
        session["extraction"] = {}
    session["extraction"].update(state.get("extraction") or {})
    old_ext_type = session["extraction"].get("appType")
    if old_ext_type != a_type:
        print(f"[MUTATION] extraction['appType']: {old_ext_type} -> {a_type} | file: step_router.py | function: _state_to_session | line: 1618 | msg: {state.get('message')}")
    session["extraction"]["appType"] = a_type
    
    session["dynamicContext"] = state.get("dynamic_context")
    
    if "deepAnswers" not in session or not isinstance(session["deepAnswers"], dict):
        session["deepAnswers"] = {}
    session["deepAnswers"].update(state.get("deep_answers") or {})
    
    session["formConfirmed"] = state.get("form_confirmed") or False
    session["modelId"] = state.get("model_id")
    session["modelCost"] = state.get("model_cost")
    session["modelName"] = state.get("model_name")
    
    if "promptData" not in session or not isinstance(session["promptData"], dict):
        session["promptData"] = {}
    session["promptData"].update(state.get("prompt_data") or {})
    
    if "seoData" not in session or not isinstance(session["seoData"], dict):
        session["seoData"] = {}
    session["seoData"].update(state.get("seo_data") or {})
    
    session["languageMode"] = state.get("language_mode", "English")
    session["confidence"] = state.get("confidence", "MEDIUM")
    session["reasoning"] = state.get("reasoning", "")
    session["triageRounds"] = state.get("triage_rounds", 0)
    session["lastSlotKey"] = state.get("last_slot_key")
    session["dynamicSlots"] = state.get("dynamic_slots") or []
    session["awaitingDeepAnswer"] = state.get("awaiting_deep_answer") or False
    session["currentDeepField"] = state.get("current_deep_field")
    session["ingestionVector"] = state.get("ingestion_vector")
    session["verificationMetadata"] = state.get("verification_metadata") or {}
    session["dynamicWorkflow"] = state.get("dynamic_workflow")
    session["behaviorGoal"] = state.get("behavior_goal")
    session["clarificationPlan"] = state.get("clarification_plan")
    session["clarificationComplete"] = state.get("clarification_complete") or False
    session["askedClarificationKeys"] = state.get("asked_clarification_keys") or []
    session["askedClarificationQuestions"] = state.get("asked_clarification_questions") or []

    session_history = []
    for m in state.get("history", []):
        role = "agent" if (m.get("role") if isinstance(m, dict) else m.type) in ("assistant", "agent") else "user"
        content = m.get("content") if isinstance(m, dict) else m.content
        session_history.append({"role": role, "content": content})
    session["history"] = session_history


def _build_session_dict_from_state(state: ConversationState) -> dict:
    session = {}
    session["step"] = state.get("current_step", 0)
    session["appType"] = state.get("app_type") or "text"
    session["extraction"] = dict(state.get("extraction") or {})
    session["extraction"]["appType"] = session["appType"]
    session["dynamicContext"] = state.get("dynamic_context")
    session["deepAnswers"] = dict(state.get("deep_answers") or {})
    session["formConfirmed"] = state.get("form_confirmed") or False
    session["modelId"] = state.get("model_id")
    session["modelCost"] = state.get("model_cost")
    session["modelName"] = state.get("model_name")
    session["promptData"] = dict(state.get("prompt_data") or {})
    session["seoData"] = dict(state.get("seo_data") or {})
    session["languageMode"] = state.get("language_mode", "English")
    session["confidence"] = state.get("confidence", "MEDIUM")
    session["reasoning"] = state.get("reasoning", "")
    session["triageRounds"] = state.get("triage_rounds", 0)
    session["lastSlotKey"] = state.get("last_slot_key")
    session["dynamicSlots"] = list(state.get("dynamic_slots") or [])
    session["awaitingDeepAnswer"] = state.get("awaiting_deep_answer") or False
    session["currentDeepField"] = state.get("current_deep_field")
    session["ingestionVector"] = state.get("ingestion_vector")
    session["verificationMetadata"] = dict(state.get("verification_metadata") or {})
    session["dynamicWorkflow"] = state.get("dynamic_workflow")
    session["behaviorGoal"] = state.get("behavior_goal")
    session["clarificationPlan"] = state.get("clarification_plan")
    session["clarificationComplete"] = state.get("clarification_complete") or False
    session["askedClarificationKeys"] = state.get("asked_clarification_keys") or []
    session["askedClarificationQuestions"] = state.get("asked_clarification_questions") or []
    
    session_history = []
    for m in state.get("history", []):
        role = "agent" if (m.get("role") if isinstance(m, dict) else m.type) in ("assistant", "agent") else "user"
        content = m.get("content") if isinstance(m, dict) else m.content
        session_history.append({"role": role, "content": content})
    session["history"] = session_history
    return session


async def intent_classifier_node(state: ConversationState, config: dict) -> dict:
    app_state = config["configurable"]["app_state"]
    message = state.get("message", "")
    text = _normalize(message)
    msg = _lower(text)
    
    session_snapshot = _build_session_dict_from_state(state)
    decision = await get_agentic_decision(app_state.llm, text, session_snapshot)
    action = decision.get("recommended_action") or "GATHER_REQUIREMENTS"
    
    _update_verification_metadata_state(state, text, decision)
    
    if action == "HANDLE_OFF_TOPIC" and _has_active_context(state):
        action = "GATHER_REQUIREMENTS"
    elif msg.startswith(("how ", "what ", "why ", "explain ")) and not _has_active_context(state):
        action = "HANDLE_OFF_TOPIC"

    app_type = decision.get("app_type") or state.get("app_type") or "text"
    state["app_type"] = app_type

    extraction = state.get("extraction") or {}
    extraction["appType"] = app_type

    return {
        "recommended_action": action,
        "app_type": app_type,
        "confidence": decision.get("confidence", "MEDIUM").upper(),
        "reasoning": decision.get("reasoning", ""),
        "extraction": extraction,
        "decision_payload": decision,
        "ingestion_vector": state.get("ingestion_vector"),
        "verification_metadata": state.get("verification_metadata") or {},
    }


async def ideation_triage_node(state: ConversationState, config: dict) -> dict:
    app_state = config["configurable"]["app_state"]
    text = state.get("message", "")
    
    pivot_notice = state.get("pivot_transition")
    if pivot_notice:
        state["pivot_transition"] = None

    is_clarifying = (
        state.get("awaiting_deep_answer")
        or state.get("current_deep_field")
    )
    if not is_clarifying:
        chip_type = _parse_chip_app_type(text)
        if chip_type:
            state["app_type"] = chip_type
            if "extraction" not in state or state["extraction"] is None:
                state["extraction"] = {}
            state["extraction"]["appType"] = chip_type

    if state.get("awaiting_deep_answer") and state.get("current_deep_field") == "ingestion_vector":
        if "deep_answers" not in state or state["deep_answers"] is None:
            state["deep_answers"] = {}
        state["deep_answers"]["ingestion_vector"] = text
        state["awaiting_deep_answer"] = False
        state["current_deep_field"] = None
        _update_verification_metadata_state(state, text)
        result = await _build_step0_response_state(state, text, app_state)
    elif state.get("awaiting_deep_answer") and state.get("current_deep_field") == "budgetPreference":
        if "deep_answers" not in state or state["deep_answers"] is None:
            state["deep_answers"] = {}
        state["deep_answers"]["budgetPreference"] = text
        if "extraction" not in state or state["extraction"] is None:
            state["extraction"] = {}
        state["extraction"]["budget"] = text
        state["awaiting_deep_answer"] = False
        state["current_deep_field"] = None
        _update_verification_metadata_state(state, text)
        result = await _show_models_state(state, app_state)
    else:
        result = await _exec_gather_requirements_state(state, text, app_state)
        
    if pivot_notice:
        result["reply"] = f"{pivot_notice}\n\n{result.get('reply', '')}"
        state["reply"] = result["reply"]
        
    state["response_payload"] = result
    return state


async def form_submission_node(state: ConversationState, config: dict) -> dict:
    app_state = config["configurable"]["app_state"]
    text = state.get("message", "")
    payload = _parse_multi_select_payload(text)
    if payload:
        if "dynamic_context" not in state or not isinstance(state["dynamic_context"], dict):
            state["dynamic_context"] = {}
        state["dynamic_context"]["options"] = payload.get("selectedOptions") or []
        state["dynamic_context"]["variables"] = [{"name": v.get("name"), "placeholder": v.get("placeholder"), "test_value": v.get("value") or ""} for v in (payload.get("variables") or []) if isinstance(v, dict)]
        _prefill_dynamic_context_variables_state(state)
        state["form_confirmed"] = True
        if "extraction" not in state or state["extraction"] is None:
            state["extraction"] = {}
        state["extraction"]["keyFeatures"] = payload.get("selectedOptions") or []
        result = await _show_models_state(state, app_state)
    else:
        result = {"reply": "Invalid Form structure Context.", "uiType": "text"}
    state["reply"] = result.get("reply")
    state["response_payload"] = result
    return state


async def off_topic_handler_node(state: ConversationState, config: dict) -> dict:
    res = {"reply": "I am your RentPrompts Architect. Let me know what app you want to configure!", "uiType": None}
    state["reply"] = res["reply"]
    state["response_payload"] = res
    return state


async def off_topic_inline_node(state: ConversationState, config: dict) -> dict:
    app_state = config["configurable"]["app_state"]
    res = await _rebuild_current_step_response_state(state, app_state, state.get("current_step", 0))
    res["reply"] = f"Let's focus on finishing your custom Rapp logic.\n\n{res.get('reply', '')}"
    state["reply"] = res["reply"]
    state["response_payload"] = res
    return state


async def model_selection_node(state: ConversationState, config: dict) -> dict:
    app_state = config["configurable"]["app_state"]
    res = await _show_models_state(state, app_state)
    state["response_payload"] = res
    return state


async def app_preview_node(state: ConversationState, config: dict) -> dict:
    app_state = config["configurable"]["app_state"]
    res = await _exec_generate_preview_state(state, state["message"], app_state)
    state["response_payload"] = res
    return state


async def modification_handler_node(state: ConversationState, config: dict) -> dict:
    app_state = config["configurable"]["app_state"]
    # When the user explicitly clicks the UI 'Edit App' button, treat clarification as complete
    # so we don't re-enter the clarification planner asking for missing behavioral dimensions
    state["clarification_complete"] = True
    state["awaiting_deep_answer"] = False
    state["current_deep_field"] = None

    res = await _exec_edit_app_state(state, state["message"], state.get("decision_payload") or {}, app_state)
    state["response_payload"] = res
    return state


async def seo_review_node(state: ConversationState, config: dict) -> dict:
    app_state = config["configurable"]["app_state"]
    res = await _exec_review_seo_state(state, app_state)
    state["response_payload"] = res
    return state


async def save_draft_node(state: ConversationState, config: dict) -> dict:
    """Handle saving SEO metadata as a draft. Accepts either a raw UI payload via 'SEO_DRAFT::' prefix
    or a plain 'Save Draft' message that uses the current `state['seo_data']`.
    """
    app_state = config["configurable"]["app_state"]
    message = str(state.get("message") or "")

    payload = {}
    if message.lower().startswith("seo_draft::"):
        try:
            raw = message[len("seo_draft::"):].strip()
            payload = json.loads(raw) if raw else {}
        except Exception:
            payload = {}

    # Merge and persist to state
    seo_data = {**(state.get("seo_data") or {}), **payload}
    state["seo_data"] = seo_data

    # Optionally persist to CMS or DB here. For now, just return confirmation.
    res = {
        "reply": "✅ Draft saved. You can continue editing or publish when ready.",
        "uiType": "seo_preview",
        "uiData": {
            "appName": seo_data.get("appName") or "Your App",
            "appDescription": seo_data.get("appDescription") or "",
            "tags": seo_data.get("tags") or [],
            "appType": state.get("app_type"),
            "modelId": state.get("model_id"),
        },
        "nextStep": state.get("current_step", 2),
    }
    state["response_payload"] = res
    return state


async def publish_app_node(state: ConversationState, config: dict) -> dict:
    app_state = config["configurable"]["app_state"]
    res = await _handle_seo_publish_state(state, {}, app_state)
    state["response_payload"] = res
    state["clear_session"] = True
    return state


async def pivot_manager_node(state: ConversationState, config: dict) -> dict:
    app_state = config["configurable"]["app_state"]
    message = state.get("message", "")
    
    from services.intent_engine import classify_intent_with_pivot_check
    current_context = {
        "appPurpose": (state.get("extraction") or {}).get("appPurpose") or ""
    }
    pivot_data = classify_intent_with_pivot_check(message, current_context)
    prev_cat = pivot_data.get("previous_category") or "unknown"
    new_cat = pivot_data.get("new_category") or "unknown"
    
    transition_text = (
        f"I noticed you've pivoted from a **{prev_cat.capitalize()}** app "
        f"to a **{new_cat.capitalize()}** app. Let me clear the old configuration "
        f"and regenerate clean requirements for you!"
    )
    
    session_id = state.get("session_id")
    if session_id:
        await app_state.session.reset_app_specific_context(session_id)
        
    new_history = list(state.get("history") or [])
    new_history.append({"role": "user", "content": message})
    new_history.append({"role": "assistant", "content": transition_text})
    
    return {
        "history": new_history,
        "app_type": "text",
        "app_purpose": message,
        "extraction": {
            "appPurpose": message,
            "appType": "text"
        },
        "dynamic_context": None,
        "deep_answers": {},
        "current_step": 0,
        "form_confirmed": False,
        "model_id": None,
        "model_cost": None,
        "model_name": None,
        "prompt_data": {},
        "seo_data": {},
        "triage_rounds": 0,
        "last_slot_key": None,
        "dynamic_slots": [],
        "awaiting_deep_answer": False,
        "current_deep_field": None,
        "ingestion_vector": None,
        "verification_metadata": {
            "app_type": "inferred",
            "ingestion_vector": "missing",
            "budget": "missing"
        },
        "clarification_complete": False,
        "behavior_goal": None,
        "clarification_plan": None,
        "asked_clarification_keys": [],
        "asked_clarification_questions": [],
        "pivot_transition": transition_text,
    }


def route_by_conversational_intent(state: ConversationState) -> str:
    message = str(state.get("message") or "").strip()
    message_lower = message.lower()
    action = state.get("recommended_action") or "GATHER_REQUIREMENTS"

    # ONLY hard protocol prefix intercepts are allowed here (case-insensitive)
    if message_lower.startswith("seo_publish::") or action == "PUBLISH_APP":
        return "publish_app_node"
    if message_lower.startswith("confirm seo::") or message_lower in ("approve app", "approve") or action == "REVIEW_SEO":
        return "seo_review_node"
    if message_lower.startswith("seo_draft::") or message_lower in ("save draft", "save_draft") or action == "SAVE_DRAFT":
        return "save_draft_node"
    if message_lower.startswith("multi_select_form::") or action == "PROCESS_FORM":
        return "form_submission_node"
    if message_lower.startswith("select ") and state.get("current_step") == 1:
        return "app_preview_node"

    # Trust the LLM. Pure dispatch.
    ACTION_TO_NODE = {
        "GATHER_REQUIREMENTS": "ideation_triage_node",
        "SHOW_MODEL_CARDS":    "model_selection_node",
        "GENERATE_PREVIEW":    "app_preview_node",
        "EDIT_APP":            "modification_handler_node",
        "PIVOT_APP":           "pivot_manager_node",
        "HANDLE_OFF_TOPIC":    "off_topic_handler_node",
        "REVIEW_SEO":          "seo_review_node",
        "PUBLISH_APP":         "publish_app_node",
    }
    return ACTION_TO_NODE.get(action, "ideation_triage_node")


def check_triage_completeness(state: ConversationState) -> str:
    # Behavioral clarification must be complete before auto-advancing
    if not state.get("clarification_complete"):
        return "wait_for_user"

    # ─── 🛡️ METADATA VERIFICATION GATING ───
    v_meta = state.get("verification_metadata") or {}
    ing_status = v_meta.get("ingestion_vector", "missing")
    budget_status = v_meta.get("budget", "missing")
    app_purpose = state.get("app_purpose") or state.get("extraction", {}).get("appPurpose") or ""
    if ((ing_status in ("missing", "inferred") and requires_input_artifact(state.get("app_type") or state.get("extraction", {}).get("appType"), app_purpose))
            or budget_status in ("missing", "inferred")):
        return "wait_for_user"

    # ─── 🛡️ BUDGET CHECK: Prevent bypassing budget collection ───
    extraction = state.get("extraction") or {}
    answers = state.get("deep_answers") or {}
    if not _session_has_budget(extraction, answers):
        return "wait_for_user"

    return "auto_advance_to_models"


def build_true_agentic_graph() -> StateGraph:
    graph = StateGraph(ConversationState)
    
    # Add nodes
    graph.add_node("intent_classifier_node", intent_classifier_node)
    graph.add_node("pivot_manager_node", pivot_manager_node)
    graph.add_node("off_topic_handler_node", off_topic_handler_node)
    graph.add_node("off_topic_inline_node", off_topic_inline_node)
    graph.add_node("ideation_triage_node", ideation_triage_node)
    graph.add_node("form_submission_node", form_submission_node)
    graph.add_node("model_selection_node", model_selection_node)
    graph.add_node("app_preview_node", app_preview_node)
    graph.add_node("modification_handler_node", modification_handler_node)
    graph.add_node("seo_review_node", seo_review_node)
    graph.add_node("save_draft_node", save_draft_node)
    graph.add_node("publish_app_node", publish_app_node)
    
    # Set entry routing logic
    graph.set_entry_point("intent_classifier_node")
    graph.add_conditional_edges("intent_classifier_node", route_by_conversational_intent, {
        "off_topic_handler_node": "off_topic_handler_node",
        "off_topic_inline_node": "off_topic_inline_node",
        "ideation_triage_node": "ideation_triage_node",
        "form_submission_node": "form_submission_node",
        "model_selection_node": "model_selection_node",
        "app_preview_node": "app_preview_node",
        "modification_handler_node": "modification_handler_node",
        "seo_review_node": "seo_review_node",
        "publish_app_node": "publish_app_node",
        "pivot_manager_node": "pivot_manager_node",
    })
    
    # Direct edge transition
    graph.add_edge("pivot_manager_node", "ideation_triage_node")
    
    # Autonomous routing evaluated right after requirements tracking
    graph.add_conditional_edges(
        "ideation_triage_node",
        check_triage_completeness,
        {
            "wait_for_user": END,                            
            "auto_advance_to_models": "model_selection_node" 
        }
    )
    
    # ─── 🛠️ THE CRITICAL EDGE REPAIR ───
    # Break the direct node-to-node bypass. Let model selection stop so cards render.
    graph.add_edge("model_selection_node", END)
    graph.add_edge("app_preview_node", END)
    
    # Other standard exit routes
    graph.add_edge("off_topic_handler_node", END)
    graph.add_edge("off_topic_inline_node", END)
    graph.add_edge("form_submission_node", END)
    graph.add_edge("modification_handler_node", END)
    graph.add_edge("seo_review_node", END)
    graph.add_edge("save_draft_node", END)
    graph.add_edge("publish_app_node", END)
    
    return graph.compile()


compiled_graph = build_true_agentic_graph()

async def route(session: dict, message: str, app_state: Any) -> dict:
    initial_state = _session_to_state(session, message)
    config = {"configurable": {"app_state": app_state}}
    final_state = await compiled_graph.ainvoke(initial_state, config=config)
    _state_to_session(final_state, session)
    return final_state.get("response_payload") or {}


async def _show_models(session: dict, app_state: Any) -> dict:
    state = _session_to_state(session, "")
    res = await _show_models_state(state, app_state)
    _state_to_session(state, session)
    return res


async def _exec_edit_app(session: dict, text: str, decision: dict, app_state: Any) -> dict:
    state = _session_to_state(session, text)
    state["decision_payload"] = decision
    res = await _exec_edit_app_state(state, text, decision, app_state)
    _state_to_session(state, session)
    return res