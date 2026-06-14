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
    triage_dynamic_context,
)
from services.intent_engine import get_agentic_decision
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
    if any(s in v for s in ("vision", "image analyzer", "analyze or understand images")):
        return "vision"
    if "video" in v:
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


def _merge_extraction(existing: dict | None, latest: dict | None, message: str) -> dict:
    if not latest:
        return existing or {}
    if not existing:
        return latest

    is_control = bool(
        _parse_selected_model_id(message, None)
        or _parse_selected_plan(message)
        or _parse_chip_app_type(message)
        or _is_yes(message)
    )
    
    # ─── 🛡️ FORMAT RECOVERY PROTECTION LOCK ───
    if existing.get("appType") and (not latest.get("appType") or latest.get("appType") == "None"):
        latest["appType"] = existing["appType"]

    keep_app_type = is_control or not latest.get("appType")
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
        "appType": existing.get("appType") if keep_app_type else latest.get("appType"),
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
            prefilled.append({
                "name": dim_name,
                "placeholder": f"Enter {dim_name.lower()}...",
                "test_value": val.strip()
            })
            existing_names.add(dim_name.lower())
            
    dynamic_context["variables"] = prefilled[:4]
    session["dynamicContext"] = dynamic_context


async def _save(session: dict, app_state: Any) -> None:
    await app_state.session.save_session(session)


async def _show_models(session: dict, app_state: Any) -> dict:
    full_text = " ".join([
        str((session.get("extraction") or {}).get("appPurpose") or ""),
        str((session.get("extraction") or {}).get("oneLineUnderstanding") or ""),
        json.dumps(session.get("deepAnswers") or {}),
    ])
    budget = (session.get("deepAnswers") or {}).get("budgetPreference") or ((session.get("extraction") or {}).get("budget"))
    model_collection = MODELS.get(session.get("appType") or "text", MODELS.get("text", []))
    
    models = _rank_models(model_collection, full_text, str(budget or ""), artistic_priority=False)

    session["step"] = 1
    session["awaitingConfirmation"] = False
    await _save(session, app_state)

    return {
        "reply": (
            f"## 🤖 AI Model Selection\n\nI've ranked the **top 3 models** for your "
            f"**{session.get('appType')}** app based on your requirements and budget.\n\n"
            "Each card shows the model's strengths, speed, and cost per run — **click any card** to select it."
        ),
        "uiType": "models",
        "uiData": {"appType": session.get("appType"), "models": models},
        "nextStep": 1,
        "coins": None,
    }


async def _build_step0_response(session: dict, text: str, app_state: Any) -> dict:
    llm = app_state.llm
    ext = session.get("extraction") or {}

    # Force fallback if local mapping gets empty values
    if not session.get("appType") or session.get("appType") == "None":
        session["appType"] = "text"
        if not session.get("extraction"): session["extraction"] = {}
        session["extraction"]["appType"] = "text"

    session["step"] = 0
    session["awaitingConfirmation"] = False

    rag_context = ""
    vector_store = getattr(app_state, "vector_store", None)
    if vector_store and hasattr(vector_store, "search"):
        try:
            matches = await vector_store.search(query=ext.get("appPurpose") or text, categories=["marketplace", "examples"], top_k=2)
            rag_context = "\n\n".join([m.get("content", "") for m in matches if m.get("content")])
        except Exception as e:
            logger.warning(f"Triage RAG lookup failed: {e}")

    triage_result = await triage_dynamic_context(
        llm, session.get("appType"), (session.get("extraction") or {}).get("appPurpose") or "",
        _detect_language_mode(session), session.get("history") or [], session.get("deepAnswers") or {}, rag_context=rag_context
    )

    required_slots = ['PRIMARY_SUBJECT', 'ENVIRONMENT_SETTING', 'ACTION_DYNAMIC', 'AESTHETIC_STYLE']
    filled_slots = [s for s in required_slots if ext.get(s)]
    
    # ─── 🛡️ PRD REQUIREMENTS SCOPING RULES: ANTI-LOOP & CAP ───
    triage_rounds = session.get("triageRounds", 0) or session.get("triage_rounds", 0)
    deep_answers = session.get("deepAnswers") or {}
    populated_keys = {k.lower().strip() for k, v in deep_answers.items() if v and str(v).strip()}
    
    triage_slot_key = str(triage_result.get("slot_key") or "").lower().strip()
    if triage_slot_key in populated_keys or any(k in populated_keys for k in ["tone", "length", "theme", "audience"] if triage_slot_key == k):
        triage_result["status"] = "ready"
        triage_result["question"] = None
        triage_result["slot_key"] = None

    if triage_rounds >= 2:
        triage_result["status"] = "ready"
        triage_result["question"] = None
        triage_result["slot_key"] = None
    
    # ─── 🛡️ STRICT PIECE: FORCE TRIAGE QUESTIONS, STOP POISONED OVERWRITES ───
    if triage_result.get("status") == "ready" and len(filled_slots) < 2 and triage_rounds < 2:
        logger.info("[Guard] Overfitting signal locked. Re-routing loop target down to context acquisition.")
        triage_result["status"] = "needs_context"
        triage_result["question"] = "What specific focus, tone, or dynamic parameters should the speech generation include?"
        triage_result["slot_key"] = "visual_style"
        triage_result["form"] = None

    if triage_result.get("status") == "needs_context":
        question = str(triage_result.get("question") or "").strip()
        slot_key = triage_result.get("slot_key") or _infer_slot_key_from_question(question)
        session["lastSlotKey"] = slot_key
        session["triageRounds"] = (session.get("triageRounds") or 0) + 1
        session["lastQuestion"] = question
        await _save(session, app_state)
        return {
            "reply": question,
            "uiType": None,
            "uiData": None,
            "nextStep": 0,
            "coins": None,
        }

    if triage_result.get("form"):
        session["dynamicContext"] = triage_result["form"]
    else:
        session["dynamicContext"] = await generate_dynamic_context(llm, session.get("appType") or "text", (session.get("extraction") or {}).get("appPurpose") or "", _detect_language_mode(session))
    
    session["formConfirmed"] = True
    session["triageRounds"] = 0
    prefill_dynamic_context_variables(session)
    await _save(session, app_state)

    deep_answers = session.get("deepAnswers") or {}
    if not _session_has_budget(session.get("extraction"), deep_answers):
        session["currentDeepField"] = "budgetPreference"
        session["awaitingDeepAnswer"] = True
        await _save(session, app_state)
        return {
            "reply": f"Excellent choice! I have saved the blueprint for your app. One last parameter to proceed: **What is your budget preference per run?**",
            "uiType": "chips",
            "uiData": {"options": BUDGET_CHIP_OPTIONS},
            "nextStep": 0,
            "coins": None,
        }

    return await _show_models(session, app_state)


async def _exec_gather_requirements(session: dict, text: str, app_state: Any) -> dict:
    if not session.get("history"):
        session["history"] = []
    latest_extraction = await extract_requirements(app_state.llm, text, session["history"])
    session["extraction"] = _merge_extraction(session.get("extraction"), latest_extraction, text)
    
    session["languageMode"] = _detect_language_mode(session)
    extraction = session.get("extraction") or {}
    if not session.get("appType") and extraction.get("appType"):
        session["appType"] = extraction["appType"]
    return await _build_step0_response(session, text, app_state)


async def _exec_generate_preview(session: dict, text: str, app_state: Any) -> dict:
    llm = app_state.llm
    app_type = session.get("appType") or "text"
    selected_model_id = _parse_selected_model_id(text, MODELS.get(app_type, []))
    if not selected_model_id:
        return {"reply": "Please **click one of the model cards** above to select the AI engine. 👆", "uiType": "text", "nextStep": 1}

    selected_model = _find_model(app_type, selected_model_id)
    if not selected_model:
        return {"reply": "I couldn't match that model. Please click one of the options above.", "uiType": "text", "nextStep": 1}

    session["modelId"] = selected_model["id"]
    session["modelCost"] = selected_model["cost"]
    session["modelName"] = selected_model["name"]
    await _save(session, app_state)

    try:
        if not session.get("dynamicContext") or not (session.get("dynamicContext") or {}).get("variables"):
            session["dynamicContext"] = await generate_dynamic_context(llm, app_type, (session.get("extraction") or {}).get("appPurpose") or "", _detect_language_mode(session))

        prefill_dynamic_context_variables(session)

        # Task 1: Grounding with Web Search Tools
        search_query = f"{app_type} model {session.get('modelName')} prompting guidelines parameters"
        try:
            search_tool = get_web_search_tool()
            search_result = await search_tool.search_and_summarize(search_query)
            session["webSearchContext"] = search_result
        except Exception as search_err:
            logger.warning(f"WebSearch grounding failed: {search_err}")

        prompt_data, seo_data = await asyncio.gather(
            generate_prompt_template(llm, session),
            generate_seo(llm, session),
        )

        session["promptData"] = prompt_data
        session["seoData"] = seo_data
        session["step"] = 2
        await _save(session, app_state)

        return {
            "reply": f"## App Preview Ready\n\nI've configured the full AI logic using **{selected_model['name']}**.\n\nTest it in the Live Preview below — click **Approve App** when ready!",
            "uiType": "app_preview",
            "uiData": {
                "appName": seo_data.get("appName"),
                "appType": app_type,
                "appDescription": seo_data.get("appDescription"),
                "cost": session.get("modelCost"),
                "systemPrompt": prompt_data.get("systemPrompt"),
                "userPrompt": prompt_data.get("userPrompt"),
                "variablesUsed": prompt_data.get("variablesUsed"),
                "variables": (session.get("dynamicContext") or {}).get("variables") or [],
                "acceptImageInput": _sanitize_accept_image_input(prompt_data.get("acceptImageInput"), app_type),
                "options": ["Approve App", "Edit App"],
                "step": 2,
            },
            "nextStep": 2,
            "coins": session.get("modelCost"),
        }
    except Exception as err:
        logger.error(f"Preview generation Crash: {err}")
        return {"reply": "Oops! hit a snag generating the config. Please retry.", "uiType": "text", "nextStep": 1}


async def _handle_seo_publish(session: dict, card_data: dict, app_state: Any) -> dict:
    prompt_data = session.get("promptData") or {}
    seo_data = {**(session.get("seoData") or {}), **card_data}
    session["seoData"] = seo_data

    app_name = seo_data.get("appName") or "Your App"
    alt_text = str(seo_data.get("appDescription") or app_name)[:500]
    media_id = None

    # Task 2: Real Media Upload to /api/media Multipart Form Call
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

    is_private = bool(session.get("isPrivate") or session.get("is_private") or card_data.get("isPrivate"))
    variables_used = prompt_data.get("variablesUsed") or []
    var_descriptions = prompt_data.get("variableDescriptions") or {}
    prompt_variables = [
        {"name": v, "description": var_descriptions.get(v, f"Enter {str(v).replace('_', ' ')}")}
        for v in variables_used
    ]

    # Task 3: Public and Private Endpoints Payload Structuring Alignment
    if is_private:
        payload = {
            "name": app_name,
            "description": seo_data.get("appDescription"),
            "modelType": session.get("appType"),
            "model": session.get("modelId"),
            "systemprompt": prompt_data.get("systemPrompt"),
            "prompt": prompt_data.get("userPrompt"),
            "negativeprompt": prompt_data.get("negativePrompt"),
            "priceapplicable": True,
            "price": session.get("modelCost"),
            "promptVariables": prompt_variables,
        }
    else:
        payload = {
            "name": app_name,
            "description": seo_data.get("appDescription"),
            "modelType": session.get("appType"),
            "model": session.get("modelId"),
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
        return {"reply": "Publishing hit a snag on our end. Please try again.", "uiType": "text"}

    session_id = session.get("sessionId")
    if session_id:
        await app_state.session.delete_session(session_id)

    return {
        "reply": f'## 🎉 Published Successfully!\n\nYour app **"{app_name}"** is now live on {"your private registry" if is_private else "the marketplace"}!',
        "uiType": "success",
        "uiData": {"appName": app_name, "modelCost": session.get("modelCost")},
        "nextStep": 0,
        "clearSession": True,
    }


# ─── EXTRA EXECUTION REDIRECT INTERFACES ───
async def _exec_review_seo(session: dict, app_state: Any) -> dict:
    session["step"] = 3
    await _save(session, app_state)
    seo_data = session.get("seoData") or {}
    return {
        "reply": "## 🎉 App Configured — Final Review\n\nReview your app metadata profile details below.",
        "uiType": "seo_preview",
        "uiData": {
            "appName": seo_data.get("appName") or "Your App",
            "appDescription": seo_data.get("appDescription") or "",
            "tags": seo_data.get("tags") or [],
            "appType": session.get("appType"),
            "modelId": session.get("modelId"),
            "costPerRun": session.get("modelCost"),
        },
        "nextStep": 3,
    }

async def _exec_handle_budget(session: dict, text: str, decision: dict, app_state: Any) -> dict:
    tier = _extract_budget_tier(text, decision)
    if tier:
        if not session.get("extraction"): session["extraction"] = {}
        session["extraction"]["budget"] = tier
        if not session.get("deepAnswers"): session["deepAnswers"] = {}
        session["deepAnswers"]["budgetPreference"] = tier
        session["awaitingDeepAnswer"] = False
        session["currentDeepField"] = None
        await _save(session, app_state)
        return await _show_models(session, app_state)
    return {"reply": "What budget tier works?", "uiType": "chips", "uiData": {"options": BUDGET_CHIP_OPTIONS}}

async def _exec_pivot_app(session: dict, text: str, decision: dict, app_state: Any) -> dict:
    new_type = decision.get("app_type") or _parse_chip_app_type(text) or "text"
    session["appType"] = new_type
    session["extraction"] = {"appPurpose": text, "appType": new_type}
    session["step"] = 0
    await _save(session, app_state)
    return await _build_step0_response(session, text, app_state)

async def _exec_edit_app(session: dict, text: str, decision: dict, app_state: Any) -> dict:
    extracted = decision.get("extracted_variables") or {}
    instruction = extracted.get("editInstruction") or text
    _apply_edit_to_session(session, instruction)
    session["step"] = 2
    await _save(session, app_state)
    return await _exec_generate_preview(session, text, app_state)

async def _rebuild_current_step_response(session: dict, app_state: Any, step: int) -> dict:
    if step >= 3: return await _exec_review_seo(session, app_state)
    if step == 2: return await _exec_generate_preview(session, "", app_state)
    if step == 1: return await _show_models(session, app_state)
    return {"reply": "Let's continue shaping your application.", "uiType": None}


# ─── LANGGRAPH CORE CONTEXT INTERFACE LIFECYCLE ───

class ConversationState(TypedDict, total=False):
    session_id: str
    message: str
    history: Annotated[list, add_messages]
    app_type: Optional[str]
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


def _session_to_state(session: dict, message: str) -> ConversationState:
    hist = []
    for m in session.get("history", []):
        role = "assistant" if m.get("role") in ("agent", "assistant") else "user"
        hist.append({"role": role, "content": m.get("content", "")})
    
    a_type = session.get("appType")
    if not a_type or a_type == "None": a_type = "text"

    return {
        "session_id": session.get("sessionId") or "",
        "message": message,
        "history": hist,
        "app_type": a_type,
        "extraction": session.get("extraction") or {"appType": a_type},
        "dynamic_context": session.get("dynamicContext"),
        "deep_answers": session.get("deepAnswers") or {},
        "current_step": session.get("step") or 0,
        "recommended_action": "",
        "confidence": session.get("confidence") or "MEDIUM",
        "reasoning": session.get("reasoning") or "",
        "response_payload": {},
        "form_confirmed": session.get("formConfirmed") or False,
        "model_id": session.get("modelId"),
        "model_cost": session.get("modelCost"),
        "model_name": session.get("modelName"),
        "prompt_data": session.get("promptData") or {},
        "seo_data": session.get("seoData") or {},
        "clear_session": False,
        "language_mode": session.get("languageMode") or "English",
    }


def _state_to_session(state: ConversationState, session: dict) -> None:
    session["step"] = state.get("current_step", 0)
    
    a_type = state.get("app_type") or "text"
    session["appType"] = a_type
    
    session["extraction"] = state.get("extraction") or {}
    session["extraction"]["appType"] = a_type
    session["dynamicContext"] = state.get("dynamic_context")
    session["deepAnswers"] = state.get("deep_answers") or {}
    session["formConfirmed"] = state.get("form_confirmed") or False
    session["modelId"] = state.get("model_id")
    session["modelCost"] = state.get("model_cost")
    session["modelName"] = state.get("model_name")
    session["promptData"] = state.get("prompt_data") or {}
    session["seoData"] = state.get("seo_data") or {}
    session["languageMode"] = state.get("language_mode", "English")
    session["confidence"] = state.get("confidence", "MEDIUM")
    session["reasoning"] = state.get("reasoning", "")

    session_history = []
    for m in state.get("history", []):
        role = "agent" if (m.get("role") if isinstance(m, dict) else m.type) in ("assistant", "agent") else "user"
        content = m.get("content") if isinstance(m, dict) else m.content
        session_history.append({"role": role, "content": content})
    session["history"] = session_history


async def intent_classifier_node(state: ConversationState, config: dict) -> dict:
    app_state = config["configurable"]["app_state"]
    message = state.get("message", "")
    text = _normalize(message)
    msg = _lower(text)
    
    if text.lower().startswith("multi_select_form::"):
        return {
            "recommended_action": "PROCESS_FORM",
            "app_type": state.get("app_type") or "text",
            "confidence": "HIGH",
            "reasoning": "Form submitted.",
        }
    if text.startswith("SEO_PUBLISH::"):
        return {
            "recommended_action": "PUBLISH_APP",
            "app_type": state.get("app_type") or "text",
            "confidence": "HIGH",
            "reasoning": "Publish requested via SEO form.",
        }
    if text.startswith("SEO_DRAFT::"):
        return {
            "recommended_action": "SAVE_DRAFT",
            "app_type": state.get("app_type") or "text",
            "confidence": "HIGH",
            "reasoning": "Save draft requested via SEO form.",
        }
    if state.get("current_step") in (2, 3) and msg == "edit app":
        return {
            "recommended_action": "EDIT_APP",
            "app_type": state.get("app_type") or "text",
            "confidence": "HIGH",
            "reasoning": "Edit app trigger.",
        }
        
    temp_session = {}
    _state_to_session(state, temp_session)
    decision = await get_agentic_decision(app_state.llm, text, temp_session)
    action = decision.get("recommended_action") or "GATHER_REQUIREMENTS"
    
    if msg.startswith(("how ", "what ", "why ", "explain ")) or action == "HANDLE_OFF_TOPIC":
        action = "HANDLE_OFF_TOPIC"

    # ─── 🛡️ SECURE PIPELINE APP_TYPE PRESERVATION LOOP ───
    app_type = state.get("app_type") or temp_session.get("appType") or "text"
    new_inferred = decision.get("app_type")
    
    # Block casual/empty text conversions unless explicitly switching via PIVOT
    if new_inferred and new_inferred != "None" and action == "PIVOT_APP":
        app_type = new_inferred
    elif app_type == "None" or not app_type:
        app_type = "text"

    extraction = state.get("extraction") or {}
    extraction["appType"] = app_type

    return {
        "recommended_action": action,
        "app_type": app_type,
        "confidence": decision.get("confidence", "MEDIUM").upper(),
        "reasoning": decision.get("reasoning", ""),
        "extraction": extraction,
        "decision_payload": decision,
    }


# ─── ASYNC PIPELINE NODE ROUTERS ───

async def ideation_triage_node(state: ConversationState, config: dict) -> dict:
    app_state = config["configurable"]["app_state"]
    text = state.get("message", "")
    temp_session = {}
    _state_to_session(state, temp_session)
    
    chip_type = _parse_chip_app_type(text)
    if chip_type:
        temp_session["appType"] = chip_type
        temp_session["extraction"]["appType"] = chip_type

    if temp_session.get("awaitingDeepAnswer") and temp_session.get("currentDeepField") == "budgetPreference":
        if not temp_session.get("deepAnswers"): temp_session["deepAnswers"] = {}
        temp_session["deepAnswers"]["budgetPreference"] = text
        temp_session["extraction"]["budget"] = text
        temp_session["awaitingDeepAnswer"] = False
        temp_session["currentDeepField"] = None
        result = await _show_models(temp_session, app_state)
    else:
        result = await _exec_gather_requirements(temp_session, text, app_state)
        
    new_state = _session_to_state(temp_session, text)
    new_state["response_payload"] = result
    return new_state


async def form_submission_node(state: ConversationState, config: dict) -> dict:
    app_state = config["configurable"]["app_state"]
    text = state.get("message", "")
    temp_session = {}
    _state_to_session(state, temp_session)
    payload = _parse_multi_select_payload(text)
    if payload:
        if not temp_session.get("dynamicContext"): temp_session["dynamicContext"] = {}
        temp_session["dynamicContext"]["options"] = payload.get("selectedOptions") or []
        temp_session["dynamicContext"]["variables"] = [{"name": v.get("name"), "placeholder": v.get("placeholder"), "test_value": v.get("value") or ""} for v in (payload.get("variables") or []) if isinstance(v, dict)]
        prefill_dynamic_context_variables(temp_session)
        temp_session["formConfirmed"] = True
        temp_session["extraction"]["keyFeatures"] = payload.get("selectedOptions") or []
        result = await _show_models(temp_session, app_state)
    else:
        result = {"reply": "Invalid Form structure Context.", "uiType": "text"}
    new_state = _session_to_state(temp_session, text)
    new_state["response_payload"] = result
    return new_state

async def off_topic_handler_node(state: ConversationState, config: dict) -> dict:
    app_state = config["configurable"]["app_state"]
    res = {"reply": "I am your RentPrompts Architect. Let me know what app you want to configure!", "uiType": None}
    return {"response_payload": res}

async def off_topic_inline_node(state: ConversationState, config: dict) -> dict:
    app_state = config["configurable"]["app_state"]
    temp_session = {}
    _state_to_session(state, temp_session)
    res = await _rebuild_current_step_response(temp_session, app_state, state.get("current_step", 0))
    res["reply"] = f"Let's focus on finishing your custom Rapp logic.\n\n{res.get('reply', '')}"
    return {"response_payload": res}

async def model_selection_node(state: ConversationState, config: dict) -> dict:
    app_state = config["configurable"]["app_state"]
    temp_session = {}
    _state_to_session(state, temp_session)
    res = await _show_models(temp_session, app_state)
    new_state = _session_to_state(temp_session, state["message"])
    new_state["response_payload"] = res
    return new_state

async def app_preview_node(state: ConversationState, config: dict) -> dict:
    app_state = config["configurable"]["app_state"]
    temp_session = {}
    _state_to_session(state, temp_session)
    res = await _exec_generate_preview(temp_session, state["message"], app_state)
    new_state = _session_to_state(temp_session, state["message"])
    new_state["response_payload"] = res
    return new_state

async def modification_handler_node(state: ConversationState, config: dict) -> dict:
    app_state = config["configurable"]["app_state"]
    temp_session = {}
    _state_to_session(state, temp_session)
    res = await _exec_edit_app(temp_session, state["message"], state.get("decision_payload") or {}, app_state)
    new_state = _session_to_state(temp_session, state["message"])
    new_state["response_payload"] = res
    return new_state

async def seo_review_node(state: ConversationState, config: dict) -> dict:
    app_state = config["configurable"]["app_state"]
    temp_session = {}
    _state_to_session(state, temp_session)
    res = await _exec_review_seo(temp_session, app_state)
    return {"response_payload": res}

async def publish_app_node(state: ConversationState, config: dict) -> dict:
    app_state = config["configurable"]["app_state"]
    temp_session = {}
    _state_to_session(state, temp_session)
    res = await _handle_seo_publish(temp_session, {}, app_state)
    new_state = _session_to_state(temp_session, state["message"])
    new_state["response_payload"] = res
    new_state["clear_session"] = True
    return new_state


def route_by_conversational_intent(state: ConversationState) -> str:
    action = state.get("recommended_action")
    message = str(state.get("message") or "").strip()
    msg = message.lower()

    # Special UI fast paths
    if msg.startswith("seo_publish::") or msg in ("publish to marketplace", "save draft") or action == "PUBLISH_APP":
        return "publish_app_node"
    if msg.startswith("confirm seo::") or msg in ("approve app", "approve") or action == "REVIEW_SEO":
        return "seo_review_node"
    if msg.startswith("multi_select_form::") or action == "PROCESS_FORM":
        return "form_submission_node"
        
    if not state.get("form_confirmed") and action not in ("GATHER_REQUIREMENTS", "PROCESS_FORM", "HANDLE_OFF_TOPIC"):
        return "ideation_triage_node"

    if action == "HANDLE_OFF_TOPIC":
        return "off_topic_inline_node" if _has_active_context(state) else "off_topic_handler_node"
    if action == "SHOW_MODEL_CARDS": 
        return "model_selection_node"
    if action == "GENERATE_PREVIEW": 
        return "app_preview_node"
    if action == "EDIT_APP": 
        return "modification_handler_node"
    
    return "ideation_triage_node"


def build_orchestrator_graph() -> StateGraph:
    graph = StateGraph(ConversationState)
    graph.add_node("intent_classifier_node", intent_classifier_node)
    graph.add_node("off_topic_handler_node", off_topic_handler_node)
    graph.add_node("off_topic_inline_node", off_topic_inline_node)
    graph.add_node("ideation_triage_node", ideation_triage_node)
    graph.add_node("form_submission_node", form_submission_node)
    graph.add_node("model_selection_node", model_selection_node)
    graph.add_node("app_preview_node", app_preview_node)
    graph.add_node("modification_handler_node", modification_handler_node)
    graph.add_node("seo_review_node", seo_review_node)
    graph.add_node("publish_app_node", publish_app_node)
    
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
    })
    
    graph.add_edge("off_topic_handler_node", END)
    graph.add_edge("off_topic_inline_node", END)
    graph.add_edge("ideation_triage_node", END)
    graph.add_edge("form_submission_node", END)
    graph.add_edge("model_selection_node", END)
    graph.add_edge("app_preview_node", END)
    graph.add_edge("modification_handler_node", END)
    graph.add_edge("seo_review_node", END)
    graph.add_edge("publish_app_node", END)
    
    return graph.compile()

compiled_graph = build_orchestrator_graph()

async def route(session: dict, message: str, app_state: Any) -> dict:
    initial_state = _session_to_state(session, message)
    config = {"configurable": {"app_state": app_state}}
    final_state = await compiled_graph.ainvoke(initial_state, config=config)
    _state_to_session(final_state, session)
    return final_state.get("response_payload") or {}