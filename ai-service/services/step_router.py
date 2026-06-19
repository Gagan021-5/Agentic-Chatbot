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
    is_personal_boilerplate,
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
    v = _lower(msg)
    if any(s in v for s in ("image", "images", "poster", "posters", "photo", "photos", "graphic", "art")):
        return "image"
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
    
    # ─── 🛡️ CRITICAL VALUE INTENT OVERRIDE INTERCEPTOR ───
    # If the text message contains formatting keywords, unconditionally prioritize them over stale history states
    msg_l = message.lower()
    if any(s in msg_l for s in ("image", "images", "poster", "posters", "photo", "photos", "graphic")):
        resolved_app_type = "image"
    elif any(s in msg_l for s in ("video", "videos", "animation", "reel", "reels", "cinematic clip")):
        resolved_app_type = "video"
    elif any(s in msg_l for s in ("audio", "voiceover", "voice", "speech aloud", "podcast", "tts")):
        resolved_app_type = "audio"
    else:
        existing_app_type = existing.get("appType")
        latest_app_type = latest.get("appType")
        specialized_formats = {"audio", "video", "image", "vision"}
        
        if latest_app_type and latest_app_type not in ("None", "text", None):
            resolved_app_type = latest_app_type
        else:
            resolved_app_type = existing_app_type or latest_app_type or "text"

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
        "appType": resolved_app_type,
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


async def _save(session: dict, app_state: Any) -> None:
    await app_state.session.save_session(session)


async def _show_models(session: dict, app_state: Any) -> dict:
    # Double validation step ensuring dynamic context doesn't bypass catalog formats
    user_history = [h for h in (session.get("history") or []) if h.get("role") in ("user", "human")]
    user_history_str = " ".join([str(h.get("content", "")).lower() for h in user_history])
    purpose_str = str((session.get("extraction") or {}).get("appPurpose") or "").lower()
    
    last_user_msg = str(user_history[-1].get("content") or "").lower() if user_history else ""
    is_text_correction = "text" in last_user_msg and any(w in last_user_msg for w in ("it is", "i want", "use", "change to", "make it", "it's", "app", "tool"))
    
    if not is_text_correction and any(s in user_history_str or s in purpose_str for s in ("image", "images", "poster", "posters", "photo")):
        if session.get("appType") not in ("audio", "video", "vision", "text") or is_text_correction:
            session["appType"] = "image"
            if session.get("extraction"):
                session["extraction"]["appType"] = "image"

    app_type_str = session.get("appType") or "text"
    full_text = " ".join([
        str((session.get("extraction") or {}).get("appPurpose") or ""),
        str((session.get("extraction") or {}).get("oneLineUnderstanding") or ""),
        json.dumps(session.get("deepAnswers") or {}),
    ])
    budget = (session.get("deepAnswers") or {}).get("budgetPreference") or ((session.get("extraction") or {}).get("budget"))
    model_collection = MODELS.get(app_type_str, MODELS.get("text", []))
    
    models = _rank_models(model_collection, full_text, str(budget or ""), artistic_priority=False)

    session["step"] = 1
    session["awaitingConfirmation"] = False
    await _save(session, app_state)

    # Build user-facing capability description instead of exposing model names
    app_type_labels = {
        "audio": {
            "speech":  ["tts", "voice", "speech", "narration", "spoken"],
            "music":   ["music", "song", "lyria", "melody", "instrumental", "beat"],
        }
    }

    def _build_capability_summary(app_type: str, ranked_models: list) -> str:
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

    capability_summary = _build_capability_summary(app_type_str, models)

    # Strip internal model names from display — show friendly labels only
    display_models = []
    for m in models:
        display_models.append({
            **m,
            # Keep id internal but replace visible name with capability label
            "displayName": m.get("name"),   # frontend can use this if it wants
            "name": m.get("name"),          # keep for internal selection
        })

    return {
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


async def _build_step0_response(session: dict, text: str, app_state: Any) -> dict:
    llm = app_state.llm
    ext = session.get("extraction") or {}

    text_l = text.lower()
    if any(s in text_l for s in ("image", "images", "poster", "posters", "photo", "photos")):
        session["appType"] = "image"
        if not session.get("extraction"): session["extraction"] = {}
        session["extraction"]["appType"] = "image"

    if not session.get("appType") or session.get("appType") == "None":
        session["appType"] = "text"
        if not session.get("extraction"): session["extraction"] = {}
        session["extraction"]["appType"] = "text"

    session["step"] = 0
    session["awaitingConfirmation"] = False

    triage_rounds = session.get("triageRounds", 0)
    app_type = session.get("appType") or "text"

    dynamic_slots = session.get("dynamicSlots") or []
    deep_answers = session.get("deepAnswers") or {}
    captured_slots = {}
    for k, v in deep_answers.items():
        if v and str(v).strip():
            captured_slots[str(k).lower().strip()] = str(v).strip()

    missing_slots = []
    for slot_obj in dynamic_slots:
        slot_key = str(slot_obj.get("key") or "").lower().strip()
        found = False
        for k in captured_slots:
            if slot_key == k or slot_key in k or k in slot_key:
                found = True
                break
        if not found:
            missing_slots.append(slot_obj)

    is_satisfied = len(dynamic_slots) > 0 and len(missing_slots) == 0

    if not dynamic_slots or (not is_satisfied and triage_rounds < 3):
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

        if not dynamic_slots:
            dynamic_slots = triage_result.get("slots") or []
            session["dynamicSlots"] = dynamic_slots
            
            missing_slots = []
            for slot_obj in dynamic_slots:
                slot_key = str(slot_obj.get("key") or "").lower().strip()
                found = False
                for k in captured_slots:
                    if slot_key == k or slot_key in k or k in slot_key:
                        found = True
                        break
                if not found:
                    missing_slots.append(slot_obj)
            is_satisfied = len(dynamic_slots) > 0 and len(missing_slots) == 0

        if not is_satisfied and triage_rounds < 3:
            if missing_slots and (triage_result.get("status") == "ready" or 
                not triage_result.get("question") or 
                triage_result.get("slot_key") not in [s.get("key") for s in missing_slots]):
                next_slot = missing_slots[0]
                triage_result["status"] = "needs_context"
                triage_result["slot_key"] = next_slot.get("key")
                triage_result["question"] = next_slot.get("question")
                triage_result["form"] = None
        else:
            triage_result["status"] = "ready"
            triage_result["question"] = None
            triage_result["slot_key"] = None
            triage_result["form"] = None
    else:
        triage_result = {
            "status": "ready",
            "question": None,
            "slot_key": None,
            "domain_identified": app_type,
            "confidence_score": 100,
            "form": None,
        }

    if triage_result.get("status") == "needs_context":
        question = str(triage_result.get("question") or "").strip()
        slot_key = triage_result.get("slot_key") or _infer_slot_key_from_question(question)
        session["lastSlotKey"] = slot_key
        session["triageRounds"] = triage_rounds + 1
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
        _app_purpose = (session.get("extraction") or {}).get("appPurpose") or ""
        _rag_context = ""
        vector_store = getattr(app_state, "vector_store", None)
        if vector_store and hasattr(vector_store, "search"):
            try:
                matches = await vector_store.search(
                    query=f"{session.get('appType')} app variables for {_app_purpose}",
                    categories=["examples", "blueprints"],
                    top_k=3,
                )
                _rag_context = "\n\n".join([m.get("content", "") for m in matches if m.get("content")])
            except Exception as e:
                logger.warning(f"RAG lookup for generate_dynamic_context failed: {e}")

        session["dynamicContext"] = await generate_dynamic_context(
            llm,
            session.get("appType") or "text",
            _app_purpose,
            _detect_language_mode(session),
            rag_context=_rag_context,
        )
    
    session["formConfirmed"] = True
    session["triageRounds"] = 0
    session["lastSlotKey"] = None
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
    # Guard: if the message is too short or looks like small talk, bail early
    if len(text.strip()) < 6 and not any(c.isalpha() for c in text[2:]):
        return {
            "reply": "I am your RentPrompts Architect. Let me know what app you want to configure!",
            "uiType": None,
            "uiData": None,
            "nextStep": 0,
        }

    if not session.get("history"):
        session["history"] = []

    last_slot = session.get("lastSlotKey")
    if last_slot and text:
        if "deepAnswers" not in session or not isinstance(session["deepAnswers"], dict):
            session["deepAnswers"] = {}
        normalized_key = str(last_slot).strip()
        session["deepAnswers"][normalized_key] = text.strip()
        session["lastSlotKey"] = None  # Consume key

    latest_extraction = await extract_requirements(app_state.llm, text, session["history"])
    session["extraction"] = _merge_extraction(session.get("extraction"), latest_extraction, text)
    
    session["languageMode"] = _detect_language_mode(session)
    extraction = session.get("extraction") or {}
    
    if any(s in text.lower() for s in ("image", "images", "poster", "posters", "photo", "photos")):
        session["appType"] = "image"
        session["extraction"]["appType"] = "image"
    else:
        session["appType"] = extraction.get("appType") or session.get("appType") or "text"
        
    return await _build_step0_response(session, text, app_state)


async def _exec_generate_preview(session: dict, text: str, app_state: Any) -> dict:
    llm = app_state.llm
    app_type = session.get("appType") or "text"
    selected_model_id = _parse_selected_model_id(text, MODELS.get(app_type, []))
    if not selected_model_id:
        selected_model_id = session.get("modelId")
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
        existing_vars = (session.get("dynamicContext") or {}).get("variables") or []
        bad_var_names = {
            "episode_title", "story_title", "case_file_details", "content",
            "description", "details", "information", "background", "overview"
        }
        has_stale_vars = any(
            v.get("name", "").lower() in bad_var_names
            for v in existing_vars
        )

        if not session.get("dynamicContext") or not existing_vars or has_stale_vars:
            _rag_context = ""
            vector_store = getattr(app_state, "vector_store", None)
            if vector_store and hasattr(vector_store, "search"):
                try:
                    matches = await vector_store.search(
                        query=f"{app_type} app variables for {(session.get('extraction') or {}).get('appPurpose') or ''}",
                        categories=["examples", "blueprints"],
                        top_k=3,
                    )
                    _rag_context = "\n\n".join([m.get("content", "") for m in matches if m.get("content")])
                except Exception as e:
                    logger.warning(f"RAG lookup for generate_dynamic_context failed: {e}")

            session["dynamicContext"] = await generate_dynamic_context(
                llm,
                app_type,
                (session.get("extraction") or {}).get("appPurpose") or "",
                _detect_language_mode(session),
                rag_context=_rag_context,
                history=session.get("history"),
                deep_answers=session.get("deepAnswers"),
            )

        prefill_dynamic_context_variables(session)

        search_query = f"{app_type} model {session.get('modelName')} prompting guidelines parameters"
        try:
            search_tool = get_web_search_tool()
            search_result = await search_tool.search_and_summarize(search_query)
            session["webSearchContext"] = search_result
        except Exception as search_err:
            logger.warning(f"WebSearch grounding failed: {search_err}")

        _prompt_rag = ""
        vector_store = getattr(app_state, "vector_store", None)
        if vector_store and hasattr(vector_store, "search"):
            try:
                matches = await vector_store.search(
                    query=f"{app_type} app prompt template for {(session.get('extraction') or {}).get('appPurpose') or ''}",
                    categories=["examples", "prompting"],
                    top_k=3,
                )
                _prompt_rag = "\n\n".join([m.get("content", "") for m in matches if m.get("content")])
                if _prompt_rag:
                    session["ragContext"] = _prompt_rag
            except Exception as e:
                logger.warning(f"RAG lookup for prompt generation failed: {e}")

        prompt_data, seo_data = await asyncio.gather(
            generate_prompt_template(llm, session),
            generate_seo(llm, session),
        )

        session["promptData"] = prompt_data
        session["seoData"] = seo_data
        session["step"] = 2
        await _save(session, app_state)

        preview_image_url = None
        if app_type in ("image", "vision"):
            try:
                compiled_prompt = (prompt_data.get("userPrompt") or "")
                dc_vars = (session.get("dynamicContext") or {}).get("variables") or []
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

                # Robust cleanup
                compiled_prompt = re.sub(r"\*\*+(?:\$\$|\[)[a-zA-Z0-9_'\s-]+?(?:\$\$|\])?\*\*+", "", compiled_prompt)
                compiled_prompt = re.sub(r"\[[a-zA-Z0-9_'\s-]+?\]", "", compiled_prompt)
                compiled_prompt = re.sub(r"\$\$[a-zA-Z0-9_']+\b", "", compiled_prompt)
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
                "previewImageUrl": preview_image_url,
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
        "reply": f'## 🎉 Published Successfully!\n\nYour app "{app_name}" is now live on {"your private registry" if is_private else "the marketplace"}!',
        "uiType": "success",
        "uiData": {"appName": app_name, "modelCost": session.get("modelCost")},
        "nextStep": 0,
        "clearSession": True,
    }


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
    session["deepAnswers"] = {}
    session["dynamicSlots"] = []
    session["step"] = 0
    session["triageRounds"] = 0
    session["lastSlotKey"] = None
    await _save(session, app_state)
    return await _build_step0_response(session, text, app_state)


async def _exec_edit_app(session: dict, text: str, decision: dict, app_state: Any) -> dict:
    instruction = text.strip()
    edit_scope = (decision.get("extracted_variables") or {}).get("edit_scope") or \
                 decision.get("edit_scope") or "PATCH_PROMPT"

    if edit_scope == "PATCH_VALUE":
        # Only patch the matching variable value — never touch schema
        dc = session.get("dynamicContext") or {}
        variables = dc.get("variables") or []
        instr_lower = instruction.lower()
        patched = False
        for var in variables:
            var_name_lower = str(var.get("name") or "").lower().replace(" ", "_")
            # Check if instruction references this variable by keyword proximity
            if any(
                kw in instr_lower
                for kw in [var_name_lower, var_name_lower.replace("_", " ")]
            ):
                # Extract the value after common phrases like "be", "to", "is", "="
                value_match = re.search(
                    r"(?:be|to|is|=|:)\s+(.+)$", instruction, re.IGNORECASE
                )
                if value_match:
                    var["test_value"] = value_match.group(1).strip()
                    patched = True
                    break
        if not patched and variables:
            # Fallback: patch the first name-like variable
            for var in variables:
                if any(w in str(var.get("name") or "").lower() for w in
                       ["name", "character", "subject", "protagonist", "hero"]):
                    value_match = re.search(
                        r"(?:be|to|is|=|:)\s+(.+)$", instruction, re.IGNORECASE
                    )
                    if value_match:
                        var["test_value"] = value_match.group(1).strip()
                    break
        session["dynamicContext"] = dc
        session["step"] = 2
        await _save(session, app_state)
        return await _exec_generate_preview(session, text, app_state)

    if edit_scope == "DOMAIN_SHIFT":
        # Full reset — re-run triage with new intent
        session["dynamicContext"] = None
        session["ragContext"] = None
        session["webSearchContext"] = None
        session["dynamicSlots"] = []
        session["triageRounds"] = 0
        session["formConfirmed"] = False
        if not session.get("extraction"):
            session["extraction"] = {}
        session["extraction"]["appPurpose"] = instruction
        session["step"] = 0
        await _save(session, app_state)
        return await _build_step0_response(session, instruction, app_state)

    # PATCH_PROMPT — update prompt direction, preserve schema entirely
    _apply_edit_to_session(session, instruction)
    session["ragContext"] = None
    session["webSearchContext"] = None
    # Do NOT clear dynamicContext
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
    triage_rounds: int
    last_slot_key: Optional[str]
    dynamic_slots: list[dict]
    awaiting_deep_answer: Optional[bool]
    current_deep_field: Optional[str]


def _session_to_state(session: dict, message: str) -> ConversationState:
    hist = []
    for m in session.get("history", []):
        role = "assistant" if m.get("role") in ("agent", "assistant") else "user"
        hist.append({"role": role, "content": m.get("content", "")})
    
    a_type = session.get("appType")
    
    # ─── 🛡️ HARD LOCK VALVE: INSTANT ENTRY OVERRIDE CHECK ───
    if any(s in message.lower() for s in ("image", "images", "poster", "posters", "photo", "photos")):
        a_type = "image"
    if not a_type or a_type == "None": a_type = "text"

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
    }


def _state_to_session(state: ConversationState, session: dict) -> None:
    session["step"] = state.get("current_step", 0)
    
    a_type = state.get("app_type") or "text"
    session["appType"] = a_type
    
    if "extraction" not in session or not isinstance(session["extraction"], dict):
        session["extraction"] = {}
    session["extraction"].update(state.get("extraction") or {})
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

    # ─── 🛡️ HARD OVERRIDE VALVE: CHAT SEPARATION INTERCEPT ───
    app_type = state.get("app_type") or temp_session.get("appType") or "text"
    
    if any(s in msg for s in ("image", "images", "poster", "posters", "photo", "photos", "graphic")):
        app_type = "image"
    else:
        new_inferred = decision.get("app_type")
        if new_inferred and new_inferred != "None" and (action == "PIVOT_APP" or decision.get("_source") == "fast_path"):
            app_type = new_inferred

    if app_type == "None" or not app_type:
        app_type = "ambiguous"

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

    if msg.startswith("seo_publish::") or msg in ("publish to marketplace", "save draft") or action == "PUBLISH_APP":
        return "publish_app_node"
    if msg.startswith("confirm seo::") or msg in ("approve app", "approve") or action == "REVIEW_SEO":
        return "seo_review_node"
    if msg.startswith("multi_select_form::") or action == "PROCESS_FORM":
        return "form_submission_node"

    # ─── 🛡️ PREVIEW SAFETY BOUNDARY GATE ───
    # If a preview is requested but no model is selected, drop back down gracefully
    if (action == "GENERATE_PREVIEW" or state.get("current_step") == 2) and not state.get("model_id"):
        if not _session_has_budget(state.get("extraction"), state.get("deep_answers")):
            return "ideation_triage_node"
        return "model_selection_node"
        
    # ─── 🛑 THE RADICAL ROUTING VALVE BYPASS RECOVERY ───
    # If the user text contains strong format changes, FORCE it through the ideation extraction loop
    # instead of blindly short-circuiting straight into model rendering via stale action schemas!
    if any(s in msg for s in ("image", "images", "poster", "posters", "photo", "photos", "graphic")):
        return "ideation_triage_node"

    if not state.get("form_confirmed") and action not in ("GATHER_REQUIREMENTS", "PROCESS_FORM", "HANDLE_OFF_TOPIC"):
        return "ideation_triage_node"

    # ─── 🛡️ BUDGET CHIP INTERCEPT ───
    # If we're awaiting a budget answer, the chip must go through ideation_triage_node
    # so the budget is saved to deepAnswers BEFORE _show_models is called.
    # Sending it to model_selection_node directly causes a double render with wrong models first.
    if state.get("awaiting_deep_answer") and state.get("current_deep_field") == "budgetPreference":
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


def check_triage_completeness(state: ConversationState) -> str:
    # Autonomously evaluate if requirements are robustly met mid-flight
    slots = state.get("dynamic_slots") or []
    answers = state.get("deep_answers") or {}
    
    # If slots are missing, halt and return control to the user
    if not slots or len(answers) < len(slots):
        return "wait_for_user"
        
    # ─── 🛡️ BUDGET CHECK: Prevent bypassing budget collection ───
    extraction = state.get("extraction") or {}
    if not _session_has_budget(extraction, answers):
        return "wait_for_user"
        
    # TRUE AUTONOMY: Move directly to model scaling without waiting for a new API request turn
    return "auto_advance_to_models"


def build_true_agentic_graph() -> StateGraph:
    graph = StateGraph(ConversationState)
    
    # Add nodes
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
    })
    
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
    graph.add_edge("publish_app_node", END)
    
    return graph.compile()


compiled_graph = build_true_agentic_graph()

async def route(session: dict, message: str, app_state: Any) -> dict:
    # ─── 🛡️ DRASTIC PIVOT INTERCEPTOR ───
    from services.intent_engine import classify_intent_with_pivot_check
    
    current_context = {
        "appPurpose": (session.get("extraction") or {}).get("appPurpose") or session.get("domainIdentified") or ""
    }
    pivot_data = classify_intent_with_pivot_check(message, current_context)
    
    if pivot_data.get("drastic_pivot"):
        # 1. Reset/purge app-specific context
        await app_state.session.reset_app_specific_context(session.get("sessionId", ""))
        
        # Read the newly reset session
        updated_session = await app_state.session.get_session(session.get("sessionId", "")) or {}
        session.clear()
        session.update(updated_session)
        
        # 2. Inject a user-friendly conversational block
        transition_text = (
            f"I noticed you've pivoted from a **{pivot_data['previous_category'].capitalize()}** app "
            f"to a **{pivot_data['new_category'].capitalize()}** app. Let me clear the old configuration "
            f"and regenerate clean requirements for you!"
        )
        
        # Append message to chat history so the user sees it in their UI
        if "history" not in session:
            session["history"] = []
        session["history"].append({"role": "user", "content": message})
        session["history"].append({"role": "agent", "content": transition_text})
        
        # Update appPurpose in extraction so step 0 triage knows the target concept!
        session["extraction"] = {
            "appPurpose": message,
            "appType": "text" # default to text, dynamic triage will refine
        }
        
        # Save session
        await app_state.session.save_session(session)
        
        # 3. Direct back to baseline slot-filling/prompt-generation phase (step 0 response)
        res = await _build_step0_response(session, message, app_state)
        res["reply"] = f"{transition_text}\n\n{res.get('reply', '')}"
        return res

    initial_state = _session_to_state(session, message)
    config = {"configurable": {"app_state": app_state}}
    final_state = await compiled_graph.ainvoke(initial_state, config=config)
    _state_to_session(final_state, session)
    return final_state.get("response_payload") or {}