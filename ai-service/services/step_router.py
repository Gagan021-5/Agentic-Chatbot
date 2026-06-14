"""
Step router for the RentPrompts chat lifecycle.
Async route(session, message, app_state) -> dict matching the React response contract.
"""

from __future__ import annotations

import asyncio
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

COST_WARNING_THRESHOLD = 100

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
    """Infer slot key from triage question text as fallback when LLM doesn't return slot_key."""
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
    """Queries model_catalog.md via ChromaDB to find model names with artistic or editing capabilities."""
    vector_store = getattr(app_state, "vector_store", None)
    if not vector_store:
        return []
    try:
        # Search models collection in ChromaDB
        matches = await vector_store.search(
            query="artistic pro-edit creative cinematic editing drawing style",
            categories=["models"],
            top_k=5
        )
        artistic_models = []
        for match in matches:
            content = match.get("content", "")
            # Locate headers (e.g. ## Model Name)
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
    
    # Ignore generic budget filtering if artistic_priority is active
    if not artistic_priority:
        b = (budget_str or "").lower()
        if b:
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
        
        # Check if matched by RAG catalog search
        if artistic_priority and artistic_model_names:
            for name in artistic_model_names:
                name_lower = name.lower()
                if name_lower in model_id or name_lower in model_name or model_id in name_lower or model_name in name_lower:
                    score += 50
                    
        # Check tags for artistic / pro-edit / creative / detailed / editing / cinematic
        model_tags = [str(t).lower() for t in (model.get("tags") or [])]
        if any(t in model_tags for t in ("artistic", "pro-edit", "creative", "detailed", "editing", "cinematic", "design")):
            if artistic_priority:
                score += 30
            else:
                score += 5
        
        # Standard tag matching from user input
        for tag in model.get("tags") or []:
            if str(tag).lower() in input_lower:
                score += 5
                
        # Standard tier matching
        if any(w in input_lower for w in ("fast", "quick", "speed")):
            if model.get("tier") == "fast":
                score += 5
        if any(w in input_lower for w in ("quality", "best", "advanced")):
            if model.get("tier") in ("premium", "ultra"):
                score += 5
                
        if not artistic_priority:
            if any(w in input_lower for w in ("cheap", "affordable")):
                score += 20 - model.get("cost", 0)
                
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
            if is_control
            or not latest.get("targetUsers")
            or latest.get("targetUsers") == "general users"
            else latest.get("targetUsers")
        ),
        "budget": latest.get("budget") if latest.get("budget") else existing.get("budget"),
        "wantsImageInput": bool(existing.get("wantsImageInput") or latest.get("wantsImageInput")),
        "detectedLanguage": (
            existing.get("detectedLanguage")
            if is_control and existing.get("detectedLanguage")
            else latest.get("detectedLanguage") or existing.get("detectedLanguage")
        ),
        "userTone": (
            existing.get("userTone")
            if is_control and existing.get("userTone")
            else latest.get("userTone") or existing.get("userTone")
        ),
        "oneLineUnderstanding": (
            existing.get("oneLineUnderstanding") or latest.get("oneLineUnderstanding")
            if is_control
            else latest.get("oneLineUnderstanding") or existing.get("oneLineUnderstanding")
        ),
        "suggestedReply": (
            existing.get("suggestedReply") or latest.get("suggestedReply")
            if is_control
            else latest.get("suggestedReply") or latest.get("suggestedReply")
        ),
        "confidence": {
            "appType": (
                existing_conf.get("appType") or latest_conf.get("appType") or "LOW"
                if keep_app_type
                else latest_conf.get("appType") or existing_conf.get("appType") or "LOW"
            ),
            "budget": (
                latest_conf.get("budget") or existing_conf.get("budget") or "LOW"
                if latest.get("budget")
                else existing_conf.get("budget") or latest_conf.get("budget") or "LOW"
            ),
        },
        "keyFeatures": (
            existing.get("keyFeatures")
            if is_control
            or not isinstance(latest.get("keyFeatures"), list)
            or not latest.get("keyFeatures")
            else latest.get("keyFeatures")
        ),
        "missingFields": list(
            set((existing.get("missingFields") or []) + (latest.get("missingFields") or []))
        ),
        "userType": latest.get("userType") or existing.get("userType"),
        "enterpriseSignals": (
            latest.get("enterpriseSignals")
            if latest.get("enterpriseSignals") is not None
            else existing.get("enterpriseSignals")
        ),
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
        wants_transparent = bool(
            re.search(r"transparent|no background|remove background|no bg", edit_instruction, re.IGNORECASE)
        )
        wants_new_background = bool(
            re.search(r"background|backdrop|scene|environment", edit_instruction, re.IGNORECASE)
        )
        no_x_match = re.search(r"no\s+(\w+)", edit_instruction, re.IGNORECASE)

        keys_to_remove: list[str] = []
        for key, value in deep_answers.items():
            kl = key.lower()
            vl = str(value or "").lower()
            if wants_transparent and re.search(
                r"scene|background|forest|nature|backdrop|environment|location|setting", kl, re.IGNORECASE
            ):
                keys_to_remove.append(key)
            elif wants_new_background and re.search(
                r"scene|background|backdrop|environment", kl, re.IGNORECASE
            ):
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
    if variables and re.search(
        r"transparent|no background|remove background|no bg", edit_instruction, re.IGNORECASE
    ):
        dynamic_context["variables"] = [
            v
            for v in variables
            if not re.search(
                r"scene|forest|nature|backdrop|environment|location|background_scene|forest_scene",
                str(v.get("name") if isinstance(v, dict) else v),
                re.IGNORECASE,
            )
        ]

    if not session.get("deepAnswers"):
        session["deepAnswers"] = {}
    session["deepAnswers"]["lastEditInstruction"] = edit_instruction.strip()

    history = session.get("history")
    if isinstance(history, list):
        history.append({"role": "user", "content": f"[EDIT REQUEST]: {edit_instruction.strip()}"})


def _is_ready_to_generate(session_context: dict) -> bool:
    # Check if the 4 universal dimensions are filled
    required_slots = ['PRIMARY_SUBJECT', 'ENVIRONMENT_SETTING', 'ACTION_DYNAMIC', 'AESTHETIC_STYLE']
    
    # Generic extraction check
    filled_slots = []
    for slot in required_slots:
        val = session_context.get(slot) or (session_context.get("extraction") or {}).get(slot)
        if val:
            filled_slots.append(slot)
            
    return len(filled_slots) >= 3 # Agar 3/4 found then bypass!


def bypass_wizard_if_ready(session_context: dict, text: str = "") -> bool:
    required_slots = ['PRIMARY_SUBJECT', 'ENVIRONMENT_SETTING', 'ACTION_DYNAMIC', 'AESTHETIC_STYLE']
    
    # Generic extraction check
    filled_slots = []
    for slot in required_slots:
        val = session_context.get(slot) or (session_context.get("extraction") or {}).get(slot)
        if val:
            filled_slots.append(slot)
            
    # Count words in prompt text
    prompt_text = text or session_context.get("message") or ""
    words = [w for w in prompt_text.split() if w]
    word_count = len(words)

    # Bypass if we have at least 3 slots filled OR (word_count > 8 and clear entities)
    has_clear_entities = (
        (session_context.get("extraction") or {}).get("PRIMARY_SUBJECT") is not None and
        (session_context.get("extraction") or {}).get("ENVIRONMENT_SETTING") is not None and
        (session_context.get("extraction") or {}).get("ACTION_DYNAMIC") is not None
    )
    
    should_bypass = len(filled_slots) >= 3 or (word_count > 8 and has_clear_entities)
    
    if should_bypass:
        session_context["status"] = "ready"
        session_context["session_status"] = "ready"
        session_context["awaitingDeepAnswer"] = False
        session_context["artisticPriority"] = True
        session_context["formConfirmed"] = True
        
        # Populate session["dynamicContext"]["variables"] if not done
        extraction = session_context.get("extraction") or session_context
        
        sub = extraction.get("PRIMARY_SUBJECT")
        env = extraction.get("ENVIRONMENT_SETTING")
        act = extraction.get("ACTION_DYNAMIC")
        aes = extraction.get("AESTHETIC_STYLE")
        
        variables = []
        if sub:
            variables.append({"name": "PRIMARY_SUBJECT", "placeholder": "Primary Subject", "test_value": sub})
        if env:
            variables.append({"name": "ENVIRONMENT_SETTING", "placeholder": "Environment Setting", "test_value": env})
        if act:
            variables.append({"name": "ACTION_DYNAMIC", "placeholder": "Action Dynamic", "test_value": act})
        if aes:
            variables.append({"name": "AESTHETIC_STYLE", "placeholder": "Aesthetic Style", "test_value": aes})
            
        if not session_context.get("dynamicContext"):
            session_context["dynamicContext"] = {}
        session_context["dynamicContext"]["variables"] = variables
        session_context["dynamicContext"]["options"] = ["Dynamic UI Blueprint", "Multi-Slot Extraction", "Artistic Override", "Ready Canvas"]
        
        if not session_context.get("deepAnswers"):
            session_context["deepAnswers"] = {}
        session_context["deepAnswers"]["budgetPreference"] = "Premium"
        session_context["currentDeepField"] = None
        session_context["lastSlotKey"] = None
        session_context["triageRounds"] = 0
        
        if not session_context.get("appType"):
            session_context["appType"] = "image"
            if "extraction" in session_context:
                session_context["extraction"]["appType"] = "image"
                
        return True
    return False


def _check_and_apply_artistic_priority(session: dict, text: str) -> None:
    bypass_wizard_if_ready(session, text)


def _get_next_deep_question(session: dict) -> dict | None:
    if session.get("artisticPriority"):
        return None
    if not session.get("deepAnswers"):
        session["deepAnswers"] = {}
    extraction = session.get("extraction") or {}
    if not extraction.get("budget") and not session["deepAnswers"].get("budgetPreference"):
        return {
            "field": "budgetPreference",
            "question": "Last step before I pick models — what's your budget per generation?",
            "options": BUDGET_CHIP_OPTIONS,
        }
    return None


async def _save(session: dict, app_state: Any) -> None:
    await app_state.session.save_session(session)


async def _show_models(session: dict, app_state: Any) -> dict:
    full_text = " ".join(
        [
            str((session.get("extraction") or {}).get("appPurpose") or ""),
            str((session.get("extraction") or {}).get("oneLineUnderstanding") or ""),
            json.dumps(session.get("deepAnswers") or {}),
        ]
    )
    budget = (session.get("deepAnswers") or {}).get("budgetPreference") or (
        (session.get("extraction") or {}).get("budget")
    )
    model_collection = MODELS.get(session.get("appType") or "", MODELS.get("text", []))
    
    artistic_priority = bool(session.get("artisticPriority"))
    artistic_model_names = []
    if artistic_priority:
        artistic_model_names = await _find_artistic_models_from_catalog(app_state)
        
    models = _rank_models(
        model_collection, 
        full_text, 
        str(budget or ""), 
        artistic_priority=artistic_priority, 
        artistic_model_names=artistic_model_names
    )

    session["step"] = 1
    session["awaitingConfirmation"] = False
    await _save(session, app_state)

    ui_data = {
        "appType": session.get("appType"), 
        "models": models
    }
    if artistic_priority:
        ui_data["artistic_priority"] = True
        ui_data["style_tags"] = session.get("styleTags") or []

    return {
        "reply": (
            f"## 🤖 AI Model Selection\n\nI've ranked the **top 3 models** for your "
            f"**{session.get('appType')}** app based on your requirements and budget.\n\n"
            "Each card shows the model's strengths, speed, and cost per run — **click any card** to select it."
        ),
        "uiType": "models",
        "uiData": ui_data,
        "nextStep": 1,
        "coins": None,
    }


async def _build_step0_response(session: dict, text: str, app_state: Any) -> dict:
    llm = app_state.llm
    ext = session.get("extraction") or {}

    _check_and_apply_artistic_priority(session, text)
    if session.get("artisticPriority"):
        session["formConfirmed"] = True
        session["awaitingDeepAnswer"] = False
        session["currentDeepField"] = None
        session["lastSlotKey"] = None
        session["triageRounds"] = 0
        await _save(session, app_state)
        return await _show_models(session, app_state)

    ambiguous_domain_signals = [
        "birthday", "greeting", "certificate", "diploma", "award",
        "wedding", "anniversary", "thank you", "congratulation",
        "wish", "card app", "card generat",
    ]
    purpose_lower_for_ambiguity = str(
        ext.get("appPurpose") or ext.get("oneLineUnderstanding") or ""
    ).lower()
    is_ambiguous_domain = any(sig in purpose_lower_for_ambiguity for sig in ambiguous_domain_signals)
    has_explicit_type_from_user = bool(
        (ext.get("confidence") or {}).get("appType") == "HIGH" or session.get("formatConfirmedByUser")
    )

    if not session.get("appType"):
        local_type_signals = {
            "image": [
                "background remov", "remove background", "bg remov", "image generat", "generate image",
                "photo generat", "interior design", "room design", "logo generat", "logo maker",
                "portrait", "art generat", "image edit", "greeting card", "birthday card", "poster", "meme"
            ],
            "video": [
                "video generat", "generate video", "video creat", "animate", "animation", "reel generat"
            ],
            "audio": [
                "audio generat", "voiceover", "text to speech", "tts", "music generat", "podcast"
            ],
            "vision": [
                "crop disease", "object detect", "image analys", "ocr", "read text from image", "analyze image"
            ],
            "text": [
                "blog", "article", "content", "email", "chatbot", "chat assistant", "resume", "cover letter",
                "proposal", "contract", "recipe", "itinerary", "planner", "workout", "meal plan", "birthday wishes"
            ],
        }

        purpose_lower = str(ext.get("appPurpose") or ext.get("oneLineUnderstanding") or "").lower()

        if is_ambiguous_domain and not has_explicit_type_from_user:
            logger.info(f'[Ambiguous Domain] "{purpose_lower[:60]}" matches ambiguous signals — skipping local inference')
        else:
            for app_type, signals in local_type_signals.items():
                if any(sig in purpose_lower for sig in signals):
                    session["appType"] = app_type
                    if not session.get("extraction"):
                        session["extraction"] = {}
                    session["extraction"]["appType"] = app_type
                    break

    if not session.get("appType"):
        has_purpose = bool(ext.get("appPurpose") and len(str(ext.get("appPurpose"))) > 5)
        if has_purpose:
            if is_ambiguous_domain and not has_explicit_type_from_user:
                logger.info("[Smart Infer] Ambiguous domain detected — NOT auto-inferring type. Triage will ask.")
            else:
                purpose_l = str(ext.get("appPurpose")).lower()
                image_signals = ["photo", "picture", "image", "card", "poster", "meme", "logo", "thumbnail"]
                video_signals = ["video", "animation", "animate", "reel", "clip"]
                audio_signals = ["audio", "voice", "music", "speech", "podcast", "tts"]
                vision_signals = ["detect", "analyze image", "scan", "ocr", "read from image"]

                inferred_type = "text"
                if any(s in purpose_l for s in image_signals): inferred_type = "image"
                elif any(s in purpose_l for s in video_signals): inferred_type = "video"
                elif any(s in purpose_l for s in audio_signals): inferred_type = "audio"
                elif any(s in purpose_l for s in vision_signals): inferred_type = "vision"

                session["appType"] = inferred_type
                if not session.get("extraction"):
                    session["extraction"] = {}
                session["extraction"]["appType"] = inferred_type

        else:
            session["step"] = 0
            await _save(session, app_state)
            options = ["टेक्स्ट", "इमेज", "ऑडियो", "वीडियो", "विज़न"] if _detect_language_mode(session) == "Hindi" else ["Text", "Image", "Audio", "Video", "Vision"]
            return {
                "reply": "What kind of output should your app produce?",
                "uiType": "chips",
                "uiData": {"options": options},
                "nextStep": 0,
                "coins": None,
            }

    session["step"] = 0
    session["awaitingConfirmation"] = False

    if not session.get("dynamicContext"):
        affirmations = ["yes", "sure", "ok", "yep", "yeah", "correct", "sounds good", "exactly", "perfect", "proceed", "looks good"]
        msg_clean = re.sub(r"[!.,?]+$", "", _lower(text).strip())
        is_affirmation = msg_clean in affirmations

        if is_affirmation and (session.get("triageRounds") or 0) > 0:
            session["triageRounds"] = 99
            await _save(session, app_state)
        else:
            # ─── 🚀 LIVE RAG EXTRACTION INTEGRATION LAYER ───
            rag_context = ""
            vector_store = getattr(app_state, "vector_store", None)
            if vector_store and hasattr(vector_store, "search"):
                try:
                    matches = await vector_store.search(
                        query=ext.get("appPurpose") or text,
                        categories=["marketplace", "examples"],
                        top_k=2,
                    )
                    rag_context = "\n\n".join(
                        [m.get("content", "") for m in matches if m.get("content")]
                    )
                    logger.info(
                        f"[RAG Grounding] Injected {len(matches)} contextual layout blueprints safely."
                    )
                except Exception as e:
                    logger.warning(f"Triage RAG lookup failed: {e}")

            # Execute context triage grounded dynamically via real platform knowledge base chunks
            triage_result = await triage_dynamic_context(
                llm,
                session.get("appType"),
                (session.get("extraction") or {}).get("appPurpose") or "",
                _detect_language_mode(session),
                session.get("history") or [],
                session.get("deepAnswers") or {},
                rag_context=rag_context,
            )

            if (session.get("triageRounds") or 0) >= 3 and triage_result.get("status") in (
                "needs_context",
                "needs_format",
            ):
                logger.info("[Triage] Limit of 3 questions reached. Forcing status to ready.")
                triage_result = {
                    "status": "ready",
                    "domain": triage_result.get("domain") or session.get("domainIdentified"),
                    "app_format": triage_result.get("corrected_app_type") or session.get("appType") or "text",
                    "form": None,
                }

            if triage_result.get("status") == "needs_context":
                question = str(triage_result.get("question") or "").strip()
                if len(question) >= 10:
                    last_q = session.get("lastQuestion") or ""
                    last_slot = session.get("lastSlotKey")
                    if last_q and text and last_slot:
                        if not session.get("deepAnswers"):
                            session["deepAnswers"] = {}
                        if not session["deepAnswers"].get(last_slot):
                            session["deepAnswers"][last_slot] = text

                    slot_key = (
                        triage_result.get("slot_key")
                        or _infer_slot_key_from_question(question)
                    )
                    session["lastSlotKey"] = slot_key
                    session["triageRounds"] = (session.get("triageRounds") or 0) + 1
                    session["lastQuestion"] = question

                    if not session.get("deepAnswers"):
                        session["deepAnswers"] = {}
                    session["deepAnswers"]["_lastTriageQuestion"] = question

                    await _save(session, app_state)
                    suggested = triage_result.get("suggested_options")
                    is_format_question = (
                        isinstance(suggested, list)
                        and len(suggested) >= 2
                        and all(
                            s.lower() in ("text", "image", "audio", "video", "vision")
                            for s in suggested
                        )
                    )
                    return {
                        "reply": question,
                        "uiType": "chips" if is_format_question else None,
                        "uiData": {"options": suggested} if is_format_question else None,
                        "nextStep": 0,
                        "coins": None,
                    }

            if triage_result.get("corrected_app_type") and triage_result["corrected_app_type"] != session.get("appType"):
                session["appType"] = triage_result["corrected_app_type"]
                if session.get("extraction"):
                    session["extraction"]["appType"] = triage_result["corrected_app_type"]

            if triage_result.get("form"):
                session["dynamicContext"] = triage_result["form"]
            else:
                session["dynamicContext"] = await generate_dynamic_context(
                    llm,
                    session.get("appType") or "text",
                    (session.get("extraction") or {}).get("appPurpose") or "",
                    _detect_language_mode(session),
                )
            session["triageRounds"] = 0
            await _save(session, app_state)

    last_slot = session.get("lastSlotKey")
    if last_slot and text:
        if not session.get("deepAnswers"):
            session["deepAnswers"] = {}
        if not session["deepAnswers"].get(last_slot):
            session["deepAnswers"][last_slot] = text

    session["formConfirmed"] = True
    session["lastSlotKey"] = None
    await _save(session, app_state)

    extraction = session.get("extraction") or {}
    deep_answers = session.get("deepAnswers") or {}
    has_budget = extraction.get("budget") or deep_answers.get("budgetPreference")
    if not has_budget:
        session["currentDeepField"] = "budgetPreference"
        session["awaitingDeepAnswer"] = True
        await _save(session, app_state)
        return {
            "reply": (
                "Got it — I have everything I need to build your app! "
                "One last thing: **what's your budget per generation?**"
            ),
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
    
    _check_and_apply_artistic_priority(session, text)
    if session.get("artisticPriority"):
        session["formConfirmed"] = True
        session["awaitingDeepAnswer"] = False
        session["currentDeepField"] = None
        session["lastSlotKey"] = None
        session["triageRounds"] = 0
        await _save(session, app_state)
        return await _show_models(session, app_state)
    session["languageMode"] = _detect_language_mode(session)
    extraction = session.get("extraction") or {}
    if extraction.get("enterpriseSignals") is not None:
        session["enterpriseSignals"] = extraction["enterpriseSignals"]
    if extraction.get("userType"):
        session["userType"] = extraction["userType"]
    if not session.get("appType") and extraction.get("appType"):
        session["appType"] = extraction["appType"]
    return await _build_step0_response(session, text, app_state)


async def _exec_render_form(session: dict, app_state: Any) -> dict:
    llm = app_state.llm
    dynamic_context = session.get("dynamicContext") or {}
    if not isinstance(dynamic_context.get("variables"), list) or not isinstance(
        dynamic_context.get("options"), list
    ):
        session["dynamicContext"] = await generate_dynamic_context(
            llm,
            session.get("appType") or "text",
            (session.get("extraction") or {}).get("appPurpose") or "",
            _detect_language_mode(session),
        )
    dynamic_context = session.get("dynamicContext") or {}
    if not isinstance(dynamic_context.get("variables"), list) or not isinstance(
        dynamic_context.get("options"), list
    ):
        session["dynamicContext"] = build_dynamic_context_fallback(
            session.get("appType") or "text",
            (session.get("extraction") or {}).get("appPurpose") or "",
            _detect_language_mode(session),
        )
    session["formConfirmed"] = True
    await _save(session, app_state)

    budget = (session.get("deepAnswers") or {}).get("budgetPreference") or (
        (session.get("extraction") or {}).get("budget")
    )
    if not budget:
        session["currentDeepField"] = "budgetPreference"
        session["awaitingDeepAnswer"] = True
        await _save(session, app_state)
        return {
            "reply": (
                "Got everything I need to build your app! One last thing — "
                "**what's your budget per generation?** This helps me pick the right AI model."
            ),
            "uiType": "chips",
            "uiData": {"options": BUDGET_CHIP_OPTIONS},
            "nextStep": 0,
            "coins": None,
        }

    return await _show_models(session, app_state)


async def _exec_generate_preview(session: dict, text: str, app_state: Any) -> dict:
    llm = app_state.llm
    app_type = session.get("appType") or "text"
    selected_model_id = _parse_selected_model_id(text, MODELS.get(app_type, []))
    if not selected_model_id:
        return {
            "reply": "Please **click one of the model cards** above to select the AI engine. 👆",
            "uiType": "text",
            "uiData": None,
            "nextStep": 1,
            "coins": None,
        }

    selected_model = _find_model(app_type, selected_model_id)
    if not selected_model:
        return {
            "reply": "I couldn't match that model. Please click one of the options above.",
            "uiType": "text",
            "uiData": None,
            "nextStep": 1,
            "coins": None,
        }

    session["modelId"] = selected_model["id"]
    session["modelCost"] = selected_model["cost"]
    session["modelName"] = selected_model["name"]
    session["awaitingConfirmation"] = False
    await _save(session, app_state)

    try:
        # Ensure dynamicContext is present and has variables
        dynamic_context = session.get("dynamicContext") or {}
        if not dynamic_context or not dynamic_context.get("variables"):
            dynamic_context = await generate_dynamic_context(
                llm,
                session.get("appType") or "text",
                (session.get("extraction") or {}).get("appPurpose") or "",
                _detect_language_mode(session),
            )
            session["dynamicContext"] = dynamic_context

        # Prefill dynamicContext variables from conversational slot answers
        deep_answers = session.get("deepAnswers") or {}
        extraction = session.get("extraction") or {}
        variables = dynamic_context.get("variables") or []

        if variables and (deep_answers or extraction):
            prefilled = []
            for var in variables:
                var_name = str(var.get("name") or "").lower().replace(" ", "_")
                # Try exact slot key match first
                prefilled_value = None
                for slot_key, slot_val in deep_answers.items():
                    if slot_key.startswith("_"):
                        continue
                    if slot_key in var_name or var_name in slot_key:
                        prefilled_value = str(slot_val)
                        break
                # Fallback: fuzzy match on keywords
                if not prefilled_value:
                    keywords = re.findall(r'[a-z]{4,}', var_name)
                    for kw in keywords:
                        for slot_key, slot_val in deep_answers.items():
                            if slot_key.startswith("_"):
                                continue
                            if kw in slot_key:
                                prefilled_value = str(slot_val)
                                break
                        if prefilled_value:
                            break
                prefilled.append({
                    **var,
                    "test_value": prefilled_value or var.get("test_value") or var.get("placeholder") or ""
                })
            
            if not session.get("dynamicContext"):
                session["dynamicContext"] = {}
            session["dynamicContext"]["variables"] = prefilled

        prompt_data, seo_data = await asyncio.gather(
            generate_prompt_template(llm, session),
            generate_seo(llm, session),
        )
        # Resolve initial ui_meta for frontend dynamic layout
        from routers.preview import _resolve_transformation_tool
        combined_context = f"{prompt_data.get('userPrompt') or ''}\n\n{prompt_data.get('systemPrompt') or ''}"
        blueprint = await _resolve_transformation_tool(combined_context, app_state)
        if blueprint:
            ui_meta = {
                "show_upload": blueprint.get("show_upload", False),
                "show_url_input": blueprint.get("show_url_input", False),
                "active_tool": blueprint.get("tool_id"),
                "layout_mode": blueprint.get("layout_mode", "static"),
                "tool_id": blueprint.get("tool_id"),
                "config": blueprint.get("config", {})
            }
        else:
            accept_img = _sanitize_accept_image_input(prompt_data.get("acceptImageInput"), app_type)
            show_url = any(kw in combined_context.lower() for kw in ["url", "fetch", "scrap", "external", "link"])
            ui_meta = {
                "show_upload": accept_img,
                "show_url_input": show_url,
                "active_tool": "bg_remover" if ("remove background" in combined_context.lower() or "bg_remover" in combined_context.lower()) else None,
                "layout_mode": "interactive" if (accept_img or show_url) else "static",
                "tool_id": "bg_remover" if ("remove background" in combined_context.lower() or "bg_remover" in combined_context.lower()) else None,
                "config": {}
            }

        session["promptData"] = prompt_data
        session["seoData"] = seo_data
        session["step"] = 2
        await _save(session, app_state)
        return {
            "reply": (
                f"## App Preview Ready\n\nI've configured the full AI logic using **{selected_model['name']}**.\n\n"
                "Test it in the Live Preview below — click **Approve App** when ready!"
            ),
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
                "acceptImageInput": _sanitize_accept_image_input(
                    prompt_data.get("acceptImageInput"), app_type
                ),
                "ui_meta": ui_meta,
                "options": ["Approve App", "Edit App"],
            },
            "nextStep": 2,
            "coins": session.get("modelCost"),
        }
    except Exception as err:
        logger.error(f"[execGeneratePreview] Error: {err}")
        return {
            "reply": "Oops! ⚠️ I hit a snag generating the config. Please try selecting the model again.",
            "uiType": "text",
            "uiData": None,
            "nextStep": 1,
            "coins": None,
        }


async def _exec_review_seo(session: dict, app_state: Any) -> dict:
    session["step"] = 3
    await _save(session, app_state)
    seo_data = session.get("seoData") or {}
    return {
        "reply": (
            "## 🎉 App Configured — Final Review\n\n"
            "Review your **app name, description, and tags** below.\n\n"
            "Edit any field before publishing — make it shine! ✨"
        ),
        "uiType": "seo_preview",
        "uiData": {
            "appName": seo_data.get("appName") or "Your App",
            "appDescription": seo_data.get("appDescription") or "",
            "category": seo_data.get("category") or "",
            "tags": seo_data.get("tags") if isinstance(seo_data.get("tags"), list) else [],
            "appType": session.get("appType") or "text",
            "modelId": session.get("modelId"),
            "costPerRun": session.get("modelCost"),
        },
        "nextStep": 3,
        "coins": session.get("modelCost"),
    }


async def _exec_pivot_app(session: dict, text: str, decision: dict, app_state: Any) -> dict:
    new_type = decision.get("app_type") or _parse_chip_app_type(text)
    if not new_type:
        match = re.search(r"(image|video|audio|text|vision)", text, re.IGNORECASE)
        if match:
            new_type = match.group(1).lower()

    extraction = session.get("extraction") or {}
    is_format_only = (
        new_type
        and new_type != session.get("appType")
        and len(str(extraction.get("appPurpose") or "").strip()) > 10
        and not decision.get("is_major_pivot")
    )

    if is_format_only:
        session["appType"] = new_type
        if session.get("extraction"):
            session["extraction"]["appType"] = new_type
        session["dynamicContext"] = None
        session["modelId"] = None
        session["modelCost"] = None
        session["step"] = 0
        session["triageRounds"] = 0
        session["awaitingPromptTweak"] = False
        session["awaitingDeepAnswer"] = False
        session["currentDeepField"] = None
        await _save(session, app_state)

        has_budget = (session.get("deepAnswers") or {}).get("budgetPreference") or extraction.get("budget")
        if has_budget:
            return await _show_models(session, app_state)

        session["currentDeepField"] = "budgetPreference"
        session["awaitingDeepAnswer"] = True
        await _save(session, app_state)
        return {
            "reply": (
                f"Got it! Switching to a **{new_type}** app — your context is preserved.\n\n"
                "What budget per generation works for you?"
            ),
            "uiType": "chips",
            "uiData": {"options": BUDGET_CHIP_OPTIONS},
            "nextStep": 0,
            "coins": None,
        }

    clean_purpose = re.sub(
        r"\b(actually|instead|change it to|switch to|something different)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    clean_purpose = re.sub(
        r"\bi want to (build|make|create) (a|an)\b",
        "",
        clean_purpose,
        flags=re.IGNORECASE,
    )
    clean_purpose = re.sub(r"\bi want (a|an)\b", "", clean_purpose, flags=re.IGNORECASE).strip()
    if not clean_purpose or len(clean_purpose) < 3:
        clean_purpose = f"{new_type} generation app" if new_type else "a new idea"

    session["dynamicContext"] = None
    session["appType"] = new_type
    session["extraction"] = {"appPurpose": clean_purpose, "confidence": {}}
    session["deepAnswers"] = {}
    session["history"] = []
    session["step"] = 0
    session["triageRounds"] = 0
    session["awaitingPromptTweak"] = False
    session["awaitingDeepAnswer"] = False
    session["currentDeepField"] = None
    session["isPivot"] = True
    await _save(session, app_state)
    return await _build_step0_response(session, text, app_state)


async def _exec_edit_app(session: dict, text: str, decision: dict, app_state: Any) -> dict:
    llm = app_state.llm
    extracted = decision.get("extracted_variables") or {}
    instruction = extracted.get("editInstruction") or text

    if session.get("step") == 0:
        _apply_edit_to_session(session, instruction)
        session["formConfirmed"] = False
        session["dynamicContext"] = None
        session["triageRounds"] = 0
        session["awaitingDeepAnswer"] = False
        session["currentDeepField"] = None
        if session.get("deepAnswers"):
            session["deepAnswers"].pop("budgetPreference", None)
        extraction = session.get("extraction")
        if extraction:
            extraction.pop("budget", None)
        await _save(session, app_state)
        return await _build_step0_response(session, text, app_state)

    session["awaitingPromptTweak"] = False
    _apply_edit_to_session(session, instruction)

    # Regenerate dynamic context based on new requirements / app purpose
    try:
        session["dynamicContext"] = await generate_dynamic_context(
            llm,
            session.get("appType") or "text",
            (session.get("extraction") or {}).get("appPurpose") or "",
            _detect_language_mode(session),
        )
    except Exception as e:
        logger.warning(f"[execEditApp] Regenerate dynamicContext failed: {e}")

    try:
        new_prompt_data, new_seo_data = await asyncio.gather(
            generate_prompt_template(llm, session),
            generate_seo(llm, session),
        )
        session["promptData"] = new_prompt_data
        session["seoData"] = new_seo_data
    except Exception as err:
        logger.warning(f"[execEditApp] Regen failed, applying safe fallback instruction: {err}")
        session["promptData"] = apply_prompt_instruction(session.get("promptData") or {}, instruction)

    session["step"] = 2
    await _save(session, app_state)

    prompt_data = session.get("promptData") or {}
    safe_image_input = (
        _sanitize_accept_image_input(prompt_data.get("acceptImageInput"), session.get("appType"))
        if prompt_data
        else False
    )
    seo_data = session.get("seoData") or {}

    return {
        "reply": (
            f'## ✅ App Updated\n\nApplied: **"{instruction}"**\n\n'
            "Here's the refreshed preview — approve when ready!"
        ),
        "uiType": "app_preview",
        "uiData": {
            "appName": seo_data.get("appName") or "Your App",
            "appType": session.get("appType") or "text",
            "appDescription": seo_data.get("appDescription") or "",
            "cost": session.get("modelCost"),
            "systemPrompt": prompt_data.get("systemPrompt") or "",
            "userPrompt": prompt_data.get("userPrompt") or "",
            "variablesUsed": prompt_data.get("variablesUsed") or [],
            "variables": (session.get("dynamicContext") or {}).get("variables") or [],
            "acceptImageInput": safe_image_input,
            "options": ["Approve App", "Edit App"],
        },
        "nextStep": 2,
        "coins": session.get("modelCost"),
    }


def _extract_budget_tier(text: str, decision: dict) -> str | None:
    tier = decision.get("budget_tier") or (decision.get("extracted_variables") or {}).get("budget")
    if tier:
        return str(tier).lower()
    match = re.search(r"\b(free|low|medium|premium)\b", text, re.IGNORECASE)
    return match.group(1).lower() if match else None


async def _exec_handle_budget(session: dict, text: str, decision: dict, app_state: Any) -> dict:
    tier = _extract_budget_tier(text, decision)
    if tier:
        if not session.get("extraction"):
            session["extraction"] = {}
        session["extraction"]["budget"] = tier
        if not session.get("deepAnswers"):
            session["deepAnswers"] = {}
        session["deepAnswers"]["budgetPreference"] = tier
        session["awaitingDeepAnswer"] = False
        session["currentDeepField"] = None
        await _save(session, app_state)
        return await _show_models(session, app_state)

    return {
        "reply": "What budget per generation would you like?",
        "uiType": "chips",
        "uiData": {"options": BUDGET_CHIP_OPTIONS},
        "nextStep": session.get("step", 0),
        "coins": None,
    }


async def _exec_change_model(session: dict, text: str, decision: dict, app_state: Any) -> dict:
    if not session.get("dynamicContext"):
        return {
            "reply": (
                "We'll pick the perfect model in a moment! First, let's finish scoping the app.\n\n"
                f'{session.get("lastQuestion") or "What specific details should the app handle?"}'
            ),
            "uiType": None,
            "uiData": None,
            "nextStep": 0,
            "coins": None,
        }

    tier = _extract_budget_tier(text, decision)
    if tier:
        if not session.get("extraction"):
            session["extraction"] = {}
        session["extraction"]["budget"] = tier
        if not session.get("deepAnswers"):
            session["deepAnswers"] = {}
        session["deepAnswers"]["budgetPreference"] = tier
        await _save(session, app_state)
        return await _show_models(session, app_state)

    if not session.get("extraction"):
        session["extraction"] = {}
    session["extraction"]["budget"] = None
    if session.get("deepAnswers"):
        session["deepAnswers"]["budgetPreference"] = None
    session["currentDeepField"] = "budgetPreference"
    session["awaitingDeepAnswer"] = True
    session["step"] = 0
    await _save(session, app_state)
    return {
        "reply": "What budget per generation would you like to switch to?",
        "uiType": "chips",
        "uiData": {"options": BUDGET_CHIP_OPTIONS},
        "nextStep": 0,
        "coins": None,
    }


async def _handle_seo_publish(session: dict, card_data: dict, app_state: Any) -> dict:
    prompt_data = session.get("promptData") or {}
    seo_data = {**(session.get("seoData") or {}), **card_data}
    session["seoData"] = seo_data

    payload = {
        "appType": session.get("appType"),
        "modelId": session.get("modelId"),
        "costPerRun": session.get("modelCost"),
        "systemPrompt": prompt_data.get("systemPrompt"),
        "userPrompt": prompt_data.get("userPrompt"),
        "negativePrompt": prompt_data.get("negativePrompt"),
        "acceptImageInput": _sanitize_accept_image_input(
            prompt_data.get("acceptImageInput"), session.get("appType")
        ),
        "appName": seo_data.get("appName"),
        "appDescription": seo_data.get("appDescription"),
        "tags": seo_data.get("tags"),
        "publishedAt": datetime.now(timezone.utc).isoformat(),
    }

    try:
        await app_state.cms.create_rapp(payload)
    except Exception as err:
        logger.error(f"[route] CMS publish failed: {err}")
        return {
            "reply": (
                "Sorry — publishing hit a snag on our end. Please try again in a moment "
                "or save as a draft."
            ),
            "uiType": "text",
            "uiData": None,
            "nextStep": session.get("step", 3),
            "coins": session.get("modelCost"),
        }

    session_id = session.get("sessionId")
    if session_id:
        await app_state.session.delete_session(session_id)

    app_name = seo_data.get("appName") or "Your App"
    return {
        "reply": (
            f'## 🎉 Published Successfully!\n\nYour app **"{app_name}"** is now live!\n\n'
            f"- **Cost per run:** {payload['costPerRun']} coins\n"
            "- **Status:** ✅ Live 🚀"
        ),
        "uiType": "success",
        "uiData": {
            "appName": app_name,
            "modelId": session.get("modelId"),
            "modelName": session.get("modelName") or session.get("modelId"),
            "costPerRun": payload["costPerRun"],
            "tags": seo_data.get("tags"),
            "mockUrl": f"https://rentprompts.ai/app/demo-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
        },
        "nextStep": 0,
        "coins": session.get("modelCost"),
        "clearSession": True,
    }


# ─── STATE DEFINITION ────────────────────────────────────────

class ConversationState(TypedDict, total=False):
    """Unified conversational state schema for LangGraph."""
    session_id: str
    message: str
    history: Annotated[list, add_messages]
    app_type: Optional[str]        # text | image | audio | video | vision
    app_purpose: Optional[str]
    extraction: Dict[str, Any]
    deep_answers: Dict[str, Any]
    primary_subject: Optional[str]
    environment_setting: Optional[str]
    action_dynamic: Optional[str]
    aesthetic_style: Optional[str]
    dynamic_context: Optional[Dict[str, Any]]
    model_id: Optional[str]
    model_name: Optional[str]
    model_cost: Optional[float]
    prompt_data: Dict[str, Any]
    seo_data: Dict[str, Any]
    current_step: int              # 0: Ideation, 1: Models, 2: Preview, 3: SEO Review
    recommended_action: str        # e.g., "GATHER_REQUIREMENTS", "HANDLE_OFF_TOPIC", "EDIT_APP"
    response_payload: Dict[str, Any] # React client contract
    
    # Internal parameters
    awaiting_confirmation: bool
    awaiting_prompt_tweak: bool
    awaiting_deep_answer: bool
    current_deep_field: Optional[str]
    triage_rounds: int
    form_confirmed: bool
    clear_session: bool
    language_mode: str
    enterprise_signals: Optional[bool]
    user_type: Optional[str]
    is_pivot: bool
    decision_payload: Dict[str, Any]
    last_slot_key: Optional[str]
    artistic_priority: Optional[bool]
    style_tags: Optional[list[str]]


# Helper state translators
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
        "primary_subject": (session.get("extraction") or {}).get("PRIMARY_SUBJECT"),
        "environment_setting": (session.get("extraction") or {}).get("ENVIRONMENT_SETTING"),
        "action_dynamic": (session.get("extraction") or {}).get("ACTION_DYNAMIC"),
        "aesthetic_style": (session.get("extraction") or {}).get("AESTHETIC_STYLE"),
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
        "model_id": session.get("modelId") or session.get("selectedModelId"),
        "model_cost": session.get("modelCost"),
        "model_name": session.get("modelName"),
        "prompt_data": session.get("promptData") or {},
        "seo_data": session.get("seoData") or {},
        "clear_session": False,
        "language_mode": session.get("languageMode") or "English",
        "enterprise_signals": session.get("enterpriseSignals"),
        "user_type": session.get("userType"),
        "is_pivot": session.get("isPivot") or False,
        "decision_payload": {},
        "last_slot_key": session.get("lastSlotKey"),
        "artistic_priority": session.get("artisticPriority") or False,
        "style_tags": session.get("styleTags") or [],
    }


def _state_to_session(state: ConversationState, session: dict) -> None:
    session["step"] = state.get("current_step", 0)
    session["appType"] = state.get("app_type")
    session["extraction"] = state.get("extraction") or {}
    
    session["extraction"]["PRIMARY_SUBJECT"] = state.get("primary_subject")
    session["extraction"]["ENVIRONMENT_SETTING"] = state.get("environment_setting")
    session["extraction"]["ACTION_DYNAMIC"] = state.get("action_dynamic")
    session["extraction"]["AESTHETIC_STYLE"] = state.get("aesthetic_style")
    
    session["dynamicContext"] = state.get("dynamic_context")
    session["deepAnswers"] = state.get("deep_answers") or {}
    session["awaitingConfirmation"] = state.get("awaiting_confirmation") or False
    session["awaitingPromptTweak"] = state.get("awaiting_prompt_tweak") or False
    session["awaitingDeepAnswer"] = state.get("awaiting_deep_answer") or False
    session["currentDeepField"] = state.get("current_deep_field")
    session["triageRounds"] = state.get("triage_rounds", 0)
    session["formConfirmed"] = state.get("form_confirmed") or False
    session["modelId"] = state.get("model_id")
    session["modelCost"] = state.get("model_cost")
    session["modelName"] = state.get("model_name")
    session["promptData"] = state.get("prompt_data") or {}
    session["seoData"] = state.get("seo_data") or {}
    session["languageMode"] = state.get("language_mode", "English")
    session["enterpriseSignals"] = state.get("enterprise_signals")
    session["userType"] = state.get("user_type")
    session["isPivot"] = state.get("is_pivot") or False
    session["lastSlotKey"] = state.get("last_slot_key")
    session["artisticPriority"] = state.get("artistic_priority") or False
    session["styleTags"] = state.get("style_tags") or []
    
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


# ─── ISOLATED ASYNC NODES ────────────────────────────────────

async def intent_classifier_node(state: ConversationState, config: dict) -> dict:
    """Classifies user intent and assigns next recommended action."""
    app_state = config["configurable"]["app_state"]
    message = state.get("message", "")
    text = _normalize(message)
    msg = _lower(text)
    
    # Check UI/Action interceptors
    if text.lower().startswith("multi_select_form::"):
        return {"recommended_action": "PROCESS_FORM"}
    if text.startswith("SEO_PUBLISH::"):
        return {"recommended_action": "PUBLISH_APP"}
    if text.startswith("SEO_DRAFT::"):
        return {"recommended_action": "SAVE_DRAFT"}
    if state.get("current_step") in (2, 3) and msg == "edit app":
        return {"recommended_action": "INITIATE_TWEAK"}
        
    temp_session = {}
    _state_to_session(state, temp_session)
    decision = await get_agentic_decision(app_state.llm, text, temp_session)
    action = decision.get("recommended_action") or "GATHER_REQUIREMENTS"
    
    # Extra robustness check for general questions/informational inquiries at any step
    msg_clean = msg.strip().lower()
    # Clean question prefixes
    question_prefixes = (
        "how does", "how do", "how is", "what is", "what are", "whats", "what's",
        "why does", "why do", "why is", "tell me about", "explain how", "explain what",
        "can you explain", "how can i", "how do i", "how to"
    )
    # Check if the user is asking an informational/general question
    is_informational_question = msg_clean.startswith(question_prefixes) or (
        ("?" in msg_clean or msg_clean.startswith(("what", "how", "why", "explain")))
        and not any(phrase in msg_clean for phrase in (
            "i want to build", "i want to create", "i want to make", "i want to start",
            "let's build", "lets build", "let's create", "lets create", "let's make", "lets make",
            "create a", "create an", "build a", "build an", "make a", "make an"
        ))
    )
    if is_informational_question or action == "HANDLE_OFF_TOPIC":
        action = "HANDLE_OFF_TOPIC"

    # Normalize decision fields to lowercase
    if decision.get("confidence"):
        decision["confidence"] = decision["confidence"].lower()
    if decision.get("budget_tier"):
        decision["budget_tier"] = decision["budget_tier"].lower()
        
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


async def off_topic_handler_node(state: ConversationState, config: dict) -> dict:
    """Answers general questions or casual chat naturally without modifying session state."""
    app_state = config["configurable"]["app_state"]
    message = state.get("message", "")
    
    system_prompt = (
        "You are RentPrompts App Architect, a conversational AI app builder. "
        "The user is asking a general question, seeking informational guidance, or having casual talk. "
        "Answer their question naturally, concisely, and friendly. "
        "Keep the focus on how RentPrompts can help them build, test, and publish custom AI apps (Rapps) "
        "for text, image, audio, video, or vision outputs. "
        "Do NOT ask configuration details (like budget, options, variables, prompts) yet unless they explicitly say they want to build an app."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    for m in state.get("history", []):
        role = m.get("role") if isinstance(m, dict) else m.type
        content = m.get("content") if isinstance(m, dict) else m.content
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})
    messages = messages[:1] + messages[-9:]
    
    try:
        completion = await app_state.llm.groq_completion(messages, model="llama-3.3-70b-versatile")
        reply = completion["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning(f"Off-topic completion failed: {e}. Falling back to default.")
        reply = (
            "I'm here to help you design, configure, and publish custom AI apps (Rapps) "
            "on the RentPrompts platform. What kind of app would you like to build?"
        )
        
    result = {
        "reply": reply,
        "uiType": None,
        "uiData": None,
        "nextStep": state.get("current_step", 0),
        "coins": None,
    }
    
    return {
        "response_payload": result,
    }


async def ideation_triage_node(state: ConversationState, config: dict) -> dict:
    """Runs Step 0 requirements scoping logic to build initial understanding."""
    app_state = config["configurable"]["app_state"]
    text = state.get("message", "")
    temp_session = {}
    _state_to_session(state, temp_session)
    
    _check_and_apply_artistic_priority(temp_session, text)
    if temp_session.get("artisticPriority"):
        temp_session["awaitingDeepAnswer"] = False
        temp_session["currentDeepField"] = None
        temp_session["formConfirmed"] = True
        temp_session["lastSlotKey"] = None
        temp_session["triageRounds"] = 0
        result = await _show_models(temp_session, app_state)
        new_state = _session_to_state(temp_session, text)
        new_state["response_payload"] = result
        return new_state
        
    # Check format input from chip override
    chip_type = _parse_chip_app_type(text)
    if chip_type:
        if not temp_session.get("appType") or (
            temp_session.get("formatAskedByTriage") and not temp_session.get("formatConfirmedByUser")
        ):
            temp_session["appType"] = chip_type
            if not temp_session.get("extraction"):
                temp_session["extraction"] = {}
            temp_session["extraction"]["appType"] = chip_type
            if temp_session.get("formatAskedByTriage"):
                temp_session["formatConfirmedByUser"] = True
                
    if temp_session.get("awaitingDeepAnswer") and temp_session.get("currentDeepField"):
        if not temp_session.get("deepAnswers"):
            temp_session["deepAnswers"] = {}
        temp_session["deepAnswers"][temp_session["currentDeepField"]] = text
        if not temp_session.get("extraction"):
            temp_session["extraction"] = {}
        if temp_session["currentDeepField"] == "budgetPreference":
            temp_session["extraction"]["budget"] = text
        temp_session["awaitingDeepAnswer"] = False
        temp_session["currentDeepField"] = None
        
        next_q = _get_next_deep_question(temp_session)
        if next_q:
            temp_session["currentDeepField"] = next_q["field"]
            temp_session["awaitingDeepAnswer"] = True
            result = {
                "reply": next_q["question"],
                "uiType": "chips",
                "uiData": {"options": next_q.get("options") or []},
                "nextStep": 0,
                "coins": None,
            }
        else:
            result = await _show_models(temp_session, app_state)
    else:
        result = await _exec_gather_requirements(temp_session, text, app_state)
        
    new_state = _session_to_state(temp_session, text)
    new_state["response_payload"] = result
    return new_state


async def form_submission_node(state: ConversationState, config: dict) -> dict:
    """Extracts features and variables from front-end multi-select forms."""
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


async def model_selection_node(state: ConversationState, config: dict) -> dict:
    """Filters and ranks model options for selection cards."""
    app_state = config["configurable"]["app_state"]
    temp_session = {}
    _state_to_session(state, temp_session)
    
    result = await _show_models(temp_session, app_state)
    
    new_state = _session_to_state(temp_session, state["message"])
    new_state["response_payload"] = result
    return new_state


async def app_preview_node(state: ConversationState, config: dict) -> dict:
    """Generates prompt template blueprints and serves the test card preview."""
    app_state = config["configurable"]["app_state"]
    text = state.get("message", "")
    temp_session = {}
    _state_to_session(state, temp_session)
    
    result = await _exec_generate_preview(temp_session, text, app_state)
    
    new_state = _session_to_state(temp_session, text)
    new_state["response_payload"] = result
    return new_state


async def modification_handler_node(state: ConversationState, config: dict) -> dict:
    """Processes prompt adjustments or AI engine selection swaps."""
    app_state = config["configurable"]["app_state"]
    text = state.get("message", "")
    temp_session = {}
    _state_to_session(state, temp_session)
    
    msg = _lower(text)
    decision = state.get("decision_payload") or {}
    if re.match(r"^change\s*:", msg, re.IGNORECASE):
        correction = re.sub(r"^change\s*:", "", text, flags=re.IGNORECASE).strip()
        if len(correction) > 1:
            extracted = decision.get("extracted_variables") or {}
            extracted["editInstruction"] = correction
            decision["extracted_variables"] = extracted
            
    result = await _exec_edit_app(temp_session, text, decision, app_state)
    
    new_state = _session_to_state(temp_session, text)
    new_state["response_payload"] = result
    return new_state


async def seo_review_node(state: ConversationState, config: dict) -> dict:
    """Presents calculated app metadata name, description, and tags for review."""
    app_state = config["configurable"]["app_state"]
    temp_session = {}
    _state_to_session(state, temp_session)
    
    result = await _exec_review_seo(temp_session, app_state)
    
    new_state = _session_to_state(temp_session, state["message"])
    new_state["response_payload"] = result
    return new_state


async def publish_app_node(state: ConversationState, config: dict) -> dict:
    """Pubishes configurations directly to the marketplace database CMS."""
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
            
    result = await _handle_seo_publish(temp_session, card_data, app_state)
    
    new_state = _session_to_state(temp_session, text)
    new_state["response_payload"] = result
    new_state["clear_session"] = result.get("clearSession", False)
    return new_state


# Additional nodes to ensure logic coverage
async def render_form_node(state: ConversationState, config: dict) -> dict:
    """Renders the dynamic option fields form."""
    app_state = config["configurable"]["app_state"]
    text = state.get("message", "")
    temp_session = {}
    _state_to_session(state, temp_session)
    
    from services.extraction import extract_requirements, _merge_extraction
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


async def handle_budget_node(state: ConversationState, config: dict) -> dict:
    app_state = config["configurable"]["app_state"]
    text = state.get("message", "")
    temp_session = {}
    _state_to_session(state, temp_session)
    decision = state.get("decision_payload") or {}
    result = await _exec_handle_budget(temp_session, text, decision, app_state)
    new_state = _session_to_state(temp_session, text)
    new_state["response_payload"] = result
    return new_state


async def change_model_node(state: ConversationState, config: dict) -> dict:
    app_state = config["configurable"]["app_state"]
    text = state.get("message", "")
    temp_session = {}
    _state_to_session(state, temp_session)
    decision = state.get("decision_payload") or {}
    result = await _exec_change_model(temp_session, text, decision, app_state)
    new_state = _session_to_state(temp_session, text)
    new_state["response_payload"] = result
    return new_state


async def pivot_app_node(state: ConversationState, config: dict) -> dict:
    app_state = config["configurable"]["app_state"]
    text = state.get("message", "")
    temp_session = {}
    _state_to_session(state, temp_session)
    decision = state.get("decision_payload") or {}
    result = await _exec_pivot_app(temp_session, text, decision, app_state)
    new_state = _session_to_state(temp_session, text)
    new_state["response_payload"] = result
    return new_state


async def save_draft_node(state: ConversationState, config: dict) -> dict:
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
    
    # Extract model and cost from card_data or session
    model_id = card_data.get("modelId") or temp_session.get("modelId") or "unknown"
    cost_per_run = card_data.get("costPerRun")
    if cost_per_run is None:
        cost_per_run = temp_session.get("modelCost")
    if cost_per_run is None:
        cost_per_run = 0
        
    tags = card_data.get("tags") or (temp_session.get("seoData") or {}).get("tags") or []
    
    result = {
        "reply": f'## 📋 Draft Saved\n\n**"{app_name}"** saved. Resume anytime from your dashboard.',
        "uiType": "success",
        "uiData": {
            "appName": app_name,
            "status": "Draft",
            "modelId": model_id,
            "costPerRun": cost_per_run,
            "selectedPlan": "draft",
            "tags": tags,
            "mockUrl": f"https://rentprompts.ai/app/demo-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
        },
        "nextStep": 0,
        "coins": None,
        "clearSession": True,
    }
    new_state = _session_to_state(temp_session, text)
    new_state["response_payload"] = result
    new_state["clear_session"] = True
    return new_state


async def initiate_tweak_node(state: ConversationState, config: dict) -> dict:
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


async def handle_greeting_node(state: ConversationState, config: dict) -> dict:
    temp_session = {}
    _state_to_session(state, temp_session)
    
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
    new_state = _session_to_state(temp_session, state.get("message", ""))
    new_state["response_payload"] = result
    return new_state


async def handle_violation_node(state: ConversationState, config: dict) -> dict:
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
    result = {
        "reply": "Hmm, I didn't quite catch that! 🤔 What type of output should your AI app generate?",
        "uiType": "chips",
        "uiData": {"options": ["Text", "Image", "Audio", "Video", "Vision"]},
        "nextStep": state.get("current_step", 0),
        "coins": None,
    }
    return {"response_payload": result}


# ─── CONDITIONAL STATE ROUTING EDGE ──────────────────────────

def route_by_conversational_intent(state: ConversationState) -> str:
    """Dynamic routing logic determining target execution state."""
    action = state.get("recommended_action")
    
    # ─── CONTEXT CHECK ───
    is_form_confirmed = bool(state.get("form_confirmed"))
    extraction = state.get("extraction") or {}
    deep_answers = state.get("deep_answers") or {}
    budget_pref = deep_answers.get("budgetPreference") or extraction.get("budget")
    
    if is_form_confirmed and budget_pref:
        if action == "GENERATE_PREVIEW" or state.get("current_step") == 2:
            return "app_preview_node"
        return "model_selection_node"

    # If the state variables indicate requirements are incomplete, bypass advance attempts
    if action in ("SHOW_MODEL_CARDS", "GENERATE_PREVIEW"):
        if not is_form_confirmed or not budget_pref:
            return "ideation_triage_node"
            
    if action == "HANDLE_OFF_TOPIC":
        return "off_topic_handler_node"
    if action == "HANDLE_VIOLATION":
        return "handle_violation_node"
    if action == "HANDLE_GIBBERISH":
        return "handle_gibberish_node"
    if action == "HANDLE_GREETING":
        return "handle_greeting_node"
    if action == "HANDLE_BUDGET":
        return "handle_budget_node"
    if action == "CHANGE_MODEL":
        return "change_model_node"
    if action == "PIVOT_APP":
        return "pivot_app_node"
    if action == "EDIT_APP":
        return "modification_handler_node"
    if action == "RENDER_FORM":
        return "render_form_node"
    if action == "SHOW_MODEL_CARDS":
        return "model_selection_node"
    if action == "GENERATE_PREVIEW":
        return "app_preview_node"
    if action == "REVIEW_SEO":
        return "seo_review_node"
    if action == "PUBLISH_APP":
        return "publish_app_node"
    if action == "SAVE_DRAFT":
        return "save_draft_node"
    if action == "INITIATE_TWEAK":
        return "initiate_tweak_node"
    if action == "PROCESS_FORM":
        return "form_submission_node"
        
    return "ideation_triage_node"


# ─── GRAPH ASSEMBLY & COMPILATION ────────────────────────────

def build_orchestrator_graph() -> StateGraph:
    """Builds and compiles the conversational routing StateGraph."""
    graph = StateGraph(ConversationState)
    
    # Add nodes
    graph.add_node("intent_classifier_node", intent_classifier_node)
    graph.add_node("off_topic_handler_node", off_topic_handler_node)
    graph.add_node("ideation_triage_node", ideation_triage_node)
    graph.add_node("form_submission_node", form_submission_node)
    graph.add_node("model_selection_node", model_selection_node)
    graph.add_node("app_preview_node", app_preview_node)
    graph.add_node("modification_handler_node", modification_handler_node)
    graph.add_node("seo_review_node", seo_review_node)
    graph.add_node("publish_app_node", publish_app_node)
    
    # Add extra nodes
    graph.add_node("render_form_node", render_form_node)
    graph.add_node("handle_budget_node", handle_budget_node)
    graph.add_node("change_model_node", change_model_node)
    graph.add_node("pivot_app_node", pivot_app_node)
    graph.add_node("save_draft_node", save_draft_node)
    graph.add_node("initiate_tweak_node", initiate_tweak_node)
    graph.add_node("handle_greeting_node", handle_greeting_node)
    graph.add_node("handle_violation_node", handle_violation_node)
    graph.add_node("handle_gibberish_node", handle_gibberish_node)
    
    # Establish Entrypoint
    graph.set_entry_point("intent_classifier_node")
    
    # Routing conditional edges
    graph.add_conditional_edges(
        "intent_classifier_node",
        route_by_conversational_intent,
        {
            "off_topic_handler_node": "off_topic_handler_node",
            "ideation_triage_node": "ideation_triage_node",
            "form_submission_node": "form_submission_node",
            "model_selection_node": "model_selection_node",
            "app_preview_node": "app_preview_node",
            "modification_handler_node": "modification_handler_node",
            "seo_review_node": "seo_review_node",
            "publish_app_node": "publish_app_node",
            
            # extra nodes mapping
            "render_form_node": "render_form_node",
            "handle_budget_node": "handle_budget_node",
            "change_model_node": "change_model_node",
            "pivot_app_node": "pivot_app_node",
            "save_draft_node": "save_draft_node",
            "initiate_tweak_node": "initiate_tweak_node",
            "handle_greeting_node": "handle_greeting_node",
            "handle_violation_node": "handle_violation_node",
            "handle_gibberish_node": "handle_gibberish_node",
        }
    )
    
    # Leaf links to END
    graph.add_edge("off_topic_handler_node", END)
    graph.add_edge("ideation_triage_node", END)
    graph.add_edge("form_submission_node", END)
    graph.add_edge("model_selection_node", END)
    graph.add_edge("app_preview_node", END)
    graph.add_edge("modification_handler_node", END)
    graph.add_edge("seo_review_node", END)
    graph.add_edge("publish_app_node", END)
    graph.add_edge("render_form_node", END)
    graph.add_edge("handle_budget_node", END)
    graph.add_edge("change_model_node", END)
    graph.add_edge("pivot_app_node", END)
    graph.add_edge("save_draft_node", END)
    graph.add_edge("initiate_tweak_node", END)
    graph.add_edge("handle_greeting_node", END)
    graph.add_edge("handle_violation_node", END)
    graph.add_edge("handle_gibberish_node", END)
    
    return graph.compile()


compiled_graph = build_orchestrator_graph()


# ─── CORE ORCHESTRATOR ENTRYPOINT ────────────────────────────

async def route(session: dict, message: str, app_state: Any) -> dict:
    """Main entrypoint invoking the event-driven LangGraph pipeline."""
    initial_state = _session_to_state(session, message)
    config = {"configurable": {"app_state": app_state}}
    
    final_state = await compiled_graph.ainvoke(initial_state, config=config)
    
    _state_to_session(final_state, session)
    return final_state.get("response_payload") or {}


# ─── FASTAPI HTTP ROUTER REFERENCE INTEGRATION ────────────────

# Pydantic templates for the client request / response payload structure
# class AgentChatRequest(BaseModel):
#     sessionId: str
#     message: str
#
# class AgentChatResponse(BaseModel):
#     reply: str
#     uiType: Optional[str] = None
#     uiData: Optional[dict] = None
#     step: int = 0
#     coins: Optional[float] = None
#     confirm: Optional[dict] = None
#
# @router.post("/api/agent/chat", response_model=AgentChatResponse)
# async def agent_chat(request: Request, body: AgentChatRequest):
#     # 1. Access request application states and services
#     session_svc = request.app.state.session
#     
#     # 2. Retrieve or initialize the session state
#     session = await session_svc.get_or_create_session(body.sessionId)
#     if not isinstance(session.get("history"), list):
#         session["history"] = []
#     
#     # 3. Add current user input to conversation history
#     session["history"].append({"role": "user", "content": body.message})
#     
#     # 4. Trigger the LangGraph State Machine route runner via .ainvoke()
#     response_payload = await route(session, body.message, request.app.state)
#     
#     # 5. Persist or clear session store changes based on execution outcomes
#     if response_payload.get("clearSession"):
#         await session_svc.delete_session(body.sessionId)
#     else:
#         session["history"].append({
#             "role": "agent", 
#             "content": response_payload.get("reply", ""), 
#             "uiType": response_payload.get("uiType")
#         })
#         await session_svc.save_session(session)
#         
#     # 6. Deliver calculated response_payload contract to client
#     return AgentChatResponse(
#         reply=response_payload.get("reply", ""),
#         uiType=response_payload.get("uiType"),
#         uiData=response_payload.get("uiData"),
#         step=response_payload.get("nextStep", session.get("step", 0)),
#         coins=response_payload.get("coins"),
#     )


