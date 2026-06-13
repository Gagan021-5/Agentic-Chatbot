"""
Step router for the RentPrompts chat lifecycle.
Async route(session, message, app_state) -> dict matching the React response contract.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Any

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


def _rank_models(available_models: list | None, user_input: str, budget_str: str) -> list:
    if not available_models:
        return []

    filtered = list(available_models)
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
        is_premium_intent = any(x in b for x in ("premium", "best", "> 20"))
        is_medium_intent = any(x in b for x in ("medium", "5-20", "5 - 20"))
        if is_premium_intent or is_medium_intent:
            return sorted(available_models, key=lambda m: m.get("cost", 0), reverse=True)[:3]
        return sorted(available_models, key=lambda m: m.get("cost", 0))[:3]

    input_lower = (user_input or "").lower()

    def score_model(model: dict) -> dict:
        score = 0
        for tag in model.get("tags") or []:
            if str(tag).lower() in input_lower:
                score += 5
        if any(w in input_lower for w in ("fast", "quick", "speed")):
            if model.get("tier") == "fast":
                score += 5
        if any(w in input_lower for w in ("quality", "best", "advanced")):
            if model.get("tier") in ("premium", "ultra"):
                score += 5
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


def _get_next_deep_question(session: dict) -> dict | None:
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
    models = _rank_models(model_collection, full_text, str(budget or ""))

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
                "portrait", "art generat", "image creat", "photo edit", "image edit",
                "visual generat", "thumbnail", "product photo", "banner maker", "illustration",
                "greeting card", "birthday card", "card maker", "card generat", "poster",
                "meme", "photo frame", "photo filter", "in the photo", "on the image",
                "text on image", "text overlay", "image with text", "invitation",
                "flyer", "avatar", "wallpaper", "sticker", "with photo", "with picture",
                "birthday card with photo", "birthday image", "birthday poster",
            ],
            "video": [
                "video generat", "generate video", "video creat", "animate", "animation",
                "photo to video", "image to video", "text to video", "reel generat", "short film",
                "video ad", "cinematic", "motion graphic",
            ],
            "audio": [
                "audio generat", "voiceover", "voice over", "text to speech", "tts", "speech generat",
                "music generat", "podcast", "narration", "voice cloning", "sound effect",
                "audio app", "speak text", "spoken audio", "verbal briefing", "audio briefing",
                "spoken summary", "audio log", "spoken log", "verbal report", "spoken report",
                "convert to audio", "read aloud", "audio file", "voice briefing", "tts app",
            ],
            "vision": [
                "crop disease", "plant disease", "detect disease", "object detect", "image analys",
                "ocr", "read text from image", "invoice scan", "document scan", "quality inspect",
                "medical image", "x-ray", "analyze image", "image recognit", "face detect",
            ],
            "text": [
                "advocate", "legal", "lawyer", "law", "blog", "article", "content", "email",
                "lesson plan", "quiz", "teacher", "educat", "farm advisor", "crop advisor",
                "chatbot", "chat assistant", "writing", "summarize", "translate", "script",
                "seo", "description generat", "report generat", "story generat",
                "resume", "cover letter", "proposal", "invoice", "contract", "newsletter",
                "recipe", "itinerary", "planner", "workout", "meal plan", "diet plan",
                "study guide", "flashcard", "essay", "thesis", "assignment", "homework",
                "letter", "memo", "document", "template", "form generat", "bio generat",
                "caption", "tagline", "slogan", "headline", "ad copy", "copywriting",
                "product description", "review generat", "feedback generat", "response generat",
                "text generat", "write", "draft", "compose", "author",
                "birthday wishes", "birthday message", "birthday poem", "birthday quote",
            ],
        }

        purpose_lower = str(ext.get("appPurpose") or ext.get("oneLineUnderstanding") or "").lower()

        if is_ambiguous_domain and not has_explicit_type_from_user:
            logger.info(
                f'[Ambiguous Domain] "{purpose_lower[:60]}" matches ambiguous signals — skipping local inference'
            )
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
                image_signals = [
                    "photo", "picture", "image", "card", "poster", "meme", "frame", "banner",
                    "flyer", "invitation", "greeting", "visual", "avatar", "portrait", "logo",
                    "thumbnail", "in the photo", "on the image", "with picture", "wallpaper", "sticker",
                ]
                video_signals = ["video", "animation", "animate", "reel", "clip", "cinematic"]
                audio_signals = ["audio", "voice", "music", "speech", "podcast", "sound", "tts"]
                vision_signals = ["detect", "analyze image", "scan", "ocr", "read from image"]

                inferred_type = "text"
                if any(s in purpose_l for s in image_signals):
                    inferred_type = "image"
                elif any(s in purpose_l for s in video_signals):
                    inferred_type = "video"
                elif any(s in purpose_l for s in audio_signals):
                    inferred_type = "audio"
                elif any(s in purpose_l for s in vision_signals):
                    inferred_type = "vision"

                session["appType"] = inferred_type
                if not session.get("extraction"):
                    session["extraction"] = {}
                session["extraction"]["appType"] = inferred_type
                logger.info(
                    f'[Smart Infer] No explicit type for "{str(ext.get("appPurpose"))[:50]}" — inferred {inferred_type}'
                )
        else:
            session["step"] = 0
            await _save(session, app_state)
            if _detect_language_mode(session) == "Hindi":
                options = ["टेक्स्ट", "इमेज", "ऑडियो", "वीडियो", "विज़न"]
            else:
                options = ["Text", "Image", "Audio", "Video", "Vision"]
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
        affirmations = [
            "yes", "sure", "ok", "yep", "yeah", "correct", "sounds good", "exactly",
            "perfect", "go ahead", "proceed", "that's right", "looks good", "right", "agreed", "great",
        ]
        msg_clean = re.sub(r"[!.,?]+$", "", _lower(text).strip())
        is_affirmation = msg_clean in affirmations

        if is_affirmation and (session.get("triageRounds") or 0) > 0:
            session["triageRounds"] = 99
            await _save(session, app_state)
        else:
            triage_result = await triage_dynamic_context(
                llm,
                session.get("appType"),
                (session.get("extraction") or {}).get("appPurpose") or "",
                _detect_language_mode(session),
                session.get("history") or [],
                session.get("deepAnswers") or {},
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
                    session["triageRounds"] = (session.get("triageRounds") or 0) + 1
                    session["lastQuestion"] = question
                    await _save(session, app_state)
                    suggested = triage_result.get("suggested_options")
                    has_chips = isinstance(suggested, list) and len(suggested) >= 2
                    return {
                        "reply": question,
                        "uiType": "chips" if has_chips else None,
                        "uiData": {"options": suggested} if has_chips else None,
                        "nextStep": 0,
                        "coins": None,
                    }

            if triage_result.get("corrected_app_type") and triage_result["corrected_app_type"] != session.get("appType"):
                logger.info(
                    f'[Triage] Type correction: {session.get("appType")} → {triage_result["corrected_app_type"]}'
                )
                session["appType"] = triage_result["corrected_app_type"]
                if session.get("extraction"):
                    session["extraction"]["appType"] = triage_result["corrected_app_type"]

            if triage_result.get("form"):
                session["dynamicContext"] = triage_result["form"]
            else:
                session["dynamicContext"] = await generate_dynamic_context(
                    llm,
                    session.get("appType") or (session.get("extraction") or {}).get("appType") or "text",
                    (session.get("extraction") or {}).get("appPurpose") or "",
                    _detect_language_mode(session),
                )
            session["triageRounds"] = 0
            session["formConfirmed"] = True
            await _save(session, app_state)

    if session.get("dynamicContext"):
        session["formConfirmed"] = True

    is_form_confirmed = bool(session.get("formConfirmed"))
    if is_form_confirmed:
        extraction = session.get("extraction") or {}
        deep_answers = session.get("deepAnswers") or {}
        if not extraction.get("budget") and not deep_answers.get("budgetPreference"):
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
    else:
        dynamic_context = session.get("dynamicContext") or {}
        if not isinstance(dynamic_context.get("variables"), list) or not isinstance(
            dynamic_context.get("options"), list
        ):
            session["dynamicContext"] = build_dynamic_context_fallback(
                session.get("appType") or "text",
                (session.get("extraction") or {}).get("appPurpose") or "",
                _detect_language_mode(session),
            )
        dynamic_context = session.get("dynamicContext") or {}
        return {
            "reply": _localized_text(
                session,
                "## 📋 Customize Your App Configuration\n\nI've generated a draft of key features and input fields based on our conversation.\n\nVerify or adjust the options below, then click **Confirm options**!",
                "## 📋 आपके ऐप का सेटअप\n\nमैंने हमारी बातचीत के आधार पर प्रमुख विशेषताओं और इनपुट फ़ील्ड्स का एक ड्राफ़्ट तैयार किया है।\n\nकृपया नीचे दिए विकल्पों की जाँच करें, फिर **Confirm options** पर क्लिक करें!",
                "## 📋 App Configuration Customise Karein\n\nMaine humari conversation ke basis par key features aur input fields ka ek draft generate kiya hai.\n\nNeeche diye options ko check/adjust karein, fir **Confirm options** par click karein!",
            ),
            "uiType": "multi_select_form",
            "uiData": {
                "options": dynamic_context.get("options") or [],
                "variables": dynamic_context.get("variables") or [],
            },
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
        prompt_data, seo_data = await asyncio.gather(
            generate_prompt_template(llm, session),
            generate_seo(llm, session),
        )
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
                "acceptImageInput": _sanitize_accept_image_input(
                    prompt_data.get("acceptImageInput"), app_type
                ),
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


async def route(session: dict, message: str, app_state: Any) -> dict:
    """Main orchestration entry for the chat state router."""
    raw_text = str(message or "")[:1000]
    text = _normalize(raw_text)
    msg = _lower(text)

    if text.lower().startswith("multi_select_form::"):
        payload = _parse_multi_select_payload(message)
        if payload:
            if not session.get("dynamicContext"):
                session["dynamicContext"] = {}
            session["dynamicContext"]["options"] = payload.get("selectedOptions") or []
            session["dynamicContext"]["variables"] = [
                {
                    "name": v.get("name"),
                    "placeholder": v.get("placeholder") or "Enter details...",
                    "test_value": v.get("value") or "",
                }
                for v in (payload.get("variables") or [])
                if isinstance(v, dict)
            ]
            session["formConfirmed"] = True
            if not session.get("extraction"):
                session["extraction"] = {}
            session["extraction"]["keyFeatures"] = payload.get("selectedOptions") or []

            budget = (session.get("deepAnswers") or {}).get("budgetPreference") or (
                (session.get("extraction") or {}).get("budget")
            )
            if budget:
                return await _show_models(session, app_state)

            session["currentDeepField"] = "budgetPreference"
            session["awaitingDeepAnswer"] = True
            await _save(session, app_state)
            return {
                "reply": "One last thing — **what's your budget per generation?**",
                "uiType": "chips",
                "uiData": {"options": BUDGET_CHIP_OPTIONS},
                "nextStep": 0,
                "coins": None,
            }

    is_seo_publish = text.startswith("SEO_PUBLISH::")
    is_seo_save_draft = text.startswith("SEO_DRAFT::")
    if is_seo_publish or is_seo_save_draft:
        try:
            json_str = text[text.index("::") + 2 :]
            card_data = json.loads(json_str)
            session["seoData"] = {**(session.get("seoData") or {}), **card_data}
            await _save(session, app_state)
        except (json.JSONDecodeError, TypeError, ValueError) as err:
            logger.warning(f"[route] Failed to parse SEO payload: {err}")

        if is_seo_publish:
            try:
                json_str = text[text.index("::") + 2 :]
                card_data = json.loads(json_str)
            except (json.JSONDecodeError, TypeError, ValueError):
                card_data = {}
            return await _handle_seo_publish(session, card_data, app_state)

        session["status"] = "draft"
        await _save(session, app_state)
        app_name = (session.get("seoData") or {}).get("appName") or "Your App"
        return {
            "reply": f'## 📋 Draft Saved\n\n**"{app_name}"** saved. Resume anytime from your dashboard.',
            "uiType": "success",
            "uiData": {"appName": app_name, "status": "Draft"},
            "nextStep": 0,
            "coins": None,
        }

    if session.get("step") in (2, 3) and msg == "edit app":
        session["awaitingPromptTweak"] = True
        await _save(session, app_state)
        return {
            "reply": (
                "I'm listening! 📝\n\n"
                "- **Tweak the prompt** — tell me what to change\n"
                "- **Switch the AI model** — pick a different engine\n"
                "- **Start fresh** — describe a completely new app idea\n\n"
                "What would you like to adjust?"
            ),
            "uiType": "text",
            "uiData": None,
            "nextStep": session.get("step"),
            "coins": session.get("modelCost"),
        }

    decision = await get_agentic_decision(app_state.llm, text, session)
    logger.info(
        f"[Router] Action: {decision.get('recommended_action')} ({decision.get('_source')}) "
        f"| confidence: {decision.get('confidence')}"
    )

    if decision.get("app_type") and decision["app_type"] != session.get("appType"):
        session["appType"] = decision["app_type"]
        if not session.get("extraction"):
            session["extraction"] = {}
        session["extraction"]["appType"] = decision["app_type"]

    action = decision.get("recommended_action")

    if action == "HANDLE_GREETING":
        if len(session.get("history") or []) < 3:
            return {
                "reply": (
                    "Hey there! 👋 I'm your **RentPrompts App Architect** — I help you design, "
                    "configure, and publish AI-powered apps in minutes.\n\n"
                    "**Just describe your app idea** and I'll handle the rest!\n\n"
                    "What would you like to build today?"
                ),
                "uiType": None,
                "uiData": None,
                "nextStep": session.get("step", 0),
                "coins": None,
            }
        return await _exec_gather_requirements(session, text, app_state)

    if action == "HANDLE_OFF_TOPIC":
        if decision.get("confidence") != "low":
            return OFF_TOPIC_RESPONSE
        return await _exec_gather_requirements(session, text, app_state)

    if action == "HANDLE_VIOLATION":
        return {
            "reply": (
                "I can only help build apps that comply with RentPrompts' safety and content "
                "guidelines. Please suggest a different idea."
            ),
            "uiType": "text",
            "uiData": None,
            "nextStep": session.get("step", 0),
            "coins": None,
        }

    if action == "HANDLE_GIBBERISH":
        return {
            "reply": "Hmm, I didn't quite catch that! 🤔 What type of output should your AI app generate?",
            "uiType": "chips",
            "uiData": {"options": ["Text", "Image", "Audio", "Video", "Vision"]},
            "nextStep": session.get("step", 0),
            "coins": None,
        }

    if action == "HANDLE_BUDGET":
        return await _exec_handle_budget(session, text, decision, app_state)

    if action == "CHANGE_MODEL":
        return await _exec_change_model(session, text, decision, app_state)

    if action == "PIVOT_APP":
        return await _exec_pivot_app(session, text, decision, app_state)

    if action == "EDIT_APP":
        if re.match(r"^change\s*:", msg, re.IGNORECASE):
            correction = re.sub(r"^change\s*:", "", text, flags=re.IGNORECASE).strip()
            if len(correction) > 1:
                extracted = decision.get("extracted_variables") or {}
                extracted["editInstruction"] = correction
                decision["extracted_variables"] = extracted
        return await _exec_edit_app(session, text, decision, app_state)

    if action == "RENDER_FORM":
        if not session.get("history"):
            session["history"] = []
        extraction = session.get("extraction") or {}
        if not extraction.get("appPurpose"):
            ext = await extract_requirements(app_state.llm, text, session["history"])
            session["extraction"] = _merge_extraction(session.get("extraction"), ext, text)
        await _save(session, app_state)
        return await _exec_render_form(session, app_state)

    if action == "SHOW_MODEL_CARDS":
        return await _show_models(session, app_state)

    if action == "GENERATE_PREVIEW":
        return await _exec_generate_preview(session, text, app_state)

    if action == "REVIEW_SEO":
        return await _exec_review_seo(session, app_state)

    if action == "PUBLISH_APP":
        if "save draft" in msg or "save to draft" in msg:
            return {
                "reply": (
                    "Done! Your progress has been saved as a draft. "
                    "Access it anytime from your RentPrompts dashboard."
                ),
                "uiType": "success",
                "uiData": {
                    "appName": (session.get("seoData") or {}).get("appName")
                    or (session.get("extraction") or {}).get("appPurpose")
                    or "Untitled Draft",
                    "status": "Draft",
                },
                "nextStep": 0,
                "coins": None,
                "clearSession": True,
            }
        if any(phrase in msg for phrase in ("start over", "restart", "reset")):
            return {
                "reply": (
                    "No problem! 🔄 Let's start fresh.\n\n"
                    "**What kind of AI app would you like to build?**"
                ),
                "uiType": "chips",
                "uiData": {
                    "options": ["Image app", "Video app", "Text app", "Audio app", "Vision app"]
                },
                "nextStep": 0,
                "coins": None,
                "clearSession": True,
            }
        return {
            "reply": "Ready to publish? Review the SEO card and hit **Publish to Marketplace**!",
            "uiType": "chips",
            "uiData": {"options": ["Publish to Marketplace", "Save Draft", "Edit App"]},
            "nextStep": session.get("step") or 3,
            "coins": session.get("modelCost"),
        }

    # GATHER_REQUIREMENTS and default
    chip_type = _parse_chip_app_type(text)
    if chip_type:
        if not session.get("appType") or (
            session.get("formatAskedByTriage") and not session.get("formatConfirmedByUser")
        ):
            session["appType"] = chip_type
            if not session.get("extraction"):
                session["extraction"] = {}
            session["extraction"]["appType"] = chip_type
            if session.get("formatAskedByTriage"):
                session["formatConfirmedByUser"] = True
                logger.info(f"[Format Override] Chip confirmed: {chip_type}")

    if session.get("awaitingDeepAnswer") and session.get("currentDeepField"):
        if not session.get("deepAnswers"):
            session["deepAnswers"] = {}
        session["deepAnswers"][session["currentDeepField"]] = text
        if not session.get("extraction"):
            session["extraction"] = {}
        if session["currentDeepField"] == "budgetPreference":
            session["extraction"]["budget"] = text
        session["awaitingDeepAnswer"] = False
        session["currentDeepField"] = None

        next_q = _get_next_deep_question(session)
        if next_q:
            session["currentDeepField"] = next_q["field"]
            session["awaitingDeepAnswer"] = True
            await _save(session, app_state)
            return {
                "reply": next_q["question"],
                "uiType": "chips",
                "uiData": {"options": next_q.get("options") or []},
                "nextStep": 0,
                "coins": None,
            }
        return await _show_models(session, app_state)

    return await _exec_gather_requirements(session, text, app_state)
