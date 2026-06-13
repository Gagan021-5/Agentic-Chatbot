"""
Orchestration brain — RentPrompts intentEngine.js.
LLM function calling with regex fallback.
"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

from services.llm import LLMService

ORCHESTRATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "orchestrate_pipeline",
        "description": (
            "Given the user's message and the full session state, decide the single next pipeline "
            "action the orchestrator should execute. Extract any runtime variables the user mentioned "
            "and correct the app type if the classification is clearly wrong."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "recommended_action": {
                    "type": "string",
                    "enum": [
                        "GATHER_REQUIREMENTS",
                        "RENDER_FORM",
                        "SHOW_MODEL_CARDS",
                        "GENERATE_PREVIEW",
                        "REVIEW_SEO",
                        "PUBLISH_APP",
                        "PIVOT_APP",
                        "EDIT_APP",
                        "CHANGE_MODEL",
                        "HANDLE_BUDGET",
                        "HANDLE_GREETING",
                        "HANDLE_OFF_TOPIC",
                        "HANDLE_VIOLATION",
                        "HANDLE_GIBBERISH",
                    ],
                },
                "extracted_variables": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
                "app_type": {
                    "type": ["string", "null"],
                    "enum": ["text", "image", "video", "audio", "vision", None],
                },
                "is_major_pivot": {"type": "boolean"},
                "budget_tier": {
                    "type": ["string", "null"],
                    "enum": ["free", "low", "medium", "premium", None],
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
            },
            "required": ["recommended_action", "is_major_pivot", "confidence"],
        },
    },
}

ORCHESTRATOR_SYSTEM_PROMPT = """You are the Orchestration Brain for RentPrompts — a platform where users CREATE and PUBLISH AI-powered apps.

On every turn you receive: the user's latest message, the conversation history, and a SESSION STATE SNAPSHOT.
Your ONLY job: call orchestrate_pipeline with the single correct next action.

═══════════════════════════════════════
PIPELINE STAGE LOGIC
═══════════════════════════════════════

Use the SESSION STATE SNAPSHOT to determine what stage the pipeline is at:

STAGE 0 — REQUIREMENTS GATHERING (session.hasPurpose=false OR session.triageComplete=false):
  → Default action: GATHER_REQUIREMENTS
  → If user supplies budget: HANDLE_BUDGET
  → If user says hi only: HANDLE_GREETING

STAGE 1 — FORM READY (session.triageComplete=true AND session.formConfirmed=false):
  → RENDER_FORM unless user just answered a budget question → SHOW_MODEL_CARDS

STAGE 2 — MODEL SELECTION (session.formConfirmed=true AND session.modelSelected=false):
  → SHOW_MODEL_CARDS or HANDLE_BUDGET

STAGE 3 — PREVIEW (session.modelSelected=true AND session.previewApproved=false):
  → GENERATE_PREVIEW (model was just selected)
  → EDIT_APP (user wants tweaks)
  → PIVOT_APP (user describes completely different app)

STAGE 4 — SEO REVIEW (session.previewApproved=true AND session.seoReviewed=false):
  → REVIEW_SEO

STAGE 5 — PUBLISH (session.seoReviewed=true):
  → PUBLISH_APP

═══════════════════════════════════════
CLASSIFICATION RULES
═══════════════════════════════════════

PIVOT vs EDIT:
  - PIVOT_APP: user describes a DIFFERENT domain entirely. Set is_major_pivot=true.
  - EDIT_APP: user wants to refine the SAME app (tone, model, prompt, variable).

APP TYPE (most common mistake — read carefully):
  - "image": output is a PICTURE — logos, cards, posters, room designs, memes, avatars
  - "text": output is WRITTEN WORDS — blogs, legal docs, recipes, emails, wishes
  - "audio": output is SOUND — TTS, voiceover, music, podcast
  - "video": output is VIDEO — reels, animations, clips
  - "vision": INPUT is an image to ANALYZE — OCR, defect detection, medical imaging
  - "birthday card with photo" = IMAGE. "birthday wishes text" = TEXT. This is the #1 error.

AFFIRMATIONS ("yes", "sure", "ok", "sounds good", "go ahead", "proceed"):
  - During triage → GATHER_REQUIREMENTS (the yes is an answer to a triage question)
  - After form shown but not confirmed → RENDER_FORM
  - After model shown → interpret as model selection if a model was displayed, else SHOW_MODEL_CARDS
  - After preview shown → REVIEW_SEO

BUDGET CHIPS ("Free models only (0 coins)", "Low (< 5 coins)", etc.):
  → HANDLE_BUDGET always

MODEL SELECTION ("Select model-id"):
  → GENERATE_PREVIEW

APPROVE ("Approve App", "approve"):
  → REVIEW_SEO

You MUST call orchestrate_pipeline. Never respond with plain text."""


def build_session_snapshot(session: dict | None) -> dict:
    s = session or {}
    extraction = s.get("extraction") or {}
    deep = s.get("deepAnswers") or {}
    app_purpose = extraction.get("appPurpose") or ""
    return {
        "hasPurpose": bool(app_purpose and len(app_purpose) > 5),
        "appPurposePreview": app_purpose[:80] or None,
        "currentAppType": s.get("appType"),
        "appTypeConfidence": (extraction.get("confidence") or {}).get("appType", "LOW"),
        "triageComplete": bool(s.get("dynamicContext")),
        "formConfirmed": bool(s.get("formConfirmed")),
        "modelSelected": bool(s.get("modelId")),
        "selectedModel": s.get("modelId"),
        "modelCost": s.get("modelCost"),
        "previewApproved": bool(s.get("step", 0) >= 2 and s.get("promptData")),
        "seoReviewed": bool(s.get("step", 0) >= 3),
        "budgetSet": bool(extraction.get("budget") or deep.get("budgetPreference")),
        "currentBudget": deep.get("budgetPreference") or extraction.get("budget"),
        "awaitingTriageAnswer": bool(s.get("awaitingTriageAnswer")),
        "awaitingDeepAnswer": bool(s.get("awaitingDeepAnswer")),
        "awaitingPromptTweak": bool(s.get("awaitingPromptTweak")),
        "triageRounds": s.get("triageRounds") or 0,
        "currentDeepField": s.get("currentDeepField"),
        "domainIdentified": s.get("domainIdentified"),
        "languageMode": s.get("languageMode") or "English",
        "isPivot": bool(s.get("isPivot")),
        "_legacyStep": s.get("step", 0),
    }


def try_fast_path(text: str, session: dict | None) -> dict | None:
    t = str(text or "").strip()
    lower = t.lower()

    if (
        t.startswith("multi_select_form::")
        or t.startswith("SEO_PUBLISH::")
        or t.startswith("SEO_DRAFT::")
        or t.startswith("SEO_EDIT::")
        or t.startswith("confirm seo::")
    ):
        return {
            "recommended_action": "PUBLISH_APP",
            "is_major_pivot": False,
            "confidence": "high",
            "_source": "fast_path",
        }

    if t.lower().startswith("edit prompt::"):
        return {
            "recommended_action": "EDIT_APP",
            "is_major_pivot": False,
            "confidence": "high",
            "_source": "fast_path",
        }

    if lower in ("approve app", "approve"):
        return {
            "recommended_action": "REVIEW_SEO",
            "is_major_pivot": False,
            "confidence": "high",
            "_source": "fast_path",
        }

    if lower in ("edit app", "edit"):
        return {
            "recommended_action": "EDIT_APP",
            "is_major_pivot": False,
            "confidence": "high",
            "_source": "fast_path",
        }

    if lower in ("publish to marketplace", "save draft"):
        return {
            "recommended_action": "PUBLISH_APP",
            "is_major_pivot": False,
            "confidence": "high",
            "_source": "fast_path",
        }

    if re.match(r"^select\s+\S", t, re.I) and lower not in (
        "select lean", "select recommended", "select full"
    ):
        return {
            "recommended_action": "GENERATE_PREVIEW",
            "is_major_pivot": False,
            "confidence": "high",
            "_source": "fast_path",
        }

    budget_map = {
        "free models only (0 coins)": "free",
        "low (< 5 coins)": "low",
        "medium (5 - 20 coins)": "medium",
        "premium (> 20 coins)": "premium",
    }
    if lower in budget_map:
        tier = budget_map[lower]
        return {
            "recommended_action": "HANDLE_BUDGET",
            "budget_tier": tier,
            "is_major_pivot": False,
            "confidence": "high",
            "extracted_variables": {"budget": tier},
            "_source": "fast_path",
        }

    chip_types = {
        "text": "text", "image": "image", "audio": "audio", "video": "video", "vision": "vision",
        "text app": "text", "image app": "image", "audio app": "audio", "video app": "video", "vision app": "vision",
        "generate images or photos": "image", "image generator": "image",
        "create videos or animations": "video", "video creator": "video",
        "write text": "text", "writing tool": "text",
        "generate voice or music": "audio", "audio generator": "audio",
        "analyze or understand images": "vision", "image analyzer": "vision",
    }
    if lower in chip_types:
        return {
            "recommended_action": "GATHER_REQUIREMENTS",
            "app_type": chip_types[lower],
            "is_major_pivot": False,
            "confidence": "high",
            "_source": "fast_path",
        }

    return None


def build_fallback_decision(message: str, session: dict | None) -> dict:
    msg = str(message or "").strip().lower()
    snapshot = build_session_snapshot(session)

    recommended_action = "GATHER_REQUIREMENTS"
    app_type = None
    budget_tier = None
    is_major_pivot = False
    extracted_variables: dict[str, str] = {}

    if re.match(r"^(hi|hello|hey|hy|hola|greetings)[\s!.]*$", msg, re.I):
        recommended_action = "HANDLE_GREETING"
    elif re.match(r"^(help)[\s!.]*$", msg, re.I):
        recommended_action = "HANDLE_OFF_TOPIC"
    elif re.search(r"\b(jailbreak|nsfw|nude|hack|exploit|bomb|weapon|illegal)\b", msg, re.I):
        recommended_action = "HANDLE_VIOLATION"
    elif re.search(r"\b(free|low|medium|premium)\b", msg, re.I) and re.search(
        r"\b(coin|budget|model)\b", msg, re.I
    ):
        recommended_action = "HANDLE_BUDGET"
        bm = re.search(r"\b(free|low|medium|premium)\b", msg, re.I)
        budget_tier = bm.group(1).lower() if bm else None
        if budget_tier:
            extracted_variables["budget"] = budget_tier
    elif re.search(r"\b(approve|approved|looks good|proceed|confirm|yes proceed)\b", msg, re.I):
        recommended_action = "PUBLISH_APP" if snapshot["previewApproved"] else "REVIEW_SEO"
    elif re.search(r"\b(publish|save draft|go live)\b", msg, re.I):
        recommended_action = "PUBLISH_APP"
    elif re.search(r"\b(change|switch|different)\b.{0,20}\b(model|ai|engine)\b", msg, re.I):
        recommended_action = "CHANGE_MODEL"
    elif (
        re.search(r"\b(i want|build|create|make)\b.{2,50}\b(app|tool|generator)\b", msg, re.I)
        and snapshot["hasPurpose"]
    ):
        recommended_action = "PIVOT_APP"
        is_major_pivot = True
    elif re.search(r"\b(change|edit|tweak|update|modify|make it|add|remove)\b", msg, re.I) and snapshot["hasPurpose"]:
        recommended_action = "EDIT_APP"
        extracted_variables["editInstruction"] = message
    elif snapshot["formConfirmed"] and snapshot["budgetSet"] and not snapshot["modelSelected"]:
        recommended_action = "SHOW_MODEL_CARDS"
    elif snapshot["triageComplete"] and not snapshot["formConfirmed"]:
        recommended_action = "RENDER_FORM"

    type_signals = {
        "image": re.compile(
            r"\b(image|photo|picture|logo|poster|card|avatar|portrait|room design|banner|meme|flyer|sticker)\b", re.I
        ),
        "audio": re.compile(r"\b(audio|voice|podcast|tts|speech|narration|music|sound)\b", re.I),
        "video": re.compile(r"\b(video|animation|animate|reel|cinematic|clip)\b", re.I),
        "vision": re.compile(r"\b(detect|analyze image|scan|ocr|read from image)\b", re.I),
        "text": re.compile(r"\b(text|blog|legal|recipe|email|story|script|plan|write|article)\b", re.I),
    }
    for t, pattern in type_signals.items():
        if pattern.search(msg):
            app_type = t
            break

    logger.info(f"[Orchestrator:Fallback] Action: {recommended_action} | Confidence: low")

    return {
        "recommended_action": recommended_action,
        "extracted_variables": extracted_variables,
        "app_type": app_type,
        "is_major_pivot": is_major_pivot,
        "budget_tier": budget_tier,
        "confidence": "low",
        "_source": "fallback_regex",
    }


async def get_agentic_decision(llm: LLMService, message: str, session: dict) -> dict:
    text = str(message or "").strip()

    fast = try_fast_path(text, session)
    if fast:
        logger.info(f"[Orchestrator] Fast-path: {fast['recommended_action']}")
        return fast

    if not llm.has_groq:
        logger.warning("[Orchestrator] GROQ_API_KEY not set — falling back to regex dispatcher")
        return build_fallback_decision(text, session)

    try:
        snapshot = build_session_snapshot(session)
        history_slice = []
        for h in (session.get("history") or [])[-8:]:
            role = "assistant" if h.get("role") == "agent" else "user"
            content = h.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content)
            history_slice.append({"role": role, "content": content[:400]})

        messages = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "system", "content": f"SESSION STATE SNAPSHOT:\n{json.dumps(snapshot, indent=2)}"},
            *history_slice,
            {"role": "user", "content": text},
        ]

        result = await llm.groq_completion(
            messages=messages,
            model="llama-3.1-8b-instant",
            max_tokens=300,
            temperature=0.1,
            tools=[ORCHESTRATOR_TOOL],
            tool_choice={"type": "function", "function": {"name": "orchestrate_pipeline"}},
        )

        tool_call = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("tool_calls", [{}])[0]
        )
        args_str = tool_call.get("function", {}).get("arguments")
        if args_str:
            parsed = json.loads(args_str)
            action = parsed.get("recommended_action") or "GATHER_REQUIREMENTS"

            safe_actions = {
                "HANDLE_GREETING", "HANDLE_OFF_TOPIC", "HANDLE_VIOLATION",
                "HANDLE_GIBBERISH", "GATHER_REQUIREMENTS",
            }
            if not snapshot["hasPurpose"] and action not in safe_actions:
                action = "GATHER_REQUIREMENTS"

            if (
                not snapshot["modelSelected"]
                and action == "GENERATE_PREVIEW"
                and not re.match(r"^select\s+", text, re.I)
            ):
                action = "SHOW_MODEL_CARDS" if snapshot["formConfirmed"] else "GATHER_REQUIREMENTS"

            decision = {
                "recommended_action": action,
                "extracted_variables": parsed.get("extracted_variables") or {},
                "app_type": parsed.get("app_type"),
                "is_major_pivot": bool(parsed.get("is_major_pivot")),
                "budget_tier": parsed.get("budget_tier").lower() if parsed.get("budget_tier") else None,
                "confidence": parsed.get("confidence").lower() if parsed.get("confidence") else "medium",
                "_source": "llm",
            }
            logger.info(
                f"[Orchestrator] Action: {decision['recommended_action']} | "
                f"AppType: {decision['app_type']} | Pivot: {decision['is_major_pivot']} | "
                f"Budget: {decision['budget_tier']} | Confidence: {decision['confidence']}"
            )
            return decision

        logger.warning("[Orchestrator] No tool call in LLM response — falling back")
        return build_fallback_decision(text, session)

    except Exception as err:
        from tenacity import RetryError
        real_err = err
        if isinstance(err, RetryError):
            attempt = getattr(err, "last_attempt", None)
            if attempt:
                try:
                    underlying = attempt.exception()
                    if underlying:
                        real_err = underlying
                except Exception:
                    pass
        
        # Log response content if it's an httpx status error
        import httpx
        if isinstance(real_err, httpx.HTTPStatusError):
            try:
                resp_text = real_err.response.text
                logger.error(f"[Orchestrator] LLM call failed with status {real_err.response.status_code}: {resp_text}")
            except Exception:
                logger.error(f"[Orchestrator] LLM call failed: {real_err}")
        else:
            logger.error(f"[Orchestrator] LLM call failed: {real_err}")
            
        return build_fallback_decision(text, session)


ACTION_MAP = {
    "start_app": "GATHER_REQUIREMENTS",
    "pivot_app": "PIVOT_APP",
    "edit_app": "EDIT_APP",
    "select_budget": "HANDLE_BUDGET",
    "select_model": "CHANGE_MODEL",
    "affirmation": "GATHER_REQUIREMENTS",
    "greeting": "HANDLE_GREETING",
    "off_topic": "HANDLE_OFF_TOPIC",
    "policy_violation": "HANDLE_VIOLATION",
    "gibberish": "HANDLE_GIBBERISH",
    "answer_question": "GATHER_REQUIREMENTS",
    "ui_action": "PUBLISH_APP",
}


async def get_agentic_intent(llm: LLMService, message: str, session: dict) -> dict:
    decision = await get_agentic_decision(llm, message, session)
    action = "answer_question"
    for k, v in ACTION_MAP.items():
        if v == decision["recommended_action"]:
            action = k
            break
    return {
        "action": action,
        "recommended_action": decision["recommended_action"],
        "app_type": decision.get("app_type"),
        "budget_tier": decision.get("budget_tier"),
        "is_major_pivot": decision.get("is_major_pivot"),
        "edit_instruction": (decision.get("extracted_variables") or {}).get("editInstruction"),
        "extracted_details": decision.get("extracted_variables") or {},
        "confidence": decision.get("confidence"),
        "_source": decision.get("_source"),
        "_decision": decision,
    }
