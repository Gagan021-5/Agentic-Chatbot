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
            "action the orchestrator should execute, and specify the app type and confidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "recommended_action": {
                    "type": "string",
                    "enum": [
                        "GATHER_REQUIREMENTS",
                        "PROCESS_FORM",
                        "SHOW_MODEL_CARDS",
                        "GENERATE_PREVIEW",
                        "EDIT_APP",
                        "PIVOT_APP",
                        "HANDLE_OFF_TOPIC",
                    ],
                },
                "app_type": {
                    "type": "string",
                    "enum": ["text", "image", "audio", "video", "vision"],
                },
                "confidence": {
                    "type": "string",
                    "enum": ["HIGH", "MEDIUM", "LOW"],
                },
                "reasoning": {
                    "type": "string",
                    "description": "Clear explanation of state alignment and why this action was selected",
                },
            },
            "required": ["recommended_action", "app_type", "confidence", "reasoning"],
        },
    },
}

ORCHESTRATOR_SYSTEM_PROMPT = """You are the Master Intent Classifier and Orchestrator Node for RentPrompts — a platform where users CREATE and PUBLISH AI-powered apps.
Your job is to read the latest user message, evaluate the multi-turn session state memory, and select the next deterministic action.

═══════════════════════════════════════
PIPELINE STAGE LOGIC
═══════════════════════════════════════

Use the SESSION STATE SNAPSHOT to determine what stage the pipeline is at:

STAGE 0 — REQUIREMENTS GATHERING (session.hasPurpose=false OR session.triageComplete=false):
  → Default action: GATHER_REQUIREMENTS
  → If user says hi only or off-topic: HANDLE_OFF_TOPIC

STAGE 1 — FORM READY (session.triageComplete=true AND session.formConfirmed=false):
  → PROCESS_FORM (triage is done, we need to show/process the form)

STAGE 2 — MODEL SELECTION (session.formConfirmed=true AND session.modelSelected=false):
  → SHOW_MODEL_CARDS (form is confirmed, now rank and select model)

STAGE 3 — PREVIEW (session.modelSelected=true AND session.previewApproved=false):
  → GENERATE_PREVIEW (model was just selected, generate live preview)
  → EDIT_APP (user wants tweaks to the prompt/template)
  → PIVOT_APP (user describes a completely different application)

═══════════════════════════════════════
PRD COMPLIANCE & STATE RULES
═══════════════════════════════════════

1. CRITICAL STATE PRESERVATION: Look at the existing session state format ('appType' or 'app_type'). If it is locked as "audio", "video", "image", or "vision", do NOT change it to "text" or "None" unless the recommended_action is explicitly "PIVOT_APP".
2. If the user is answering a clarification follow-up question, you MUST continue returning "GATHER_REQUIREMENTS".
3. Do not jump to "SHOW_MODEL_CARDS" or "GENERATE_PREVIEW" prematurely unless the core application domain is well-scoped with at least 3 distinct metadata attributes captured in deepAnswers or state parameters (such as primary subject, setting, style, tone, length, theme, audience, etc.).

Return ONLY the function call orchestrate_pipeline. Never respond with plain text."""


def enforce_prd_rules(decision: dict, session: dict) -> dict:
    decision = dict(decision)
    action = decision.get("recommended_action") or "GATHER_REQUIREMENTS"
    
    # Rule 1: CRITICAL STATE PRESERVATION
    locked_types = {"audio", "video", "image", "vision"}
    current_app_type = session.get("appType") or session.get("app_type")
    
    if current_app_type in locked_types and action != "PIVOT_APP":
        decision["app_type"] = current_app_type
        
    # Rule 2: Clarification follow-up question answering check
    is_answering_clarification = (
        session.get("awaitingTriageAnswer") 
        or session.get("awaitingDeepAnswer") 
        or session.get("currentDeepField")
        or (session.get("step", 0) == 0 and session.get("lastQuestion"))
    )
    if is_answering_clarification:
        action = "GATHER_REQUIREMENTS"
        decision["recommended_action"] = action

    # Rule 3: Premature progression check
    # We need at least 3 distinct metadata attributes captured before SHOW_MODEL_CARDS/GENERATE_PREVIEW.
    captured_attributes = set()
    deep_answers = session.get("deepAnswers") or {}
    for k, v in deep_answers.items():
        if not k.startswith("_") and v and str(v).strip():
            captured_attributes.add(k.lower().strip())
            
    extraction = session.get("extraction") or {}
    for k in ["PRIMARY_SUBJECT", "ENVIRONMENT_SETTING", "ACTION_DYNAMIC", "AESTHETIC_STYLE", "budget", "targetUsers"]:
        if extraction.get(k) and str(extraction.get(k)).strip():
            captured_attributes.add(k.lower().strip())
            
    for k in ["model_id"]:
        if session.get(k) and str(session.get(k)).strip():
            captured_attributes.add(k.lower().strip())
    for k in ["modelId"]:
        if session.get(k) and str(session.get(k)).strip():
            captured_attributes.add(k.lower().strip())

    if len(captured_attributes) < 3 and action in ("SHOW_MODEL_CARDS", "GENERATE_PREVIEW"):
        action = "GATHER_REQUIREMENTS"
        decision["recommended_action"] = action
        
    # Normalize confidence to uppercase
    conf = str(decision.get("confidence") or "MEDIUM").upper()
    if conf not in ("HIGH", "MEDIUM", "LOW"):
        conf = "MEDIUM"
    decision["confidence"] = conf
    
    if not decision.get("reasoning"):
        decision["reasoning"] = f"Aligned state with action {action} and app type {decision.get('app_type')} based on PRD rules."
        
    return decision



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
    s = session or {}
    app_type = s.get("appType") or "text"

    if (
        t.startswith("multi_select_form::")
        or t.startswith("SEO_PUBLISH::")
        or t.startswith("SEO_DRAFT::")
        or t.startswith("SEO_EDIT::")
        or t.startswith("confirm seo::")
    ):
        return {
            "recommended_action": "GENERATE_PREVIEW",
            "app_type": app_type,
            "confidence": "HIGH",
            "reasoning": "UI transition fast-path triggered.",
            "_source": "fast_path",
        }

    if t.lower().startswith("edit prompt::"):
        return {
            "recommended_action": "EDIT_APP",
            "app_type": app_type,
            "confidence": "HIGH",
            "reasoning": "User prompt edit instruction fast-path.",
            "_source": "fast_path",
        }

    if lower in ("approve app", "approve"):
        return {
            "recommended_action": "GENERATE_PREVIEW",
            "app_type": app_type,
            "confidence": "HIGH",
            "reasoning": "User approval of app configuration.",
            "_source": "fast_path",
        }

    if lower in ("edit app", "edit"):
        return {
            "recommended_action": "EDIT_APP",
            "app_type": app_type,
            "confidence": "HIGH",
            "reasoning": "User requested edit of application config.",
            "_source": "fast_path",
        }

    if lower in ("publish to marketplace", "save draft"):
        return {
            "recommended_action": "GENERATE_PREVIEW",
            "app_type": app_type,
            "confidence": "HIGH",
            "reasoning": "Publish/Save UI action triggered.",
            "_source": "fast_path",
        }

    if re.match(r"^select\s+\S", t, re.I) and lower not in (
        "select lean", "select recommended", "select full"
    ):
        return {
            "recommended_action": "GENERATE_PREVIEW",
            "app_type": app_type,
            "confidence": "HIGH",
            "reasoning": "Model selection fast-path.",
            "_source": "fast_path",
        }

    budget_map = {
        "free models only (0 coins)": "free",
        "low (< 5 coins)": "low",
        "medium (5 - 20 coins)": "medium",
        "premium (> 20 coins)": "premium",
    }
    if lower in budget_map:
        return {
            "recommended_action": "SHOW_MODEL_CARDS" if s.get("formConfirmed") else "GATHER_REQUIREMENTS",
            "app_type": app_type,
            "confidence": "HIGH",
            "reasoning": "Budget selection chip selection.",
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
            "confidence": "HIGH",
            "reasoning": "App type chip selected by user.",
            "_source": "fast_path",
        }

    return None


def build_fallback_decision(message: str, session: dict | None) -> dict:
    msg = str(message or "").strip().lower()
    snapshot = build_session_snapshot(session)
    s = session or {}

    recommended_action = "GATHER_REQUIREMENTS"
    app_type = snapshot.get("currentAppType") or "text"
    is_major_pivot = False

    if re.match(r"^(hi|hello|hey|hy|hola|greetings)[\s!.]*$", msg, re.I):
        recommended_action = "HANDLE_OFF_TOPIC"
    elif re.match(r"^(help)[\s!.]*$", msg, re.I):
        recommended_action = "HANDLE_OFF_TOPIC"
    elif re.search(r"\b(jailbreak|nsfw|nude|hack|exploit|bomb|weapon|illegal)\b", msg, re.I):
        recommended_action = "HANDLE_OFF_TOPIC"
    elif re.search(r"\b(free|low|medium|premium)\b", msg, re.I) and re.search(
        r"\b(coin|budget|model)\b", msg, re.I
    ):
        recommended_action = "SHOW_MODEL_CARDS" if snapshot["formConfirmed"] else "GATHER_REQUIREMENTS"
    elif re.search(r"\b(approve|approved|looks good|proceed|confirm|yes proceed)\b", msg, re.I):
        recommended_action = "GENERATE_PREVIEW"
    elif re.search(r"\b(publish|save draft|go live)\b", msg, re.I):
        recommended_action = "GENERATE_PREVIEW"
    elif re.search(r"\b(change|switch|different)\b.{0,20}\b(model|ai|engine)\b", msg, re.I):
        recommended_action = "SHOW_MODEL_CARDS"
    elif (
        re.search(r"\b(i want|build|create|make)\b.{2,50}\b(app|tool|generator)\b", msg, re.I)
        and snapshot["hasPurpose"]
    ):
        recommended_action = "PIVOT_APP"
        is_major_pivot = True
    elif re.search(r"\b(change|edit|tweak|update|modify|make it|add|remove)\b", msg, re.I) and snapshot["hasPurpose"]:
        recommended_action = "EDIT_APP"
    elif snapshot["formConfirmed"] and snapshot["budgetSet"] and not snapshot["modelSelected"]:
        recommended_action = "SHOW_MODEL_CARDS"
    elif snapshot["triageComplete"] and not snapshot["formConfirmed"]:
        recommended_action = "PROCESS_FORM"

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

    logger.info(f"[Orchestrator:Fallback] Action: {recommended_action} | Confidence: LOW")

    decision = {
        "recommended_action": recommended_action,
        "app_type": app_type,
        "confidence": "LOW",
        "reasoning": f"Regex fallback matched action {recommended_action} and app type {app_type}.",
        "extracted_variables": {},
        "is_major_pivot": is_major_pivot,
        "_source": "fallback_regex",
    }
    return enforce_prd_rules(decision, s)


async def get_agentic_decision(llm: LLMService, message: str, session: dict) -> dict:
    text = str(message or "").strip()

    fast = try_fast_path(text, session)
    if fast:
        logger.info(f"[Orchestrator] Fast-path: {fast['recommended_action']}")
        return enforce_prd_rules(fast, session)

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
            app_type = parsed.get("app_type") or "text"
            confidence = parsed.get("confidence") or "MEDIUM"
            reasoning = parsed.get("reasoning") or ""

            decision = {
                "recommended_action": action,
                "app_type": app_type,
                "confidence": confidence,
                "reasoning": reasoning,
                "extracted_variables": {},
                "is_major_pivot": action == "PIVOT_APP",
                "_source": "llm",
            }
            decision = enforce_prd_rules(decision, session)
            logger.info(
                f"[Orchestrator] Action: {decision['recommended_action']} | "
                f"AppType: {decision['app_type']} | Confidence: {decision['confidence']}"
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
    "select_budget": "GATHER_REQUIREMENTS",
    "select_model": "SHOW_MODEL_CARDS",
    "affirmation": "GATHER_REQUIREMENTS",
    "greeting": "HANDLE_OFF_TOPIC",
    "off_topic": "HANDLE_OFF_TOPIC",
    "policy_violation": "HANDLE_OFF_TOPIC",
    "gibberish": "HANDLE_OFF_TOPIC",
    "answer_question": "GATHER_REQUIREMENTS",
    "ui_action": "GENERATE_PREVIEW",
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
