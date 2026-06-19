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
                    "enum": ["text", "image", "audio", "video", "vision", "ambiguous"],
                },
                "confidence": {
                    "type": "string",
                    "enum": ["HIGH", "MEDIUM", "LOW"],
                },
                "reasoning": {
                    "type": "string",
                    "description": "Clear explanation of state alignment and why this action was selected",
                },
                "edit_scope": {
                    "type": "string",
                    "enum": ["PATCH_VALUE", "PATCH_PROMPT", "DOMAIN_SHIFT"],
                    "description": (
                        "When action is EDIT_APP, classify the edit scope: "
                        "PATCH_VALUE = user is changing a single variable value (name, location, etc), "
                        "PATCH_PROMPT = user wants tone/style/length change but same domain, "
                        "DOMAIN_SHIFT = user wants a completely different kind of app"
                    )
                },
            },
            "required": ["recommended_action", "app_type", "confidence", "reasoning", "edit_scope"],
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
EDIT SCOPE CLASSIFICATION (when action = EDIT_APP)
═══════════════════════════════════════
You MUST classify edit_scope for every EDIT_APP action:

PATCH_VALUE  → user changes a specific input value only
               (a name, a place, a number, a character, any atomic field)
               Schema stays IDENTICAL. Only the value is patched.
               Example: "i want the name to be jack sparrow"
               Example: "change location to Tokyo"

PATCH_PROMPT → user changes creative direction but stays in same domain
               (tone, style, pacing, mood, length)
               Schema stays IDENTICAL. Prompt is regenerated.
               Example: "make it more intense"
               Example: "shorter episodes"

DOMAIN_SHIFT → user wants a completely different app type or purpose
               Schema is RESET. Full triage re-runs.
               Example: "change to a music app instead"
               Example: "actually I want image generation"

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
    
    if current_app_type in locked_types and action != "PIVOT_APP" and decision.get("_source") != "fast_path":
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
    captured_attributes = set()
    deep_answers = session.get("deepAnswers") or {}
    for k, v in deep_answers.items():
        if not k.startswith("_") and v and str(v).strip():
            captured_attributes.add(k.lower().strip())
            
    extraction = session.get("extraction") or {}
    for k in ["PRIMARY_SUBJECT", "ENVIRONMENT_SETTING", "ACTION_DYNAMIC", "AESTHETIC_STYLE", "budget", "targetUsers"]:
        if extraction.get(k) and str(extraction.get(k)).strip():
            captured_attributes.add(k.lower().strip())
            
    for k in ["model_id", "modelId"]:
        if session.get(k) and str(session.get(k)).strip():
            captured_attributes.add(k.lower().strip())

    # ─── 🛡️ PRD REGISTRY FIX: PREVENT OVER-ENFORCEMENT FOR TEXT APPS ───
    current_type = (session.get("appType") or "text").lower()
    min_required = 3 if current_type in ("image", "video", "vision") else 1
    
    if len(captured_attributes) < min_required and action in ("SHOW_MODEL_CARDS", "GENERATE_PREVIEW"):
        action = "GATHER_REQUIREMENTS"
        decision["recommended_action"] = action
        
    # ─── 🛡️ FIXED PROGRESSION RULE: ENFORCE COMPLETE DYNAMIC TRIAGE ───
    dynamic_slots = session.get("dynamicSlots") or []
    deep_answers = session.get("deepAnswers") or {}
    
    # If the triage node has defined dynamic requirement slots for a domain,
    # we MUST force the user to answer them before allowing progression to model selection or previews.
    if dynamic_slots and len(deep_answers) < len(dynamic_slots):
        if action in ("SHOW_MODEL_CARDS", "GENERATE_PREVIEW"):
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

    # Explicit modality correction and negation handling (e.g. "it is a text app", "i want a text app", "not an image app")
    modality_pattern = r"\b(?:it is|i want|use|change to|make it|it\'s|want a|need a|not an?|instead of)\s+a?n?\s*(text|image|audio|video|vision)\s*(?:app|tool|generator|model|models)?\b"
    modality_match = re.search(modality_pattern, lower)
    if modality_match:
        matched_type = modality_match.group(1)
        
        # Handle negation / rejection phrases (e.g., "not an image", "instead of vision")
        negation_match = re.search(r"\b(?:not an?|instead of|stop making)\s+a?n?\s*" + re.escape(matched_type), lower)
        if negation_match:
            # If the user rejected this matched_type, try to find another modality in the string they are shifting toward
            other_types = [t_key for t_key in ("text", "image", "audio", "video", "vision") if t_key != matched_type]
            found_other = None
            for ot in other_types:
                if ot in lower:
                    found_other = ot
                    break
            if found_other:
                matched_type = found_other
            else:
                matched_type = "text" if matched_type == "image" else "image"
                
        action = "SHOW_MODEL_CARDS" if s.get("formConfirmed") else "GATHER_REQUIREMENTS"
        return {
            "recommended_action": action,
            "app_type": matched_type,
            "confidence": "HIGH",
            "reasoning": f"Explicit user correction of app type to {matched_type}.",
            "_source": "fast_path",
        }

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

    # ─── 🛡️ FAST-PATH MATCH PAYLOAD CONTRACTS ───
    if lower in ("approve app", "approve") or re.match(r"^selected\s+model", lower):
        return {
            "recommended_action": "GENERATE_PREVIEW",
            "app_type": app_type,
            "confidence": "HIGH",
            "reasoning": "User approval or model confirmation of app configuration.",
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
            "recommended_action": "SHOW_MODEL_CARDS" if s.get("formConfirmed") else "GATHER_REQUIREMENTS",
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

    if re.match(r"^(h+e+l+[lo]+|h+[iy]+|hey+|hola|greetings?|yo|sup|namaste)[\s!?.]*$", msg, re.I):
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

    # Avoid substring leaks by using strict word boundaries if length is tiny
    type_signals = {
        "image": re.compile(r"\b(image|images|photo|photos|picture|logo|poster|posters|card|avatar|portrait|banner|meme|flyer|art)\b", re.I),
        "audio": re.compile(r"\b(audio|voice|podcast|tts|speech|narration|music|sound)\b", re.I),
        "video": re.compile(r"\b(video|animation|animate|reel|cinematic|clip)\b", re.I),
        "vision": re.compile(r"\b(detect|analyze image|scan|ocr|read from image)\b", re.I),
        "text": re.compile(r"\b(text|blog|legal|recipe|email|story|script|plan|write|article|debate)\b", re.I),
    }
    for t, pattern in type_signals.items():
        if pattern.search(msg):
            # Anti-contamination: Double ensure it's not matching 'artificial'
            if t == "image" and "artificial" in msg and not re.search(r"\bart\b", msg):
                continue
            app_type = t
            break

    edit_scope = "PATCH_PROMPT"
    if recommended_action == "EDIT_APP":
        if any(w in msg for w in ["music app", "song", "image generation", "instead", "actually"]):
            edit_scope = "DOMAIN_SHIFT"
        elif any(w in msg for w in ["name be", "location to", "change location", "change name", "subject to", "character to"]):
            edit_scope = "PATCH_VALUE"

    logger.info(f"[Orchestrator:Fallback] Action: {recommended_action} | Confidence: LOW")

    decision = {
        "recommended_action": recommended_action,
        "app_type": app_type,
        "confidence": "LOW",
        "reasoning": f"Regex fallback matched action {recommended_action} and app type {app_type}.",
        "extracted_variables": {},
        "is_major_pivot": is_major_pivot,
        "edit_scope": edit_scope,
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
            edit_scope = parsed.get("edit_scope") or "PATCH_PROMPT"

            decision = {
                "recommended_action": action,
                "app_type": app_type,
                "confidence": confidence,
                "reasoning": reasoning,
                "edit_scope": edit_scope,
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


def infer_category_from_purpose(purpose: str) -> str:
    if not purpose:
        return "unknown"
    lower = purpose.lower()
    if any(w in lower for w in ("quest", "mystery", "murder", "map", "fantasy", "dragon", "story", "game", "fiction", "character")):
        return "creative"
    if any(w in lower for w in ("study", "plan", "workout", "meal", "diet", "tutor", "code", "learn", "course", "fit", "health", "exercise")):
        return "functional"
    if any(w in lower for w in ("image", "photo", "picture", "poster", "logo", "banner", "art")):
        return "visual"
    if any(w in lower for w in ("audio", "voice", "podcast", "speech", "tts", "music", "song")):
        return "audio"
    if any(w in lower for w in ("video", "animation", "animate", "clip")):
        return "video"
    if any(w in lower for w in ("law", "legal", "ipc", "court", "judge")):
        return "legal"
    return "text"


def detect_vertical_mismatch(user_input: str, previous_category: str) -> bool:
    if not previous_category or previous_category == "unknown":
        return False
        
    lower = user_input.lower()
    new_app_intent = any(w in lower for w in ("i want", "build", "create", "make", "change to", "switch to", "instead of", "how about", "new app"))
    if not new_app_intent:
        return False
        
    new_cat = infer_category_from_purpose(user_input)
    if new_cat != "unknown" and new_cat != previous_category:
        return True
    return False


def classify_intent_with_pivot_check(user_input: str, current_app_context: dict) -> dict:
    prev_purpose = current_app_context.get("appPurpose") or ""
    prev_category = infer_category_from_purpose(prev_purpose)
    new_category = infer_category_from_purpose(user_input)
    
    is_pivot = detect_vertical_mismatch(user_input, prev_category)
    
    return {
        "intent": "create_new_app" if is_pivot else "continue",
        "drastic_pivot": is_pivot,
        "previous_category": prev_category,
        "new_category": new_category
    }