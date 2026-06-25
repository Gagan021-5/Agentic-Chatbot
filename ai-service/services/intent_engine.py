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
from services.extraction import _slot_is_captured
from services.artifact_utils import requires_input_artifact, is_creation_workflow

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
                        "PUBLISH_APP",
                        "SAVE_DRAFT",
                        "REVIEW_SEO",
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
                "ingestion_vector": {
                    "type": "string",
                    "enum": ["url", "screenshots", "source_code", "figma_files", "plain_text", "missing"],
                    "description": "How the application receives its main input data (URL, screenshots, figma, source code, plain_text, or missing)."
                },
                "ingestion_vector_status": {
                    "type": "string",
                    "enum": ["explicit", "inferred", "missing"],
                    "description": "Whether the ingestion vector was explicitly declared by the user, inferred from context, or is missing/unspecified."
                },
                "app_type_status": {
                    "type": "string",
                    "enum": ["explicit", "inferred", "missing"],
                    "description": "Whether the app modality was explicitly stated by the user, inferred, or missing."
                },
                "budget_tier": {
                    "type": "string",
                    "enum": ["free", "low", "medium", "premium", "ultra", "missing"],
                    "description": "Selected or inferred budget tier limit preference."
                },
                "budget_status": {
                    "type": "string",
                    "enum": ["explicit", "inferred", "missing"],
                    "description": "Whether the budget tier was explicitly declared by the user, inferred, or is missing."
                },
                "ui_event": {
                    "type": "string",
                    "enum": ["multi_select_form", "seo_publish", "seo_draft", "confirm_seo", "model_select", "none"],
                    "description": "Classify the UI action event type if this message represents a UI action prefix trigger."
                },
                "ui_payload": {
                    "type": "string",
                    "description": "The JSON payload or metadata associated with the UI event."
                }
            },
            "required": [
                "recommended_action", "app_type", "confidence", "reasoning", 
                "edit_scope", "ingestion_vector", "ingestion_vector_status",
                "app_type_status", "budget_tier", "budget_status", "ui_event", "ui_payload"
            ],
        },
    },
}

ORCHESTRATOR_SYSTEM_PROMPT = """You are the Master Intent Classifier and Orchestrator Node for RentPrompts — a platform where users CREATE and PUBLISH AI-powered apps.
Your job is to read the latest user message, evaluate the multi-turn session state memory, and select the next deterministic action.

═══════════════════════════════════════
ARCHITECTURAL DISCOVERY DIRECTIVE
═══════════════════════════════════════
You are an architectural discovery orchestrator. Descriptive adjectives like 'visual' or 'SEO' define the target application goals, NOT the ingestion boundary. You must treat the input delivery transmission vector (how data gets into the application) as an unresolved missing parameter until the user explicitly dictates it. Do not recommend PROCESS_FORM or SHOW_MODEL_CARDS while ingestion vectors are ambiguous.

═══════════════════════════════════════
CONFIRMATION METRICS DIRECTIVE
═══════════════════════════════════════
You must classify and track the confirmation status of critical application fields:
1. APP TYPE STATUS (app_type_status):
   - 'explicit': The user has explicitly stated their preferred modality (e.g. "I want a vision app", "it is a text app", "make it text app instead").
   - 'inferred': Modality has been inferred from the task context (e.g. "reviews menus" implies vision).
     - "evaluates pitch decks" → app_type = "vision", ingestion_vector = "screenshots" (user uploads the deck)
     - "scores resumes" → app_type = "vision", ingestion_vector = "screenshots"
     - "reviews LinkedIn profiles" → app_type = "vision", ingestion_vector = "screenshots"
     - "reads invoices / receipts" → app_type = "vision", ingestion_vector = "screenshots"
     RULE: If the app ANALYZES or EVALUATES an UPLOADED file/document/profile → vision + ingestion_vector = "screenshots"
   - 'missing': No modality has been established yet.

2. INGESTION VECTOR STATUS (ingestion_vector_status):
   - 'explicit': The user has explicitly stated the data delivery method (e.g. "uploaded photos", "PDF menus", "Github URL").
   - 'inferred': The delivery method is guessed but not verified.
   - 'missing': No ingestion delivery method has been specified.

3. BUDGET STATUS (budget_status):
   - 'explicit': The user has selected a specific budget preference/tier (e.g. "Free models only", "Low budget", or chosen from options).
   - 'inferred': Budget is guessed but not verified.
   - 'missing': Budget preference has not been specified.

═══════════════════════════════════════
CRITICAL ANTI-INFERENCE RULE
═══════════════════════════════════════
- Do NOT mark app_type_status as 'explicit' based on descriptive adjectives like 'visual', 'design', 'analytics', 'SEO', 'creative', or 'professional'. These describe the application's goal domain, NOT the AI modality.
- 'explicit' requires the user to DIRECTLY and UNAMBIGUOUSLY name the modality: 'it is a vision app', 'use text', 'I want image generation', 'make it an audio app'.
- Descriptive goals like 'analyze visual design appeal' should set app_type to 'vision' with app_type_status 'inferred', NOT 'explicit'.
- Similarly, do NOT mark ingestion_vector_status as 'explicit' unless the user has directly stated the delivery method (e.g. 'upload screenshots', 'paste the URL', 'from my GitHub repo').
- If the user describes WHAT they want analyzed but not HOW data arrives, ingestion_vector_status must remain 'missing'.
- EXCEPTION: For pure text generation apps (blog writers, rewriters, content generators, workout planners, meal planners, email writers) where the app GENERATES output rather than ANALYZES uploaded content:
  Set ingestion_vector = "plain_text" and ingestion_vector_status = "explicit" automatically. These apps do not require file upload discovery.
- If user says "text", "it is of text", "paste", "plain text", "type it", "I'll type" in response to an input format question:
  ingestion_vector = "plain_text", ingestion_vector_status = "explicit".
  NEVER ask about input format again after this.

═══════════════════════════════════════
UI EVENT PAYLOAD PARSING
═══════════════════════════════════════
You must parse structured UI events and payloads when they are provided as user messages:
- If the message starts with 'multi_select_form::', select recommended_action = 'PROCESS_FORM', ui_event = 'multi_select_form', and set ui_payload to the trailing JSON string.
- If the message starts with 'SEO_PUBLISH::', select recommended_action = 'PUBLISH_APP', ui_event = 'seo_publish', and set ui_payload to the trailing JSON string.
- If the message starts with 'SEO_DRAFT::', select recommended_action = 'SAVE_DRAFT', ui_event = 'seo_draft', and set ui_payload to the trailing JSON string.
- If the message starts with 'confirm seo::' or matches 'approve app' / 'approve', select recommended_action = 'REVIEW_SEO', ui_event = 'confirm_seo', and set ui_payload to the user message.
- If the message starts with 'select ' or is 'selected model: <model_id>', select recommended_action = 'GENERATE_PREVIEW', ui_event = 'model_select', and set ui_payload to the user message.
- If the message starts with 'edit prompt::' or matches 'edit app', select recommended_action = 'EDIT_APP', ui_event = 'none', and set ui_payload to the user message.

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
3. Do not jump to "SHOW_MODEL_CARDS" or "GENERATE_PREVIEW" prematurely unless the core application domain is well-scoped with at least 3 distinct metadata attributes (for image, video, vision apps) or at least 1 distinct metadata attribute (for text, audio apps) captured in deepAnswers or state parameters (such as primary subject, setting, style, tone, length, theme, audience, etc.).


═══════════════════════════════════════
READINESS HEURISTIC — WHEN TO STOP GATHERING
═══════════════════════════════════════
Transition to SHOW_MODEL_CARDS when ALL of the following are true:
1. app_type_status = "explicit" (user confirmed or stated the modality)
2. ingestion_vector_status = "explicit" OR app type is text/audio with no file upload
3. budget_status = "explicit" (user chose from budget chips)
4. session.triageRounds >= 2 AND at least one domain-specific slot is captured
   in deepAnswers (anything other than budget and ingestion_vector)

Do NOT keep gathering if you are just waiting for "more detail" on a purpose 
that is already actionable. "I want a vintage logo generator" is enough. 
"I want an app that reviews restaurant menus from photos" is enough.
If the app purpose is a complete sentence with a subject and a verb, it is enough.

Return ONLY the function call orchestrate_pipeline. Never respond with plain text."""


def enforce_prd_rules(decision: dict, session: dict) -> dict:
    decision = dict(decision)
    action = decision.get("recommended_action") or "GATHER_REQUIREMENTS"
    
    # Rule 1: CRITICAL STATE PRESERVATION
    locked_types = {"audio", "video", "image", "vision"}
    current_app_type = session.get("appType") or session.get("app_type")
    # Allow a targeted override: if the session's stated purpose clearly indicates
    # an image creation workflow (logo, thumbnail, poster, etc.) and the current
    # locked type is 'vision' (analysis), permit switching to 'image' generation.
    session_purpose = (session.get("extraction") or {}).get("appPurpose") or ""
    is_creation = is_creation_workflow(session_purpose)

    if current_app_type in locked_types and action != "PIVOT_APP" and decision.get("_source") != "fast_path":
        if not (current_app_type == "vision" and is_creation):
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
        if current_app_type:
            decision["app_type"] = current_app_type
            decision["app_type_status"] = "explicit"

    # Rule 4: Preview stage state preservation
    # If we are at the app preview stage (step == 2) and a model is selected,
    # any attempt to go back to GATHER_REQUIREMENTS should be redirected to EDIT_APP
    # unless it's a major pivot.
    if session.get("step") == 2 and session.get("modelId") and action == "GATHER_REQUIREMENTS":
        action = "EDIT_APP"
        decision["recommended_action"] = action
        decision["edit_scope"] = "PATCH_PROMPT"

    # ─── 🛡️ ARCHITECTURAL DISCOVERY OVERRIDE ───
    v_meta = session.get("verificationMetadata") or {}

    # Check session-persisted ingestion vector FIRST before reading fresh decision
    # This prevents re-asking after user already answered in a previous turn
    persisted_ing_vec = session.get("ingestionVector")
    persisted_ing_status = v_meta.get("ingestion_vector") or "missing"

    # If session already has an explicit ingestion vector, trust it — never re-ask
    if persisted_ing_vec and persisted_ing_status == "explicit":
        ing_status = "explicit"
    else:
        ing_status = (
            decision.get("ingestion_vector_status")
            or persisted_ing_status
            or "missing"
        )

    # For pure generation apps, ingestion vector should not block progression.
    app_type_for_check = (
        decision.get("app_type")
        or session.get("appType")
        or "text"
    )
    extraction = session.get("extraction") or {}
    wants_image_input = bool(extraction.get("wantsImageInput"))
    PURE_GENERATION_TYPES = {"text", "audio"}

    # If this is an image creation workflow (logo/thumbnail/etc.), mark ingestion not required
    # so we don't prompt for uploads during progression.
    session_purpose = (session.get("extraction") or {}).get("appPurpose") or ""
    if is_creation_workflow(session_purpose) and str(app_type_for_check) == "image":
        decision["ingestion_vector"] = "not_required"
        decision["ingestion_vector_status"] = "explicit"
        v_meta["ingestion_vector"] = "not_required"

    if app_type_for_check in PURE_GENERATION_TYPES and not wants_image_input:
        decision["ingestion_vector"] = "plain_text"
        decision["ingestion_vector_status"] = "explicit"
        # Update v_meta too
        v_meta["ingestion_vector"] = "not_required"
        ing_status = "not_required"

    budget_status = decision.get("budget_status") or v_meta.get("budget") or "missing"

    # Only block progression for missing ingestion vectors when the workflow actually
    # requires an uploaded artifact. Generation workflows should not be blocked.
    session_purpose = (session.get("extraction") or {}).get("appPurpose") or ""
    if ((ing_status in ("missing", "inferred") and requires_input_artifact(app_type_for_check, session_purpose))
            or budget_status in ("missing", "inferred")):
        if action in ("PROCESS_FORM", "SHOW_MODEL_CARDS", "GENERATE_PREVIEW"):
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
    
    is_confirmed = bool(session.get("formConfirmed") or session.get("form_confirmed"))
    if not is_confirmed:
        dynamic_slots = session.get("dynamicSlots") or []
        deep_answers = session.get("deepAnswers") or {}
        
        # If the triage node has defined dynamic requirement slots for a domain,
        # we MUST force the user to answer them before allowing progression to model selection or previews.
        if dynamic_slots:
            # Build normalized key-value mapping of answers
            captured_answers = {str(k).lower().strip(): str(v).strip() for k, v in deep_answers.items() if v and str(v).strip()}
            missing_any = False
            for slot_obj in dynamic_slots:
                slot_key = str(slot_obj.get("key") or "").lower().strip()
                if not _slot_is_captured(slot_key, captured_answers):
                    missing_any = True
                    break
            if missing_any:
                if action in ("SHOW_MODEL_CARDS", "GENERATE_PREVIEW"):
                    action = "GATHER_REQUIREMENTS"
                    decision["recommended_action"] = action
        else:
            if len(captured_attributes) < min_required and action in ("SHOW_MODEL_CARDS", "GENERATE_PREVIEW"):
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


def parse_ui_event_payload_to_schema(text: str, session: dict | None) -> dict | None:
    t = str(text or "").strip()
    lower = t.lower()
    s = session or {}
    app_type = s.get("appType") or "text"

    if t.lower().startswith("multi_select_form::"):
        return {
            "recommended_action": "PROCESS_FORM",
            "app_type": app_type,
            "confidence": "HIGH",
            "reasoning": "Form submitted.",
            "edit_scope": "PATCH_PROMPT",
            "ingestion_vector": s.get("ingestionVector") or "missing",
            "ingestion_vector_status": s.get("verificationMetadata", {}).get("ingestion_vector") or "missing",
            "app_type_status": s.get("verificationMetadata", {}).get("app_type") or "inferred",
            "budget_tier": s.get("deepAnswers", {}).get("budgetPreference") or "missing",
            "budget_status": s.get("verificationMetadata", {}).get("budget") or "missing",
            "ui_event": "multi_select_form",
            "ui_payload": t[len("multi_select_form::"):],
        }

    if t.startswith("SEO_PUBLISH::"):
        return {
            "recommended_action": "PUBLISH_APP",
            "app_type": app_type,
            "confidence": "HIGH",
            "reasoning": "Publish app event.",
            "edit_scope": "PATCH_PROMPT",
            "ingestion_vector": s.get("ingestionVector") or "missing",
            "ingestion_vector_status": s.get("verificationMetadata", {}).get("ingestion_vector") or "missing",
            "app_type_status": s.get("verificationMetadata", {}).get("app_type") or "inferred",
            "budget_tier": s.get("deepAnswers", {}).get("budgetPreference") or "missing",
            "budget_status": s.get("verificationMetadata", {}).get("budget") or "missing",
            "ui_event": "seo_publish",
            "ui_payload": t[len("SEO_PUBLISH::"):],
        }

    if t.startswith("SEO_DRAFT::"):
        return {
            "recommended_action": "SAVE_DRAFT",
            "app_type": app_type,
            "confidence": "HIGH",
            "reasoning": "Save draft event.",
            "edit_scope": "PATCH_PROMPT",
            "ingestion_vector": s.get("ingestionVector") or "missing",
            "ingestion_vector_status": s.get("verificationMetadata", {}).get("ingestion_vector") or "missing",
            "app_type_status": s.get("verificationMetadata", {}).get("app_type") or "inferred",
            "budget_tier": s.get("deepAnswers", {}).get("budgetPreference") or "missing",
            "budget_status": s.get("verificationMetadata", {}).get("budget") or "missing",
            "ui_event": "seo_draft",
            "ui_payload": t[len("SEO_DRAFT::"):],
        }

    if t.startswith("confirm seo::") or lower in ("approve app", "approve"):
        return {
            "recommended_action": "REVIEW_SEO",
            "app_type": app_type,
            "confidence": "HIGH",
            "reasoning": "Review SEO or Approve App event.",
            "edit_scope": "PATCH_PROMPT",
            "ingestion_vector": s.get("ingestionVector") or "missing",
            "ingestion_vector_status": s.get("verificationMetadata", {}).get("ingestion_vector") or "missing",
            "app_type_status": s.get("verificationMetadata", {}).get("app_type") or "inferred",
            "budget_tier": s.get("deepAnswers", {}).get("budgetPreference") or "missing",
            "budget_status": s.get("verificationMetadata", {}).get("budget") or "missing",
            "ui_event": "confirm_seo",
            "ui_payload": t,
        }

    if t.lower().startswith("select ") or t.lower().startswith("selected model") or re.match(r"^selected\s+model", lower):
        return {
            "recommended_action": "GENERATE_PREVIEW",
            "app_type": app_type,
            "confidence": "HIGH",
            "reasoning": "Model selection event.",
            "edit_scope": "PATCH_PROMPT",
            "ingestion_vector": s.get("ingestionVector") or "missing",
            "ingestion_vector_status": s.get("verificationMetadata", {}).get("ingestion_vector") or "missing",
            "app_type_status": s.get("verificationMetadata", {}).get("app_type") or "inferred",
            "budget_tier": s.get("deepAnswers", {}).get("budgetPreference") or "missing",
            "budget_status": s.get("verificationMetadata", {}).get("budget") or "missing",
            "ui_event": "model_select",
            "ui_payload": t,
        }

    # Plain 'Edit App' button or message should directly map to EDIT_APP
    if lower in ("edit app", "edit"):
        v_meta = s.get("verificationMetadata") or {}
        return {
            "recommended_action": "EDIT_APP",
            "app_type": app_type,
            "confidence": "HIGH",
            "reasoning": "User requested to edit the app (UI button).",
            "edit_scope": "PATCH_PROMPT",
            "ingestion_vector": s.get("ingestionVector") or "missing",
            "ingestion_vector_status": v_meta.get("ingestion_vector") or "missing",
            "app_type_status": v_meta.get("app_type") or "inferred",
            "ui_event": "none",
            "ui_payload": t,
        }

    return None


def try_fast_path(text: str, session: dict | None) -> dict | None:
    t = str(text or "").strip()
    lower = t.lower()
    s = session or {}
    app_type = s.get("appType") or "text"

    if t.lower().startswith("edit prompt::"):
        v_meta = s.get("verificationMetadata") or {}
        return {
            "recommended_action": "EDIT_APP",
            "app_type": app_type,
            "confidence": "HIGH",
            "reasoning": "User prompt edit instruction fast-path.",
            "_source": "fast_path",
            "app_type_status": v_meta.get("app_type") or "explicit",
            "ingestion_vector": s.get("ingestionVector") or "missing",
            "ingestion_vector_status": v_meta.get("ingestion_vector") or "missing",
        }

    budget_map = {
        "free models only (0 coins)": "free",
        "low (< 5 coins)": "low",
        "medium (5 - 20 coins)": "medium",
        "premium (> 20 coins)": "premium",
    }
    if lower in budget_map:
        v_meta = s.get("verificationMetadata") or {}
        return {
            "recommended_action": "SHOW_MODEL_CARDS" if s.get("formConfirmed") else "GATHER_REQUIREMENTS",
            "app_type": app_type,
            "confidence": "HIGH",
            "reasoning": "Budget selection chip selection.",
            "_source": "fast_path",
            "app_type_status": v_meta.get("app_type") or "inferred",
            "ingestion_vector": s.get("ingestionVector") or "missing",
            "ingestion_vector_status": v_meta.get("ingestion_vector") or "missing",
            "budget_tier": budget_map[lower],
            "budget_status": "explicit",
        }

    return None


def build_fallback_decision(message: str, session: dict | None) -> dict:
    s = session or {}
    decision = {
        "recommended_action": "GATHER_REQUIREMENTS",
        "app_type": s.get("appType") or "text",
        "confidence": "LOW",
        "reasoning": "LLM unavailable, defaulting to requirement gathering.",
        "ingestion_vector": s.get("ingestionVector") or "missing",
        "ingestion_vector_status": "missing",
        "app_type_status": "inferred",
        "budget_tier": "missing",
        "budget_status": "missing",
        "edit_scope": "PATCH_PROMPT",
        "_source": "fallback_regex",
    }
    return enforce_prd_rules(decision, s)


async def get_agentic_decision(llm: LLMService, message: str, session: dict) -> dict:
    text = str(message or "").strip()

    ui_parsed = parse_ui_event_payload_to_schema(text, session)
    if ui_parsed:
        logger.info(f"[Orchestrator] UI Event parsed: {ui_parsed['recommended_action']}")
        return enforce_prd_rules(ui_parsed, session)

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
            ingestion_vector = parsed.get("ingestion_vector") or "missing"
            ingestion_vector_status = parsed.get("ingestion_vector_status") or "missing"
            app_type_status = parsed.get("app_type_status") or ("explicit" if confidence == "HIGH" else "inferred")

            decision = {
                "recommended_action": action,
                "app_type": app_type,
                "confidence": confidence,
                "reasoning": reasoning,
                "edit_scope": edit_scope,
                "ingestion_vector": ingestion_vector,
                "ingestion_vector_status": ingestion_vector_status,
                "app_type_status": app_type_status,
                "budget_tier": parsed.get("budget_tier") or "missing",
                "budget_status": parsed.get("budget_status") or "missing",
                "ui_event": parsed.get("ui_event") or "none",
                "ui_payload": parsed.get("ui_payload") or "",
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
        err_msg = ""
        if isinstance(real_err, httpx.HTTPStatusError):
            try:
                err_msg = real_err.response.text
            except Exception:
                pass
        if not err_msg:
            err_msg = str(real_err)

        logger.warning(f"[Orchestrator] Groq tool call failed: {err_msg}. Trying OpenRouter fallback...")

        if llm.has_openrouter:
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
                    {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT + "\n\nYou MUST return a JSON object containing the orchestrate_pipeline arguments: recommended_action, app_type, confidence, reasoning, edit_scope, ingestion_vector, ingestion_vector_status, app_type_status, budget_tier, budget_status, ui_event, ui_payload."},
                    {"role": "system", "content": f"SESSION STATE SNAPSHOT:\n{json.dumps(snapshot, indent=2)}"},
                    *history_slice,
                    {"role": "user", "content": text},
                ]

                raw = await llm.openrouter_completion(
                    messages=messages,
                    model="meta-llama/llama-3.3-70b-instruct",
                    max_tokens=400,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                parsed = json.loads(raw)
                action = parsed.get("recommended_action") or "GATHER_REQUIREMENTS"
                app_type = parsed.get("app_type") or "text"
                confidence = parsed.get("confidence") or "MEDIUM"
                reasoning = parsed.get("reasoning") or ""
                edit_scope = parsed.get("edit_scope") or "PATCH_PROMPT"
                ingestion_vector = parsed.get("ingestion_vector") or "missing"
                ingestion_vector_status = parsed.get("ingestion_vector_status") or "missing"
                app_type_status = parsed.get("app_type_status") or ("explicit" if confidence == "HIGH" else "inferred")

                decision = {
                    "recommended_action": action,
                    "app_type": app_type,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "edit_scope": edit_scope,
                    "ingestion_vector": ingestion_vector,
                    "ingestion_vector_status": ingestion_vector_status,
                    "app_type_status": app_type_status,
                    "budget_tier": parsed.get("budget_tier") or "missing",
                    "budget_status": parsed.get("budget_status") or "missing",
                    "ui_event": parsed.get("ui_event") or "none",
                    "ui_payload": parsed.get("ui_payload") or "",
                    "extracted_variables": {},
                    "is_major_pivot": action == "PIVOT_APP",
                    "_source": "openrouter_fallback",
                }
                decision = enforce_prd_rules(decision, session)
                logger.info(
                    f"[Orchestrator] OpenRouter fallback Action: {decision['recommended_action']} | "
                    f"AppType: {decision['app_type']} | Confidence: {decision['confidence']}"
                )
                return decision
            except Exception as or_err:
                logger.error(f"[Orchestrator] OpenRouter fallback failed: {or_err}")
            
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
        "ingestion_vector": decision.get("ingestion_vector"),
        "ingestion_vector_status": decision.get("ingestion_vector_status"),
        "app_type_status": decision.get("app_type_status"),
        "budget_status": decision.get("budget_status"),
        "ui_event": decision.get("ui_event"),
        "ui_payload": decision.get("ui_payload"),
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
    if any(w in lower for w in (
        "pitch deck", "resume", "cv", "invoice", "receipt",
        "evaluate", "score", "audit", "analyze", "profile review", "linkedin"
    )):
        return "document_analysis"
    return "text"


def detect_vertical_mismatch(user_input: str, previous_category: str) -> bool:
    if not previous_category or previous_category == "unknown":
        return False
        
    lower = user_input.lower()
    
    new_app_patterns = [
        r"\bi\s+want\s+(?:to\s+)?(?:build|create|make|change|switch)\b",
        r"\bi\s+want\s+(?:an?\s+)?(?:app|tool|generator|builder|system|platform|website|service|program|new|different|another)\b",
        r"\b(?:build|create|make|switch\s+to|change\s+to)\s+(?:a|an|another|new|different)\s+(?:app|tool|generator|music|song|video|image|text|audio|vision)\b",
        r"\bnew\s+app\b",
        r"\binstead\s+of\b",
        r"\bhow\s+about\s+a\b"
    ]
    
    new_app_intent = any(re.search(pat, lower) for pat in new_app_patterns)
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