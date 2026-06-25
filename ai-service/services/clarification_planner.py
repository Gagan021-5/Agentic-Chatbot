"""
Behavior-driven clarification planner for RentPrompts.

Replaces generic slot-filling (input_source → primary_goal → output_expectation)
with goal-oriented planning: infer what materially changes how the app works,
then ask one high-value question at a time until enough behavioral context exists.
"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

from services.llm import LLMService

MAX_CLARIFICATION_TURNS = 3


def get_max_clarification_turns(app_type: str | None) -> int:
    return 3


GENERIC_SLOT_KEYS = frozenset({
    "input_source", "input_format", "input_method", "output_format",
    "output_expectation", "output_type", "primary_goal", "budget",
    "budget_preference", "intended_audience", "target_audience",
    "audience", "target_users", "target_user",
})

GENERIC_QUESTION_SIGNALS = (
    "how will you provide",
    "how will users provide",
    "how should users provide",
    "what output format",
    "output format should",
    "what format should the output",
    "what is your budget",
    "budget preference",
    "intended audience",
    "target audience",
    "who is the audience",
    "who will use this app",
    "who will use the app",
    "what is the primary goal of the app",
    "what is the expected output",
    "most important detail",
    "tell me more",
    "how should it work",
    "how the app should work",
    "how this app should work",
    "describe what this app",
)

BEHAVIOR_JUSTIFICATION_MARKERS = (
    "materially",
    "determines execution",
    "determines how",
    "affects execution",
    "affects how",
    "changes how",
    "changes execution",
    "required for this app",
    "required for this specific",
    "genuinely affects",
    "behavioral requirement",
    "operational requirement",
    "analysis requires",
    "evaluation requires",
    "upload method affects",
    "ingestion affects",
)

CLARIFICATION_PLANNER_PROMPT = """You are the User-Goal-Driven Clarification Planner for RentPrompts.

Your job: infer the app's BEHAVIORAL WORKFLOW and identify high-value behavioral decisions (high/medium priority) required to configure the app.

═══════════════════════════════════════
CRITICAL RULE: BAN GENERIC META-QUESTIONS
═══════════════════════════════════════
You MUST NEVER ask open-ended, generic discovery or meta-questions.
BANNED QUESTIONS:
- "What's the single most important detail about how this app should work?"
- "Tell me more about the app."
- "How should the app work?"
- "Can you describe what this app does?"
- "What is the primary goal of the app?"

Every question you formulate must correspond to a concrete, missing behavioral dimension.

═══════════════════════════════════════
STEP 1 — INFER PRODUCT WORKFLOW
═══════════════════════════════════════
Do NOT use general action classes (Generation, Transformation, Evaluation, Planning).
Instead, infer a concrete user-goal workflow:
- workflow_name: e.g. content_to_video, horoscope_generation, logo_generation, resume_analysis
- subject: e.g. blog posts, astrological charts, brand information, resumes
- action: e.g. convert, generate, review, audit
- output: e.g. youtube videos, horoscope predictions, logo designs, feedback reports

═══════════════════════════════════════
STEP 2 — BEHAVIORAL DIMENSIONS
═══════════════════════════════════════
Define 2-3 high-priority and 1-2 medium-priority behavioral dimensions critical to the specific workflow.
For each dimension, specify:
- key: snake_case identifier (e.g. video_style, narration_style, astrology_system, logo_style, target_role, feedback_focus)
- description: what this dimension controls
- value: string value if already stated in the App Purpose or known history, otherwise null
- question: a natural question to ask the user if this dimension is missing

Examples:
- Convert blogs to YouTube videos:
  * High-priority: video_style (visual aesthetic), video_duration (shorts vs long-form)
  * Medium-priority: narration_style (voice type/pacing)
- Resume reviewer:
  * High-priority: target_role (target job title/level), feedback_focus (ATS scan vs grammar/formatting)
  * Medium-priority: ats_level (strictness of ATS match)
- Audiobook generator:
  * High-priority: narration_style (narrative tone/expressiveness), voice_type (AI voice gender/accent)
  * Medium-priority: language

CRITICAL INPUT MODALITY RULE:
- Do NOT ask about: upload method, ingestion vector, input source, file format, PDF/image/url/text, screenshots, website URL, paste text, EPUB, TXT, etc.
- You are strictly FORBIDDEN from including or asking about these dimensions (input_source, ingestion_vector, upload_type, file_format, content_source, input_method) during the clarification phase, unless the app is an artifact analysis/review workflow where the uploaded file/image itself is the actual subject of analysis (e.g. resume review, menu image reviewer, website audit, document analysis, OCR).
- For pure generation/generator, writing, planning, astrology, logo generation, audiobook generation, or converter apps, PDF vs URL vs Text does not change the app behavior. DO NOT ask about input modality/sources. Only ask about behavior, narration style, target audience, visual aesthetic, industry, or rules.

═══════════════════════════════════════
STEP 3 — DECIDE READINESS & SELECT ONE QUESTION
═══════════════════════════════════════
Rate your confidence (0.0 to 1.0) in configuring the app based on known information.
If confidence >= 0.8 OR no high-priority dimensions are missing, set ready = true and selected_key/selected_question = null.
Otherwise, set ready = false, select the highest-priority missing dimension, and set:
- selected_key: key of the selected missing dimension
- selected_question: the natural question for that dimension. Every question must correspond to a concrete missing dimension.

Return strict JSON only matching the schema:
{
  "workflow": {
    "workflow_name": "string",
    "subject": "string",
    "action": "string",
    "output": "string"
  },
  "behavioral_dimensions": {
    "high_priority_dimensions": [
      {"key": "string", "description": "string", "value": "string or null", "question": "string"}
    ],
    "medium_priority_dimensions": [
      {"key": "string", "description": "string", "value": "string or null", "question": "string"}
    ]
  },
  "confidence": float,
  "ready": boolean,
  "selected_key": "string or null",
  "selected_question": "string or null",
  "reason": "string"
}"""


def recover_app_purpose(
    extraction: dict | None,
    user_message: str,
    conversation_history: list | None = None,
) -> tuple[str, bool]:
    """Recover appPurpose from user message or history when extraction is empty."""
    ext = extraction or {}
    purpose = str(ext.get("appPurpose") or "").strip()
    if purpose and len(purpose) >= 8:
        return purpose, False

    msg = str(user_message or "").strip()
    if len(msg) >= 8 and any(c.isalpha() for c in msg):
        if not msg.lower().startswith(("select ", "multi_select_form::", "confirm seo::")):
            return msg, True

    for entry in conversation_history or []:
        if isinstance(entry, dict):
            role = str(entry.get("role") or "").lower()
            content = str(entry.get("content") or "").strip()
        else:
            role = "agent" if getattr(entry, "type", "human") in ("assistant", "agent") else "user"
            content = str(getattr(entry, "content", "")).strip()

        if role not in ("user", "human"):
            continue
        if len(content) >= 8 and any(c.isalpha() for c in content):
            if not content.lower().startswith(("select ", "multi_select_form::")):
                return content, True

    return purpose or msg, bool(not purpose and msg)


def build_known_information(
    extraction: dict | None,
    deep_answers: dict | None,
    app_purpose: str = "",
) -> dict[str, str]:
    """Merge extraction fields and deep answers into a flat known-information map."""
    known: dict[str, str] = {}

    skip_keys = {
        "apppurpose", "apptype", "targetusers", "budget", "budgettier",
        "detectedlanguage", "usertone", "onelineunderstanding", "confidence",
        "missingfields", "keyfeatures", "wantsimageinput", "enterprisesignals",
        "usertype", "primary_subject", "environment_setting", "action_dynamic",
        "aesthetic_style", "suggestedreply",
    }

    for k, v in (extraction or {}).items():
        if not isinstance(v, str) or not v.strip():
            continue
        key = str(k).lower().strip()
        if key in skip_keys:
            continue
        known[key] = v.strip()

    for k, v in (deep_answers or {}).items():
        if k.startswith("_") or not v or not str(v).strip():
            continue
        key = str(k).lower().strip()
        if key in ("ingestion_vector", "budgetpreference", "budgettier"):
            continue
        known[key] = str(v).strip()

    return known


def normalize_question_text(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", str(text or "").lower())
    return " ".join(cleaned.split())


def question_content_words(text: str) -> set[str]:
    stop = {
        "what", "how", "will", "you", "the", "a", "an", "is", "are", "should",
        "would", "like", "to", "for", "of", "this", "that", "app", "users",
        "user", "provide", "be", "do", "does", "your", "their", "or", "and",
    }
    words: set[str] = set()
    for w in normalize_question_text(text).split():
        if w.isdigit():
            words.add(w)
        elif w not in stop and len(w) > 2:
            words.add(w)
    return words


def questions_semantically_equivalent(q1: str, q2: str) -> bool:
    n1, n2 = normalize_question_text(q1), normalize_question_text(q2)
    if not n1 or not n2:
        return False
    if n1 == n2:
        return True
    if n1 in n2 or n2 in n1:
        return True
    w1, w2 = question_content_words(q1), question_content_words(q2)
    if not w1 or not w2:
        return False
    overlap = len(w1 & w2) / len(w1 | w2)
    return overlap >= 0.65


def has_behavior_justification(reason: str) -> bool:
    r = str(reason or "").lower().strip()
    if len(r) < 30:
        return False
    return any(marker in r for marker in BEHAVIOR_JUSTIFICATION_MARKERS)
FORBIDDEN_INPUT_KEYS = frozenset({
    "input_source", "ingestion_vector", "upload_type", "file_format", "content_source",
    "input_method", "upload_method", "file_type", "input_format", "content_format"
})

# Use central artifact helper to decide whether uploaded artifacts are required
from services.artifact_utils import requires_input_artifact as _requires_input_artifact


def is_input_modality_question(key: str, question: str) -> bool:
    k_l = str(key or "").lower().strip()
    q_l = str(question or "").lower().strip()

    if k_l in FORBIDDEN_INPUT_KEYS or any(f in k_l for f in ("input_source", "ingestion_vector", "upload_type", "file_format", "content_source")):
        return True

    input_signals = (
        "how will users provide",
        "how will you provide",
        "upload pdf",
        "upload file",
        "paste text",
        "website url",
        "website link",
        "screenshot",
        "pdf or text",
        "epub or txt",
        "what format is",
        "what format are",
        "input method",
        "upload method",
        "ingestion vector",
        "file format",
        "content source",
        "upload format"
    )
    if any(sig in q_l for sig in input_signals):
        return True

    return False


def is_generic_question_rejected(key: str, question: str, reason: str) -> bool:
    """
    Return True if the question should be rejected (filtered out).
    Generic slots/questions are rejected unless explicitly behavior-justified.
    """
    key_l = str(key or "").lower().strip()
    q_l = str(question or "").lower().strip()

    is_generic_key = key_l in GENERIC_SLOT_KEYS or any(
        g in key_l for g in ("input_source", "output_format", "audience", "budget")
    )
    is_generic_question = any(sig in q_l for sig in GENERIC_QUESTION_SIGNALS)

    if not is_generic_key and not is_generic_question:
        return False

    return not has_behavior_justification(reason)


def is_duplicate_clarification(
    key: str,
    question: str,
    asked_keys: list[str],
    asked_questions: list[str],
    known: dict[str, str],
) -> bool:
    """Return True if this key/question was already asked or is semantically equivalent."""
    from services.extraction import _slot_is_captured

    key_l = str(key or "").lower().strip()
    asked_keys_l = [str(k).lower().strip() for k in asked_keys]

    if key_l in asked_keys_l:
        return True

    for asked_key in asked_keys_l:
        if _slot_is_captured(key_l, {asked_key: "asked"}):
            return True
        if _slot_is_captured(asked_key, {key_l: "asked"}):
            return True

    if key_l in known or _slot_is_captured(key_l, known):
        return True

    for asked_q in asked_questions:
        if questions_semantically_equivalent(question, asked_q):
            return True

    return False


def clarification_is_complete(plan: dict) -> bool:
    """
    Exact termination logic for clarification_complete = True:

    1. plan["status"] == "ready" — planner (or safeguards) determined enough
       behavioral information exists, OR no valid questions remain after filtering.
    2. plan.get("forced_complete") — MAX_CLARIFICATION_TURNS reached; proceed
       best-effort.
    """
    if plan.get("forced_complete"):
        return True
    return plan.get("status") == "ready"


def log_clarification_trace(
    *,
    app_purpose: str,
    plan: dict,
    clarification_complete: bool | None = None,
) -> None:
    known_items = plan.get("known_information") or []
    missing_items = plan.get("missing_information") or []

    if isinstance(known_items[0] if known_items else None, dict):
        known_out = [{"key": k.get("key"), "value": k.get("value")} for k in known_items]
    else:
        known_out = known_items

    if isinstance(missing_items[0] if missing_items else None, dict):
        missing_out = [
            {"key": m.get("key"), "question": m.get("question"), "reason": m.get("reason")}
            for m in missing_items
        ]
    else:
        missing_out = missing_items

    complete = clarification_complete if clarification_complete is not None else clarification_is_complete(plan)

    trace = {
        "app_purpose": app_purpose,
        "behavior_goal": plan.get("behavior_goal"),
        "known_information": known_out,
        "missing_information": missing_out,
        "selected_question": plan.get("selected_question"),
        "reason": plan.get("reason"),
        "clarification_complete": complete,
    }
    logger.info(f"[CLARIFICATION] {json.dumps(trace, indent=2)}")


def apply_plan_safeguards(
    plan: dict,
    *,
    known: dict[str, str],
    asked_keys: list[str] | None = None,
    asked_questions: list[str] | None = None,
    triage_rounds: int = 0,
    app_type: str = "text",
    app_purpose: str = "",
) -> dict:
    """
    Apply max-turn cap, duplicate protection, confidence score check,
    high-priority dimension completion, and question quality filter.

    Sets plan["clarification_complete"] and may force plan["status"] = "ready".
    """
    asked_keys = asked_keys or []
    asked_questions = asked_questions or []
    plan = dict(plan)

    max_turns = get_max_clarification_turns(app_type)
    
    # ─── 🛡️ PROGRAMMATIC EXIT SAFEGUARDS ───
    # 1. Turn ceiling check
    if triage_rounds >= max_turns:
        logger.warning(
            f"[CLARIFICATION] MAX_CLARIFICATION_TURNS ({max_turns}) reached "
            f"— forcing best-effort app configuration"
        )
        plan["status"] = "ready"
        plan["forced_complete"] = True
        plan["clarification_complete"] = True
        plan["selected_key"] = None
        plan["selected_question"] = None
        plan["missing_information"] = []
        plan["reason"] = (
            f"Max clarification turns ({max_turns}) reached — "
            "proceeding with best-effort app configuration"
        )
        return plan

    # 2. Confidence score check
    confidence = plan.get("confidence", 0.0)
    if confidence >= 0.8:
        logger.info(
            f"[CLARIFICATION] Confidence score {confidence} >= 0.8 "
            f"— forcing readiness"
        )
        plan["status"] = "ready"
        plan["forced_complete"] = False
        plan["clarification_complete"] = True
        plan["selected_key"] = None
        plan["selected_question"] = None
        plan["missing_information"] = []
        if not plan.get("reason"):
            plan["reason"] = f"Confidence score ({confidence}) is sufficient."
        return plan

    filtered_missing: list[dict] = []
    for item in plan.get("missing_information") or []:
        key = str(item.get("key") or "").lower().strip()
        question = str(item.get("question") or "").strip()
        reason = str(item.get("reason") or "").strip()

        # Reject input-modality questions if the app does not require an uploaded artifact as the actual subject of analysis
        if not _requires_input_artifact(app_type, app_purpose):
            if is_input_modality_question(key, question):
                logger.info(
                    f"[CLARIFICATION] Rejected input modality question for non-artifact analysis app (key={key}): {question[:100]}"
                )
                continue

        if is_generic_question_rejected(key, question, reason):
            logger.info(
                f"[CLARIFICATION] Rejected generic question (key={key}): {question[:100]}"
            )
            continue

        if is_duplicate_clarification(key, question, asked_keys, asked_questions, known):
            logger.info(
                f"[CLARIFICATION] Rejected duplicate question (key={key}): {question[:100]}"
            )
            continue

        filtered_missing.append({
            "key": key,
            "question": question,
            "reason": reason,
        })

    # 3. Check if no high-priority behavioral dimensions remain missing
    high_keys = {str(k).lower().strip() for k in (plan.get("high_priority_keys") or [])}
    if not high_keys and app_purpose and len(app_purpose) >= 8:
        wf = _match_workflow(app_purpose) or _get_generic_workflow(app_purpose, app_type)
        high_keys = {d["key"].lower().strip() for d in wf.get("high_priority_dimensions") or []}

    has_missing_high = any(item.get("key") in high_keys for item in filtered_missing)

    if not has_missing_high:
        logger.info(
            f"[CLARIFICATION] No high-priority behavioral dimensions remain missing "
            f"— forcing readiness"
        )
        plan["status"] = "ready"
        plan["clarification_complete"] = True
        plan["forced_complete"] = False
        plan["selected_key"] = None
        plan["selected_question"] = None
        plan["missing_information"] = []
        plan["reason"] = "No high-priority behavioral dimensions remain missing after filtering."
        return plan

    plan["missing_information"] = filtered_missing

    if filtered_missing:
        plan["status"] = "needs_clarification"
        plan["clarification_complete"] = False
        plan["forced_complete"] = False
        plan["selected_key"] = filtered_missing[0]["key"]
        plan["selected_question"] = filtered_missing[0]["question"]
        if not plan.get("reason"):
            plan["reason"] = filtered_missing[0].get("reason", "")
    else:
        plan["status"] = "ready"
        plan["clarification_complete"] = True
        plan["forced_complete"] = False
        plan["selected_key"] = None
        plan["selected_question"] = None
        if not plan.get("reason"):
            plan["reason"] = "All behavioral requirements satisfied or no valid questions remain"

    return plan


def _normalize_plan(raw: dict) -> dict:
    """Normalize workflow-centric LLM output into standard plan structure."""
    workflow = raw.get("workflow") or {}
    workflow_name = workflow.get("workflow_name") or ""
    action = workflow.get("action") or ""
    subject = workflow.get("subject") or ""
    output_target = workflow.get("output") or ""
    
    behavior_goal = f"{action} {subject} to {output_target}".strip()
    if not behavior_goal or behavior_goal == "to":
        behavior_goal = raw.get("behavior_goal") or f"run {workflow_name}"
        
    bd = raw.get("behavioral_dimensions") or {}
    high_dims = bd.get("high_priority_dimensions") or []
    med_dims = bd.get("medium_priority_dimensions") or []
    
    known_info = []
    missing_info = []
    
    for item in high_dims:
        if not isinstance(item, dict) or not item.get("key"):
            continue
        key = str(item["key"]).lower().strip()
        val = item.get("value")
        if val and str(val).strip() and str(val).lower() != "null":
            known_info.append({"key": key, "value": str(val).strip()})
        else:
            missing_info.append({
                "key": key,
                "question": str(item.get("question") or f"What is the {key}?").strip(),
                "reason": "High priority workflow dimension"
            })
            
    for item in med_dims:
        if not isinstance(item, dict) or not item.get("key"):
            continue
        key = str(item["key"]).lower().strip()
        val = item.get("value")
        if val and str(val).strip() and str(val).lower() != "null":
            known_info.append({"key": key, "value": str(val).strip()})
        else:
            missing_info.append({
                "key": key,
                "question": str(item.get("question") or f"What is the {key}?").strip(),
                "reason": "Medium priority workflow dimension"
            })
            
    confidence = float(raw.get("confidence") or 0.0)
    ready = bool(raw.get("ready"))
    
    # If confidence is high, or no missing high-priority info, force ready
    has_missing_high = any(
        any(h["key"] == m["key"] for h in high_dims)
        for m in missing_info
    )
    if confidence >= 0.8 or not has_missing_high:
        ready = True
        
    selected_key = raw.get("selected_key")
    selected_question = raw.get("selected_question")
    
    if not ready and missing_info:
        if not selected_key:
            selected_key = missing_info[0]["key"]
        if not selected_question:
            selected_question = missing_info[0]["question"]
        selected_key = str(selected_key).lower().strip()
        selected_question = str(selected_question).strip()
    else:
        ready = True
        selected_key = None
        selected_question = None
        
    high_keys = []
    for item in high_dims:
        if isinstance(item, dict) and item.get("key"):
            high_keys.append(str(item["key"]).lower().strip())

    return {
        "status": "ready" if ready else "needs_clarification",
        "behavior_goal": behavior_goal,
        "known_information": known_info,
        "missing_information": missing_info,
        "selected_key": selected_key,
        "selected_question": selected_question,
        "reason": str(raw.get("reason") or "").strip(),
        "clarification_complete": ready,
        "forced_complete": False,
        "confidence": confidence,
        "has_missing_high": has_missing_high,
        "high_priority_keys": high_keys
    }


KNOWN_WORKFLOWS = {
    "astrology": {
        "workflow_name": "horoscope_generation",
        "subject": "astrological calculations",
        "action": "generate",
        "output": "horoscope predictions",
        "high_priority_dimensions": [
            {"key": "astrology_system", "description": "System of astrology (e.g. Western, Vedic, Chinese)", "question": "What system of astrology or chart calculations should the app use (e.g., Western, Vedic, Chinese)?"},
            {"key": "prediction_scope", "description": "Scope of predictions (e.g. daily, weekly, monthly, birth chart)", "question": "What specific scope of predictions or calculations should be covered (e.g. daily predictions, birth chart houses)?"}
        ],
        "medium_priority_dimensions": [
            {"key": "target_audience", "description": "Target readers (e.g. beginners, advanced practitioners)", "question": "Who is the target audience or reader for the generated horoscope?"}
        ]
    },
    "blog_to_video": {
        "workflow_name": "content_to_video",
        "subject": "blog posts",
        "action": "convert",
        "output": "youtube videos",
        "high_priority_dimensions": [
            {"key": "video_style", "description": "Visual style of the video (e.g. animated, stock footage, cinematic)", "question": "What visual style should the generated video adopt (e.g. animated, stock footage, cinematic)?"},
            {"key": "video_duration", "description": "Maximum video duration (e.g. under 1 minute for shorts, 5-10 minutes for long-form)", "question": "What maximum video duration should be generated (e.g. short-form under 60 seconds, or long-form 5-10 minutes)?"}
        ],
        "medium_priority_dimensions": [
            {"key": "narration_style", "description": "Voice type and narration style (e.g. energetic, professional, multi-speaker)", "question": "What narration style should the generated video use (e.g. energetic voice, professional narration)?"}
        ]
    },
    "logo_generator": {
        "workflow_name": "logo_generation",
        "subject": "brand information",
        "action": "generate",
        "output": "logo designs",
        "high_priority_dimensions": [
            {"key": "logo_style", "description": "Aesthetic style of the logo (e.g. minimalist, vintage, modern)", "question": "What visual style or aesthetic theme should the generated logos follow (e.g. minimalist, vintage, modern)?"},
            {"key": "industry", "description": "Company industry or domain", "question": "What industry or domain is your company or brand in?"}
        ],
        "medium_priority_dimensions": [
            {"key": "icon_preference", "description": "Preference for icons vs text-only logos", "question": "Do you prefer icon-based logos, typography-only logos, or emblem designs?"}
        ]
    },
    "resume_reviewer": {
        "workflow_name": "resume_analysis",
        "subject": "resumes",
        "action": "review",
        "output": "feedback reports",
        "high_priority_dimensions": [
            {"key": "target_role", "description": "Target job or industry role", "question": "What specific job title, level, or industry should the resume review target?"},
            {"key": "feedback_focus", "description": "Main aspect of review (e.g. ATS optimization, grammar, content structure)", "question": "What specific criteria should the review focus on (e.g., ATS optimization, formatting, experience description)?"}
        ],
        "medium_priority_dimensions": [
            {"key": "ats_level", "description": "Strictness level of ATS scanning match", "question": "How strict should the ATS compliance check be (e.g. standard match, executive level keyword keyword parsing)?"}
        ]
    },
    "audiobook_generator": {
        "workflow_name": "audiobook_generation",
        "subject": "written books",
        "action": "generate",
        "output": "spoken audiobooks",
        "high_priority_dimensions": [
            {"key": "narration_style", "description": "Narration tone and pacing", "question": "What narration style should the audiobook use (e.g. professional narration, expressive storytelling)?"},
            {"key": "voice_type", "description": "Gender, age, and accent of the AI voice", "question": "What type of voice would you like to generate (e.g., warm male voice, professional female voice, specific accent)?"}
        ],
        "medium_priority_dimensions": [
            {"key": "language", "description": "Language of the audiobook", "question": "Which language should the generated narration target?"}
        ]
    }
}


def _match_workflow(app_purpose: str) -> dict | None:
    p = str(app_purpose or "").lower().strip()
    if "astrology" in p or "horoscope" in p or "birth chart" in p or "zodiac" in p:
        return KNOWN_WORKFLOWS["astrology"]
    if "blog" in p and "video" in p:
        return KNOWN_WORKFLOWS["blog_to_video"]
    if "logo" in p:
        return KNOWN_WORKFLOWS["logo_generator"]
    if "resume" in p and ("review" in p or "analyze" in p or "feedback" in p or "score" in p):
        return KNOWN_WORKFLOWS["resume_reviewer"]
    if "audiobook" in p or "audio book" in p:
        return KNOWN_WORKFLOWS["audiobook_generator"]
    return None


def _get_generic_workflow(app_purpose: str, app_type: str) -> dict:
    purpose = str(app_purpose or "").strip()
    atype = str(app_type or "text").strip().lower()
    return {
        "workflow_name": "custom_workflow",
        "subject": "user input",
        "action": "process",
        "output": f"{atype} output",
        "high_priority_dimensions": [
            {"key": "content_style", "description": f"Visual, structural, or writing style of the {atype} output", "question": f"What specific style, format, or structure should the generated {atype} use?"}
        ],
        "medium_priority_dimensions": [
            {"key": "target_audience", "description": "Target audience for the app's output", "question": f"Who is the target audience or user for the generated {atype}?"}
        ]
    }


def _purpose_specific_fallback(app_purpose: str, known: dict[str, str], app_type: str = "text") -> dict:
    """Minimal fallback when LLM is unavailable — asks specific behavioral dimension questions."""
    known_list = [{"key": k, "value": v} for k, v in known.items()]

    from services.extraction import _slot_is_captured

    wf = _match_workflow(app_purpose) or _get_generic_workflow(app_purpose, app_type)
    
    # Extract dimensions
    high_dims = wf["high_priority_dimensions"]
    med_dims = wf["medium_priority_dimensions"]
    
    missing_info = []
    
    # Add high priority missing dimensions first
    for d in high_dims:
        k = d["key"]
        if not _slot_is_captured(k, known):
            missing_info.append({
                "key": k,
                "question": d["question"],
                "reason": "High priority dimension fallback"
            })
            
    # Add medium priority missing dimensions second
    for d in med_dims:
        k = d["key"]
        if not _slot_is_captured(k, known):
            missing_info.append({
                "key": k,
                "question": d["question"],
                "reason": "Medium priority dimension fallback"
            })

    if not missing_info:
        return _normalize_plan({
            "ready": True,
            "confidence": 0.9,
            "reason": "Sufficient behavioral context from collected answers",
            "workflow": {
                "workflow_name": wf["workflow_name"],
                "subject": wf["subject"],
                "action": wf["action"],
                "output": wf["output"]
            },
            "behavioral_dimensions": {
                "high_priority_dimensions": [{"key": d["key"], "description": d["description"], "value": known.get(d["key"])} for d in high_dims],
                "medium_priority_dimensions": [{"key": d["key"], "description": d["description"], "value": known.get(d["key"])} for d in med_dims]
            }
        })

    selected = missing_info[0]
    return _normalize_plan({
        "ready": False,
        "confidence": 0.5,
        "selected_key": selected["key"],
        "selected_question": selected["question"],
        "reason": f"Planner LLM unavailable — asking high-value question for '{selected['key']}'",
        "workflow": {
            "workflow_name": wf["workflow_name"],
            "subject": wf["subject"],
            "action": wf["action"],
            "output": wf["output"]
        },
        "behavioral_dimensions": {
            "high_priority_dimensions": [{"key": d["key"], "description": d["description"], "value": known.get(d["key"])} for d in high_dims],
            "medium_priority_dimensions": [{"key": d["key"], "description": d["description"], "value": known.get(d["key"])} for d in med_dims]
        }
    })


async def _call_planner_llm(llm: LLMService, user_prompt: str) -> dict | None:
    if llm.has_groq:
        try:
            result = await llm.groq_completion(
                messages=[
                    {"role": "system", "content": CLARIFICATION_PLANNER_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"},
            )
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            parsed = json.loads(content)
            if parsed.get("behavior_goal") is not None or parsed.get("ready") is not None:
                return parsed
        except Exception as error:
            logger.warning(f"Groq clarification planner failed, trying OpenRouter: {error}")

    if llm.has_openrouter:
        try:
            raw = await llm.openrouter_completion(
                messages=[
                    {"role": "system", "content": CLARIFICATION_PLANNER_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                model="meta-llama/llama-3.3-70b-instruct",
                response_format={"type": "json_object"},
            )
            parsed = json.loads(raw)
            if parsed.get("behavior_goal") is not None or parsed.get("ready") is not None:
                return parsed
        except Exception as error:
            logger.error(f"OpenRouter clarification planner failed: {error}")

    return None


async def plan_clarification(
    llm: LLMService,
    app_purpose: str,
    app_type: str = "text",
    known_information: dict[str, str] | None = None,
    conversation_history: list | None = None,
    asked_keys: list[str] | None = None,
    asked_questions: list[str] | None = None,
    triage_rounds: int = 0,
) -> dict:
    """
    Plan the next clarification turn for an app idea.

    Termination (clarification_complete = True) occurs when:
    - LLM sets ready=true and no missing behavioral requirements remain, OR
    - All missing items are already captured in known_information, OR
    - Safeguards filter out all remaining questions (duplicates / generic), OR
    - triage_rounds >= MAX_CLARIFICATION_TURNS (forced best-effort completion).
    """
    known = dict(known_information or {})
    safe_purpose = str(app_purpose or "").strip()

    if not safe_purpose or len(safe_purpose) < 8:
        plan = {
            "status": "needs_clarification",
            "behavior_goal": "",
            "known_information": [],
            "missing_information": [{
                "key": "app_purpose",
                "question": "What kind of app would you like to build, and what should it do?",
                "reason": "Cannot plan without an app purpose",
            }],
            "selected_key": "app_purpose",
            "selected_question": "What kind of app would you like to build, and what should it do?",
            "reason": "App purpose is missing or too short",
            "clarification_complete": False,
            "forced_complete": False,
        }
        return apply_plan_safeguards(
            plan,
            known=known,
            asked_keys=asked_keys,
            asked_questions=asked_questions,
            triage_rounds=triage_rounds,
            app_type=app_type,
            app_purpose=safe_purpose,
        )

    asked_block = ""
    if asked_keys or asked_questions:
        asked_block = (
            f"\nAlready asked clarification keys: {json.dumps(asked_keys or [])}\n"
            f"Already asked questions: {json.dumps(asked_questions or [])}\n"
            "Do NOT repeat or rephrase these.\n"
        )

    known_block = json.dumps(
        [{"key": k, "value": v} for k, v in known.items()],
        indent=2,
    ) if known else "[]"

    history_block = ""
    if conversation_history:
        recent = conversation_history[-6:]
        lines = []
        for entry in recent:
            if isinstance(entry, dict):
                role = entry.get("role", "user")
                content = str(entry.get("content") or "")[:300]
            else:
                role = "agent" if getattr(entry, "type", "human") in ("assistant", "agent") else "user"
                content = str(getattr(entry, "content", ""))[:300]
            lines.append(f"{role}: {content}")
        history_block = "\n".join(lines)

    user_prompt = (
        f"App purpose: {safe_purpose}\n"
        f"App type (modality): {app_type}\n"
        f"Known information from user answers:\n{known_block}\n"
        f"{asked_block}"
    )
    if history_block:
        user_prompt += f"\nRecent conversation:\n{history_block}\n"
    user_prompt += "\nPlan the next clarification step."

    raw = await _call_planner_llm(llm, user_prompt)
    if raw:
        plan = _normalize_plan(raw)
    else:
        plan = _purpose_specific_fallback(safe_purpose, known, app_type)

    from services.extraction import _slot_is_captured

    # Filter out items already captured in known information
    filtered_missing: list[dict] = []
    for item in plan.get("missing_information") or []:
        key = item.get("key", "")
        if not _slot_is_captured(key, known):
            filtered_missing.append(item)

    # Materiality scoring heuristic: base on declared priority and textual signals
    def _materiality_score(item: dict) -> float:
        score = 0.0
        key = str(item.get("key") or "").lower()
        reason = str(item.get("reason") or "")

        # Base weight by reason/priority: 'High' -> 2, else 1
        if "high" in reason.lower():
            score += 2.0
        else:
            score += 1.0

        # Boost if the key appears in the app purpose
        if key and key in safe_purpose.lower():
            score += 1.5

        # Boost if the reason contains behavioral justification markers
        if has_behavior_justification(reason):
            score += 1.0

        # Penalize generic or input-modality questions
        if is_input_modality_question(item.get("key", ""), item.get("question", "")):
            score -= 2.0
        if is_generic_question_rejected(item.get("key", ""), item.get("question", ""), reason):
            score -= 1.5

        return float(score)

    annotated_missing = []
    for it in filtered_missing:
        k = it.get("key")
        q = it.get("question")
        if not k or not q:
            continue
        if is_duplicate_clarification(k, q, asked_keys or [], asked_questions or [], known):
            continue
        annotated = dict(it)
        annotated["materiality"] = max(0.0, _materiality_score(annotated))
        annotated_missing.append(annotated)

    # Sort missing items by materiality (descending)
    annotated_missing.sort(key=lambda x: x.get("materiality", 0.0), reverse=True)
    plan["missing_information"] = annotated_missing

    # Planner state per spec
    planner_state = {
        "goal": safe_purpose,
        "behavioral_workflow": plan.get("behavior_goal") or (plan.get("workflow") or {}).get("workflow_name"),
        "known_information": known,
        "missing_information": annotated_missing,
        "readiness_score": float(plan.get("confidence") or 0.0),
        "clarification_rounds": int(triage_rounds or 0),
        "next_question": None,
        "reasoning": plan.get("reason") or "",
    }

    if annotated_missing:
        next_item = annotated_missing[0]
        plan["status"] = "needs_clarification"
        plan["clarification_complete"] = False
        plan["forced_complete"] = False
        plan["selected_key"] = next_item.get("key")
        plan["selected_question"] = next_item.get("question")
        planner_state["next_question"] = {"key": next_item.get("key"), "question": next_item.get("question"), "materiality": next_item.get("materiality")}
        if not plan.get("reason"):
            plan["reason"] = next_item.get("reason") or "Missing high-value behavioral dimension"
    else:
        plan["status"] = "ready"
        plan["clarification_complete"] = True
        plan["forced_complete"] = False
        plan["selected_key"] = None
        plan["selected_question"] = None
        planner_state["next_question"] = None
        if not plan.get("reason"):
            plan["reason"] = "All high-value behavioral dimensions satisfied or filtered out"

    plan["planner_state"] = planner_state

    # Apply programmatic safeguards (max turns, confidence override, input modality filters)
    plan = apply_plan_safeguards(
        plan,
        known=known,
        asked_keys=asked_keys,
        asked_questions=asked_questions,
        triage_rounds=triage_rounds,
        app_type=app_type,
        app_purpose=safe_purpose,
    )

    # Refresh planner_state after safeguards
    ps = plan.get("planner_state") or planner_state
    ps["readiness_score"] = float(plan.get("confidence") or ps.get("readiness_score") or 0.0)
    ps["clarification_rounds"] = int(triage_rounds or ps.get("clarification_rounds", 0))
    if plan.get("selected_key"):
        ps["next_question"] = {"key": plan.get("selected_key"), "question": plan.get("selected_question")}
    else:
        ps["next_question"] = None
    plan["planner_state"] = ps

    # Structured debug output
    try:
        debug_trace = {
            "goal": ps.get("goal"),
            "behavioral_workflow": ps.get("behavioral_workflow"),
            "known_information": ps.get("known_information"),
            "missing_information": [{"key": m.get("key"), "materiality": m.get("materiality"), "reason": m.get("reason")} for m in ps.get("missing_information", [])],
            "readiness_score": ps.get("readiness_score"),
            "selected_question": ps.get("next_question"),
            "reasoning": ps.get("reasoning"),
        }
        logger.info(f"[PLANNER_STATE] {json.dumps(debug_trace)}")
    except Exception:
        logger.exception("Failed to log planner_state debug trace")

    return plan
