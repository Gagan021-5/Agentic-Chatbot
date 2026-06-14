"""
Requirement extraction and dynamic context — RentPrompts groq.js.
Uses LLMService (Groq primary, OpenRouter fallback).
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
from loguru import logger

from services.language_directive import LANGUAGE_MIRROR_DIRECTIVE
from services.extraction_normalizer import normalize_extraction
from services.llm import LLMService

GROQ_SYSTEM_PROMPT = f"""You are a strict data extraction engine for RentPrompts — a platform where users CREATE and PUBLISH AI-powered apps.

Users describe an app they want to build.
Your ONLY job: extract what they said. Never invent.
{LANGUAGE_MIRROR_DIRECTIVE}

APP TYPE RULES — read every word carefully:
- "image" app: generates images, photos, portraits, transforms photos, superhero filter, avatar maker, logo maker, greeting cards, birthday cards, posters, memes, photo frames, invitations, flyers, any app where the OUTPUT is a PICTURE or VISUAL
- "video" app: creates videos, animations, reels, cinematic clips, animates photos, talking avatars
- "text" app: generates written content — blogs, emails, captions, scripts, stories, reports, product descriptions, resumes, cover letters, proposals, invoices, contracts, workout PLANS, meal PLANS, diet plans, study guides, itineraries, recipes, newsletters, SOPs, any document or written plan output
- "audio" app: voice, music, speech, podcast, sound effects, text to speech, transcription
- "vision" app: analyzes images, reads text from images, detects objects, medical image analysis

CRITICAL IMAGE vs TEXT — this is the #1 mistake to avoid:
- If the user mentions "photo", "picture", "image", "card" and the OUTPUT is a VISUAL (image with text on it, greeting card, poster, meme) → appType = "image"
- If the OUTPUT is pure written text (birthday wishes as text, written poems) → appType = "text"
- "birthday app with photo and text on it" = IMAGE (output is a picture)
- "birthday wishes generator" = TEXT (output is written words)
- "meme with text" = IMAGE. "caption generator" = TEXT.
- When user says "in the photo" or "on the image" or "with picture" → almost always IMAGE

CRITICAL: A "planner" app (workout planner, meal planner, travel planner) = appType "text" BUT appPurpose must describe WHAT it plans, not just say "text generation"

ANTI-HALLUCINATION:
1. If user said nothing about target users → null
2. If user said nothing about budget → null
3. oneLineUnderstanding = rephrase ONLY what user said
4. suggestedReply = one warm question about the MOST important unknown detail. Always a question.
5. Never say "building my app" or repeat vague phrases
6. Never invent features not mentioned by user

Enterprise detection rules:
- enterpriseSignals = true if message contains ANY of: company, team, employees, scale, API, integrate, bulk, enterprise, SaaS, B2B, workflow, automate our, our company, our team, organization, department, staff
- userType = "enterprise" if enterpriseSignals is true and message mentions large scale or enterprise context
- userType = "business" if enterpriseSignals is true but no enterprise-specific language
- userType = "developer" if message mentions developers, API, SDK, code, or integration work
- userType = "normal" if message describes personal or creator use case with no business/developer signals
- userType = "unknown" if no clear signals detected

Return ONLY valid JSON. No markdown. No explanation.
{{
  "appType": "text|image|audio|video|vision|null",
  "appPurpose": "describe what this app generates/does",
  "targetUsers": "string or null",
  "keyFeatures": [],
  "budget": "free|low|medium|high|ultra|null",
  "wantsImageInput": false,
  "detectedLanguage": "english",
  "userType": "enterprise|business|developer|normal|unknown",
  "enterpriseSignals": false,
  "userTone": "urgent|casual|formal|unsure",
  "confidence": {{
    "appType": "HIGH|MEDIUM|LOW",
    "budget": "HIGH|MEDIUM|LOW"
  }},
  "missingFields": [],
  "oneLineUnderstanding": "only rephrase what user said",
  "suggestedReply": "one warm follow-up question"
}}"""

ALLOWED_TRIAGE_APP_FORMATS = ["text", "image", "audio", "video", "vision"]

TRIAGE_INSTRUCTION = """You are RentPrompts App Intelligence Engine. Your job is to convert any user idea into a structured AI application definition using strict domain-first reasoning.

You MUST classify the application into one of these domains: text, image, audio, video, vision, or hybrid.
You must NOT use generic assistant categories as a fallback under any circumstance.

🚨 HARD RULE: NO GENERIC ASSISTANT MODE
You are strictly forbidden from using or displaying generic assistant templates such as:
- basic tasks (calendar, reminders, notes)
- translation
- summarization
- general productivity assistant
- chatbot assistant menus
These are INVALID unless the user explicitly requests a "general assistant app".

If the user provides ANY domain-specific intent (e.g. astrologer app, resume builder, legal advisor, logo generator, fitness planner, text-to-audio tool), you must immediately:
1. Identify the correct domain
2. Ignore all generic assistant flows
3. Build domain-specific variables only

🧠 DOMAIN CLASSIFICATION RULES:
- TEXT: resume builders, legal apps, astrologers, chatbots, planners, analyzers, document tools
- IMAGE: generation, editing, design, logos, visuals
- AUDIO: speech, TTS, music, voice
- VIDEO: motion generation, clips, animations
- VISION: image understanding, OCR, analysis
Default priority if unclear: TEXT > IMAGE > AUDIO > VIDEO > VISION

 
📊 CONFIDENCE SYSTEM:
Assign confidence_score (0–100):
- ≥ 85 → all key details (style, purpose, target audience, specific features) are clear and specific. Proceed to ready (status = "ready"), question MUST be omitted or null.
- < 85 → if the user's idea is broad (e.g. "logo creator", "resume builder", "workout planner") or some key details are missing. You MUST ask exactly one friendly clarification question about their preferences (status = "needs_context"). Never ask more than one question per turn. Omit variables in this state.

🔁 ADAPTIVE INTENT RULE:
If user changes idea mid-conversation:
- Discard previous domain
- Recompute domain + confidence immediately
- Never continue old flow

🧾 VARIABLE EXTRACTION RULES:
After domain is confirmed, extract 3–6 variables:
- Must be user-facing (non-technical)
- Must be independent inputs
- Must directly affect output
- NEVER include model names, internal parameters, legal codes, or system settings
- 🚨 CRITICAL SCHEMA STRUCTURING CONSTRAINT: Every extracted variable name MUST be translated into explicit human language. You are strictly prohibited from generating variables named 'input', 'text', 'data', 'variables', 'param', or 'main_input'. If the application parses legal domains, map to fields like 'incident_details' or 'dispute_context'. If editing visual elements, map strictly to fields like 'target_aesthetic' or 'canvas_dimensions'.

🤖 CHATBOT BEHAVIOR & QUESTION RULE:
- Behave like a natural, friendly chatbot (like Claude, Grok, ChatGPT). Use conversational, helpful phrasing in your questions.
- If the output format/app type is ambiguous, or you are confused/uncertain, set status to "needs_context", confidence_score < 80, ask a friendly question to clarify the type of output they want, and return suggested_options = ["Text", "Image", "Audio", "Video", "Vision"].
- Only include suggested_options when asking about output format (text/image/audio/video/vision). For all other questions (style, audience, use-case), ask naturally in conversational prose with examples inline (e.g. "like portraits, products, or landscapes?") — do NOT return suggested_options.
- Otherwise, if confidence >= 80, set status = "ready", question = null, and suggested_options = null.

Every response must be valid JSON only and include:
- status ("needs_context" or "ready")
- domain_identified ("text" | "image" | "audio" | "video" | "vision" | "hybrid")
- confidence_score (0-100)
- corrected_app_type (optional, only if initial classification was wrong: "text" | "image" | "audio" | "video" | "vision")
- variables (3-6 variables, each with name, placeholder, and realistic test_value)
- question (a single conversational question when confidence is below 80, otherwise null or omit)
- suggested_options (optional, list of 2-5 simple option strings to resolve ambiguity, e.g. ["Text", "Image", "Audio", "Video", "Vision"] if format is unclear)

Do not include any explanation or markdown outside the valid JSON object."""


def _is_rate_limit_error(error: Exception) -> bool:
    from tenacity import RetryError
    if isinstance(error, RetryError):
        attempt = getattr(error, "last_attempt", None)
        if attempt:
            try:
                underlying = attempt.exception()
                if underlying:
                    error = underlying
            except Exception:
                pass
    if isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 429:
        return True
    msg = str(error)
    return "429" in msg or "rate_limit" in msg.lower() or "too many requests" in msg.lower()



def _normalize_language_hint(language_hint: str | None) -> str:
    normalized = str(language_hint or "").lower()
    if "hinglish" in normalized:
        return "Hinglish"
    if "hindi" in normalized:
        return "Hindi"
    return "English"


def _sanitize_string_list(
    items: Any, min_len: int, max_len: int, fallback: list[str]
) -> list[str]:
    cleaned: list[str] = []
    if isinstance(items, list):
        seen: set[str] = set()
        for item in items:
            s = str(item or "").strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(s)
        cleaned = cleaned[:max_len]
    return cleaned if len(cleaned) >= min_len else fallback[:max_len]


def _prettify_variable_name(name: str, app_type: str) -> str:
    clean = str(name or "").strip()
    if not clean:
        return ""
    lower = clean.lower()
    type_lower = str(app_type or "text").lower()
    mapping = {
        "main_input": {
            "text": "Topic / Details",
            "image": "Visual Subject",
            "audio": "Script Text",
            "video": "Video Concept",
            "vision": "Image Analysis Goal",
        },
        "input_text": "Topic / Details",
        "user_input": "Topic / Details",
        "input": "Topic / Details",
        "context": "Additional Context",
        "background": "Additional Context",
        "output_style": "Preferred Style",
        "style": "Preferred Style",
        "details": "Specific Requirements",
    }
    if lower in mapping:
        val = mapping[lower]
        if isinstance(val, dict):
            return val.get(type_lower, "Topic / Details")
        return val
    clean = clean.replace("_", " ")
    clean = re.sub(r"([a-z])([A-Z])", r"\1 \2", clean)
    return " ".join(w.capitalize() for w in clean.split())


def _sanitize_variable_objects(
    items: Any, min_len: int, max_len: int, fallback: list[dict], app_type: str
) -> list[dict]:
    normalized: list[dict] = []
    if isinstance(items, list):
        seen: set[str] = set()
        for item in items:
            if isinstance(item, str):
                obj = {
                    "name": _prettify_variable_name(item.strip(), app_type),
                    "placeholder": "Enter details...",
                    "test_value": "",
                }
            elif isinstance(item, dict):
                obj = {
                    "name": _prettify_variable_name(str(item.get("name", "")).strip(), app_type),
                    "placeholder": str(item.get("placeholder") or "Enter details...").strip(),
                    "test_value": str(item.get("test_value") or "").strip(),
                }
            else:
                continue
            if not obj["name"]:
                continue
            key = obj["name"].lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(obj)
        normalized = normalized[:max_len]
    return normalized if len(normalized) >= min_len else fallback[:max_len]


def build_dynamic_context_fallback(
    app_type: str, app_purpose: str = "", language_hint: str = "English"
) -> dict:
    safe_type = str(app_type or "text").lower()
    type_defaults = {
        "image": {
            "options": ["Style control", "High-quality output", "Composition guidance", "Format flexibility"],
            "variables": [
                {"name": "Visual Subject", "placeholder": "What should appear in the image?", "test_value": "A majestic lion on a rock"},
                {"name": "Visual Style", "placeholder": "Visual style or aesthetic (e.g. photorealistic, anime)", "test_value": "photorealistic"},
                {"name": "Specific Requirements", "placeholder": "Any special colors, lighting, or details", "test_value": "golden hour lighting"},
            ],
        },
        "video": {
            "options": ["Scene control", "Style consistency", "Platform-ready output", "Motion effects"],
            "variables": [
                {"name": "Video Concept", "placeholder": "What story or scene to create?", "test_value": "A spaceship launching into a nebula"},
                {"name": "Visual Style", "placeholder": "Cinematic, vlog, animation...", "test_value": "Cinematic sci-fi"},
                {"name": "Target Platform", "placeholder": "YouTube, Instagram, TikTok...", "test_value": "YouTube"},
            ],
        },
        "audio": {
            "options": ["Voice selection", "Language support", "Pacing control", "Emotion control"],
            "variables": [
                {"name": "Script Content", "placeholder": "Text or script to convert to speech", "test_value": "Welcome back to another episode of our history podcast."},
                {"name": "Voice Tone", "placeholder": "Male/Female, energetic, calm, accent...", "test_value": "Male, clear and energetic"},
                {"name": "Audio Language", "placeholder": "English, Hindi, Spanish...", "test_value": "English"},
            ],
        },
        "vision": {
            "options": ["Accurate extraction", "Structured output", "Confidence scoring", "Use-case analysis"],
            "variables": [
                {"name": "Source Image", "placeholder": "Upload your image", "test_value": "photo of product"},
                {"name": "Image Analysis Goal", "placeholder": "What details should the AI detect in the image?", "test_value": "detect any scratches or defects"},
                {"name": "Output Format", "placeholder": "JSON, plain text, bullets...", "test_value": "JSON report"},
            ],
        },
        "text": {
            "options": ["Tone control", "Structured output", "Goal-focused generation", "Context awareness"],
            "variables": [
                {"name": "Topic / Details", "placeholder": "Describe what you want the AI to write about", "test_value": "The importance of learning to cook at home"},
                {"name": "Additional Context", "placeholder": "Background details or target audience", "test_value": "targeted at college students"},
                {"name": "Preferred Style", "placeholder": "Format, tone, length preferences", "test_value": "3-paragraph email, casual tone"},
            ],
        },
    }
    return type_defaults.get(safe_type) or type_defaults["text"]


def _parse_dynamic_context_payload(
    raw_content: str, app_type: str, app_purpose: str, language_hint: str
) -> dict:
    fallback = build_dynamic_context_fallback(app_type, app_purpose, language_hint)
    try:
        cleaned = re.sub(r"```json|```", "", str(raw_content or "{}"), flags=re.I).strip()
        parsed = json.loads(cleaned)
        return {
            "options": _sanitize_string_list(parsed.get("options"), 4, 4, fallback["options"]),
            "variables": _sanitize_variable_objects(
                parsed.get("variables"), 3, 8, fallback["variables"], app_type
            ),
        }
    except Exception:
        return fallback


def _map_history_to_triage_messages(conversation_history: list) -> list[dict]:
    recent = conversation_history[-8:] if isinstance(conversation_history, list) else []
    messages = []
    for m in recent:
        if not m:
            continue
        role_raw = str(m.get("role", "")).lower()
        role = "user" if role_raw == "user" else "assistant"
        content = str(m.get("content") or m.get("text") or "").strip()
        if content:
            messages.append({"role": role, "content": content})
    return messages


def _parse_triage_response(
    raw_content: str, format_fallback: str, app_purpose: str, language_hint: str
) -> dict:
    fallback_type = format_fallback if format_fallback in ALLOWED_TRIAGE_APP_FORMATS else "text"
    safe_purpose = str(app_purpose or "").strip()

    def ready_shape(domain, app_format, form, confidence=100):
        return {
            "status": "ready",
            "domain": domain,
            "confidence_score": confidence or 100,
            "question": None,
            "app_format": app_format,
            "form": form,
        }

    try:
        cleaned = re.sub(r"```json|```", "", str(raw_content or "{}"), flags=re.I).strip()
        parsed = json.loads(cleaned)
        if not parsed or not parsed.get("status"):
            return ready_shape(
                None, fallback_type,
                build_dynamic_context_fallback(fallback_type, safe_purpose, language_hint),
            )

        domain = str(parsed.get("domain_identified") or parsed.get("domain") or "").strip() or None
        confidence = float(parsed.get("confidence_score") or parsed.get("confidence") or 100)

        corrected_raw = str(parsed.get("corrected_app_type") or "").strip().lower()
        corrected_type = corrected_raw if corrected_raw in ALLOWED_TRIAGE_APP_FORMATS else None
        effective_type = corrected_type or fallback_type

        if parsed.get("status") == "needs_context":
            question = str(parsed.get("question") or "").strip()
            if not question or len(question) < 10:
                fb_form = build_dynamic_context_fallback(fallback_type, safe_purpose, language_hint)
                return ready_shape(domain, fallback_type, fb_form, confidence)
            suggested = None
            if isinstance(parsed.get("suggested_options"), list):
                opts = [str(o or "").strip() for o in parsed["suggested_options"] if str(o or "").strip()]
                suggested = opts[:6] if len(opts) >= 2 else None
            return {
                "status": "needs_context",
                "domain": domain,
                "confidence_score": confidence,
                "question": question,
                "suggested_options": suggested,
                "corrected_app_type": corrected_type,
                "form": None,
                "app_format": None,
            }

        fallback_form = build_dynamic_context_fallback(effective_type, safe_purpose, language_hint)
        form = parsed.get("form") if isinstance(parsed.get("form"), dict) else {}
        variables_raw = parsed.get("variables") if isinstance(parsed.get("variables"), list) else form.get("variables")
        options_raw = parsed.get("options") if isinstance(parsed.get("options"), list) else form.get("options")
        return ready_shape(
            domain,
            effective_type,
            {
                "options": _sanitize_string_list(options_raw, 4, 4, fallback_form["options"]),
                "variables": _sanitize_variable_objects(
                    variables_raw, 3, 8, fallback_form["variables"], effective_type
                ),
            },
            confidence,
        )
    except Exception as e:
        logger.error(f"[parse_triage_response] Fallback parse failed: {e}")
        return ready_shape(
            None, fallback_type,
            build_dynamic_context_fallback(fallback_type, "", language_hint),
        )


async def _extract_with_openrouter_fallback(
    llm: LLMService, message: str, history: list
) -> dict:
    try:
        raw = await llm.openrouter_completion(
            messages=[
                {"role": "system", "content": GROQ_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({
                    "message": message,
                    "history": history[-8:] if isinstance(history, list) else [],
                })},
            ],
            model="meta-llama/llama-3.3-70b-instruct",
            response_format={"type": "json_object"},
        )
        cleaned = re.sub(r"```json|```", "", raw or "{}", flags=re.I).strip()
        return normalize_extraction(json.loads(cleaned), message)
    except Exception as e:
        logger.error(f"OpenRouter extraction fallback failed: {e}")
        return normalize_extraction(None, message)


async def extract_requirements(llm: LLMService, message: str, history: list) -> dict:
    if not llm.has_groq:
        return await _extract_with_openrouter_fallback(llm, message, history)

    try:
        result = await llm.groq_completion(
            messages=[
                {"role": "system", "content": GROQ_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({
                    "message": message,
                    "history": history[-8:] if isinstance(history, list) else [],
                })},
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"},
        )
        content = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "{}")
        )
        return normalize_extraction(json.loads(content), message)
    except Exception as error:
        if _is_rate_limit_error(error):
            logger.info("[Groq] 429 hit, falling back to OpenRouter...")
        else:
            logger.error(f"Groq extraction error, falling back to OpenRouter: {error}")
        return await _extract_with_openrouter_fallback(llm, message, history)


async def generate_dynamic_context(
    llm: LLMService,
    app_type: str,
    app_purpose: str,
    language_hint: str = "English",
) -> dict:
    safe_type = app_type if app_type in ALLOWED_TRIAGE_APP_FORMATS else "text"
    safe_purpose = str(app_purpose or "").strip() or "general assistant app"
    safe_lang = _normalize_language_hint(language_hint)

    system_prompt = f"""You are a strict JSON generator.
Generate compact, practical setup suggestions for an AI app idea.
{LANGUAGE_MIRROR_DIRECTIVE}
When generating 'variables', you MUST include a 'placeholder' key.
- If the variable name contains 'Date', placeholder MUST be 'DD/MM/YYYY'.
- If the variable name contains 'Time', placeholder MUST be 'HH:MM AM/PM'.
- If the variable name contains 'Place' or 'Location', placeholder MUST be 'City, Country'.
- For everything else, use a relevant example.
NEVER use 'Enter details...' as a placeholder for date, time, or location fields.
- 🚨 CRITICAL SCHEMA STRUCTURING CONSTRAINT: Every extracted variable name MUST be translated into explicit human language. You are strictly prohibited from generating variables named 'input', 'text', 'data', 'variables', 'param', or 'main_input'. If the application parses legal domains, map to fields like 'incident_details' or 'dispute_context'. If editing visual elements, map strictly to fields like 'target_aesthetic' or 'canvas_dimensions'.
Output must be strict JSON with this exact shape:
{{"options":["4 concise feature options"],"variables":[{{"name":"Date of Birth","placeholder":"DD/MM/YYYY"}},{{"name":"Location","placeholder":"City, Country"}}]}}
No markdown. No prose."""

    user_prompt = (
        f"The user wants to build a {safe_type} app for: {safe_purpose}.\n"
        f"Language mode: {safe_lang}.\n"
        f"Generate 4 highly relevant specific features and 4-8 input variables needed for the app.\n"
        f"For each variable include name and helpful placeholder."
    )

    if llm.has_groq:
        try:
            result = await llm.groq_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model="llama-3.1-8b-instant",
                response_format={"type": "json_object"},
            )
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            return _parse_dynamic_context_payload(content, safe_type, safe_purpose, safe_lang)
        except Exception as error:
            if not _is_rate_limit_error(error):
                logger.error(f"Groq dynamic context error, falling back: {error}")

    if llm.has_openrouter:
        try:
            raw = await llm.openrouter_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model="meta-llama/llama-3.3-70b-instruct",
                response_format={"type": "json_object"},
            )
            return _parse_dynamic_context_payload(raw, safe_type, safe_purpose, safe_lang)
        except Exception as error:
            logger.error(f"OpenRouter dynamic context fallback failed: {error}")

    return build_dynamic_context_fallback(safe_type, safe_purpose, safe_lang)


async def triage_dynamic_context(
    llm: LLMService,
    app_type: str | None,
    app_purpose: str,
    language_hint: str = "English",
    conversation_history: list | None = None,
    deep_answers: dict | None = None,
) -> dict:
    committed = None
    if app_type and str(app_type).strip().lower() in ALLOWED_TRIAGE_APP_FORMATS:
        committed = str(app_type).strip().lower()
    format_fallback = committed or "text"
    safe_purpose = str(app_purpose or "").strip() or "general assistant app"
    safe_lang = _normalize_language_hint(language_hint)
    history = conversation_history or []

    answered_context = ""
    if deep_answers and len(deep_answers) > 0:
        clean = {k: v for k, v in deep_answers.items() if not k.startswith("_")}
        last_q = deep_answers.get("_lastTriageQuestion", "")
        answered_context = (
            f"\nAlready answered by user: {json.dumps(clean)}"
            f"\nLast question you asked: {last_q}"
        ) if clean or last_q else ""

    last_question_asked = ""
    if history:
        for m in reversed(history):
            role = str(m.get("role","")).lower()
            if role in ("assistant", "agent"):
                last_question_asked = str(m.get("content",""))
                break

    user_task = (
        f'Current app idea: "{safe_purpose}"\n'
        f'Current app type: "{format_fallback}".\n'
        f"Language: {safe_lang}.{answered_context}\n\n"
        f'IMPORTANT: The conversation history above already contains user answers. '
        f'{"Last question you asked: " + last_question_asked if last_question_asked else ""}\n'
        f'DO NOT repeat a question already answered in history.\n'
        f'If user already answered use_case/style/target — count that as known, move to next unknown or go ready.\n'
        "Return status='ready' once you have: use_case + at least one preference detail.\n"
        "Otherwise ask ONE new question about something not yet answered."
    )

    triage_messages = [
        {"role": "system", "content": TRIAGE_INSTRUCTION},
        *_map_history_to_triage_messages(history),
        {"role": "user", "content": user_task},
    ]

    if llm.has_groq:
        try:
            result = await llm.groq_completion(
                messages=triage_messages,
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"},
            )
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            return _parse_triage_response(content, format_fallback, safe_purpose, safe_lang)
        except Exception as error:
            if not _is_rate_limit_error(error):
                logger.error(f"Groq triage error, falling back: {error}")

    if llm.has_openrouter:
        try:
            raw = await llm.openrouter_completion(
                messages=triage_messages,
                model="meta-llama/llama-3.3-70b-instruct",
                response_format={"type": "json_object"},
            )
            return _parse_triage_response(raw, format_fallback, safe_purpose, safe_lang)
        except Exception as error:
            logger.error(f"OpenRouter triage fallback failed: {error}")

    return {
        "status": "ready",
        "domain": None,
        "question": None,
        "app_format": format_fallback,
        "form": build_dynamic_context_fallback(format_fallback, safe_purpose, safe_lang),
    }
