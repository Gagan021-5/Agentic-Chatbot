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

UNIVERSAL DIMENSIONS:
- PRIMARY_SUBJECT: Who/What is the main focus (e.g. Motor racing, Kaito Yamato)
- ENVIRONMENT_SETTING: Where is the scene taking place (e.g. Beach, Mumbai streets)
- ACTION_DYNAMIC: What is happening - fighting, racing, sleeping, fire power (e.g. High-speed driving, Fire manipulation)
- AESTHETIC_STYLE: The visual mood - cinematic, 8k, hyper-realistic, sketch, anime, Naruto style (e.g. Cinematic, Naruto style)

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
  "suggestedReply": "one warm follow-up question",
  "PRIMARY_SUBJECT": "string or null",
  "ENVIRONMENT_SETTING": "string or null",
  "ACTION_DYNAMIC": "string or null",
  "AESTHETIC_STYLE": "string or null"
}}"""

ALLOWED_TRIAGE_APP_FORMATS = ["text", "image", "audio", "video", "vision"]

TRIAGE_INSTRUCTION = """You are the Dynamic Context Triage Node inside a LangGraph framework grounded dynamically via a ChromaDB vector store.
Your job is to dynamically analyze the user's application concept ('appPurpose') and determine exactly 3 relevant domain-specific parameter slots required to build the blueprint for this custom app concept.

Act like an expert Product Manager from OpenAI or Anthropic. Your goal is to figure out the target blueprint requirements through dynamic fluid conversation.

═══════════════════════════════════════
SLOT SELECTION MANDATE BY APP FORMAT TYPE
═══════════════════════════════════════
You must align your 3 parameter slots strictly with the output format of the application. Gathering generic marketing metadata or demographic profiles for highly creative workflows is a critical system failure.

1. FOR "image" AND "video" FORMATS:
   - CRITICAL: You are STRICTLY FORBIDDEN from asking about "target users", "intended audience", "demographics", or "marketing goals".
   - MANDATORY SLOTS: You must select slots exclusively from this production-critical visual list:
     * movie_title / banner_headline (highly recommended for posters, covers, or ads)
     * main_character / visual_subject
     * setting_background / environment
     * visual_style / artistic_aesthetic
     * mood_lighting
     * composition_framing
   - RATIONALE: Image generation microservices cannot parse metadata like "fans aged 18-40"; they require descriptive, physical visual attributes to populate prompt templates.

2. FOR "audio" FORMATS:
   - CRITICAL: Do not ask about target users or demographics.
   - MANDATORY SLOTS: Focus exclusively on acoustic properties:
     * voice_gender_age
     * vocal_tone_mood
     * background_music_genre
     * pacing_delivery_speed

3. FOR "text" FORMATS:
   - You are permitted to select audience parameters, document length boundaries, business tone constraints, and target section layouts.

CRITICAL BEHAVIOR POLICIES:
- NEVER format your question as a multiple-choice menu, numbered checklist, bulleted list, or structured option bracket.
- ALWAYS return the question as natural, flowing conversational prose.
- NEVER append mechanical directive text strings like "Pick one option" or "Choose from below".
- Always ask exactly ONE friendly, conversational, open-ended question tailored precisely to the selected slot.
- If the user's concept is clear and all 3 dynamic slots are satisfied in the already captured attributes, set status = "ready".

ANTI-LOOP PROTECTION:
- You must read the already captured attributes. If a key is already populated in the captured attributes, you are strictly forbidden from re-asking it.
- Ensure you choose a different missing attribute, or if all are satisfied, set status to "ready".

Return strict JSON only (no markdown, no other text):
{{
  "status": "needs_context|ready",
  "question": "Clear natural flowing prose follow-up question if needs_context, else null",
  "slot_key": "The snake_case key variable name for state mapping, else null",
  "slots": [
    {{
      "key": "slot_key_name_1",
      "question": "Default dynamic friendly question for slot 1"
    }},
    {{
      "key": "slot_key_name_2",
      "question": "Default dynamic friendly question for slot 2"
    }},
    {{
      "key": "slot_key_name_3",
      "question": "Default dynamic friendly question for slot 3"
    }}
  ],
  "corrected_app_type": "Retain the incoming specialized app type string (text|image|audio|video|vision)",
  "domain_identified": "text|image|audio|video|vision|hybrid",
  "confidence_score": 0-100,
  "form": {{
    "options": ["4 concise feature options"],
    "variables": [
      {{
        "name": "variable name",
        "placeholder": "realistic placeholder",
        "test_value": "realistic test value"
      }}
    ]
  }}
}}

VARIABLE CONFIGURATION RULES (when status is "ready"):
- Extract 3–6 variables. They must be user-facing (non-technical) and directly affect output.
- NEVER include model names, internal parameters, or system settings.
- Variable name constraint: Every variable name MUST be translated into explicit human language. You are strictly prohibited from generating variables named 'input', 'text', 'data', 'variables', 'param', or 'main_input'.
- Placeholders:
  - If the variable name contains 'Date', placeholder MUST be 'DD/MM/YYYY'.
  - If the variable name contains 'Time', placeholder MUST be 'HH:MM AM/PM'.
  - If the variable name contains 'Place' or 'Location', placeholder MUST be 'City, Country'.
  - For everything else, use a relevant example.
  - NEVER use 'Enter details...' as a placeholder for date, time, or location fields.
}}"""


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


def is_personal_boilerplate(name: str, app_purpose: str) -> bool:
    n = name.lower()
    p = app_purpose.lower()
    boilerplate = [
        "user name", "username", "date of birth", "dob", "birth date", "age",
        "creation date", "current date", "today's date", "date of creation", "creationdate",
        "model name", "model variant", "llm provider", "model version", "api key"
    ]
    for kw in boilerplate:
        if kw in n:
            if kw not in p:
                return True
    return False


def _sanitize_variable_objects(
    items: Any, min_len: int, max_len: int, fallback: list[dict], app_type: str, app_purpose: str = ""
) -> list[dict]:
    normalized: list[dict] = []
    has_boilerplate_scrubbed = False
    
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
            if is_personal_boilerplate(obj["name"], app_purpose):
                has_boilerplate_scrubbed = True
                continue
            key = obj["name"].lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(obj)

        # ─── 🛡️ PLACEHOLDER SANITY INTERCEPTOR ───
        # Catch LLM-hallucinated placeholders like "e.g., v3", "e.g., 1.0", "e.g., model-name"
        # and replace them with domain-sensible values derived from the variable name.
        VERSION_PATTERN = re.compile(r"^\s*e\.?g\.?,?\s*v?\d", re.I)
        for obj in normalized:
            placeholder = str(obj.get("placeholder") or "")
            if VERSION_PATTERN.match(placeholder) or placeholder.strip().lower() in ("", "e.g., v3", "e.g., 1.0", "n/a"):
                var_name = str(obj.get("name") or "").lower()
                # Replace with a sensible domain-aware placeholder
                if "style" in var_name:
                    obj["placeholder"] = "e.g., Modern Minimalist, Futuristic, Corporate, Geometric"
                    obj["test_value"] = obj.get("test_value") or "Modern Minimalist"
                elif "color" in var_name:
                    obj["placeholder"] = "e.g., Monochromatic, Blue and White, Earthy tones"
                    obj["test_value"] = obj.get("test_value") or "Monochromatic blue and white"
                elif "industry" in var_name or "sector" in var_name:
                    obj["placeholder"] = "e.g., Technology, Healthcare, Finance, Retail"
                    obj["test_value"] = obj.get("test_value") or "Technology"
                elif "name" in var_name or "company" in var_name or "brand" in var_name:
                    obj["placeholder"] = "e.g., Nexora AI, BrightPath, TechVault"
                    obj["test_value"] = obj.get("test_value") or "Nexora AI"
                else:
                    obj["placeholder"] = f"Enter {obj.get('name', 'value').lower()}"
                    obj["test_value"] = obj.get("test_value") or ""
        
        # ─── 🛡️ PRD DOMAIN-AWARE ATTRIBUTE ENFORCEMENT & INTERCEPTOR ───
        if has_boilerplate_scrubbed or len(normalized) < min_len:
            domain_fields = []
            purpose_lower = app_purpose.lower()
            type_lower = app_type.lower()
            
            if "motivation" in purpose_lower or "speech" in purpose_lower or type_lower in ("audio", "text"):
                domain_fields = [
                    {"name": "Script Topic", "placeholder": "What should the speech/content focus on?", "test_value": "Overcoming interview anxiety and building confidence"},
                    {"name": "Target Industry", "placeholder": "e.g. Technology, Finance, General...", "test_value": "Technology"},
                    {"name": "Speaker Accent", "placeholder": "e.g. Male, British accent, clear voice...", "test_value": "Male, clear and energetic"}
                ]
            elif type_lower == "image":
                p = purpose_lower
                if any(w in p for w in ("logo", "brand", "identity", "icon")):
                    domain_fields = [
                        {"name": "Company Name", "placeholder": "e.g., Nexora AI, BrightPath, TechVault", "test_value": "Nexora AI"},
                        {"name": "Industry", "placeholder": "e.g., Technology, Healthcare, Finance", "test_value": "Artificial Intelligence"},
                        {"name": "Logo Style", "placeholder": "e.g., Modern Minimalist, Futuristic, Corporate, Geometric", "test_value": "Modern Minimalist"},
                        {"name": "Color Scheme", "placeholder": "e.g., Monochromatic, Blue and White, Earthy tones", "test_value": "Monochromatic blue and white"},
                    ]
                elif any(w in p for w in ("poster", "flyer", "banner", "ad", "marketing")):
                    domain_fields = [
                        {"name": "Headline Text", "placeholder": "Main text to display on the poster", "test_value": "Summer Sale — 50% Off"},
                        {"name": "Visual Theme", "placeholder": "e.g., Bold and vibrant, Clean and minimal", "test_value": "Bold and vibrant"},
                        {"name": "Color Scheme", "placeholder": "e.g., Red and black, Pastel tones", "test_value": "Red and black"},
                    ]
                elif any(w in p for w in ("portrait", "avatar", "headshot", "profile")):
                    domain_fields = [
                        {"name": "Subject Description", "placeholder": "Describe the person or character", "test_value": "Professional woman in her 30s"},
                        {"name": "Art Style", "placeholder": "e.g., Photorealistic, Anime, Oil painting", "test_value": "Photorealistic"},
                        {"name": "Background", "placeholder": "e.g., Office, Studio, Outdoor", "test_value": "Clean studio background"},
                    ]
                else:
                    domain_fields = [
                        {"name": "Visual Subject", "placeholder": "Main subject of the image", "test_value": "A futuristic city at sunset"},
                        {"name": "Aesthetic Style", "placeholder": "e.g., Photorealistic, Digital art, Oil painting", "test_value": "Cinematic digital art"},
                        {"name": "Lighting Style", "placeholder": "e.g., Golden hour, Neon, Studio lighting", "test_value": "Golden hour"},
                    ]
            elif type_lower == "video":
                domain_fields = [
                    {"name": "Video Concept", "placeholder": "Overall concept or storyline", "test_value": "Drone shot of ocean waves"},
                    {"name": "Motion Style", "placeholder": "Camera motion description", "test_value": "slow pan left"},
                    {"name": "Target Platform", "placeholder": "YouTube, Reels, TikTok...", "test_value": "YouTube Reels"}
                ]
            elif type_lower == "vision":
                domain_fields = [
                    {"name": "Image Analysis Goal", "placeholder": "What to analyze/detect", "test_value": "text on receipt"},
                    {"name": "Output Format", "placeholder": "Desired format (e.g. JSON, markdown)", "test_value": "JSON"}
                ]
            else:
                domain_fields = [
                    {"name": "Script Topic", "placeholder": "Topic or focus of the application", "test_value": "General topic"},
                    {"name": "Target Industry", "placeholder": "Target domain or area", "test_value": "Technology"},
                    {"name": "Speaker Accent", "placeholder": "Accent or voice style", "test_value": "Neutral tone"}
                ]
            
            for df in domain_fields:
                key = df["name"].lower()
                if key not in seen:
                    seen.add(key)
                    normalized.append(df)
                    
        normalized = normalized[:max_len]
        
    final_list = []
    for item in normalized:
        if not is_personal_boilerplate(item["name"], app_purpose):
            final_list.append(item)
            
    return final_list if len(final_list) >= min_len else [f for f in fallback if not is_personal_boilerplate(f["name"], app_purpose)][:max_len]


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
                parsed.get("variables"), 3, 8, fallback["variables"], app_type, app_purpose
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

    def ready_shape(domain, app_format, form, slots, confidence=100):
        return {
            "status": "ready",
            "domain": domain,
            "confidence_score": confidence or 100,
            "question": None,
            "app_format": app_format,
            "form": form,
            "slots": slots or [],
        }

    try:
        cleaned = re.sub(r"```json|```", "", str(raw_content or "{}"), flags=re.I).strip()
        parsed = json.loads(cleaned)
        
        # Extract dynamic slots list
        slots = parsed.get("slots") or []
        if not isinstance(slots, list):
            slots = []

        if not parsed or not parsed.get("status"):
            return ready_shape(
                None, fallback_type,
                build_dynamic_context_fallback(fallback_type, safe_purpose, language_hint),
                slots,
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
                return ready_shape(domain, fallback_type, fb_form, slots, confidence)
            suggested = None
            if isinstance(parsed.get("suggested_options"), list):
                opts = [str(o or "").strip() for o in parsed["suggested_options"] if str(o or "").strip()]
                suggested = opts[:6] if len(opts) >= 2 else None
            slot_key = str(parsed.get("slot_key") or "").strip().lower() or None
            return {
                "status": "needs_context",
                "domain": domain,
                "confidence_score": confidence,
                "question": question,
                "slot_key": slot_key,
                "suggested_options": suggested,
                "corrected_app_type": corrected_type,
                "form": None,
                "app_format": None,
                "slots": slots,
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
                    variables_raw, 3, 8, fallback_form["variables"], effective_type, safe_purpose
                ),
            },
            slots,
            confidence,
        )
    except Exception as e:
        logger.error(f"[parse_triage_response] Fallback parse failed: {e}")
        return ready_shape(
            None, fallback_type,
            build_dynamic_context_fallback(fallback_type, "", language_hint),
            [],
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


class UniversalSlotExtractor:
    def __init__(self, llm: LLMService):
        self.llm = llm

    async def extract(self, system_prompt: str | None, user_prompt: str | None, message: str) -> dict:
        sys_p = system_prompt or ""
        usr_p = user_prompt or ""
        msg = message or ""
        
        user_content = f"System Prompt Context: {sys_p}\nUser Prompt Context: {usr_p}\nIncoming Message: {msg}"
        
        system_content = """You are a Universal Dimension Extractor.
Analyze the provided system prompt, user prompt, and message to extract these 4 Universal Dimensions:
1. PRIMARY_SUBJECT: What is the focus (e.g. Motor racing, Kaito Yamato)
2. ENVIRONMENT_SETTING: Where is it (e.g. Beach, Mumbai streets)
3. ACTION_DYNAMIC: What is happening (e.g. High-speed driving, Fire manipulation)
4. AESTHETIC_STYLE: How does it look (e.g. Cinematic, Naruto style)

Output strictly valid JSON with these keys and no other text:
{
  "PRIMARY_SUBJECT": "extracted value or null",
  "ENVIRONMENT_SETTING": "extracted value or null",
  "ACTION_DYNAMIC": "extracted value or null",
  "AESTHETIC_STYLE": "extracted value or null"
}"""

        try:
            result = await self.llm.groq_completion(
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                model="llama-3.1-8b-instant",
                response_format={"type": "json_object"}
            )
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            return json.loads(content)
        except Exception as e:
            logger.error(f"UniversalSlotExtractor LLM call failed: {e}")
            return {}


async def extract_requirements(llm: LLMService, message: str, history: list) -> dict:
    # Instantiate and run the UniversalSlotExtractor
    extractor = UniversalSlotExtractor(llm)
    extracted_slots = await extractor.extract(None, None, message)

    if not llm.has_groq:
        standard_extracted = await _extract_with_openrouter_fallback(llm, message, history)
        merged = {**standard_extracted, **extracted_slots}
        return normalize_extraction(merged, message)

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
        standard_extracted = json.loads(content)
        merged = {**standard_extracted, **extracted_slots}
        return normalize_extraction(merged, message)
    except Exception as error:
        if _is_rate_limit_error(error):
            logger.info("[Groq] 429 hit, falling back to OpenRouter...")
        else:
            logger.error(f"Groq extraction error, falling back to OpenRouter: {error}")
        standard_extracted = await _extract_with_openrouter_fallback(llm, message, history)
        merged = {**standard_extracted, **extracted_slots}
        return normalize_extraction(merged, message)


async def generate_dynamic_context(
    llm: LLMService,
    app_type: str,
    app_purpose: str,
    language_hint: str = "English",
    rag_context: str = "",
    deep_answers: dict | None = None,
    existing_variables: list | None = None,
    conversation_history: list | None = None,
) -> dict:
    safe_type = app_type if app_type in ALLOWED_TRIAGE_APP_FORMATS else "text"
    safe_purpose = str(app_purpose or "").strip() or "general assistant app"
    safe_lang = _normalize_language_hint(language_hint)
    language_mode = safe_lang

    system_prompt = """
You are an expert AI App Variable Architect for the RentPrompts marketplace.
Analyze the app purpose, conversation history, and RAG context to generate
the most domain-specific, atomic, reusable prompt variables possible.

CRITICAL RULES:
1. NEVER generate pre-filled content variables where user writes full content:
   BAD:  episode_title → user types "The Mysterious Disappearance at Ravenswood Manor"
   GOOD: location, victim_name, case_type → AI generates title FROM these

2. NEVER merge multiple facts into one field:
   BAD:  case_file_details → "Victim: Sarah Lee, 28, missing 3 days, last seen at cafe"
   GOOD: victim_name, victim_age, case_duration, last_seen_location

3. Variables must be ATOMIC (one fact per variable), REUSABLE (works for any
   instance of this app), GENERATIVE (AI builds output FROM these inputs).

4. Use RAG context + conversation history to infer domain-specific variables:
   - True crime audio    → location, victim_name, case_type, suspect_count,
                           narrative_tone, episode_length, detective_style
   - Fantasy character   → race, weapon_type, armor_style, power_class,
                           world_setting, art_style, lighting_mood
   - Legal advisor       → incident_type, jurisdiction, party_role,
                           desired_outcome, case_urgency
   - Interior design     → room_type, design_style, color_theme, budget_range
   - Marketing copy      → product_name, target_audience, tone, platform
   Always infer from actual app purpose — never use these as defaults.

5. Generate exactly 4-7 variables. Quality over quantity.

6. Each variable must have all 5 fields:
   - name:        snake_case key used in prompt template
   - label:       Human readable label shown in the UI form
   - placeholder: Realistic SHORT example (NOT pre-written content sentences)
   - description: One line telling user what to input here
   - test_value:  Concrete sample value used in live preview generation

Return ONLY valid JSON, no markdown, no explanation:
{
  "variables": [
    {
      "name": "victim_name",
      "label": "Victim Name",
      "placeholder": "e.g. Sarah Lee",
      "description": "First and last name of the victim or missing person",
      "test_value": "Emily Carter"
    }
  ],
  "options": [],
  "title": "Short descriptive title for this app input schema"
}
"""

    history_block = ""
    if conversation_history:
        history_block = "\nConversation so far:\n" + "\n".join(
            f"{m.get('role','user')}: {m.get('content','')}"
            for m in (conversation_history or [])[-6:]
        )

    captured_block = ""
    if deep_answers:
        clean = {k: v for k, v in deep_answers.items() if not k.startswith("_") and v}
        if clean:
            captured_block = "\nAlready captured from user:\n" + json.dumps(clean)

    schema_lock_block = ""
    if existing_variables:
        schema_lock_block = (
            "\nEXISTING SCHEMA (LOCKED — preserve these exact variable names, "
            "only update test_value if a captured answer matches):\n"
            + json.dumps(existing_variables)
        )

    rag_block = f"\nRAG Context:\n{rag_context}" if rag_context else ""

    user_prompt = (
        f"App type: {safe_type}\n"
        f"App purpose: {safe_purpose}\n"
        f"Language: {safe_lang}\n"
        f"{history_block}"
        f"{captured_block}"
        f"{schema_lock_block}"
        f"{rag_block}\n"
        f"Generate 4 feature flags and 3-4 input variables precisely matching "
        f"this app's domain. Use the conversation history and captured answers "
        f"to prefill test_value fields. If existing schema is provided, "
        f"return it unchanged — only update test_values."
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
    rag_context: str = "", # 🚀 ADDED: Dynamic grounding database parameter slot
) -> dict:
    committed = None
    if app_type and str(app_type).strip().lower() in ALLOWED_TRIAGE_APP_FORMATS:
        committed = str(app_type).strip().lower()
    format_fallback = committed or "text"
    safe_purpose = str(app_purpose or "").strip() or "general assistant app"
    safe_lang = _normalize_language_hint(language_hint)
    history = conversation_history or []

    clean_answers = {k: v for k, v in deep_answers.items() if not k.startswith("_")} if deep_answers else {}
    answered_context = f"\nAlready captured attributes: {json.dumps(clean_answers)}" if clean_answers else ""
    rag_metadata = f"\nMarketplace Reference Guidelines:\n{rag_context}" if rag_context else ""

    user_task = (
        f"Active App Concept: {safe_purpose}\n"
        f"Format Type: {format_fallback}\n"
        f"Language: {safe_lang}.{answered_context}{rag_metadata}\n\n"
        "Evaluate the history. If the operational target is clear, mark ready. Otherwise issue ONE open-ended inquiry."
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
        "slots": [],
        "form": build_dynamic_context_fallback(format_fallback, safe_purpose, safe_lang),
    }
