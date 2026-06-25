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


def _msg_content(m: Any) -> str:
    try:
        if hasattr(m, "get"):
            return m.get("content", "") or ""
        if hasattr(m, "content"):
            return getattr(m, "content") or ""
        if isinstance(m, str):
            return m
        return str(m)
    except Exception:
        return ""


def _msg_role(m: Any) -> str:
    try:
        if hasattr(m, "get"):
            return m.get("role", "user") or "user"
        if hasattr(m, "role"):
            return getattr(m, "role") or "user"
        if hasattr(m, "type"):
            return getattr(m, "type") or "user"
        return "user"
    except Exception:
        return "user"

GROQ_SYSTEM_PROMPT = f"""You are a strict data extraction engine for RentPrompts — a platform where users CREATE and PUBLISH AI-powered apps.

Users describe an app they want to build.
Your ONLY job: extract what they said. Never invent.
{LANGUAGE_MIRROR_DIRECTIVE}

APP TYPE RULES — read every word carefully:
- "image" app: generates images, photos, portraits, transforms photos, superhero filter, avatar maker, logo maker, greeting cards, birthday cards, posters, memes, photo frames, invitations, flyers, any app where the OUTPUT is a PICTURE or VISUAL
- "video" app: creates videos, animations, reels, cinematic clips, animates photos, talking avatars
- "text" app: generates written content — blogs, emails, captions, scripts, stories, reports, product descriptions, resumes, cover letters, proposals, invoices, contracts, workout PLANS, meal PLANS, diet plans, study guides, itineraries, recipes, newsletters, SOPs, any document or written plan output
- "audio" app: voice, music, speech, podcast, sound effects, text to speech, transcription
- "vision" app: analyzes images, reads text from images, detects objects, medical image analysis,
  evaluates documents/PDFs/pitch decks/resumes/presentations/screenshots, reviews profiles or
  webpages by analyzing their visual content. If the user wants to ANALYZE or EVALUATE something
  they will UPLOAD (a file, deck, resume, profile, screenshot, webpage), it is "vision".

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

TRIAGE_INSTRUCTION = """You are the Dynamic Context Triage Node inside a LangGraph pipeline grounded via ChromaDB.

Your job: given an app concept, determine the 3 most critical parameter slots needed to build a tailored prompt, then ask for them one at a time.

═══════════════════════════════════════
STEP 1 — CLASSIFY THE APP ON 3 AXES
═══════════════════════════════════════

AXIS 1 — INPUT TYPE (what does the user provide to the app?):
- UPLOAD    → user submits a file, image, PDF, screenshot, deck, resume, webpage, document
- STRUCTURED → user enters specific data points (birth date, name, number, location)
- FREE_TEXT  → user describes something in natural language (topic, idea, story)
- NONE       → app generates purely from parameter selections

AXIS 2 — OUTPUT TYPE (what does the app produce?):
- ANALYSIS     → scores, evaluates, critiques, extracts from something provided
- GENERATION   → creates new content (text, image, audio, video)
- TRANSFORMATION → converts input to a different format or style
- EXTRACTION   → pulls structured data out of unstructured input

AXIS 3 — DOMAIN SENSITIVITY:
- PROFESSIONAL → legal, medical, financial, academic, enterprise
- CREATIVE     → fiction, art, entertainment, gaming
- PERSONAL     → health, relationships, self-improvement, astrology
- GENERAL      → everything else

═══════════════════════════════════════
STEP 2 — DETERMINE FIRST QUESTION
═══════════════════════════════════════

Use the INPUT ARTIFACT field provided in the user task below.
That field contains the exact noun to use in your question.

If INPUT ARTIFACT is provided:
  → Ask how users will provide it. Use that exact noun. Nothing else.

If INPUT ARTIFACT is null:
  → App is pure generation. Ask about the most domain-critical slot first.
  → Never ask about input format for generation apps.
  → Never use "intended audience" or "target audience" as slot 3.
    It is too generic. Always prefer a domain-specific slot instead:
    TTS/audio converter  → ask content_type (educational/story/news/corporate)
    Image generator      → ask visual style or aspect ratio
    Text generator       → ask tone or output format
    Video generator      → ask scene style or platform target

AXIS ordering priority:
  UPLOAD + ANALYSIS → ask input format first (use INPUT ARTIFACT noun)
  STRUCTURED        → ask for the most critical data field first
  GENERATION        → ask about output style or constraints first


═══════════════════════════════════════
STEP 3 — USE RAG TO NAME THE SLOTS
═══════════════════════════════════════

The Marketplace Reference Guidelines (if provided) give you domain-specific slot names and examples.
Use them to name variables precisely. Do NOT copy their literal content into questions.
Ground every question 100% in THIS app's concept. Never blend concepts from different domains.

ANTI-CONTAMINATION RULE: If the RAG context mentions "podcast episodes", "dating profiles", "crop diseases", 
"coding interviews" — and the user's app is NOT about those things — ignore those terms completely.

═══════════════════════════════════════
CRITICAL BEHAVIORAL RULES
═══════════════════════════════════════
- Ask exactly ONE question at a time, as natural conversational prose. No menus, no numbered lists.
- NEVER blend two domains into one question.
- NEVER ask about a slot that is already answered in "Already captured attributes".
- SEMANTIC ANSWER RECOGNITION: When checking if a slot is answered,
  treat these as equivalent:
  "tone" slot → answered by: formal, casual, friendly, professional,
                conversational, technical, simple, format, clear
  "output_format" slot → answered by: paragraph, bullets, summary,
                         structured, short, long, format, brief
  "content_type" slot → answered by: any domain word the user provides
  If the user's answer is a style/format word, treat the tone/format
  slot as ANSWERED. Do not re-ask it. Move to the next missing slot.
- TYPO AND NEAR-MATCH HANDLING: If the user's answer is a near-match
  or common typo of a valid answer, treat it as answered:
  "format" when asked about tone → treat as "formal"
  "casul" → "casual"
  "profesional" → "professional"
  "freindly" → "friendly"
  
  RULE: If the user's answer is within 1-2 characters of a valid
  option AND the slot question was just asked → accept it as answered.
  Never re-ask a slot because of a minor typo.
- SINGLE WORD SHORT ANSWERS: If the user gives a single word answer
  to any slot question, always treat it as answering that slot —
  even if it's ambiguous. Store it and move on.
  Never leave a slot as unanswered because the answer was too short.
- You MUST define exactly 3 distinct slots in the "slots" array at all times. Never return fewer than 3 slots.
- If all 3 slots are answered, set status = "ready".
- BANNED generic slot 3 fallbacks — never use these as a third question
  unless the user's app is explicitly about that topic:
  × "Who is the intended audience?"
  × "What is the target audience?"
  × "Who will use this app?"
  × "What is the purpose of the app?"
  These are meaningless for slot 3. Always ask something operational:
  content type, output format, language, length, speed, or domain topic.

Return strict JSON only (no markdown, no other text):
{
  "status": "needs_context|ready",
  "question": "Single natural prose question, or null if ready",
  "slot_key": "snake_case key for this slot, or null",
  "slots": [
    {"key": "slot_key_1", "question": "question for slot 1"},
    {"key": "slot_key_2", "question": "question for slot 2"},
    {"key": "slot_key_3", "question": "question for slot 3"}
  ],
  "corrected_app_type": "text|image|audio|video|vision",
  "domain_identified": "text|image|audio|video|vision|hybrid",
  "confidence_score": 0-100,
  "form": {
    "options": ["4 concise feature options"],
    "variables": [
      {
        "name": "variable name",
        "placeholder": "realistic placeholder",
        "test_value": "realistic test value"
      }
    ]
  }
}

VARIABLE CONFIGURATION RULES (when status = "ready"):
- 3–6 variables, atomic (one fact per variable), user-facing, directly fuel the prompt template.
- Title case names. Never use: input, text, data, param, details.
- Realistic placeholders. Date → 'DD/MM/YYYY', Location → 'City, Country'."""


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
    
    # 🛡️ General profile/calculation domain exceptions
    personal_input_domains = ("astrology", "horoscope", "birth chart", "zodiac", "compatibility", "numerology", "divination")
    if any(kw in p for kw in personal_input_domains):
        if any(term in n for term in ("birth date", "date of birth", "dob", "age", "birth time", "birth location")):
            return False
            
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
        
        # ─── 🛡️ THE PRODUCTION FIX: REPLACING HARDCODED DOMAIN BIAS ───
        if has_boilerplate_scrubbed or len(normalized) < min_len:
            type_lower = str(app_type or "text").lower()
            
            # If the LLM returns fewer variables than min_len, we gracefully pad it
            # with abstract structural properties, leaving the LLM's unique vertical fields alone.
            abstract_fallbacks = [
                {
                    "name": "Core Subject", 
                    "placeholder": f"Describe the primary topic or focal point of your {type_lower} app", 
                    "test_value": "General Domain Context"
                },
                {
                    "name": "Target Audience", 
                    "placeholder": "Who is the primary consumer or target user of this output?", 
                    "test_value": "General Audience"
                },
                {
                    "name": "Output Tone", 
                    "placeholder": "e.g., Professional, creative, conversational, technical...", 
                    "test_value": "Conversational"
                }
            ]
            
            for fallback_field in abstract_fallbacks:
                if len(normalized) >= max_len:
                    break
                
                key = fallback_field["name"].lower()
                if not any(k in key or key in k for k in seen):
                    seen.add(key)
                    normalized.append(fallback_field)
                    
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


async def extract_domain_noun(app_purpose: str, llm) -> str | None:
    """
    Dynamically extract the primary input artifact noun from app_purpose using LLM.
    This replaces all hardcoded domain noun lists permanently.
    Works for ANY domain without ever needing code changes.
    """
    if not app_purpose:
        return None

    messages = [
        {
            "role": "system",
            "content": (
                "You extract the primary input artifact a user will provide or upload to an AI app. "
                "If the app purpose is to generate, write, or create something new (e.g. workout plans, blog posts, essays) "
                "from a general text description or prompt rather than analyzing an uploaded input document, "
                "return null. The input artifact must be something the user provides or uploads to the app. "
                "Return ONLY valid JSON with one key: input_artifact (string or null). "
                "No explanation. No markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                f'App purpose: "{app_purpose}"\n\n'
                "What is the PRIMARY INPUT ARTIFACT the user uploads or provides to be analyzed/scored/processed? "
                "Remember: if it is a pure generation app (e.g. creates workout plans, writes blog posts, generates code) "
                "where the user only provides a topic prompt rather than an input document to analyze, return null.\n"
                "Examples:\n"
                '- "analyzes restaurant menu photos" → {"input_artifact": "menu"}\n'
                '- "evaluates startup pitch decks" → {"input_artifact": "pitch deck"}\n'
                '- "reads lab test reports" → {"input_artifact": "lab report"}\n'
                '- "reviews Airbnb listings" → {"input_artifact": "listing"}\n'
                '- "analyzes cricket scorecards" → {"input_artifact": "scorecard"}\n'
                '- "generates blog posts about travel" → {"input_artifact": null}\n'
                '- "creates workout plans" → {"input_artifact": null}\n'
                '- "generates workout plans" → {"input_artifact": null}\n'
                '- "writes blog posts" → {"input_artifact": null}\n\n'
                "Return JSON only: {\"input_artifact\": \"noun or null\"}"
            ),
        },
    ]

    if llm.has_groq:
        try:
            result = await llm.groq_completion(
                messages=messages,
                model="llama-3.1-8b-instant",
                response_format={"type": "json_object"},
                max_tokens=30,
                temperature=0.0,
            )
            content = result["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            noun = parsed.get("input_artifact")
            return str(noun).strip().lower() if noun and str(noun).strip().lower() != "null" else None
        except Exception as e:
            logger.warning(f"extract_domain_noun (Groq) failed: {e}. Trying OpenRouter fallback...")

    if llm.has_openrouter:
        try:
            content = await llm.openrouter_completion(
                messages=messages,
                model="meta-llama/llama-3.3-70b-instruct",
                max_tokens=30,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(content)
            noun = parsed.get("input_artifact")
            return str(noun).strip().lower() if noun and str(noun).strip().lower() != "null" else None
        except Exception as e:
            logger.warning(f"extract_domain_noun (OpenRouter fallback) failed: {e}")

    return None


def clean_leaked_rag_terms(text: str, app_purpose: str, domain_noun: str | None = None) -> str:
    """
    Replace foreign domain nouns in LLM output with the user's actual domain noun.
    domain_noun comes from extract_domain_noun() — no hardcoded lists needed.
    """
    if not text:
        return text
    if not domain_noun:
        return text

    # These are known RAG example nouns that frequently leak.
    # This list only needs to contain nouns that EXIST IN YOUR CHROMADB EXAMPLES.
    # It does NOT need to cover every possible user domain — that's handled by domain_noun.
    KNOWN_RAG_ARTIFACT_NOUNS = [
        "pitch deck", "startup deck", "investor deck",
        "resume", "cv",
        "invoice", "receipt",
        "podcast", "podcast episode", "episode",
        "dating profile",
        "crop disease", "plant disease",
        "linkedin profile",
        "menu",
        "screenshot",
        "product photo",
    ]

    domain_lower = domain_noun.lower().strip()

    for noun in KNOWN_RAG_ARTIFACT_NOUNS:
        # Only replace if it's a foreign noun (not the user's own domain noun)
        if noun.lower() == domain_lower:
            continue
        # Skip if the noun is actually part of the user's app purpose
        if noun.lower() in app_purpose.lower():
            continue
        pattern = re.compile(r"\b" + re.escape(noun) + r"\b", re.I)
        if pattern.search(text):
            text = pattern.sub(domain_noun, text)

    return text


def _parse_dynamic_context_payload(
    raw_content: str, app_type: str, app_purpose: str, language_hint: str
) -> dict:
    fallback = build_dynamic_context_fallback(app_type, app_purpose, language_hint)
    try:
        cleaned = re.sub(r"```json|```", "", str(raw_content or "{}"), flags=re.I).strip()
        parsed = json.loads(cleaned)
        
        # Clean leaked RAG terms inside variables list
        variables = parsed.get("variables") or []
        if isinstance(variables, list):
            for v in variables:
                if isinstance(v, dict):
                    for k in ["name", "label", "placeholder", "description", "test_value"]:
                        if v.get(k):
                            v[k] = clean_leaked_rag_terms(v[k], app_purpose)
                            
        return {
            "options": _sanitize_string_list(parsed.get("options"), 4, 4, fallback["options"]),
            "variables": _sanitize_variable_objects(
                variables, 3, 8, fallback["variables"], app_type, app_purpose
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
    raw_content: str, format_fallback: str, app_purpose: str, language_hint: str, domain_noun: str | None = None
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
            # Clean leaked RAG terms in main question
            question = clean_leaked_rag_terms(question, safe_purpose, domain_noun)
            
            if not question or len(question) < 10:
                fb_form = build_dynamic_context_fallback(fallback_type, safe_purpose, language_hint)
                return ready_shape(domain, fallback_type, fb_form, slots, confidence)
            suggested = None
            if isinstance(parsed.get("suggested_options"), list):
                opts = [str(o or "").strip() for o in parsed["suggested_options"] if str(o or "").strip()]
                suggested = opts[:6] if len(opts) >= 2 else None
            slot_key = str(parsed.get("slot_key") or "").strip().lower() or None
            
            # Clean leaked RAG terms in slots questions/keys
            for s in slots:
                if isinstance(s, dict):
                    if s.get("question"):
                        s["question"] = clean_leaked_rag_terms(s["question"], safe_purpose, domain_noun)
                    if s.get("key"):
                        s["key"] = clean_leaked_rag_terms(s["key"], safe_purpose, domain_noun).replace(" ", "_").lower()
            if slot_key:
                slot_key = clean_leaked_rag_terms(slot_key, safe_purpose, domain_noun).replace(" ", "_").lower()
                
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
   - Astrology app       → birth_date, birth_time, birth_location, zodiac_sign,
                           astrology_type, output_tone
   - Audiobook app       → book_genre, narrator_gender, narration_pace,
                           audio_format, target_listeners
   - Text to audio / TTS converter → content_type (educational/news/story/corporate),
                           voice_style, output_language, audio_speed, content_topic
   - Podcast script generator → episode_topic, episode_length, host_style,
                           target_audience, segment_structure
   - Study material to audio → subject_area, difficulty_level, narration_pace,
                           voice_gender, output_language
   - News/article reader → content_source, voice_tone, playback_speed,
                           output_language, summary_length
   Always infer from actual app purpose — never use these as defaults.

CRITICAL RAG ISOLATION & NO-LEAK RULE:
- The reference guidelines and examples (e.g., podcast episodes, resumes, pitch decks, dating profiles) are for structural style reference ONLY.
- Do NOT copy their literal domain terms, variable names, or concepts.
- Ground all variables 100% in the user's specific app concept. E.g., if the user wants an audiobook app, do NOT use "episode_length" or "podcast_genre" — use "narration_speed", "voice_style", etc.

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
            f"{_msg_role(m)}: {_msg_content(m)}"
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
    rag_context: str = "",
) -> dict:
    from services.clarification_planner import (
        recover_app_purpose,
        build_known_information,
        plan_clarification,
        log_clarification_trace,
        clarification_is_complete,
    )

    fallback_type = app_type if app_type in ALLOWED_TRIAGE_APP_FORMATS else "text"
    safe_purpose, recovered = recover_app_purpose(
        {"appPurpose": app_purpose}, app_purpose, conversation_history
    )
    if recovered:
        logger.info(f"[WORKFLOW] appPurpose recovered: {safe_purpose}")
    safe_purpose = str(safe_purpose or "").strip() or "general assistant app"
    safe_lang = _normalize_language_hint(language_hint)

    known = build_known_information({"appPurpose": safe_purpose}, deep_answers, safe_purpose)

    plan = await plan_clarification(
        llm=llm,
        app_purpose=safe_purpose,
        app_type=fallback_type,
        known_information=known,
        conversation_history=conversation_history,
        asked_keys=[],
        asked_questions=[],
        triage_rounds=0,
    )

    missing_items = plan.get("missing_information") or []
    known_items = plan.get("known_information") or []
    slots = [
        {"key": m["key"], "question": m["question"]} for m in missing_items
    ] + [
        {"key": k["key"], "question": k.get("value", "")} for k in known_items
    ]

    complete = clarification_is_complete(plan)
    log_clarification_trace(
        app_purpose=safe_purpose,
        plan=plan,
        clarification_complete=complete,
    )

    if not complete and plan.get("selected_question"):
        next_slot_key = plan["selected_key"]
        next_question = plan["selected_question"]

        domain_noun = await extract_domain_noun(safe_purpose, llm) if safe_purpose else None
        next_question = clean_leaked_rag_terms(next_question, safe_purpose, domain_noun)
        for s in slots:
            if s.get("question"):
                s["question"] = clean_leaked_rag_terms(s["question"], safe_purpose, domain_noun)
            if s.get("key"):
                s["key"] = clean_leaked_rag_terms(s["key"], safe_purpose, domain_noun).replace(" ", "_").lower()
        next_slot_key = clean_leaked_rag_terms(next_slot_key, safe_purpose, domain_noun).replace(" ", "_").lower()

        return {
            "status": "needs_context",
            "domain": "behavior_driven",
            "confidence_score": 100.0,
            "question": next_question,
            "slot_key": next_slot_key,
            "suggested_options": None,
            "corrected_app_type": fallback_type,
            "form": None,
            "app_format": None,
            "slots": slots,
        }

    form = await generate_dynamic_context(
        llm=llm,
        app_type=fallback_type,
        app_purpose=safe_purpose,
        language_hint=safe_lang,
        rag_context=rag_context,
        deep_answers=deep_answers,
        conversation_history=conversation_history,
    )

    return {
        "status": "ready",
        "domain": "behavior_driven",
        "confidence_score": 100.0,
        "question": None,
        "app_format": fallback_type,
        "slots": slots,
        "form": form,
    }


async def generate_dynamic_workflow(app_purpose: str, llm: LLMService) -> dict:
    """
    Legacy adapter — delegates to behavior-driven clarification planner.
    Returns workflow-shaped dict for backward compatibility.
    """
    from services.clarification_planner import plan_clarification, build_known_information

    plan = await plan_clarification(
        llm=llm,
        app_purpose=app_purpose,
        known_information=build_known_information({"appPurpose": app_purpose}, {}),
    )

    missing = plan.get("missing_information") or []
    known = plan.get("known_information") or []

    required_fields = [m["key"] for m in missing] + [k["key"] for k in known]
    field_questions = {m["key"]: m["question"] for m in missing}

    if not required_fields and not plan.get("ready"):
        return {
            "behavior_goal": plan.get("behavior_goal", ""),
            "required_fields": [],
            "field_questions": {},
        }

    return {
        "behavior_goal": plan.get("behavior_goal", ""),
        "required_fields": required_fields,
        "field_questions": field_questions,
    }


def _slot_is_captured(slot_key: str, captured_slots: dict) -> bool:
    """Check if a slot has been answered, using semantic key matching."""
    slot_key = slot_key.lower().strip()
    
    # Direct key match
    if slot_key in captured_slots:
        return True
    
    # Substring match on key names
    for k in captured_slots:
        if slot_key in k or k in slot_key:
            return True
    
    # Semantic alias matching — common slot name variations
    SEMANTIC_ALIASES = {
        "tone": ["tone", "writing_tone", "explanation_tone", "voice_tone",
                 "style", "writing_style", "output_tone", "format",
                 "formal", "casual", "friendly", "professional"],
        "voice_tone": ["tone", "voice", "style", "format", "writing_style"],
        "writing_tone": ["tone", "style", "format", "writing_style"],
        "output_format": ["format", "output_format", "output_style",
                           "structure", "layout"],
        "content_type": ["type", "content_type", "category", "domain",
                         "subject", "topic_type"],
        "input_format": ["format", "input", "how", "provide", "delivery"],
        "language": ["language", "lang", "locale"],
        "length": ["length", "word_count", "size", "long", "short"],
    }
    
    # Check if any alias of this slot_key matches a captured key
    for canonical, aliases in SEMANTIC_ALIASES.items():
        if slot_key == canonical or slot_key in aliases:
            for alias in aliases:
                if alias in captured_slots:
                    return True
                for k in captured_slots:
                    if alias in k or k in alias:
                        return True
    
    return False

