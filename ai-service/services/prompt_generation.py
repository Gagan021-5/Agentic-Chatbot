"""
Prompt template and SEO generation — RentPrompts gemini.js.
Uses LLMService OpenRouter.
"""

from __future__ import annotations

import json
import time
from typing import Any

from loguru import logger

from services.language_directive import LANGUAGE_MIRROR_DIRECTIVE
from services.llm import LLMService

UPLOAD_KEYWORDS = [
    "background remov", "background replac", "room design", "interior design",
    "redesign", "style transfer", "face swap", "portrait", "enhance photo",
    "edit photo", "photo editing", "image editing", "remove background",
    "plant disease", "crop disease", "object detect", "image analys",
    "my photo", "my image", "my room", "my product", "photo to", "image to",
]

GENERATION_KEYWORDS = [
    "logo", "poster", "flyer", "banner", "thumbnail creator", "album art",
    "icon generator", "brand logo", "text to image", "generate image",
    "create image", "fantasy", "landscape generator", "ai art generator",
]


def apply_prompt_instruction(prompt_data: dict | None, instruction: str) -> dict:
    clean = (instruction or "").strip()
    base = prompt_data or {
        "systemPrompt": "",
        "userPrompt": "",
        "negativePrompt": None,
        "acceptImageInput": False,
        "variablesUsed": [],
        "variableDescriptions": {},
    }
    if not clean:
        return base
    current = str(base.get("userPrompt") or "")
    suffix = f" Additional instruction: {clean}."
    next_prompt = current if clean in current else f"{current}{suffix}"
    return {**base, "userPrompt": next_prompt}


def build_prompt_template_from_session(session: dict) -> dict:
    return {
        "userPrompt": "Write about $$topic in a $$tone tone for $$audience at $$length length, optimized for $$goal.",
        "negativePrompt": None,
        "acceptImageInput": False,
        "variablesUsed": ["topic", "tone", "audience", "length", "goal"],
        "advancedSettings": {"aspectRatio": None, "quality": "balanced"},
    }
async def run_prompt_test(
    llm: LLMService,
    system_prompt: str,
    user_prompt: str,
    test_inputs: dict | None = None,
    model_hint: str | None = None,
) -> dict:
    started = time.time()
    model = model_hint or "google/gemini-1.5-flash"
    inputs = test_inputs if isinstance(test_inputs, dict) else {}
    resolved = str(user_prompt or "")
    
    # Support both $$variable and [variable] resolving case-insensitively
    for key, value in inputs.items():
        val_str = str(value or "")
        resolved = re.sub(re.escape(f"$${key}"), val_str, resolved, flags=re.I)
        resolved = re.sub(re.escape(f"[{key}]"), val_str, resolved, flags=re.I)
        
        # Also try replacing underscores in key to cover both styles
        key_alt = key.replace(" ", "_")
        resolved = re.sub(re.escape(f"$${key_alt}"), val_str, resolved, flags=re.I)
        resolved = re.sub(re.escape(f"[{key_alt}]"), val_str, resolved, flags=re.I)

    raw = await llm.openrouter_chat(
        system_prompt=str(system_prompt or "You are a helpful AI assistant."),
        user_content=resolved,
        model=model,
        temperature=0.4,
        max_tokens=700,
    )
    return {
        "output": str(raw)[:3000],
        "modelUsed": model,
        "latencyMs": int((time.time() - started) * 1000),
        "tokens": None,
    }


async def generate_prompt_template(llm: LLMService, session: dict) -> dict:
    edit_instruction = ""
    if session.get("deepAnswers", {}).get("lastEditInstruction"):
        edit_instruction = (
            f"\n- EDIT INSTRUCTION FROM CREATOR: {session['deepAnswers']['lastEditInstruction']}"
        )

    system_prompt = f"""You are an Elite AI Prompt Engineer specializing in production-ready app prompts for the RentPrompts marketplace.
Your job: generate a HIGHLY SPECIFIC, DOMAIN-AWARE prompt configuration for the app described below.
{LANGUAGE_MIRROR_DIRECTIVE}

QUALITY RULES:
1. systemPrompt: Define a tight AI persona. Include: role, domain expertise, tone, output format rules, and constraints. 3-5 sentences. It MUST be written in the second-person ("You are...") rather than first-person ("I am...").
2. userPrompt — FORMAT DEPENDS ON APP TYPE:
   ▸ For TEXT and AUDIO apps: Must be HIGHLY DETAILED (200-400 words). Structure it as:
     a) Instructions for the LLM as the expert using the [Variable_Name] variables ("Analyze the user's [Variable_Name]...", "Based on the topic [Variable_Name]...")
     b) Step-by-step processing logic
     c) Output format specification (headings, bullets, structure)
     d) Constraints (what NOT to do)
     DO NOT use markdown headers (##) inside the prompt — write it as flowing prose with labeled sections.
   ▸ For IMAGE and VIDEO apps: Must be a CONCISE VISUAL DESCRIPTION (50-120 words). Write it as:
     - A single flowing visual prompt describing the desired output
     - Incorporate [Variable_Name] variables naturally: "A [style] [subject] with [details]"
     - Include art direction: lighting, camera angle, color palette, mood
     - End with quality keywords: "professional photography, 8K, ultra-detailed"
     - DO NOT use ## headers, numbered steps, or "Processing Logic" — image generators don't read structured prompts
3. Use [Variable_Name] bracket syntax ONLY for the REQUIRED INPUT VARIABLES listed below. Do not use $$ prefix or invent extra variables.
4. negativePrompt: For image/video apps, write a detailed negative prompt. For text/audio/vision, set to null.
5. acceptImageInput: SMART DETECTION — do NOT blindly set true for all image apps.
6. NO META-PLATFORM DETAILS: NEVER mention the user's budget, coin cost, or model name in the systemPrompt or userPrompt.
7. DOMAIN GROUNDING: If the app is legal/medical/agricultural, explicitly define domain-specific terms in systemPrompt.
8. DO NOT write generic prompts. Every sentence must be specific to THIS app's purpose.
9. NEVER include in prompts: model names, coin costs, budget tiers, platform names ("RentPrompts"), or any internal metadata.
10. USER PERSPECTIVE PRINCIPLE (CRITICAL): Variables must reflect what a NON-EXPERT end-user can actually provide.
11. CONFLICT DETECTION & ANTI-HALLUCINATION (CRITICAL for legal/medical/expert apps).
12. EXPERT INSTRUCTIONS (CRITICAL): For text apps, the userPrompt must instruct the model to act as the expert and process the input variables (e.g. "Analyze the [birth_sign] and [dob] to generate daily horoscope..."). Do NOT write the prompt in the first-person perspective (do NOT write "I want...", "My sign is...").

Return ONLY valid JSON:
{{
  "reasoning": "Brief explanation of your design choices.",
  "systemPrompt": "Tight 3-5 sentence AI persona with role, domain, tone, format rules.",
  "userPrompt": "Highly detailed prompt with context, processing logic, output format, and constraints.",
  "negativePrompt": "Detailed negative prompt or null",
  "acceptImageInput": true or false,
  "variablesUsed": ["[var1]", "[var2]"],
  "variableDescriptions": {{ "[var1]": "What the user enters here" }}
}}"""

    vars_list = (session.get("dynamicContext") or {}).get("variables") or []
    var_lines = []
    for v in vars_list:
        if isinstance(v, dict):
            var_lines.append(f"[{v.get('name', '')}]: {v.get('placeholder', '')}")
        else:
            var_lines.append(f"[{v}]")
    var_list = "\n".join(var_lines)

    extraction = session.get("extraction") or {}
    requirements = session.get("requirements") or {}
    user_content = (
        f"Generate a production-ready prompt for this app:\n"
        f"- App Type: {session.get('appType')}\n"
        f"- App Purpose: {requirements.get('appPurpose') or extraction.get('appPurpose') or 'Not specified'}\n"
        f"- Target Users: {requirements.get('targetUsers') or extraction.get('targetUsers') or 'General Public'}\n"
        f"- Domain Context from Chat: {json.dumps(session.get('deepAnswers') or {})}\n"
        f"- Conversation Summary: {json.dumps([(h.get('content')) for h in (session.get('history') or [])[-6:]])}\n"
        f"- REQUIRED INPUT VARIABLES (use EXACTLY these, no more, no less):\n"
        f"{var_list or 'Use the most logical 3-4 variables for this app type and purpose.'}"
        f"{edit_instruction}"
    )

    try:
        import re
        return await llm.openrouter_json(system_prompt, user_content)
    except Exception as err:
        logger.error(f"[generate_prompt_template] Error: {err}")
        app_purpose = (
            requirements.get("appPurpose") or extraction.get("appPurpose") or ""
        ).lower()
        needs_upload = any(k in app_purpose for k in UPLOAD_KEYWORDS)
        is_generation = any(k in app_purpose for k in GENERATION_KEYWORDS)
        accept_image = (
            session.get("appType") == "vision"
            or (session.get("appType") == "image" and needs_upload and not is_generation)
        )
        main_var = vars_list[0].get("name") if vars_list and isinstance(vars_list[0], dict) else "main_input"
        return {
            "reasoning": "Fallback triggered.",
            "systemPrompt": (
                f"You are a highly specialized AI assistant for {session.get('appType') or 'content'} generation. "
                "Focus exclusively on the app's stated purpose. Provide structured, accurate, and domain-specific outputs only."
            ),
            "userPrompt": (
                "Based on the following inputs, perform the requested task precisely:\n\n"
                f"[{main_var}]\n\n"
                "Provide a detailed, well-structured response that directly addresses the request. "
                "Do not add unrelated information."
            ),
            "negativePrompt": (
                "blurry, low quality, distorted, watermark, text overlay, pixelated, overexposed, underexposed"
                if session.get("appType") in ("image", "vision")
                else None
            ),
            "acceptImageInput": accept_image,
            "variablesUsed": [
                f"[{v.get('name') if isinstance(v, dict) else v}]"
                for v in vars_list[:3]
            ],
            "variableDescriptions": {
                f"[{v.get('name') if isinstance(v, dict) else v}]": (
                    v.get("placeholder") if isinstance(v, dict) else "Enter details"
                )
                for v in vars_list[:3]
            },
        }


async def generate_seo(llm: LLMService, session: dict) -> dict:
    system_prompt = f"""You are an expert Product Marketer and App Store Optimization (ASO) specialist for a premium AI app marketplace.
Your job: generate irresistible, high-converting marketplace metadata that makes users WANT to try the app instantly.
{LANGUAGE_MIRROR_DIRECTIVE}

STRICT RULES — follow every single one:

1. APP NAME (2-4 words MAXIMUM): catchy, memorable, premium SaaS product name. Max 55 characters.
2. DESCRIPTION (under 150 characters): benefit-driven copy, NOT a feature list.
3. TAGS (EXACTLY 7 tags): lowercase, hyphenated, no # prefix.
4. CATEGORY: one of: creative, business, education, healthcare, entertainment, productivity, social, other

Return ONLY valid JSON:
{{
  "appName": "string, 2-4 words, max 55 chars",
  "appDescription": "string, benefit-driven, max 150 chars",
  "tags": ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7"],
  "category": "creative | business | education | healthcare | entertainment | productivity | social | other"
}}"""

    extraction = session.get("extraction") or {}
    app_purpose = extraction.get("appPurpose") or extraction.get("oneLineUnderstanding") or ""
    deep_answers = session.get("deepAnswers") or {}
    triage_history = " | ".join(
        str(m.get("content", ""))
        for m in (session.get("history") or [])
        if m.get("role") == "user"
    )[:600]

    user_content = (
        f"Generate premium, high-converting marketplace metadata for this AI app:\n"
        f"- What the app does: {app_purpose}\n"
        f"- App type: {session.get('appType')}\n"
        f"- User answers during setup: {json.dumps(deep_answers)}\n"
        f"- Conversation context: {triage_history}\n"
        f"- Model used (DO NOT use as app name — this is internal only): {session.get('modelId') or 'unknown'}\n\n"
        "Remember: App name must sound like a premium SaaS product (2-4 words). "
        "Description must sell the benefit to the user in under 150 characters. Exactly 7 action-oriented tags."
    )

    try:
        result = await llm.openrouter_json(system_prompt, user_content)
        logger.info("[generate_seo] SEO generated OK")
        return result
    except Exception as err:
        logger.error(f"[generate_seo] Error: {err}")
        purpose_clean = (app_purpose or "").strip()
        type_label = session.get("appType") or "creative"
        name_map = {
            "image": "PixelForge AI",
            "text": "CopyFlow AI",
            "audio": "VoiceCraft AI",
            "video": "ReelSpark AI",
            "vision": "InsightLens AI",
        }
        desc_map = {
            "image": f"Create stunning AI-powered visuals{' — ' + purpose_clean if purpose_clean else ''} in seconds.",
            "text": f"Craft polished, professional content{' — ' + purpose_clean if purpose_clean else ''} with AI.",
            "audio": f"Transform text into natural, professional audio{' — ' + purpose_clean if purpose_clean else ''} instantly.",
            "video": f"Produce scroll-stopping AI videos{' — ' + purpose_clean if purpose_clean else ''} effortlessly.",
            "vision": f"Analyze and extract insights from images{' — ' + purpose_clean if purpose_clean else ''} with AI.",
        }
        tag_map = {
            "image": ["ai-design", "visual-creator", "brand-design", "image-craft", "creative-ai", "instant-design", "smart-visuals"],
            "text": ["ai-writing", "content-creator", "smart-copy", "professional-writing", "instant-content", "creative-ai", "productivity-boost"],
            "audio": ["ai-voice", "text-to-speech", "audio-creator", "voice-studio", "podcast-tool", "smart-audio", "creative-ai"],
            "video": ["ai-video", "reel-maker", "video-creator", "motion-design", "content-studio", "creative-ai", "instant-video"],
            "vision": ["ai-vision", "image-analysis", "smart-scan", "visual-ai", "insight-tool", "data-extraction", "intelligent-scan"],
        }
        return {
            "appName": name_map.get(type_label, "SparkAI Studio"),
            "appDescription": (desc_map.get(type_label) or f"Unlock AI-powered {type_label} creation — fast, polished, professional results.")[:150],
            "tags": tag_map.get(type_label) or [
                "ai-powered", "smart-tool", "instant-results", "creative-ai",
                "productivity", "no-code", "professional",
            ],
            "category": "creative",
        }
