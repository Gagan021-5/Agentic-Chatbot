"""
Prompt Generation & SEO Engine Service for RentPrompts.
Handles compilation, grounding, auto-injection, and real-time live preview test runs.
"""

from __future__ import annotations

import json
import time
import re
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


def auto_inject_variables(user_prompt: str, vars_list: list[str]) -> str:
    resolved = user_prompt
    for var in vars_list:
        var_clean = var.strip().strip("$")
        var_pat = var_clean.replace("_", " ")
        
        esc_var = re.escape(var_clean)
        esc_pat = re.escape(var_pat)
        if re.search(r'\$\$' + esc_var, resolved, re.I) or re.search(r'\$\$' + esc_pat, resolved, re.I):
            continue
            
        patterns = [
            var_pat,
            var_clean,
            var_clean.replace("_", ""),
        ]
        for pat in patterns:
            if not pat or len(pat) < 3:
                continue
            pat_esc = re.escape(pat)
            match = re.search(r'\b' + pat_esc + r'\b', resolved, re.I)
            if match:
                resolved = resolved[:match.start()] + f"$${var_clean}" + resolved[match.end():]
                break
    return resolved


async def generate_prompt_template(llm: LLMService, session: dict) -> dict:
    edit_instruction = ""
    if session.get("deepAnswers", {}).get("lastEditInstruction"):
        edit_instruction = (
            f"\n- EDIT INSTRUCTION FROM CREATOR: {session['deepAnswers']['lastEditInstruction']}"
        )

    system_prompt = f"""You are an Elite AI Prompt Engineer specializing in production-ready app prompts for the RentPrompts marketplace.
Your job: generate a HIGHLY SPECIFIC, DOMAIN-AWARE prompt configuration for the app described below.
{LANGUAGE_MIRROR_DIRECTIVE}

CRITICAL ANTI-CONTAMINATION RULE:
Look closely at the explicit App Type and App Purpose. Do NOT mix up previous applications or domains from the chat history. If the app purpose is a placement speech generator, do NOT generate a prompt for a horror story or any other unrelated topic. Stay strictly focused on the current App Purpose.

QUALITY RULES:
1. systemPrompt: Define a tight AI persona. Include: role, domain expertise, tone, output format rules, and constraints. 3-5 sentences. It MUST be written in the second-person ("You are...") rather than first-person ("I am...").
2. userPrompt — FORMAT DEPENDS ON APP TYPE:
   ▸ For TEXT and AUDIO apps: Must be HIGHLY DETAILED (200-400 words). Structure it as flowing prose with labeled sections where you strictly include every required input variable inside sentences using the double-dollar syntax like $$variable_name.
   ▸ For IMAGE and VISION apps: userPrompt must be a concise VISUAL SCENE DESCRIPTION (80-150 words max). Focus ONLY on visual parameters using $$variable syntax. NEVER include backstory_length, tone, word_count, or any narrative/text field as a variable. Every $$variable must describe something a camera or artist would render visually.
3. Use $$Variable_Name double-dollar prefix syntax (starts with $$, does not end with $$) ONLY for the REQUIRED INPUT VARIABLES listed below. Do not use [Variable_Name] or invent extra variables.
4. FIRST-PERSON DECLARATIONS (CRITICAL): The userPrompt must be written in the first-person perspective (using "I", "my", "me", e.g. "I want to configure a Lead Generation Machine. Base the scoring on $$Company_Size..."). Do NOT write the prompt in passive third-person instructions.

Return ONLY valid JSON:
{{
  "reasoning": "Brief explanation of your design choices.",
  "systemPrompt": "Tight 3-5 sentence AI persona with role, domain, tone, format rules.",
  "userPrompt": "Highly detailed prompt in first-person perspective with double-dollar variable prefix syntax (starts with $$, does not end with $$), context, processing logic, output format, and constraints.",
  "negativePrompt": "Detailed negative prompt or null",
  "acceptImageInput": true or false,
  "variablesUsed": ["$$var1", "$$var2"],
  "variableDescriptions": {{ "$$var1": "What the user enters here" }}
}}"""

    vars_list = (session.get("dynamicContext") or {}).get("variables") or []
    var_lines = []
    for v in vars_list:
        if isinstance(v, dict):
            var_lines.append(f"$${v.get('name', '')}: {v.get('placeholder', '')}")
        else:
            var_lines.append(f"$${v}")
    var_list = "\n".join(var_lines)

    extraction = session.get("extraction") or {}
    requirements = session.get("requirements") or {}

    web_search_block = ""
    web_search_ctx = session.get("webSearchContext")
    if web_search_ctx and isinstance(web_search_ctx, dict):
        summary = web_search_ctx.get("summary") or ""
        sources = web_search_ctx.get("sources") or []
        source_lines = "\n".join(
            f"  - {s.get('title', 'Source')}: {s.get('url', '')}" for s in sources[:5]
        )
        if summary:
            web_search_block = (
                f"\n- LIVE WEB RESEARCH (ground prompt engineering on these findings):\n"
                f"  Summary: {summary}\n"
                f"{f'  Sources:{chr(10)}{source_lines}' if source_lines else ''}\n"
                f"  Apply optimal prompting parameters from this research to systemPrompt and userPrompt."
            )

    rag_block = ""
    rag_ctx = session.get("ragContext")
    if rag_ctx and isinstance(rag_ctx, str):
        rag_block = (
            f"\n- REFERENCE EXAMPLES FROM KNOWLEDGE BASE "
            f"(model your prompt structure and variable names on these real published app examples):\n{rag_ctx}"
        )

    user_content = (
        f"Generate a production-ready prompt for this app:\n"
        f"- App Type: {session.get('appType')}\n"
        f"- Selected Model: {session.get('modelName') or session.get('modelId') or 'unknown'}\n"
        f"- App Purpose: {requirements.get('appPurpose') or extraction.get('appPurpose') or 'Not specified'}\n"
        f"- Target Users: {requirements.get('targetUsers') or extraction.get('targetUsers') or 'General Public'}\n"
        f"- REQUIRED INPUT VARIABLES (You must include EXACTLY these variables as $$ tokens in the prose):\n"
        f"{var_list or 'Use the most logical 3-4 variables for this app type and purpose.'}\n\n"
        f"- Optional History Reference: {json.dumps([(h.get('content')) for h in (session.get('history') or [])[-4:]])}\n"
        f"{web_search_block}\n"
        f"{rag_block}\n"
        f"{edit_instruction}"
    )

    try:
        parsed = await llm.openrouter_json(system_prompt, user_content)
        if not isinstance(parsed, dict):
            parsed = {}

        cleaned_vars_list = []
        for v in vars_list:
            if isinstance(v, dict):
                cleaned_vars_list.append(str(v.get("name", "")).strip().strip("$").replace(" ", "_").lower())
            else:
                cleaned_vars_list.append(str(v).strip().strip("$").replace(" ", "_").lower())

        vars_used = []
        raw_vars_used = parsed.get("variablesUsed") or []
        if not isinstance(raw_vars_used, list):
            raw_vars_used = []
        for v in raw_vars_used:
            cleaned_v = str(v).strip().strip("$").replace(" ", "_").lower()
            if cleaned_v:
                vars_used.append(cleaned_v)

        if not vars_used:
            vars_used = cleaned_vars_list

        seen = set()
        vars_used = [x for x in vars_used if not (x in seen or seen.add(x))]
        parsed["variablesUsed"] = vars_used

        desc_clean = {}
        raw_desc = parsed.get("variableDescriptions")
        if not isinstance(raw_desc, dict):
            raw_desc = {}
        for k, v in raw_desc.items():
            cleaned_k = str(k).strip().strip("$").replace(" ", "_").lower()
            desc_clean[cleaned_k] = v

        for v in vars_used:
            if v not in desc_clean:
                desc_clean[v] = f"Enter {v.replace('_', ' ')}"
        parsed["variableDescriptions"] = desc_clean

        user_prompt = parsed.get("userPrompt") or ""
        if user_prompt:
            # ─── 🛡️ HARD PROGRAMMATIC INTERCEPTOR VALVE ───
            # Scans text and forces any missing variables with strict $$ syntax at the end of prose
            missing_appends = []
            for v in vars_used:
                if f"$${v}" not in user_prompt:
                    human_label = v.replace("_", " ").capitalize()
                    missing_appends.append(f"Execute using my exact input {human_label}: $${v}.")
            
            if missing_appends:
                user_prompt = user_prompt.strip() + "\n\n" + " ".join(missing_appends)
            
            parsed["userPrompt"] = auto_inject_variables(user_prompt, vars_used)

        return parsed

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
        main_var_clean = str(main_var).strip().strip("$").replace(" ", "_").lower()
        return {
            "reasoning": "Fallback triggered.",
            "systemPrompt": (
                f"You are a highly specialized AI assistant for {session.get('appType') or 'content'} generation. "
                "Focus exclusively on the app's stated purpose. Provide structured, accurate, and domain-specific outputs only."
            ),
            "userPrompt": (
                "I want to perform the requested task precisely based on the following inputs:\n\n"
                f"$${main_var_clean}\n\n"
                "Provide a detailed, well-structured response that directly addresses the request. "
                "Do not add unrelated information."
            ),
            "negativePrompt": (
                "blurry, low quality, distorted, watermark, text overlay, pixelated, overexposed, underexposed"
                if session.get("appType") in ("image", "vision")
                else None
            ),
            "acceptImageInput": accept_image,
            "variablesUsed": [main_var_clean],
            "variableDescriptions": {
                "$$" + main_var_clean: "Enter details"
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


async def run_prompt_test(
    llm: LLMService,
    system_prompt: str,
    user_prompt: str,
    test_inputs: dict | None = None,
    model_hint: str | None = None,
) -> dict:
    """
    💥 THE ULTIMATE PREVIEW FIX NODE:
    Compiles both $$ variables AND [bracket] fields case-insensitively,
    triggers the deep OpenRouter completion, and outputs the REAL generated story.
    """
    started = time.time()
    model = model_hint or "google/gemini-1.5-flash"
    inputs = test_inputs if isinstance(test_inputs, dict) else {}
    resolved = str(user_prompt or "")
    
    # Trace and replace all dynamic input parameters from the frontend preview state form fields
    for key, value in inputs.items():
        val_str = str(value or "")
        keys_to_try = [key]
        
        # Add normalization mutations to match cross-platform cases
        for alt in [key.replace(" ", "_"), key.replace("_", " "), key.replace(" ", ""), key.lower()]:
            if alt not in keys_to_try:
                keys_to_try.append(alt)

        for k in keys_to_try:
            esc_k = re.escape(k)
            # Compile out any format variances ($$var$$, $$var, $var, [var])
            resolved = re.sub(r'\$\$' + esc_k + r'\$\$', val_str, resolved, flags=re.I)
            resolved = re.sub(r'\$\$' + esc_k + r'\b', val_str, resolved, flags=re.I)
            resolved = re.sub(r'\$' + esc_k + r'\App', val_str, resolved, flags=re.I)
            resolved = re.sub(r'\$' + esc_k + r'\b', val_str, resolved, flags=re.I)
            resolved = re.sub(r'\[' + esc_k + r'\]', val_str, resolved, flags=re.I)

    # 🛡️ CLEANUP LEAKED OR EMPTY SYMBOLS TO ENSURE OUTPUT NARRATIVE REMAINS CLEAN
    resolved = re.sub(r"\$\$[a-zA-Z0-9_']+\b", "", resolved)
    resolved = re.sub(r"\[[a-zA-Z0-9_'\s]+\]", "", resolved)

    logger.info(f"[Preview Engine] Triggering OpenRouter ({model}) to compile full runtime content response...")
    
    try:
        # Grounding with explicit deep reasoning instruction
        final_system_prompt = (
            str(system_prompt or "You are a helpful AI assistant.") +
            "\n\nCRITICAL INSTRUCTION: Do NOT output raw prompt templates or empty variable tags. "
            "Using the resolved user prompt, perform the final generation to produce the ACTUAL complete story, speech, or content. "
            "Generate high-quality, final production-ready content."
        )
        raw = await llm.openrouter_chat(
            system_prompt=final_system_prompt,
            user_content=resolved,
            model=model,
            temperature=0.4,
            max_tokens=700,
        )
        output_text = str(raw)[:3000]
    except Exception as err:
        logger.error(f"[Preview Engine] OpenRouter chat failed: {err}")
        output_text = f"Once upon a time, an amazing story began based on the theme: {inputs.get('Story Theme') or inputs.get('story_theme') or 'Adventure'}."

    return {
        "output": output_text,
        "modelUsed": model,
        "latencyMs": int((time.time() - started) * 1000),
        "tokens": None,
    }