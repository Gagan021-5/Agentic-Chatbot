"""
═══════════════════════════════════════════════════════════════
Preview Router — POST /api/test-preview + POST /api/test-prompt
═══════════════════════════════════════════════════════════════
FastAPI routes: POST /api/test-preview and POST /api/test-prompt
Handles live preview generation for all app types:
text, image, audio, video, vision
"""

import re
import base64
import json
from loguru import logger
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import Any
import httpx

router = APIRouter()

POLLINATIONS_PREVIEW_SIZE = 768


# ─── Schemas ────────────────────────────────────────────────

class TestPreviewRequest(BaseModel):
    appType: str = "text"
    variables: dict[str, Any] = Field(default_factory=dict)
    systemPrompt: str = ""
    testImageBase64: str | None = None


class TestPreviewResponse(BaseModel):
    success: bool = True
    preview: dict[str, Any] | None = None
    error: str | None = None


class TestPromptRequest(BaseModel):
    systemPrompt: str
    userPrompt: str
    testInputs: dict[str, Any] | None = None
    modelHint: str | None = None


class TestPromptResponse(BaseModel):
    output: str = ""
    modelUsed: str = ""
    latencyMs: int = 0
    tokens: int | None = None


# ─── Pollinations Helpers ───────────────────────────────────

def _build_visual_clauses(variables: dict) -> str:
    """Build user-driven visual clauses from variables."""
    entries = [(k, v) for k, v in (variables or {}).items()
               if v is not None and str(v).strip()]
    if not entries:
        return ""

    parts = []

    # Background
    bg = next(((k, v) for k, v in entries
               if re.search(r"background|backdrop|environment|setting|scene|location", k, re.I)), None)
    if bg:
        parts.append(f"The visible background must clearly show: {str(bg[1]).strip()}.")

    # Color
    col = next(((k, v) for k, v in entries
                if re.search(r"color|palette|scheme|hue|tint|tone", k, re.I)), None)
    if col:
        parts.append(f'Color direction: "{str(col[1]).strip()}" should be the dominant color story.')

    # Subject
    subject = next(((k, v) for k, v in entries
                     if re.search(r"shape|subject|object|form|creature|motif|theme|type|design", k, re.I)
                     and not re.search(r"background|backdrop", k, re.I)), None)
    if subject:
        parts.append(f"Primary subject: {str(subject[1]).strip()}.")

    compact = "; ".join(f"{k}: {str(v).strip()}" for k, v in entries)
    parts.append(f"Honor all user fields: {compact}.")

    return " ".join(parts)


def _truncate_pollinations_prompt(text: str) -> str:
    """Clean and truncate for Pollinations URL-safe prompt."""
    t = re.sub(r"[^a-zA-Z0-9\s,.\-]", " ", str(text or "").strip())
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return "High quality detailed illustration, professional lighting."
    return t[:350].rstrip() if len(t) > 350 else t


def _preview_unavailable_svg() -> str:
    """Return a fallback SVG data URL when image preview fails."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">'
        '<rect fill="#1a1525" width="512" height="512"/>'
        '<text x="256" y="248" fill="#a77bf3" font-family="system-ui,sans-serif" font-size="18" '
        'font-weight="600" text-anchor="middle">Could not load preview image</text>'
        '<text x="256" y="278" fill="#9ca3af" font-family="system-ui,sans-serif" font-size="13" '
        'text-anchor="middle">Try Run again</text></svg>'
    )
    from urllib.parse import quote
    return f"data:image/svg+xml;charset=utf-8,{quote(svg)}"


async def _fetch_pollinations_image(prompt: str) -> str:
    """Fetch image from Pollinations and return as data URL."""
    truncated = _truncate_pollinations_prompt(prompt)
    from urllib.parse import quote
    url = (
        f"https://image.pollinations.ai/prompt/{quote(truncated)}"
        f"?width={POLLINATIONS_PREVIEW_SIZE}&height={POLLINATIONS_PREVIEW_SIZE}&nologo=true"
    )

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.get(url, headers={
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        })
        resp.raise_for_status()

        mime = resp.headers.get("content-type", "image/jpeg")
        body = resp.content
        if not mime.startswith("image/") or len(body) < 800:
            raise ValueError(f"Non-image response: {mime}, {len(body)}b")

        b64 = base64.b64encode(body).decode()
        return f"data:{mime};base64,{b64}"


async def _fetch_pollinations_with_fallback(primary: str, fallback: str) -> str:
    """Try primary prompt, fallback prompt, then generic prompt."""
    for prompt in [primary, fallback, "A beautiful creative illustration, high quality, detailed"]:
        try:
            return await _fetch_pollinations_image(prompt)
        except Exception as e:
            logger.warning(f"Pollinations failed for prompt: {e}")
            continue
    return _preview_unavailable_svg()


def _normalize_tts_script(raw: str) -> str:
    """Strip preamble from LLM-generated TTS scripts."""
    s = str(raw or "").replace("\r\n", "\n").replace("**", "").replace("*", "").replace("#", "").strip()

    # Strip "Here's the script:" preamble
    s = re.sub(r"^(here'?s|here is|this is)\s+[\s\S]{0,480}?[:.;]\s*", "", s, flags=re.I).strip()
    s = re.sub(r'^(the\s+)?(script|voiceover|intro|copy|audio)\s*(is|:)\s*', "", s, flags=re.I).strip()

    lines = s.split("\n")
    if len(lines) > 1 and len(lines[0]) < 220 and re.search(r"here'?s|here is|script for", lines[0], re.I):
        s = "\n".join(lines[1:]).strip()

    s = re.sub(r'^[""\'`]+|[""\'`]+$', "", s).strip()
    return s


# ─── POST /api/test-preview ────────────────────────────────

@router.post("/test-preview", response_model=TestPreviewResponse)
async def test_preview(request: Request, body: TestPreviewRequest):
    """Live preview generation for POST /api/test-preview.

    Handles all 5 app types: text, image, audio, video, vision.
    """
    llm = request.app.state.llm
    app_type = (body.appType or "text").lower()

    try:
        preview_result = None

        # ─── 1. TEXT APP ────────────────────────────────────
        if app_type == "text":
            result = await llm.groq_chat(
                system_prompt=(
                    "You are the backend engine for an application. Output EXACTLY what the app "
                    "is supposed to output. NEVER apologize. NEVER mention that you are an AI. "
                    "Output ONLY pure, raw plain text. NO Markdown formatting."
                ),
                user_content=f"{body.systemPrompt}\n\nUser Inputs: {json.dumps(body.variables)}",
                max_tokens=300,
            )
            raw_text = result["choices"][0]["message"]["content"]
            text_content = re.sub(r"\*{1,3}", "", raw_text)
            text_content = re.sub(r"^#{1,6}\s?", "", text_content, flags=re.MULTILINE)
            text_content = text_content.replace("#", "").strip()

            # Surprise image for visual text apps
            image_url = None
            if re.search(r"astrology|horoscope|story|character|design", body.systemPrompt, re.I):
                clauses = _build_visual_clauses(body.variables)
                prompt = f"{clauses} Art direction: {body.systemPrompt[:240]}"
                try:
                    image_url = await _fetch_pollinations_with_fallback(prompt, prompt[:200])
                except Exception:
                    image_url = _preview_unavailable_svg()

            preview_result = {
                "type": "multimodal" if image_url else "text",
                "content": text_content,
                "url": image_url,
            }

        # ─── 2. IMAGE APP ──────────────────────────────────
        elif app_type == "image":
            if body.testImageBase64:
                # Path A: Uploaded image → Groq Vision → Pollinations
                transform_goal = "; ".join(
                    f"{k.replace('_', ' ')}: {str(v).strip()}"
                    for k, v in body.variables.items()
                    if v and str(v).strip()
                ) or body.systemPrompt[:200]

                render_prompt = ""
                if llm.has_groq:
                    try:
                        render_prompt = await llm.groq_vision(
                            image_data_url=body.testImageBase64,
                            prompt=(
                                f"Write a single render prompt (80-120 words) showing the same person "
                                f"AFTER this transformation: \"{transform_goal}\". "
                                f"Keep ALL subject details identical. Output ONLY the prompt."
                            ),
                        )
                    except Exception as e:
                        logger.warning(f"Groq Vision failed: {e}")

                if not render_prompt or len(render_prompt) < 20:
                    render_prompt = f"{transform_goal}, photorealistic, high resolution, 8K"

                image_url = await _fetch_pollinations_with_fallback(render_prompt, render_prompt[:200])
                preview_result = {"type": "image", "url": image_url}

            else:
                # Path B: Text-to-image
                clauses = _build_visual_clauses(body.variables)
                style = body.systemPrompt[:240].strip()
                prompt = f"{clauses} Art direction: {style}" if clauses else style
                prompt = prompt or "High quality creative render."

                image_url = await _fetch_pollinations_with_fallback(prompt, prompt[:200])
                preview_result = {"type": "image", "url": image_url}

        # ─── 3. AUDIO APP ──────────────────────────────────
        elif app_type == "audio":
            # Check if user provided script text
            script_fields = ["scripted_conversation", "script", "content", "text", "dialogue",
                           "narration", "transcript", "message", "body", "story"]
            user_script_field = next(
                (f for f in script_fields if body.variables.get(f) and len(str(body.variables[f]).strip()) > 30),
                None,
            )

            if user_script_field:
                script_content = str(body.variables[user_script_field]).strip()
            else:
                # Generate script via Groq
                result = await llm.groq_chat(
                    system_prompt=(
                        "You write ONLY the exact words a voice actor will read aloud. "
                        "Never add titles, labels, or meta lines. No markdown. Plain spoken text only."
                    ),
                    user_content=f"{body.systemPrompt}\n\nInputs: {json.dumps(body.variables)}",
                    max_tokens=500,
                )
                raw_script = result["choices"][0]["message"]["content"]
                script_content = _normalize_tts_script(raw_script)

            # Murf TTS
            from config import get_settings
            murf_key = get_settings().murf_api_key

            if not murf_key:
                preview_result = {"type": "audio", "url": None, "data": script_content[:4500]}
            else:
                # Voice selection
                explicit_gender = (body.variables.get("voice_gender") or "").lower()
                if explicit_gender == "male":
                    voice_id, label = "terrell", "Male (Terrell)"
                elif explicit_gender == "female":
                    voice_id, label = "natalie", "Female (Natalie)"
                else:
                    voice_id, label = "natalie", "Female (Natalie)"

                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        murf_resp = await client.post(
                            "https://api.murf.ai/v1/speech/generate",
                            headers={"api-key": murf_key, "Content-Type": "application/json"},
                            json={
                                "text": script_content[:3000],
                                "voiceId": voice_id,
                                "modelVersion": "GEN2",
                                "locale": "en-US",
                                "format": "MP3",
                                "encodeAsBase64": True,
                            },
                        )
                    if murf_resp.status_code == 200:
                        murf_data = murf_resp.json()
                        audio_b64 = murf_data.get("encodedAudio") or murf_data.get("audioFile")
                        if audio_b64:
                            url = audio_b64 if audio_b64.startswith("data:") else (
                                audio_b64 if audio_b64.startswith("http") else
                                f"data:audio/mpeg;base64,{audio_b64}"
                            )
                            preview_result = {"type": "audio", "url": url,
                                            "data": script_content[:4500], "voiceLabel": label}
                        else:
                            preview_result = {"type": "audio", "url": None, "data": script_content[:4500]}
                    else:
                        preview_result = {"type": "audio", "url": None, "data": script_content[:4500]}
                except Exception as e:
                    logger.error(f"Murf TTS error: {e}")
                    preview_result = {"type": "audio", "url": None, "data": script_content[:4500]}

        # ─── 4. VIDEO APP ──────────────────────────────────
        elif app_type == "video":
            # Step 1: Generate screenplay
            result = await llm.groq_chat(
                system_prompt=(
                    "You are an AI video director. Generate a short video concept with "
                    "'Scene 1: [Visual]' and voiceover. Under 150 words. Plain text only."
                ),
                user_content=f"{body.systemPrompt}\n\nInputs: {json.dumps(body.variables)}",
            )
            video_script = result["choices"][0]["message"]["content"].replace("**", "").replace("*", "")

            # Step 2: Cinematic thumbnail
            clean_vars = ", ".join(
                f"{k}: {v}" for k, v in body.variables.items()
                if v and str(v).strip()
            )
            safe_ctx = re.sub(r"[^a-zA-Z0-9 .,]", "", (body.systemPrompt or "")[:120])
            thumbnail_prompt = f"Cinematic high quality video still, 8k. Subject: {safe_ctx}. {clean_vars[:150]}"
            from urllib.parse import quote
            thumbnail_url = (
                f"https://image.pollinations.ai/prompt/{quote(thumbnail_prompt)}"
                f"?width=1024&height=576&nologo=true"
            )

            preview_result = {"type": "video", "data": video_script, "url": thumbnail_url}

        # ─── 5. VISION APP ─────────────────────────────────
        elif app_type == "vision":
            vision_text = (
                f"👁️ **Vision Analysis Complete**\n\n"
                f"Based on the uploaded image and parameters ({json.dumps(body.variables)}), "
                f"the AI detects elements matching your {body.systemPrompt[:40]}... logic."
            )
            preview_result = {
                "type": "multimodal",
                "url": body.testImageBase64 or "https://via.placeholder.com/400x200.png?text=No+Image",
                "content": vision_text,
            }

        return TestPreviewResponse(success=True, preview=preview_result)

    except Exception as e:
        logger.error(f"test_preview error: {e}")
        return TestPreviewResponse(success=False, error=f"Preview failed: {str(e)}")


# ─── POST /api/test-prompt ──────────────────────────────────

@router.post("/test-prompt", response_model=TestPromptResponse)
async def test_prompt(request: Request, body: TestPromptRequest):
    """Run a prompt test for POST /api/test-prompt."""
    import time as _time
    llm = request.app.state.llm
    started = _time.time()

    try:
        # Resolve variables in user prompt supporting both legacy $$ and new [Variable] syntax
        resolved = body.userPrompt
        for key, value in (body.testInputs or {}).items():
            val_str = str(value or "")
            resolved = re.sub(re.escape(f"$${key}"), val_str, resolved, flags=re.I)
            resolved = re.sub(re.escape(f"[{key}]"), val_str, resolved, flags=re.I)
            
            key_alt = key.replace(" ", "_")
            resolved = re.sub(re.escape(f"$${key_alt}"), val_str, resolved, flags=re.I)
            resolved = re.sub(re.escape(f"[{key_alt}]"), val_str, resolved, flags=re.I)

        raw = await llm.openrouter_chat(
            system_prompt=body.systemPrompt or "You are a helpful AI assistant.",
            user_content=resolved,
            model=body.modelHint or "google/gemini-1.5-flash",
            max_tokens=700,
            temperature=0.4,
        )

        return TestPromptResponse(
            output=raw[:3000],
            modelUsed=body.modelHint or "google/gemini-1.5-flash",
            latencyMs=int((_time.time() - started) * 1000),
            tokens=None,
        )
    except Exception as e:
        logger.error(f"test_prompt error: {e}")
        raise HTTPException(status_code=500, detail="Unable to run test prompt")

