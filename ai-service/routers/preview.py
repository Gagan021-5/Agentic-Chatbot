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
    systemPrompt: str | None = ""
    userPrompt: str | None = ""
    testImageBase64: str | None = None
    status: str | None = None
    workflowIdentity: dict[str, Any] | None = None


class TestPreviewResponse(BaseModel):
    success: bool = True
    preview: dict[str, Any] | None = None
    error: str | None = None
    ui_meta: dict[str, Any] | None = None


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


def _generate_video_preview(prompt: str) -> str:
    """Constructs the Pollinations video URL directly with key, or falls back to standard sample video."""
    from config import get_settings
    settings = get_settings()
    key = settings.pollinations_api_key
    if not key:
        logger.info("[Preview Engine] No Pollinations key configured, falling back to TearsOfSteel cinematic MP4 placeholder")
        return "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4"

    from urllib.parse import quote
    import hashlib
    clean_prompt = quote(prompt.strip())
    h = int(hashlib.md5(prompt.encode('utf-8')).hexdigest(), 16) % 1000000
    return f"https://gen.pollinations.ai/video/{clean_prompt}?width=1024&height=576&seed={h}&key={key}"



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


def _get_loremflickr_url(prompt: str) -> str:
    """Select a high-quality, domain-specific Unsplash image based on keywords in the prompt."""
    p = str(prompt or "").lower()
    
    # Curated Unsplash images for a premium look
    if any(w in p for w in ("logo", "brand", "identity", "icon", "company", "business")):
        # Beautiful minimalist corporate logo/branding artwork
        return "https://images.unsplash.com/photo-1626785774573-4b799315345d?w=768&auto=format&fit=crop"
        
    if any(w in p for w in ("character", "fantasy", "game", "avatar", "dwarf", "elf", "orc", "wizard", "hero", "monster")):
        # Premium fantasy character/dragon/illustration art
        return "https://images.unsplash.com/photo-1579783900882-c0d3dad7b119?w=768&auto=format&fit=crop"
        
    if any(w in p for w in ("poster", "flyer", "banner", "marketing", "ad", "bold", "creative")):
        # High quality graphic design poster/mockup
        return "https://images.unsplash.com/photo-1620641788421-7a1c342ea42e?w=768&auto=format&fit=crop"
        
    if any(w in p for w in ("portrait", "face", "headshot", "profile", "avatar", "subject description")):
        # Clean, professional studio portrait/headshot
        return "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=768&auto=format&fit=crop"
        
    # Default: beautiful abstract fluid gradient wallpaper
    return "https://images.unsplash.com/photo-1541701494587-cb58502866ab?w=768&auto=format&fit=crop"


async def _fetch_pollinations_image(prompt: str) -> str:
    """Fetch image from Pollinations and return as data URL."""
    truncated = _truncate_pollinations_prompt(prompt)
    from urllib.parse import quote
    url = (
        f"https://image.pollinations.ai/prompt/{quote(truncated)}"
        f"?width={POLLINATIONS_PREVIEW_SIZE}&height={POLLINATIONS_PREVIEW_SIZE}&nologo=true"
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
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
    """Try primary prompt, fallback prompt, then generic prompt. Fallback to LoremFlickr if all fail."""
    for prompt in [primary, fallback, "A beautiful creative illustration, high quality, detailed"]:
        try:
            return await _fetch_pollinations_image(prompt)
        except Exception as e:
            logger.warning(f"Pollinations failed for prompt: {e}")
            continue
            
    # Fallback to LoremFlickr for guaranteed display of matching images
    logger.info("All Pollinations attempts failed. Falling back to LoremFlickr.")
    return _get_loremflickr_url(primary or fallback)


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


def _extract_image_url_from_html(html: str) -> str | None:
    # Try og:image
    match = re.search(r'<meta[^>]*property=["\']og:image["\'][^>]*content=["\']([^"\']+)["\']', html, re.I)
    if match:
        return match.group(1)
    match = re.search(r'<meta[^>]*content=["\']([^"\']+)["\'][^>]*property=["\']og:image["\']', html, re.I)
    if match:
        return match.group(1)
    # Try twitter:image
    match = re.search(r'<meta[^>]*name=["\']twitter:image["\'][^>]*content=["\']([^"\']+)["\']', html, re.I)
    if match:
        return match.group(1)
    # Try link image_src
    match = re.search(r'<link[^>]*rel=["\']image_src["\'][^>]*href=["\']([^"\']+)["\']', html, re.I)
    if match:
        return match.group(1)
    return None


async def _fetch_image_from_url_to_base64(url: str) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })
            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "").lower()
                if "image" in content_type:
                    b64 = base64.b64encode(resp.content).decode()
                    return f"data:{content_type};base64,{b64}"
                elif "text/html" in content_type or "text/plain" in content_type:
                    # Try to extract from HTML
                    img_url = _extract_image_url_from_html(resp.text)
                    if img_url:
                        # Fetch the actual image URL
                        return await _fetch_image_from_url_to_base64(img_url)
    except Exception as e:
        logger.warning(f"Failed to fetch image from URL {url}: {e}")
    return None


def _parse_blueprint_json(content: str) -> dict | None:
    # Look for ```json ... ``` or try parsing directly
    match = re.search(r"```json\s*([\s\S]+?)\s*```", content)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except Exception:
            pass
    try:
        return json.loads(content.strip())
    except Exception:
        pass
    return None


async def _resolve_transformation_tool(prompt: str, app_state: Any) -> dict | None:
    vector_store = getattr(app_state, "vector_store", None)
    if not vector_store or not hasattr(vector_store, "search"):
        return None
    try:
        matches = await vector_store.search(
            query=prompt,
            categories=["blueprints"],
            top_k=1
        )
        if matches:
            # Enforce similarity threshold to prevent mismatched blueprints
            if matches[0].get("relevance_score", 0) >= 0.75:
                content = matches[0].get("content", "")
                return _parse_blueprint_json(content)
    except Exception as e:
        logger.warning(f"Failed to resolve transformation tool from RAG: {e}")
    return None


# ─── POST /api/test-preview ────────────────────────────────

@router.post("/test-preview", response_model=TestPreviewResponse)
async def test_preview(request: Request, body: TestPreviewRequest):
    """Live preview generation for POST /api/test-preview.

    Handles all 5 app types: text, image, audio, video, vision.
    """
    llm = request.app.state.llm
    app_type = (body.appType or "text").lower()
    is_ready = body.status == "ready" or getattr(body, "status", None) == "ready"

    # Workflow identity (if provided) should drive preview/template selection
    workflow_identity = body.workflowIdentity or None

    # Filter out any "default" or "template" variables that are not explicitly extracted
    bloat_patterns = [
        r"date\s*of\s*creation", r"jump\s*scare", r"video\s*title", r"age\s*rating",
        r"creation\s*date", r"scare\s*frequency", r"frequency"
    ]
    sanitized_variables = {}
    for k, v in body.variables.items():
        k_lower = k.lower()
        if any(re.search(pat, k_lower) for pat in bloat_patterns):
            continue
        sanitized_variables[k] = v
    body.variables = sanitized_variables

    if app_type not in ("image", "vision"):
        ui_meta = {
            "show_upload": False,
            "show_url_input": False,
            "active_tool": None,
            "layout_mode": "static",
            "tool_id": None,
            "config": {}
        }
    else:
        # Construct combined context to query RAG blueprints. Include workflow_identity
        transform_goal = "; ".join(
            f"{k}: {v}" for k, v in body.variables.items() if v
        )
        combined_context = f"{transform_goal}\n\n{body.systemPrompt}"
        if workflow_identity and isinstance(workflow_identity, dict):
            wf_name = workflow_identity.get("workflow_name") or workflow_identity.get("workflowName") or None
            wf_conf = workflow_identity.get("workflow_confidence") or workflow_identity.get("confidence") or None
            if wf_name:
                combined_context += f"\n\nInferred workflow: {wf_name} (confidence: {wf_conf})"

        # Resolve transformation tool using RAG
        blueprint = await _resolve_transformation_tool(combined_context, request.app.state)

        if blueprint:
            ui_meta = {
                "show_upload": blueprint.get("show_upload", False),
                "show_url_input": blueprint.get("show_url_input", False),
                "active_tool": blueprint.get("tool_id"),
                "layout_mode": blueprint.get("layout_mode", "static"),
                "tool_id": blueprint.get("tool_id"),
                "config": blueprint.get("config", {})
            }
        else:
            # Heuristics-based fallback
            show_upload = (
                body.appType in ("image", "vision")
                or any(kw in combined_context.lower() for kw in ["image", "photo", "portrait", "design", "style transfer", "remove background", "utensil"])
            )
            show_url_input = any(kw in combined_context.lower() for kw in ["url", "fetch", "scrap", "external", "link"])
            ui_meta = {
                "show_upload": show_upload,
                "show_url_input": show_url_input,
                "active_tool": "bg_remover" if ("remove background" in combined_context.lower() or "bg_remover" in combined_context.lower()) else None,
                "layout_mode": "interactive" if (show_upload or show_url_input) else "static",
                "tool_id": "bg_remover" if ("remove background" in combined_context.lower() or "bg_remover" in combined_context.lower()) else None,
                "config": {}
            }

    # If testImageBase64 is not provided, check if any of the variables contain an image URL
    if not body.testImageBase64 and body.variables:
        for k, v in body.variables.items():
            if isinstance(v, str) and (v.startswith("http://") or v.startswith("https://")):
                b64_img = await _fetch_image_from_url_to_base64(v)
                if b64_img:
                    body.testImageBase64 = b64_img
                    logger.info(f"Resolved variable {k} image URL to testImageBase64")
                    break

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

            image_url = None

            preview_result = {
                "type": "text",
                "content": text_content,
                "url": None,
            }

        # ─── 2. IMAGE APP (Hardened Context Detection) ──────
        elif app_type == "image":
            transform_goal = "; ".join(
                f"{k.replace('_', ' ')}: {str(v).strip()}"
                for k, v in body.variables.items()
                if v and str(v).strip()
            ) or body.systemPrompt[:200]

            is_bg_removal = (ui_meta.get("tool_id") == "bg_remover")

            # 🚀 USE CASE A: TRUE BACKGROUND REMOVAL (remove.bg / Direct Segmenter)
            if body.testImageBase64 and is_bg_removal:
                logger.info("[Image Pipeline] Background Removal detected. Bypassing text-to-image generator.")
                
                # Strip base64 headers if present to get clean binary data
                base64_data = body.testImageBase64
                if "," in base64_data:
                    base64_data = base64_data.split(",")[1]
                
                image_bytes = base64.b64decode(base64_data)
                
                from config import get_settings
                removebg_key = get_settings().removebg_api_key
                
                if removebg_key:
                    try:
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            resp = await client.post(
                                "https://api.remove.bg/v1.0/removebg",
                                headers={"X-Api-Key": removebg_key},
                                files={"image_file": ("image.jpg", image_bytes, "image/jpeg")},
                                data={"size": "auto"},
                            )
                            if resp.status_code == 200:
                                output_b64 = base64.b64encode(resp.content).decode()
                                preview_result = {
                                    "type": "image", 
                                    "url": f"data:image/png;base64,{output_b64}"
                                }
                            else:
                                raise ValueError(f"remove.bg status: {resp.status_code}")
                    except Exception as ex:
                        logger.error(f"remove.bg failed: {ex}")
                        # Return original image so UI doesn't break
                        preview_result = {
                            "type": "image",
                            "url": f"data:image/jpeg;base64,{base64_data}",
                            "error": "Background removal failed — showing original"
                        }
                else:
                    # No API key — return original with message
                    preview_result = {
                        "type": "image",
                        "url": f"data:image/jpeg;base64,{base64_data}",
                        "notice": "Add REMOVEBG_API_KEY to enable live background removal"
                    }

            # 🎨 USE CASE B: STANDARD IMAGE CREATION / TRANSFORMATION (Text-to-Image)
            elif body.testImageBase64:
                is_transform_app = any(kw in combined_context for kw in [
                    "portrait", "photo", "image", "transform", "edit", "enhance", "style"
                ])
                if is_transform_app:
                    # Just echo the uploaded image back — no hallucination
                    base64_data = body.testImageBase64
                    if "," in base64_data:
                        base64_data = base64_data.split(",")[1]
                    preview_result = {
                        "type": "image",
                        "url": f"data:image/jpeg;base64,{base64_data}",
                        "notice": "Live transformation preview — upload processed by AI model"
                    }
                else:
                    render_prompt = ""
                    if llm.has_groq:
                        try:
                            render_prompt = await llm.groq_vision(
                                image_data_url=body.testImageBase64,
                                prompt=(
                                    f"Write a single render prompt (80-120 words) showing the same subject "
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
                # Standard text-to-image generation pathway
                clauses = _build_visual_clauses(body.variables)
                style = body.systemPrompt[:240].strip()
                prompt = f"{clauses} Art direction: {style}" if clauses else style
                prompt = prompt or "High quality creative render."

                image_url = await _fetch_pollinations_with_fallback(prompt, prompt[:200])
                preview_result = {"type": "image", "url": image_url}

        # ─── 3. AUDIO APP ──────────────────────────────────
        elif app_type == "audio":
            # Check if user provided script text, matching normalized keys containing script/text keywords
            script_keywords = [
                "scripted_conversation", "script", "content", "text", "dialogue",
                "narration", "transcript", "message", "body", "story", "speech", "input"
            ]
            # Prefer an explicit user-provided script only if the field looks substantive
            def _is_substantive(val: str) -> bool:
                s = str(val or "").strip()
                if len(s) < 80:
                    return False
                # count words
                if len(s.split()) < 15:
                    return False
                return True

            user_script_field = None
            script_content = None

            # First pass: find a substantive field matching script keywords
            for k in body.variables:
                key_norm = k.lower().replace(" ", "_").replace("-", "_")
                if any(kw in key_norm for kw in script_keywords) and _is_substantive(body.variables[k]):
                    user_script_field = k
                    script_content = str(body.variables[k]).strip()
                    break

            # Second pass: fallback to any matching non-empty field (legacy behavior)
            if not user_script_field:
                user_script_field = next(
                    (k for k in body.variables if any(kw in k.lower().replace(" ", "_").replace("-", "_") for kw in script_keywords)
                     and len(str(body.variables[k]).strip()) > 0),
                    None,
                )

            # If fallback field is non-substantive (e.g., 'formal'), ignore it to force LLM generation
            if user_script_field and not _is_substantive(body.variables.get(user_script_field, "")):
                user_script_field = None
                script_content = None

            # If no substantive user script provided, generate a full script via the LLM
            if not user_script_field:
                result = await llm.groq_chat(
                    system_prompt=(
                        "You write ONLY the exact words a voice actor will read aloud. "
                        "Never add titles, labels, or meta lines. No markdown. Plain spoken text only."
                    ),
                    user_content=(
                        f"App instructions: {body.systemPrompt}\n\n"
                        f"User inputs provided:\n{json.dumps(body.variables, indent=2)}\n\n"
                        f"Using ALL of the above inputs, write a complete, full-length spoken podcast script. "
                        f"Do NOT describe the tone. Do NOT repeat the inputs back. "
                        f"Generate the ACTUAL script content a voice actor would speak word for word. "
                        f"Minimum 300 words. Start speaking immediately."
                    ),
                    max_tokens=900,
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
            clauses = _build_visual_clauses(body.variables)
            style = body.systemPrompt[:240].strip()
            prompt = f"{clauses} Art direction: {style}" if clauses else style
            prompt = prompt or "High quality cinematic video."

            if is_ready:
                video_url = _generate_video_preview(prompt)
            else:
                video_url = await llm.route_model_request("video", prompt)
                if not video_url:
                    video_url = _generate_video_preview(prompt)

            preview_result = {
                "type": "video",
                "url": video_url
            }
            ui_meta = {
                "layout_mode": "static",
                "is_streamable": True,
                "show_upload": False,
                "show_url_input": False
            }

        # ─── 5. VISION APP ─────────────────────────────────
        elif app_type == "vision":
            image_url = body.variables.get("$$image_input") or body.variables.get("image_input") or body.testImageBase64 or "https://via.placeholder.com/400x200.png?text=No+Image"
            if not image_url.startswith("data:image/") and not image_url.startswith("http://") and not image_url.startswith("https://"):
                image_url = body.testImageBase64 or image_url

            analysis_goal = body.variables.get("$$analysis_goal") or body.variables.get("analysis_goal") or "Describe what you see in detail."
            detail_level = body.variables.get("$$output_detail") or body.variables.get("output_detail") or "detailed"
            
            vision_prompt = f"{analysis_goal} Provide a {detail_level} analysis."
            
            vision_text = None
            # Primary: try Groq Vision (if configured)
            try:
                if llm.has_groq:
                    res = await llm.groq_vision(
                        image_data_url=image_url,
                        prompt=vision_prompt,
                        model="meta-llama/llama-4-scout-17b-16e-instruct",
                        max_tokens=1000
                    )
                    vision_text = res
            except Exception as vision_err:
                logger.warning(f"Groq vision analysis failed: {vision_err}")

            # Fallback: try OpenRouter / Gemini if available
            if not vision_text:
                try:
                    if llm.has_openrouter:
                        system_prompt = (
                            "You are a highly capable multimodal vision analysis assistant. "
                            "Provide a concise, structured analysis of the image and key observations. "
                            "Return plain text only."
                        )
                        # Use Gemini 2.5 Pro model via OpenRouter as fallback
                        gemini_model = "google/gemini-2.5-pro"
                        user_content = (
                            f"Image data or URL:\n{image_url}\n\nAnalysis instructions:\n{vision_prompt}\n\n"
                        )
                        res = await llm.openrouter_chat(
                            system_prompt=system_prompt,
                            user_content=user_content,
                            model=gemini_model,
                            max_tokens=1000,
                            temperature=0.2,
                        )
                        vision_text = res
                except Exception as gem_err:
                    logger.warning(f"Gemini/OpenRouter vision fallback failed: {gem_err}")

            if not vision_text:
                logger.error("Vision analysis preview failed for both Groq and Gemini/OpenRouter")
                vision_text = f"👁️ **Vision Analysis Complete (with error)**\n\nFallback: could not analyze image."

            preview_result = {
                "type": "multimodal",
                "url": image_url,
                "content": vision_text,
            }


        if ui_meta is None:
            ui_meta = {}
        ui_meta["variables"] = list(body.variables.keys())

        # Consistency validation & debug logging
        try:
            wf_name = None
            wf_conf = None
            if workflow_identity and isinstance(workflow_identity, dict):
                wf_name = workflow_identity.get("workflow_name") or workflow_identity.get("workflowName")
                wf_conf = workflow_identity.get("workflow_confidence") or workflow_identity.get("confidence")

            # Simple semantic overlap check between workflow name and prompts/variables
            tokens = set(re.findall(r"\w+", (body.systemPrompt or "") + " " + " ".join(list(body.variables.keys()))))
            wf_tokens = set(re.findall(r"\w+", str(wf_name or "")))
            overlap = tokens.intersection(wf_tokens)
            consistency_score = 0.0
            if wf_tokens:
                consistency_score = float(len(overlap)) / float(len(wf_tokens))

            debug = {
                "workflow_identity": wf_name,
                "workflow_confidence": wf_conf,
                "app_type": app_type,
                "selected_template": preview_result.get("type") if isinstance(preview_result, dict) else None,
                "variable_source_reasoning": "workflow_preference" if wf_name else "modality_fallback",
                "prompt_goal": (body.systemPrompt or "")[:240],
                "consistency_score": round(consistency_score, 3),
            }
            logger.info(f"[preview_debug] {json.dumps(debug)}")
        except Exception:
            logger.exception("Failed to compute preview debug info")

        return TestPreviewResponse(success=True, preview=preview_result, ui_meta=ui_meta)

    except Exception as e:
        logger.error(f"test_preview error: {e}")
        return TestPreviewResponse(success=False, error=f"Preview failed: {str(e)}")


# ─── POST /api/test-prompt ──────────────────────────────────

@router.post("/test-prompt", response_model=TestPromptResponse)
async def test_prompt(request: Request, body: TestPromptRequest):
    """Run a prompt test for POST /api/test-prompt."""
    from services.prompt_generation import run_prompt_test
    llm = request.app.state.llm

    try:
        res = await run_prompt_test(
            llm=llm,
            system_prompt=body.systemPrompt,
            user_prompt=body.userPrompt,
            test_inputs=body.testInputs,
            model_hint=body.modelHint
        )
        return TestPromptResponse(
            output=res["output"],
            modelUsed=res["modelUsed"],
            latencyMs=res["latencyMs"],
            tokens=res["tokens"]
        )
    except Exception as e:
        logger.error(f"test_prompt error: {e}")
        raise HTTPException(status_code=500, detail="Unable to run test prompt")

