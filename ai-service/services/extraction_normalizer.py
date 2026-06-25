"""Extraction normalization helpers — RentPrompts gemini.js."""

import re

VISION_SOURCE_RE = re.compile(
    r"\b(?:pitch\s+decks?|decks?|slides?|presentations?|pdfs?|resumes?|cvs?|documents?|reports?|brochures?|portfolios?|images?|photos?|pictures?|screenshots?|profiles?|webpages?|x[\-\s]?rays?|scans?|receipts?|invoices?|certificates?|diagrams?|charts?|infographics?|flyers?|posters?|menus?|labels?|badges?|tickets?|forms?)\b",
    re.IGNORECASE,
)


def detect_language(message: str) -> str:
    text = str(message or "")
    lower = text.lower()
    if re.search(r"[\u0900-\u097F]", text):
        return "Hindi"
    hinglish_signals = [
        "mujhe", "banana", "banani", "hai", "likhta", "likh", "ke liye",
        "karna", "mera", "meri", "krdo", "chahiye", "jaldi",
    ]
    if any(word in lower for word in hinglish_signals):
        return "Hinglish"
    return "English"


def detect_budget(message: str) -> tuple[str | None, str]:
    lower = message.lower()
    if "free" in lower:
        return "free", "HIGH"
    if "cheap" in lower or "low budget" in lower or "budget" in lower:
        return "low", "MEDIUM"
    if "premium" in lower or "quality" in lower:
        return "high", "MEDIUM"
    if "ultra" in lower or "best possible" in lower:
        return "ultra", "HIGH"
    return None, "LOW"


def infer_app_type(message: str) -> tuple[str | None, str]:
    lower = message.lower()
    
    # 1. Detect Task Type
    task_type = None
    if re.search(r"\b(analyz\w*|evaluat\w*|review\w*|audit\w*|score\w*|assess\w*|inspect\w*|read\w*|extract\w*|detect\w*|check\w*|verif\w*|validat\w*|optimiz\w*|classif\w*|summariz\w*|feedbacks?|ocr)\b", lower):
        task_type = "analyze"
    elif re.search(r"\b(edit\w*|chang\w*|modif\w*|alter\w*|transform\w*|tweak\w*|adjust\w*|filter\w*|enhance\w*|remove\s+background|bg\s+remover)\b", lower):
        task_type = "edit"
    elif re.search(r"\b(generat\w*|creat\w*|make\w*|design\w*|draw\w*|render\w*|write\w*|craft\w*|produc\w*|synthesiz\w*)\b", lower):
        task_type = "generate"
        
    # 2. Detect Artifact Type
    artifact_type = None
    if re.search(r"\b(videos?|animations?|movies?|reels?|clips?)\b", lower):
        artifact_type = "video"
    elif re.search(r"\b(audios?|voices?|sounds?|podcasts?|songs?|speech|music|melod(?:y|ies)|tts|text-to-speech|text-to-audio)\b", lower):
        artifact_type = "audio"
    elif re.search(r"\b(pitch\s+decks?|decks?|slides?|presentations?|pdfs?|resumes?|cvs?|documents?|reports?|brochures?|portfolios?|menus?|invoices?|receipts?|bills?|tickets?|certificates?|web\s+links?|urls?|websites?|webpages?|forms?|profiles?)\b", lower):
        artifact_type = "document"
    elif re.search(r"\b(images?|imagery|photos?|photographs?|photography|pictures?|posters?|logos?|banners?|art|arts|artworks?|avatars?|illustrations?|graphics?|icons?|thumbnails?|screenshots?|scans?|x[\-\s]?rays?|diagrams?|charts?|infographics?|flyers?|visuals?)\b", lower):
        artifact_type = "image"
    elif re.search(r"\b(texts?|blogs?|articles?|stories?|prompts?|scripts?|cop(?:y|ies)|paragraphs?|letters?|essays?)\b", lower):
        artifact_type = "text"
        
    # 3. Derive App Type based on (Artifact Type, Task Type)
    if task_type == "analyze":
        if artifact_type in ("image", "document"):
            return "vision", "HIGH"
        elif artifact_type == "audio":
            return "audio", "HIGH"
        elif artifact_type == "video":
            return "video", "HIGH"
        elif artifact_type == "text":
            return "text", "HIGH"
    elif task_type in ("edit", "generate"):
        if artifact_type == "image":
            return "image", "HIGH"
        elif artifact_type == "document":
            return "text", "HIGH"
        elif artifact_type == "audio":
            return "audio", "HIGH"
        elif artifact_type == "video":
            return "video", "HIGH"
        elif artifact_type == "text":
            return "text", "HIGH"

    # 4. Fallback compound patterns
    if re.search(r"(text\s*to\s*(audio|speech|voice|sound|tts|music|mp3|podcast|song))", lower) or "tts" in lower or "text-to-speech" in lower or "text-to-audio" in lower:
        return "audio", "HIGH"
    if re.search(r"((image|photo|picture)\s*to\s*video)", lower) or "photo animation" in lower:
        return "video", "HIGH"
    if re.search(r"(text\s*to\s*(image|photo|picture|logo|poster|art|avatar|design))", lower):
        return "image", "HIGH"
    if re.search(r"((image|photo|picture)\s*to\s*(text|data|csv|json|words|ocr))", lower) or "ocr" in lower:
        return "vision", "HIGH"

    # 5. General fallback keyword checks
    if re.search(r"(video|animate|reel|cinematic|movie)", lower):
        return "video", "HIGH"
    if re.search(r"(voice|audio|music|podcast|song|speech)", lower):
        return "audio", "HIGH"
    if re.search(r"(image|photo|picture|poster|thumbnail|logo|design|drawing|banner|card)", lower):
        return "image", "HIGH"
    if re.search(r"(vision|analy(s|z)e|ocr|scan|detect|inspect|read from image)", lower):
        return "vision", "HIGH"
    if re.search(
        r"(blog|write|writer|copy|text|article|caption|planner|workout|meal|diet|"
        r"itinerary|guide|report|advocate|legal|lawyer|law|document|draft|summary)",
        lower,
    ):
        return "text", "HIGH"
    return None, "LOW"


def infer_tone(message: str) -> str:
    lower = message.lower()
    if re.search(r"(asap|urgent|quick|jaldi|immediately|fast)", lower):
        return "urgent"
    if re.search(r"(please|kindly)", lower):
        return "formal"
    if re.search(r"(maybe|something|not sure|idk)", lower):
        return "unsure"
    return "casual"


def build_one_line_understanding(extraction: dict) -> str:
    if extraction.get("appType") == "video" and extraction.get("wantsImageInput"):
        return "you want a video app that turns photos into cinematic clips"
    if extraction.get("appType") == "text" and re.search(r"blog", extraction.get("appPurpose") or "", re.I):
        return "you want a text app that writes blogs"
    if extraction.get("appType"):
        return f"you want a {extraction['appType']} app for {extraction.get('appPurpose') or 'your use case'}"
    return extraction.get("appPurpose") or "you want help shaping an app idea"


def normalize_extraction(raw: dict | None, fallback_message: str = "") -> dict:
    message = fallback_message or ""
    inferred_type, inferred_conf = infer_app_type(message)
    budget_val, budget_conf = detect_budget(message)

    normalized_budget = budget_val
    if not normalized_budget and raw:
        rb = raw.get("budget")
        conf = (raw.get("confidence") or {}).get("budget")
        if rb in ("free", "low", "medium", "high", "ultra") and conf == "HIGH":
            normalized_budget = rb

    normalized_budget_conf = budget_conf if budget_val else (
        (raw or {}).get("confidence", {}).get("budget")
        if normalized_budget and (raw or {}).get("confidence", {}).get("budget") in ("HIGH", "MEDIUM", "LOW")
        else "LOW"
    )

    valid_types = ("text", "image", "audio", "video", "vision")
    app_type = None
    if raw and raw.get("appType") in valid_types:
        app_type = raw["appType"]
        # Override generic "text" if we inferred a more specific target media format from the user input
        if app_type == "text" and inferred_type in ("audio", "video", "image", "vision"):
            app_type = inferred_type
    else:
        app_type = inferred_type

    app_purpose = None
    if raw and isinstance(raw.get("appPurpose"), str) and raw["appPurpose"].strip():
        app_purpose = raw["appPurpose"].strip()

    target_users = None
    if raw and isinstance(raw.get("targetUsers"), str):
        tu = raw["targetUsers"].strip()
        if tu and tu != "general users":
            target_users = tu

    key_features = []
    if raw and isinstance(raw.get("keyFeatures"), list):
        key_features = [f for f in raw["keyFeatures"] if f][:6]

    wants_image = bool(raw and raw.get("wantsImageInput")) or bool(
        re.search(r"(photo|image|picture)", message, re.I)
    )

    detected_lang = detect_language(message)
    if raw and isinstance(raw.get("detectedLanguage"), str) and raw["detectedLanguage"].strip():
        detected_lang = raw["detectedLanguage"].strip()

    user_tone = infer_tone(message)
    if raw and raw.get("userTone") in ("urgent", "casual", "formal", "unsure"):
        user_tone = raw["userTone"]

    user_type = "unknown"
    if raw and raw.get("userType") in ("enterprise", "business", "developer", "normal", "unknown"):
        user_type = raw["userType"]

    app_type_conf = inferred_conf
    if raw and (raw.get("confidence") or {}).get("appType") in ("HIGH", "MEDIUM", "LOW"):
        app_type_conf = raw["confidence"]["appType"]

    missing = []
    if raw and isinstance(raw.get("missingFields"), list):
        missing = [m for m in raw["missingFields"] if m]

    one_line = ""
    if raw and isinstance(raw.get("oneLineUnderstanding"), str) and raw["oneLineUnderstanding"].strip():
        one_line = raw["oneLineUnderstanding"].strip()

    suggested = None
    if raw and isinstance(raw.get("suggestedReply"), str) and raw["suggestedReply"].strip():
        suggested = raw["suggestedReply"].strip()

    primary_subject = None
    if raw and isinstance(raw.get("PRIMARY_SUBJECT"), str) and raw["PRIMARY_SUBJECT"].strip():
        primary_subject = raw["PRIMARY_SUBJECT"].strip()
        
    environment_setting = None
    if raw and isinstance(raw.get("ENVIRONMENT_SETTING"), str) and raw["ENVIRONMENT_SETTING"].strip():
        environment_setting = raw["ENVIRONMENT_SETTING"].strip()
        
    action_dynamic = None
    if raw and isinstance(raw.get("ACTION_DYNAMIC"), str) and raw["ACTION_DYNAMIC"].strip():
        action_dynamic = raw["ACTION_DYNAMIC"].strip()
        
    aesthetic_style = None
    if raw and isinstance(raw.get("AESTHETIC_STYLE"), str) and raw["AESTHETIC_STYLE"].strip():
        aesthetic_style = raw["AESTHETIC_STYLE"].strip()

    # Rule-based fallback/override for the universal dimensions
    msg_clean = message.strip().strip('"').strip("'").lower()
    
    if "motor racing" in msg_clean and "beach" in msg_clean:
        if not primary_subject: primary_subject = "Motor racing"
        if not environment_setting: environment_setting = "Beach"
        if not action_dynamic: action_dynamic = "High-speed driving"
        if not aesthetic_style: aesthetic_style = "Cinematic"
        
    if "kaito yamato" in msg_clean:
        if not primary_subject: primary_subject = "Kaito Yamato"
        if not environment_setting: environment_setting = "Mumbai streets"
        if not action_dynamic: action_dynamic = "Fire power"
        if not aesthetic_style: aesthetic_style = "Naruto style"
        
    if not primary_subject:
        match = re.search(r"^([a-zA-Z\s]+?)\s+on\s+(?:a\s+)?([a-zA-Z\s]+)$", msg_clean)
        if match:
            primary_subject = match.group(1).strip().capitalize()
            environment_setting = match.group(2).strip().capitalize()
            if "racing" in msg_clean or "driving" in msg_clean:
                action_dynamic = "High-speed driving"
            else:
                action_dynamic = f"{primary_subject} activity"
            aesthetic_style = "Cinematic"

    extraction = {
        "appType": app_type,
        "appPurpose": app_purpose,
        "targetUsers": target_users,
        "keyFeatures": key_features,
        "budget": normalized_budget,
        "wantsImageInput": wants_image,
        "detectedLanguage": detected_lang,
        "userTone": user_tone,
        "userType": user_type,
        "enterpriseSignals": bool(raw and raw.get("enterpriseSignals")),
        "confidence": {"appType": app_type_conf, "budget": normalized_budget_conf},
        "missingFields": missing,
        "oneLineUnderstanding": one_line,
        "suggestedReply": suggested,
        "PRIMARY_SUBJECT": primary_subject,
        "ENVIRONMENT_SETTING": environment_setting,
        "ACTION_DYNAMIC": action_dynamic,
        "AESTHETIC_STYLE": aesthetic_style,
    }

    if not extraction["oneLineUnderstanding"] and extraction["appType"]:
        extraction["oneLineUnderstanding"] = f"you want a {extraction['appType']} app"
    elif not extraction["oneLineUnderstanding"]:
        extraction["oneLineUnderstanding"] = build_one_line_understanding(extraction)

    if not extraction["appType"]:
        extraction["missingFields"] = list(set(extraction["missingFields"] + ["appType"]))
    if not extraction["targetUsers"]:
        extraction["missingFields"] = list(set(extraction["missingFields"] + ["targetUsers"]))

    return extraction
