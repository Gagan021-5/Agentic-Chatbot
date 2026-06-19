"""Off-topic detection — RentPrompts requirementRouter.js."""

import re

GREETING_PATTERN = re.compile(
    r"^(h+e+l+[lo]+|h+[iy]+|hey+|hola|greetings?|howdy|sup|yo|what'?s up|namaste|hi+)[\s!?.]*$",
    re.IGNORECASE,
)

OFF_TOPIC_KEYWORDS = [
    "gravity", "physics", "chemistry", "biology", "history",
    "math", "equation", "formula", "theorem", "law of",
    "what is", "who is", "who was", "when did", "where is",
    "capital of", "population of", "how far", "how old",
    "weather", "news", "stock", "price of bitcoin",
    "temperature", "forecast",
    "fix my code", "debug", "error in my code",
    "how to install", "npm install", "python",
    "i am sad", "i am happy", "i love you", "you are",
    "are you human", "are you ai", "who made you",
    "hack", "crack", "illegal", "kill", "weapon",
]

QUESTION_STARTERS = [
    "what is the", "what are the", "who is", "who was",
    "when is", "when was", "where is", "how does gravity",
    "explain the", "define the", "tell me about",
    "what causes", "why does the sun",
]

APP_CONTEXT_WORDS = [
    "app", "build", "create", "make", "generate", "tool",
    "rentprompts", "marketplace", "publish", "image app",
    "video app", "text app", "audio app", "vision app",
    "i want", "i need", "my app", "our app", "for my",
    "for our", "product", "clinic", "company", "business",
    "generator", "creator", "builder",
]

OFF_TOPIC_RESPONSE = {
    "reply": (
        "I'm RentPrompts Agent — I help you create and publish AI-powered apps on RentPrompts marketplace.\n\n"
        "I can help you build apps that generate:\n"
        "- 🖼️ Images — portraits, art, product photos\n"
        "- 🎥 Videos — animations, cinematic clips, reels\n"
        "- 📝 Text — blogs, emails, scripts, stories\n"
        "- 🔊 Audio — voiceovers, music, speech\n"
        "- 👁️ Vision — image analysis, object detection\n\n"
        "What kind of AI app would you like to create?"
    ),
    "uiType": "chips",
    "uiData": {"options": ["Image app", "Video app", "Text app", "Audio app", "Vision app"]},
    "nextStep": 0,
    "coins": None,
}


def is_off_topic(message: str, session: dict | None) -> bool:
    msg = str(message or "").lower().strip()
    if len(msg) < 2:
        return False

    # Catch greetings and typo variants FIRST, before any other check
    if GREETING_PATTERN.match(msg):
        return True

    if session:
        has_history = isinstance(session.get("history"), list) and len(session["history"]) > 1
        has_triage = (session.get("triageRounds") or 0) > 0 or session.get("awaitingTriageAnswer") is True
        has_context = session.get("dynamicContext") or session.get("appType")
        if has_history or has_triage or has_context:
            return False

    if any(w in msg for w in APP_CONTEXT_WORDS):
        return False

    if any(msg.startswith(s) for s in QUESTION_STARTERS):
        return True
    return any(kw in msg for kw in OFF_TOPIC_KEYWORDS)
