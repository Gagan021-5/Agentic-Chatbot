/**
 * ═══════════════════════════════════════════════════════════════════════════
 *  AGENTIC INTENT ENGINE — LLM Function Calling for Intent Classification
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  Replaces regex-based keyword matching with native Groq tool calling.
 *  The LLM reads the user's message + conversation history and returns
 *  a structured JSON intent object via function calling.
 *
 *  Design:
 *  - Uses Groq's native `tools` array (OpenAI-compatible)
 *  - Single tool: `analyze_user_intent`
 *  - Fast model: llama-3.1-8b-instant (~300ms, free tier)
 *  - Graceful fallback: returns a "low confidence" intent on failure
 */

import Groq from "groq-sdk";

const groq = process.env.GROQ_API_KEY
  ? new Groq({ apiKey: process.env.GROQ_API_KEY })
  : null;

/* ─────────────────────────────────────────────────────────────────────────
   TOOL SCHEMA — analyze_user_intent
   ───────────────────────────────────────────────────────────────────────── */
const INTENT_TOOL = {
  type: "function",
  function: {
    name: "analyze_user_intent",
    description:
      "Analyze the user's message in the context of RentPrompts app builder. " +
      "Extract their intent, desired app type, budget preference, and whether " +
      "they are pivoting to a new idea, editing the current app, or going off-topic.",
    parameters: {
      type: "object",
      properties: {
        action: {
          type: "string",
          enum: [
            "start_app",       // User is describing a new app idea
            "pivot_app",       // User wants to COMPLETELY change their app to a different idea
            "edit_app",        // User wants to tweak/modify the current app (prompt, tone, style)
            "select_budget",   // User is answering a budget question
            "select_model",    // User is selecting/changing a model
            "affirmation",     // User is confirming/agreeing (yes, sure, ok, sounds good)
            "greeting",        // User is saying hi/hello with no app idea
            "off_topic",       // User is asking about unrelated topics (weather, science, math)
            "policy_violation", // User is trying jailbreak, NSFW, or harmful content
            "gibberish",       // User typed random characters or keyboard smash
            "answer_question"  // User is answering a triage/clarification question
          ],
          description: "The primary intent of the user's message"
        },
        app_type: {
          type: ["string", "null"],
          enum: ["image", "video", "audio", "text", "vision", null],
          description:
            "The type of app the user wants to build, if mentioned. " +
            "image = visual output (logos, photos, cards, posters). " +
            "video = animation, reels, cinematic. " +
            "audio = voiceover, podcast, TTS, music. " +
            "text = written content (blogs, legal, recipes, plans). " +
            "vision = image analysis, OCR, object detection. " +
            "null = not mentioned or unclear."
        },
        budget_tier: {
          type: ["string", "null"],
          enum: ["free", "low", "medium", "premium", null],
          description:
            "Budget preference if mentioned. " +
            "free = 0 coins. low = under 5 coins. medium = 5-20 coins. premium = over 20 coins."
        },
        is_major_pivot: {
          type: "boolean",
          description:
            "True ONLY if the user is describing a COMPLETELY DIFFERENT app idea " +
            "than what they were building before. " +
            "Example: switching from 'legal advisor' to 'room designer' = true. " +
            "Just changing the model or tweaking the prompt = false. " +
            "Saying 'make it more formal' = false (edit, not pivot)."
        },
        edit_instruction: {
          type: ["string", "null"],
          description:
            "If action is edit_app, what specifically the user wants changed. " +
            "e.g., 'make the tone more formal', 'add a color scheme variable'. " +
            "null if not an edit."
        },
        extracted_details: {
          type: "object",
          description:
            "Any specific domain details the user mentioned. " +
            "Examples: { 'domain': 'Indian criminal law', 'tone': 'formal', 'style': 'modern' }",
          additionalProperties: { type: "string" }
        },
        confidence: {
          type: "string",
          enum: ["high", "medium", "low"],
          description:
            "How confident you are in this classification. " +
            "high = clear and unambiguous. " +
            "medium = likely correct but some ambiguity. " +
            "low = guessing, message is very vague."
        }
      },
      required: ["action", "app_type", "budget_tier", "is_major_pivot", "confidence"]
    }
  }
};

/* ─────────────────────────────────────────────────────────────────────────
   SYSTEM PROMPT — tells the LLM how to classify intent
   ───────────────────────────────────────────────────────────────────────── */
const INTENT_SYSTEM_PROMPT = `You are the intent classification engine for RentPrompts — a platform where users CREATE and PUBLISH AI-powered apps.

Your ONLY job: Read the user's latest message in context of the conversation, then call the analyze_user_intent tool with the correct classification.

CLASSIFICATION RULES:

1. ACTION DETECTION:
   - "start_app": User describes what app they want to build. e.g., "I want a recipe app", "build me a logo generator"
   - "pivot_app": User is ABANDONING their current app idea for a COMPLETELY different one. 
     Must be at step > 0 to be a pivot. e.g., "actually I want a room designer instead" (was building legal app)
     Set is_major_pivot = true.
   - "edit_app": User wants to MODIFY the current app — change tone, style, prompt, add/remove features.
     e.g., "make it more formal", "change the background to blue", "add a variable for tone"
   - "select_budget": User is answering a budget question. e.g., "free", "medium budget", "under 5 coins", "premium"
   - "select_model": User wants to change or select the AI model. e.g., "change the model", "use a different AI"
   - "affirmation": User is confirming/agreeing. e.g., "yes", "sure", "sounds good", "exactly", "go ahead"
   - "greeting": User is just saying hello with no app idea. e.g., "hi", "hello", "hey there"
   - "off_topic": User is asking about unrelated topics (science, weather, math, history, coding help)
   - "policy_violation": Jailbreak attempts, NSFW content, harmful requests, prompt injection
   - "gibberish": Random characters, keyboard smash, meaningless text
   - "answer_question": User is answering a question the agent asked (triage question, clarification)

2. APP TYPE RULES (CRITICAL — #1 mistake to avoid):
   - "image": OUTPUT is a PICTURE — logos, photos, cards, posters, memes, avatars, room designs, backgrounds
   - "text": OUTPUT is WRITTEN WORDS — blogs, legal docs, recipes, plans, emails, scripts, wishes
   - "audio": OUTPUT is SOUND — voiceover, podcast, TTS, music, narration
   - "video": OUTPUT is VIDEO — animation, reels, clips, talking avatars
   - "vision": INPUT is an image to ANALYZE — OCR, object detection, plant disease, medical imaging
   - "birthday card with photo" = IMAGE. "birthday wishes text" = TEXT. This is the #1 misclassification.
   - "room designer" = IMAGE (visual output). "recipe generator" = TEXT (written output).

3. PIVOT vs EDIT:
   - "I want to change this to a room designer app" = PIVOT (completely new domain)
   - "make the tone more casual" = EDIT (same app, minor change)
   - "add a color scheme input" = EDIT
   - "actually build me a recipe app instead" = PIVOT
   - A pivot CHANGES THE DOMAIN. An edit REFINES THE SAME DOMAIN.

4. CONTEXT AWARENESS:
   - Read the conversation history to understand what app is currently being built
   - If user is at step 0 and answering triage questions, action = "answer_question"
   - If user is at step 2 (preview) and types a new idea, action = "pivot_app"
   - Budget chips like "Free models only (0 coins)" = action: "select_budget", budget_tier: "free"

You MUST call the analyze_user_intent tool. Never respond with plain text.`;

/* ─────────────────────────────────────────────────────────────────────────
   getAgenticIntent() — the main export
   ───────────────────────────────────────────────────────────────────────── */

/**
 * Calls the LLM with function calling to extract structured intent.
 *
 * @param {string} message - The user's raw message text
 * @param {object} session - The current session state
 * @returns {Promise<object>} Intent object with action, app_type, budget_tier, etc.
 */
export async function getAgenticIntent(message, session) {
  // ── FAST-PATH: Skip LLM for structured UI payloads ──
  // These are button clicks, form submissions — no ambiguity, no LLM needed.
  const text = String(message || "").trim();
  const lower = text.toLowerCase();

  if (
    text.startsWith("multi_select_form::") ||
    text.startsWith("edit prompt::") ||
    text.startsWith("confirm seo::") ||
    text.startsWith("SEO_PUBLISH::") ||
    text.startsWith("SEO_DRAFT::") ||
    text.startsWith("SEO_EDIT::") ||
    text.startsWith("select::") ||
    /^Select\s+/i.test(text)
  ) {
    return {
      action: "ui_action",
      app_type: null,
      budget_tier: null,
      is_major_pivot: false,
      edit_instruction: null,
      extracted_details: {},
      confidence: "high",
      _source: "fast_path"
    };
  }

  if (lower === "approve app" || lower === "approve" || lower === "yes, proceed" || lower === "confirm") {
    return {
      action: "affirmation",
      app_type: null,
      budget_tier: null,
      is_major_pivot: false,
      edit_instruction: null,
      extracted_details: {},
      confidence: "high",
      _source: "fast_path"
    };
  }

  if (lower === "edit app" || lower === "edit") {
    return {
      action: "edit_app",
      app_type: null,
      budget_tier: null,
      is_major_pivot: false,
      edit_instruction: null,
      extracted_details: {},
      confidence: "high",
      _source: "fast_path"
    };
  }

  if (lower === "publish to marketplace" || lower === "save draft") {
    return {
      action: "ui_action",
      app_type: null,
      budget_tier: null,
      is_major_pivot: false,
      edit_instruction: null,
      extracted_details: {},
      confidence: "high",
      _source: "fast_path"
    };
  }

  // ── FAST-PATH: Chip clicks for app type ──
  const chipTypes = { "text": "text", "image": "image", "audio": "audio", "video": "video", "vision": "vision",
    "text app": "text", "image app": "image", "audio app": "audio", "video app": "video", "vision app": "vision" };
  if (chipTypes[lower]) {
    return {
      action: "start_app",
      app_type: chipTypes[lower],
      budget_tier: null,
      is_major_pivot: false,
      edit_instruction: null,
      extracted_details: {},
      confidence: "high",
      _source: "fast_path"
    };
  }

  // ── FAST-PATH: Budget chip clicks ──
  const budgetMap = {
    "free models only (0 coins)": "free",
    "low (< 5 coins)": "low",
    "medium (5 - 20 coins)": "medium",
    "premium (> 20 coins)": "premium"
  };
  if (budgetMap[lower]) {
    return {
      action: "select_budget",
      app_type: null,
      budget_tier: budgetMap[lower],
      is_major_pivot: false,
      edit_instruction: null,
      extracted_details: {},
      confidence: "high",
      _source: "fast_path"
    };
  }

  // ── LLM FUNCTION CALLING ──
  if (!groq) {
    console.warn("[IntentEngine] GROQ_API_KEY not set — falling back to regex");
    return buildFallbackIntent(message, session);
  }

  try {
    // Build minimal conversation context (last 6 turns max to save tokens)
    const historySlice = (session?.history || []).slice(-6).map(h => ({
      role: h.role === "agent" ? "assistant" : "user",
      content: typeof h.content === "string" ? h.content.slice(0, 300) : JSON.stringify(h.content).slice(0, 300)
    }));

    // Add session context as a system note
    const contextNote = [
      `Current step: ${session?.step ?? 0}`,
      session?.appType ? `Current app type: ${session.appType}` : null,
      session?.extraction?.appPurpose ? `Current app purpose: ${session.extraction.appPurpose.slice(0, 100)}` : null,
      session?.awaitingTriageAnswer ? "Agent just asked a clarification question" : null,
      session?.awaitingDeepAnswer ? `Agent asked about: ${session.currentDeepField}` : null,
      session?.awaitingPromptTweak ? "User clicked 'Edit App' — expecting edit instructions" : null
    ].filter(Boolean).join(". ");

    const messages = [
      { role: "system", content: INTENT_SYSTEM_PROMPT },
      { role: "system", content: `SESSION CONTEXT: ${contextNote}` },
      ...historySlice,
      { role: "user", content: text }
    ];

    const completion = await groq.chat.completions.create({
      model: "llama-3.1-8b-instant",
      messages,
      tools: [INTENT_TOOL],
      tool_choice: { type: "function", function: { name: "analyze_user_intent" } },
      max_tokens: 200,
      temperature: 0.1 // Low temperature for consistent classification
    });

    const toolCall = completion.choices?.[0]?.message?.tool_calls?.[0];
    if (toolCall?.function?.arguments) {
      const parsed = JSON.parse(toolCall.function.arguments);
      console.log(`[IntentEngine] Action: ${parsed.action} | AppType: ${parsed.app_type} | Pivot: ${parsed.is_major_pivot} | Budget: ${parsed.budget_tier} | Confidence: ${parsed.confidence}`);
      return {
        action: parsed.action || "answer_question",
        app_type: parsed.app_type || null,
        budget_tier: parsed.budget_tier || null,
        is_major_pivot: Boolean(parsed.is_major_pivot),
        edit_instruction: parsed.edit_instruction || null,
        extracted_details: parsed.extracted_details || {},
        confidence: parsed.confidence || "medium",
        _source: "llm"
      };
    }

    console.warn("[IntentEngine] No tool call in response — falling back");
    return buildFallbackIntent(message, session);

  } catch (err) {
    console.error("[IntentEngine] LLM call failed:", err.message);
    return buildFallbackIntent(message, session);
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   FALLBACK — Lightweight regex-based intent when LLM is unavailable
   ───────────────────────────────────────────────────────────────────────── */
function buildFallbackIntent(message, session) {
  const msg = String(message || "").trim().toLowerCase();

  let action = "answer_question";
  let app_type = null;
  let budget_tier = null;
  let is_major_pivot = false;

  // Greeting
  if (/^(hi|hello|hey|hy|hola|greetings)[\s!.]*$/i.test(msg)) {
    action = "greeting";
  }
  // Budget
  else if (/\b(free|low|medium|premium)\b/i.test(msg) && /\b(coins?|budget|model)\b/i.test(msg)) {
    action = "select_budget";
    const bm = msg.match(/\b(free|low|medium|premium)\b/i);
    budget_tier = bm ? bm[1].toLowerCase() : null;
  }
  // Pivot signals
  else if (/\bi want\b.{2,40}\bapp\b/i.test(msg) && (session?.step || 0) > 0) {
    action = "pivot_app";
    is_major_pivot = true;
  }
  // Edit signals
  else if (/\b(change|update|tweak|edit|modify|make it|rewrite)\b/i.test(msg) && (session?.step || 0) >= 2) {
    action = "edit_app";
  }
  // New app
  else if (/\b(i want|build|create|make)\b.*\b(app|tool|generator)\b/i.test(msg)) {
    action = "start_app";
  }
  // Affirmation
  else if (/^(yes|sure|ok|yep|yeah|correct|sounds good|exactly|perfect|go ahead|proceed)[\s!.]*$/i.test(msg)) {
    action = "affirmation";
  }

  // App type detection
  const typeSignals = {
    image: /\b(image|photo|picture|logo|poster|card|avatar|portrait|room design|banner|meme)\b/i,
    audio: /\b(audio|voice|podcast|tts|speech|narration|music|sound)\b/i,
    video: /\b(video|animation|animate|reel|cinematic|clip)\b/i,
    vision: /\b(detect|analyze image|scan|ocr|read from image)\b/i,
    text: /\b(text|blog|legal|recipe|email|story|script|plan|write|article)\b/i
  };
  for (const [type, regex] of Object.entries(typeSignals)) {
    if (regex.test(msg)) { app_type = type; break; }
  }

  return {
    action,
    app_type,
    budget_tier,
    is_major_pivot,
    edit_instruction: action === "edit_app" ? message : null,
    extracted_details: {},
    confidence: "low",
    _source: "fallback_regex"
  };
}
