import Groq from "groq-sdk";
import OpenAI from "openai";
import { normalizeExtraction } from "./gemini.js";
import { LANGUAGE_MIRROR_DIRECTIVE } from "./languageDirective.js";

const groqApiKey = process.env.GROQ_API_KEY;

const GROQ_SYSTEM_PROMPT = `You are a strict data extraction engine for RentPrompts — a platform where users CREATE and PUBLISH AI-powered apps.

Users describe an app they want to build.
Your ONLY job: extract what they said. Never invent.
${LANGUAGE_MIRROR_DIRECTIVE}

APP TYPE RULES — read every word carefully:
- "image" app: generates images, photos, portraits, transforms photos, superhero filter, avatar maker, logo maker, any visual output
- "video" app: creates videos, animations, reels, cinematic clips, animates photos, talking avatars
- "text" app: generates written content — blogs, emails, captions, scripts, stories, reports, product descriptions, workout PLANS, meal PLANS, diet plans, study guides, itineraries, any document or written plan output
- "audio" app: voice, music, speech, podcast, sound effects, text to speech, transcription
- "vision" app: analyzes images, reads text from images, detects objects, medical image analysis

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
{
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
  "confidence": {
    "appType": "HIGH|MEDIUM|LOW",
    "budget": "HIGH|MEDIUM|LOW"
  },
  "missingFields": [],
  "oneLineUnderstanding": "only rephrase what user said",
  "suggestedReply": "one warm follow-up question"
}`;

function hasRealValue(value) {
  return Boolean(value && !/^your_.+_here$/i.test(value));
}

const groq = hasRealValue(groqApiKey) ? new Groq({ apiKey: groqApiKey }) : null;

let openRouterClient = null;

function getOpenRouterClient() {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!hasRealValue(apiKey)) {
    throw new Error("OPENROUTER_API_KEY is not configured");
  }
  if (!openRouterClient) {
    openRouterClient = new OpenAI({
      baseURL: "https://openrouter.ai/api/v1",
      apiKey,
      defaultHeaders: {
        "HTTP-Referer": "http://localhost:5173",
        "X-Title": "RentPrompts Agent Demo"
      }
    });
  }
  return openRouterClient;
}

function isRateLimitError(error) {
  return error && (error.status === 429 || error.statusCode === 429 || error.code === 429 || (error.message && error.message.includes('429')));
}

function normalizeLanguageHint(languageHint) {
  const normalized = String(languageHint || "").toLowerCase();
  if (normalized.includes("hinglish")) return "Hinglish";
  if (normalized.includes("hindi")) return "Hindi";
  return "English";
}

function sanitizeStringList(list, minLen, maxLen, fallback) {
  const cleaned = Array.isArray(list)
    ? list
      .map((item) => String(item || "").trim())
      .filter(Boolean)
      .filter((item, idx, arr) => arr.findIndex((x) => x.toLowerCase() === item.toLowerCase()) === idx)
      .slice(0, maxLen)
    : [];

  if (cleaned.length >= minLen) return cleaned;
  return fallback.slice(0, maxLen);
}

function sanitizeVariableObjects(list, minLen, maxLen, fallback) {
  const normalized = Array.isArray(list)
    ? list
      .map((item) => {
        if (typeof item === "string") {
          return { name: item.trim(), placeholder: "Enter details...", test_value: "" };
        }
        if (!item || typeof item !== "object") return null;
        return {
          name: String(item.name || "").trim(),
          placeholder: String(item.placeholder || "Enter details...").trim(),
          test_value: String(item.test_value || "").trim()
        };
      })
      .filter((item) => item && item.name)
      .filter((item, idx, arr) => arr.findIndex((x) => x.name.toLowerCase() === item.name.toLowerCase()) === idx)
      .slice(0, maxLen)
    : [];

  if (normalized.length >= minLen) return normalized;
  return fallback.slice(0, maxLen);
}

function buildDynamicContextFallback(appType, languageHint) {
  const lang = normalizeLanguageHint(languageHint);
  const isHindi = lang === "Hindi";
  const isHinglish = lang === "Hinglish";

  if (isHindi) {
    return {
      options: ["उपयोगकर्ता के लिए पर्सनल परिणाम", "स्पष्ट और संरचित आउटपुट", "तेज और विश्वसनीय प्रतिक्रिया", "कस्टम इनपुट आधारित जनरेशन"],
      variables: [
        { name: "मुख्य इनपुट", placeholder: "अपनी आवश्यकता लिखें..." },
        { name: "संदर्भ", placeholder: "कॉन्टेक्स्ट या पृष्ठभूमि जोड़ें..." },
        { name: "पसंदीदा स्टाइल", placeholder: "जैसे: प्रोफेशनल, फ्रेंडली..." }
      ]
    };
  }
  if (isHinglish) {
    return {
      options: ["Personalized output for user", "Structured and clear result", "Fast and reliable response", "Custom input based generation"],
      variables: [
        { name: "Main input", placeholder: "Aap kya generate karna chahte ho?" },
        { name: "Context", placeholder: "Background ya extra details" },
        { name: "Preferred style", placeholder: "Jaise: Formal, Friendly" }
      ]
    };
  }

  const typeSpecific = {
    text: { options: ["Personalized response style", "Structured output format", "Tone control", "Goal-focused generation"], variables: [{ name: "Main topic", placeholder: "What is the main topic?" }, { name: "Audience", placeholder: "Who is this for?" }, { name: "Tone", placeholder: "Professional, friendly, motivational..." }] },
    image: { options: ["Style consistency", "Composition control", "High-detail output", "Prompt safety guardrails"], variables: [{ name: "Subject", placeholder: "What should appear in the image?" }, { name: "Style", placeholder: "Anime, realistic, cinematic..." }, { name: "Reference details", placeholder: "Colors, mood, composition" }] },
    video: { options: ["Scene pacing control", "Shot/style consistency", "Duration control", "Platform-ready output"], variables: [{ name: "Concept", placeholder: "What story or scene?" }, { name: "Visual style", placeholder: "Cinematic, vlog, ad..." }, { name: "Video duration", placeholder: "e.g., 10 seconds" }] },
    audio: { options: ["Voice style selection", "Language support", "Pacing control", "Clean output format"], variables: [{ name: "Script", placeholder: "Paste narration text" }, { name: "Voice style", placeholder: "Male/Female, energetic/calm..." }, { name: "Language", placeholder: "e.g., English, Hindi" }] },
    vision: { options: ["Accurate object/text extraction", "Structured response mode", "Confidence-aware output", "Use-case specific analysis"], variables: [{ name: "Image input", placeholder: "Image URL or description" }, { name: "Task type", placeholder: "OCR, object detection, QA..." }, { name: "Output format", placeholder: "JSON, summary, plain text" }] }
  };
  return typeSpecific[appType] || { options: ["Personalized output", "Structured result", "Fast response", "Custom input support"], variables: [{ name: "Main input", placeholder: "Enter primary input" }, { name: "Context", placeholder: "Add context details" }, { name: "Style", placeholder: "Choose preferred style" }] };
}

function parseDynamicContextPayload(rawContent, appType, languageHint) {
  const fallback = buildDynamicContextFallback(appType, languageHint);
  try {
    const parsed = JSON.parse(String(rawContent || "{}").replace(/```json/gi, "").replace(/```/g, "").trim());
    return {
      options: sanitizeStringList(parsed.options, 4, 4, fallback.options),
      variables: sanitizeVariableObjects(parsed.variables, 3, 4, fallback.variables)
    };
  } catch {
    return fallback;
  }
}

async function extractRequirements(message, history) {
  // If Groq is not configured, go straight to OpenRouter fallback
  if (!groq) {
    return extractWithOpenRouterFallback(message, history);
  }

  try {
    const completion = await groq.chat.completions.create({
      model: "llama-3.3-70b-versatile",
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content: GROQ_SYSTEM_PROMPT
        },
        {
          role: "user",
          content: JSON.stringify({
            message,
            history: Array.isArray(history) ? history.slice(-8) : []
          })
        }
      ]
    });

    const content = completion.choices && completion.choices[0] && completion.choices[0].message ? completion.choices[0].message.content : "{}";
    return normalizeExtraction(JSON.parse(content), message);
  } catch (error) {
    if (isRateLimitError(error)) {
      console.log("[Groq] 429 hit, falling back to OpenRouter...");
      return extractWithOpenRouterFallback(message, history);
    }

    console.error("Groq extraction error, falling back to OpenRouter:", error.message);
    return extractWithOpenRouterFallback(message, history);
  }
}

async function generateDynamicContext({ appType, appPurpose, languageHint }) {
  const safeType = ["text", "image", "video", "audio", "vision"].includes(appType) ? appType : "text";
  const safePurpose = String(appPurpose || "").trim() || "general assistant app";
  const safeLang = normalizeLanguageHint(languageHint);
  const systemPrompt = `You are a strict JSON generator.
Generate compact, practical setup suggestions for an AI app idea.
${LANGUAGE_MIRROR_DIRECTIVE}
When generating 'variables', you MUST include a 'placeholder' key.
- If the variable name contains 'Date', placeholder MUST be 'DD/MM/YYYY'.
- If the variable name contains 'Time', placeholder MUST be 'HH:MM AM/PM'.
- If the variable name contains 'Place' or 'Location', placeholder MUST be 'City, Country'.
- For everything else, use a relevant example.
NEVER use 'Enter details...' as a placeholder for date, time, or location fields.
Output must be strict JSON with this exact shape:
{"options":["4 concise feature options"],"variables":[{"name":"Date of Birth","placeholder":"DD/MM/YYYY"},{"name":"Location","placeholder":"City, Country"}]}
No markdown. No prose.`;
  const userPrompt = `The user wants to build a ${safeType} app for: ${safePurpose}.
Language mode: ${safeLang}.
Generate 4 highly relevant specific features and 3-4 input variables needed for the app.
For each variable include name and helpful placeholder.`;

  try {
    if (groq) {
      const completion = await groq.chat.completions.create({
        model: "llama-3.3-70b-versatile",
        response_format: { type: "json_object" },
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt }
        ]
      });
      const content = completion.choices?.[0]?.message?.content || "{}";
      return parseDynamicContextPayload(content, safeType, safeLang);
    }
  } catch (error) {
    if (!isRateLimitError(error)) {
      console.error("Groq dynamic context error, falling back:", error.message);
    }
  }

  try {
    const fallback = await getOpenRouterClient().chat.completions.create({
      model: "meta-llama/llama-3.3-70b-instruct",
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt }
      ]
    });
    const content = fallback.choices?.[0]?.message?.content || "{}";
    return parseDynamicContextPayload(content, safeType, safeLang);
  } catch (error) {
    console.error("OpenRouter dynamic context fallback failed:", error.message);
    return buildDynamicContextFallback(safeType, safeLang);
  }
}

async function extractWithOpenRouterFallback(message, history) {
  try {
    const fallback = await getOpenRouterClient().chat.completions.create({
      model: 'meta-llama/llama-3.3-70b-instruct',
      messages: [
        { role: 'system', content: GROQ_SYSTEM_PROMPT },
        { role: 'user', content: JSON.stringify({ message, history: Array.isArray(history) ? history.slice(-8) : [] }) }
      ],
      response_format: { type: 'json_object' },
    });

    const raw = fallback.choices[0].message.content;
    const parsed = JSON.parse(raw.replace(/```json/gi, '').replace(/```/g, '').trim());
    return normalizeExtraction(parsed, message);
  } catch (fallbackError) {
    console.error("OpenRouter fallback also failed:", fallbackError.message);
    return normalizeExtraction(null, message);
  }
}

/* ────────────────────────────────────────────
   AGENTIC TRIAGE — evaluate specificity before generating form
   ──────────────────────────────────────────── */
const TRIAGE_INSTRUCTION = `
You are a professional, highly intelligent Universal AI Technical Architect. The person you are talking to (the App Creator) wants to build an application. 
Your goal is to scope the app by identifying the specific USE CASE and the FACTORS the app will analyze before proceeding.

CRITICAL UX RULES - PROACTIVE SUGGESTIONS & LATEST INTENT:
1. Keep your tone strictly professional, direct, and helpful. 
2. NEVER ask the lazy literal question: "what inputs/variables should the app collect?". 
3. UNIVERSAL ADAPTABILITY: Adapt your suggested factors to the user's specific profession or domain instantly.
4. PROACTIVELY SUGGEST FACTORS: It is your job to suggest the business logic. Actively suggest relevant factors tailored to their specific profession (e.g., "Should the app consider the scam type, location, and date?").
5. LATEST INTENT PRIORITIZATION: If the App Creator pivots their idea (e.g., they started talking about "fundamental rights" but their latest message is about "finding a scam IPC/BNS section"), prioritize their LATEST actionable goal. Discard old concepts that no longer fit the current workflow. Do not generate variables for discarded concepts.

STEP 1: THE "ZERO CONTEXT" RULE
- If the App Creator gives NO specific domain at all, return "needs_context" and ask what industry/topic the app is for.

STEP 2: THE "NEEDS CONTEXT" RULE (Broad Domain)
- If they give a broad domain (e.g., "safety app for factory" or "lesson planner"), return "needs_context".
- Ask specific questions about the app's output AND proactively suggest domain-specific factors to analyze based on their profession.

STEP 3: UNIVERSAL VARIABLE RULES (ONLY IF READY)
- If the status is "ready" (meaning they explicitly confirmed the workflow and the factors you suggested OR they explicitly provided the inputs themselves):
- You MUST autonomously generate 3-4 layman-friendly variables based on the factors you discussed.
- TEST DATA: You MUST generate a highly specific, realistic 'test_value' for each variable.

Return STRICTLY as JSON:
{
  "status": "needs_context" | "ready",
  "domain_identified": "e.g., Legal, Education, Industrial, Business",
  "question": "Only filled if needs_context is true. Ask about the output AND actively suggest domain-specific factors they might want the app to analyze.",
  "form": {
    "options": ["Feature 1", "Feature 2"],
    "variables": [{"name": "Suggested Factor", "placeholder": "...", "test_value": "..."}]
  }
}
`;

const ALLOWED_TRIAGE_APP_FORMATS = ["text", "image", "audio", "video", "vision"];

function normalizeTriageAppFormat(raw, fallbackType) {
  const fb = ALLOWED_TRIAGE_APP_FORMATS.includes(fallbackType) ? fallbackType : "text";
  const v = String(raw == null ? "" : raw).trim().toLowerCase();
  if (ALLOWED_TRIAGE_APP_FORMATS.includes(v)) return v;
  if (String(raw || "").trim()) console.warn("[Triage] Invalid or unknown app_format:", raw, "→ using", fb);
  return fb;
}

function mapHistoryToTriageMessages(conversationHistory) {
  // Last 8 turns so chip / format clicks in the previous user turn stay visible to the model
  const recent = Array.isArray(conversationHistory) ? conversationHistory.slice(-8) : [];
  return recent
    .map((m) => {
      if (!m) return null;
      const raw = String(m.role || "").toLowerCase();
      const role = raw === "user" ? "user" : "assistant";
      const content = String(m.content ?? m.text ?? "").trim();
      if (!content) return null;
      return { role, content };
    })
    .filter(Boolean);
}

function parseTriageResponse(rawContent, formatFallback, languageHint) {
  const fallbackType = ALLOWED_TRIAGE_APP_FORMATS.includes(formatFallback) ? formatFallback : "text";

  const readyShape = (domain, appFormat, form) => ({
    status: "ready",
    domain,
    question: null,
    app_format: appFormat,
    form
  });

  try {
    const cleaned = String(rawContent || "{}").replace(/```json/gi, "").replace(/```/g, "").trim();
    const parsed = JSON.parse(cleaned);

    if (!parsed || !parsed.status) {
      return readyShape(null, fallbackType, buildDynamicContextFallback(fallbackType, languageHint));
    }

    const domain =
      String(parsed.domain_identified || parsed.domain || "").trim() || null;

    if (parsed.status === "needs_context") {
      const question = String(parsed.question || "").trim();
      if (!question || question.length < 10) {
        // Empty question — fall through to ready with fallback form
        const fbForm = buildDynamicContextFallback(fallbackType, languageHint);
        return readyShape(domain, fallbackType, fbForm);
      }
      return {
        status: "needs_context",
        domain,
        question,
        form: null,
        app_format: null
      };
    }

    if (parsed.status !== "ready") {
      return readyShape(domain, fallbackType, buildDynamicContextFallback(fallbackType, languageHint));
    }

    const fallbackForm = buildDynamicContextFallback(fallbackType, languageHint);
    const form = parsed.form && typeof parsed.form === "object" ? parsed.form : {};
    return readyShape(domain, fallbackType, {
      options: sanitizeStringList(form.options, 4, 4, fallbackForm.options),
      variables: sanitizeVariableObjects(form.variables, 3, 4, fallbackForm.variables)
    });
  } catch {
    return readyShape(null, fallbackType, buildDynamicContextFallback(fallbackType, languageHint));
  }
}

async function triageDynamicContext({ appType, appPurpose, languageHint, conversationHistory }) {
  const committed =
    appType != null &&
    String(appType).trim() &&
    ALLOWED_TRIAGE_APP_FORMATS.includes(String(appType).trim().toLowerCase())
      ? String(appType).trim().toLowerCase()
      : null;
  const formatFallback = committed || "text";
  const safePurpose = String(appPurpose || "").trim() || "general assistant app";
  const safeLang = normalizeLanguageHint(languageHint);

  const userTaskPrompt = `The App Creator wants to build a ${formatFallback} app.
Their description: "${safePurpose}"
Language mode: ${safeLang}.

The app type is already locked to "${formatFallback}". Prioritize their LATEST message in the conversation history over older topics if they pivoted.
- Zero context: return "needs_context" — ask industry/topic only; do NOT invent random styles.
- Broad domain: return "needs_context" — ask about output AND proactively suggest profession-specific factors to analyze (never ask "what variables should we collect?").
- Workflow + factors confirmed (or they supplied inputs): return "ready" and autonomously generate layman-friendly variables with test_value in the form.`;

  // System + mapped history + task user message (Groq / OpenRouter)
  const triageMessages = [
    { role: "system", content: TRIAGE_INSTRUCTION },
    ...mapHistoryToTriageMessages(conversationHistory),
    { role: "user", content: userTaskPrompt }
  ];

  // Try Groq first
  try {
    if (groq) {
      const completion = await groq.chat.completions.create({
        model: "llama-3.3-70b-versatile",
        response_format: { type: "json_object" },
        messages: triageMessages
      });
      const content = completion.choices?.[0]?.message?.content || "{}";
      return parseTriageResponse(content, formatFallback, safeLang);
    }
  } catch (error) {
    if (!isRateLimitError(error)) {
      console.error("Groq triage error, falling back:", error.message);
    }
  }

  // OpenRouter fallback
  try {
    const fallback = await getOpenRouterClient().chat.completions.create({
      model: "meta-llama/llama-3.3-70b-instruct",
      response_format: { type: "json_object" },
      messages: triageMessages
    });
    const content = fallback.choices?.[0]?.message?.content || "{}";
    return parseTriageResponse(content, formatFallback, safeLang);
  } catch (error) {
    console.error("OpenRouter triage fallback failed:", error.message);
    return {
      status: "ready",
      domain: null,
      question: null,
      app_format: formatFallback,
      form: buildDynamicContextFallback(formatFallback, safeLang)
    };
  }
}

export { extractRequirements, generateDynamicContext, triageDynamicContext };
