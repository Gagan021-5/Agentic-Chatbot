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

const openRouterClient = new OpenAI({
  baseURL: 'https://openrouter.ai/api/v1',
  apiKey: process.env.OPENROUTER_API_KEY,
  defaultHeaders: {
    'HTTP-Referer': 'http://localhost:5173',
    'X-Title': 'RentPrompts Agent Demo',
  }
});

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
    const fallback = await openRouterClient.chat.completions.create({
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
    const fallback = await openRouterClient.chat.completions.create({
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
You are an elite, conversational AI Architect. The user is describing an app they want to build.
Your goal is to converse naturally, deduce their DOMAIN, and figure out the APP FORMAT (Text, Image, Audio, Video, Vision).

Trust but verify (avoid the over-automation trap): still deduce the most likely format, but the user may want something different in the same domain (e.g. astrology as text vs tarot images vs zodiac video). You will state your guess in downstream UX; prioritize explicit user intent over stereotypes. Always read the conversation messages you are given and never re-ask for details or format already stated there.

STEP 1: FORMAT & DOMAIN DEDUCTION
- Analyze the user's request. What is the domain?
- What format is the output?
  - e.g., "Astrologer app" often defaults to 'text' ONLY when the user did not ask for visuals, audio, or video.
  - e.g., "Generate a picture of a cake" is 'image'.
  - e.g., "Voice cloning" is 'audio'.
- EXPLICIT OVERRIDE: If the user's exact words include the format words from STEP 2 ("image", "photo", "text", "video", "audio", "voice") or clear equivalents they typed (e.g. "pictures", "speech"), set app_format accordingly. Domain hints alone never bypass STEP 2's EXPLICIT CHECK.
- CORRECTIONS: If the user corrects the format (e.g., "No, make it an image app instead"), you must immediately set app_format to match their request and regenerate variables and options to match the new format.

STEP 2: ROUTING & FORMAT VERIFICATION (NEUTRAL STANCE)
- Vague: If you genuinely cannot tell what the app outputs, set status to "needs_context" and ask a clarifying question.
- EXPLICIT CHECK (Strict Rule): Look closely at the user's exact words. Did they explicitly type the word "image", "photo", "text", "video", "audio", or "voice"?
  - If NO: You MUST set status to "needs_format" to ask the user.
    - HUMILITY RULE: DO NOT state your assumption out loud. DO NOT act like you know what format they want.
    - Simply generate a friendly response praising their idea, and ask them neutrally what format they want to use.
    - Good Example: "A futuristic product generator sounds amazing! What type of output should this app generate?"
    - Bad Example (Do NOT do this): "I assume you want 3D images, is that correct?"
  - If YES (or if they just clicked a UI format button in the history): Set status to "ready" and immediately generate the JSON form variables.

STEP 3: VARIABLES (If Ready)
- Generate 3-4 highly relevant input variables based strictly on the SPECIFIC DOMAIN.
- Design/Marketing Domain (e.g., Posters, Ads, Logos): Ask for design-focused variables like "Brand Colors", "Event Theme", or "Target Audience".
- Concept Art Domain (e.g., Product Generator, Characters): Ask for visual properties like "Materials/Finish", "Lighting", or "Art Style".
- NEVER ask for backend logistical variables like "Release Date", "Budget", or "Physical Location" for any visual app.

${LANGUAGE_MIRROR_DIRECTIVE}

Return STRICTLY as JSON:
{
  "status": "needs_context" | "needs_format" | "ready",
  "domain": "e.g., Cooking",
  "app_format": "text" | "image" | "audio" | "video" | "vision" | "unknown",
  "question": "Only filled if needs_context or needs_format is true",
  "form": {
    "options": ["Feature 1"],
    "variables": [{"name": "Var", "placeholder": "...", "test_value": "..."}]
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
      String(parsed.domain || parsed.domain_identified || "").trim() || null;

    if (parsed.status === "needs_context" || parsed.status === "needs_format") {
      const question = String(parsed.question || "").trim();
      if (!question || question.length < 10) {
        const af = normalizeTriageAppFormat(parsed.app_format, fallbackType);
        const fbForm = buildDynamicContextFallback(af, languageHint);
        return readyShape(domain, af, fbForm);
      }
      const assumedFormat =
        parsed.status === "needs_format"
          ? normalizeTriageAppFormat(parsed.app_format, fallbackType)
          : null;
      return {
        status: parsed.status,
        domain,
        question,
        form: null,
        app_format: assumedFormat
      };
    }

    if (parsed.status !== "ready") {
      return readyShape(domain, fallbackType, buildDynamicContextFallback(fallbackType, languageHint));
    }

    const appFormat = normalizeTriageAppFormat(parsed.app_format, fallbackType);
    const fallbackForm = buildDynamicContextFallback(appFormat, languageHint);
    const form = parsed.form && typeof parsed.form === "object" ? parsed.form : {};
    return readyShape(domain, appFormat, {
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

  const historyHint =
    "Recent conversation turns are included as separate messages above—read them to see if the user already clicked or typed a format (Text, Image, etc.).";

  const userTaskPrompt = committed
    ? `The user wants to build a ${committed} app.
Their description (from extraction / latest turn): "${safePurpose}"
Language mode: ${safeLang}.

${historyHint}
The product has already locked output format to "${committed}" (often from a chip). If history shows they just chose this format, you MUST set status to "ready", set app_format to "${committed}", and output the full form—do NOT use needs_format to ask again. Use needs_context only if the goal is too vague to define variables.`
    : `The user is describing an app they want to build.
Their description (from extraction / latest turn): "${safePurpose}"
Language mode: ${safeLang}.

${historyHint}
Apply STEP 2 strictly: use needs_format unless (a) the user's exact words include an explicit format word from the checklist (image, photo, text, video, audio, voice) or (b) conversation history shows they just used a UI format chip. Domain guesses alone are never enough for "ready". Use needs_context if outputs are unknown. Use ready with the full form only when the domain is clear AND (a) or (b) is true.

When status is needs_format, the "question" string must follow the HUMILITY RULE: never state your format guess—only friendly praise of their idea and a neutral ask for what type of output they want (chips will offer choices).`;

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
    const fallback = await openRouterClient.chat.completions.create({
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
