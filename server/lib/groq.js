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
- "image" app: generates images, photos, portraits, transforms photos, superhero filter, avatar maker, logo maker, greeting cards, birthday cards, posters, memes, photo frames, invitations, flyers, any app where the OUTPUT is a PICTURE or VISUAL
- "video" app: creates videos, animations, reels, cinematic clips, animates photos, talking avatars
- "text" app: generates written content — blogs, emails, captions, scripts, stories, reports, product descriptions, resumes, cover letters, proposals, invoices, contracts, workout PLANS, meal PLANS, diet plans, study guides, itineraries, recipes, newsletters, SOPs, any document or written plan output
- "audio" app: voice, music, speech, podcast, sound effects, text to speech, transcription
- "vision" app: analyzes images, reads text from images, detects objects, medical image analysis

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

function prettifyVariableName(name, appType) {
  let clean = String(name || "").trim();
  if (!clean) return "";
  const lower = clean.toLowerCase();
  
  // Map developer jargon directly
  if (lower === "main_input" || lower === "input_text" || lower === "user_input" || lower === "input") {
    const type = String(appType || "text").toLowerCase();
    if (type === "text") return "Topic / Details";
    if (type === "image") return "Visual Subject";
    if (type === "audio") return "Script Text";
    if (type === "video") return "Video Concept";
    if (type === "vision") return "Image Analysis Goal";
    return "Topic / Details";
  }
  if (lower === "context" || lower === "background") {
    return "Additional Context";
  }
  if (lower === "output_style" || lower === "style") {
    return "Preferred Style";
  }
  if (lower === "details") {
    return "Specific Requirements";
  }

  // Replace snake_case and camelCase to human readable title case
  clean = clean.replace(/_/g, " ");
  clean = clean.replace(/([a-z])([A-Z])/g, "$1 $2");
  return clean.split(/\s+/).map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
}

function sanitizeVariableObjects(list, minLen, maxLen, fallback, appType) {
  const normalized = Array.isArray(list)
    ? list
      .map((item) => {
        if (typeof item === "string") {
          return { name: prettifyVariableName(item.trim(), appType), placeholder: "Enter details...", test_value: "" };
        }
        if (!item || typeof item !== "object") return null;
        return {
          name: prettifyVariableName(String(item.name || "").trim(), appType),
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

function buildDynamicContextFallback(appType, appPurpose, languageHint) {
  const safeType = String(appType || "text").toLowerCase();

  // Prettified default fallbacks
  const typeDefaults = {
    image: {
      options: ["Style control", "High-quality output", "Composition guidance", "Format flexibility"],
      variables: [
        { name: "Visual Subject", placeholder: "What should appear in the image?", test_value: "A majestic lion on a rock" },
        { name: "Visual Style", placeholder: "Visual style or aesthetic (e.g. photorealistic, anime)", test_value: "photorealistic" },
        { name: "Specific Requirements", placeholder: "Any special colors, lighting, or details", test_value: "golden hour lighting" }
      ]
    },
    video: {
      options: ["Scene control", "Style consistency", "Platform-ready output", "Motion effects"],
      variables: [
        { name: "Video Concept", placeholder: "What story or scene to create?", test_value: "A spaceship launching into a nebula" },
        { name: "Visual Style", placeholder: "Cinematic, vlog, animation...", test_value: "Cinematic sci-fi" },
        { name: "Target Platform", placeholder: "YouTube, Instagram, TikTok...", test_value: "YouTube" }
      ]
    },
    audio: {
      options: ["Voice selection", "Language support", "Pacing control", "Emotion control"],
      variables: [
        { name: "Script Content", placeholder: "Text or script to convert to speech", test_value: "Welcome back to another episode of our history podcast." },
        { name: "Voice Tone", placeholder: "Male/Female, energetic, calm, accent...", test_value: "Male, clear and energetic" },
        { name: "Audio Language", placeholder: "English, Hindi, Spanish...", test_value: "English" }
      ]
    },
    vision: {
      options: ["Accurate extraction", "Structured output", "Confidence scoring", "Use-case analysis"],
      variables: [
        { name: "Source Image", placeholder: "Upload your image", test_value: "photo of product" },
        { name: "Image Analysis Goal", placeholder: "What details should the AI detect in the image?", test_value: "detect any scratches or defects" },
        { name: "Output Format", placeholder: "JSON, plain text, bullets...", test_value: "JSON report" }
      ]
    },
    text: {
      options: ["Tone control", "Structured output", "Goal-focused generation", "Context awareness"],
      variables: [
        { name: "Topic / Details", placeholder: "Describe what you want the AI to write about", test_value: "The importance of learning to cook at home" },
        { name: "Additional Context", placeholder: "Background details or target audience", test_value: "targeted at college students" },
        { name: "Preferred Style", placeholder: "Format, tone, length preferences", test_value: "3-paragraph email, casual tone" }
      ]
    }
  };

  return typeDefaults[safeType] || typeDefaults.text;
}

function parseDynamicContextPayload(rawContent, appType, appPurpose, languageHint) {
  const fallback = buildDynamicContextFallback(appType, appPurpose, languageHint);
  try {
    const parsed = JSON.parse(String(rawContent || "{}").replace(/```json/gi, "").replace(/```/g, "").trim());
    return {
      options: sanitizeStringList(parsed.options, 4, 4, fallback.options),
      variables: sanitizeVariableObjects(parsed.variables, 3, 8, fallback.variables, appType)
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
      model: "llama-3.1-8b-instant",
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
- 🚨 CRITICAL SCHEMA STRUCTURING CONSTRAINT: Every extracted variable name MUST be translated into explicit human language. You are strictly prohibited from generating variables named 'input', 'text', 'data', 'variables', 'param', or 'main_input'. If the application parses legal domains, map to fields like 'incident_details' or 'dispute_context'. If editing visual elements, map strictly to fields like 'target_aesthetic' or 'canvas_dimensions'.
Output must be strict JSON with this exact shape:
{"options":["4 concise feature options"],"variables":[{"name":"Date of Birth","placeholder":"DD/MM/YYYY"},{"name":"Location","placeholder":"City, Country"}]}
No markdown. No prose.`;
  const userPrompt = `The user wants to build a ${safeType} app for: ${safePurpose}.
Language mode: ${safeLang}.
Generate 4 highly relevant specific features and 4-8 input variables needed for the app.
For each variable include name and helpful placeholder.`;

  try {
    if (groq) {
      const completion = await groq.chat.completions.create({
        model: "llama-3.1-8b-instant",
        response_format: { type: "json_object" },
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt }
        ]
      });
      const content = completion.choices?.[0]?.message?.content || "{}";
      return parseDynamicContextPayload(content, safeType, safePurpose, safeLang);
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
    return parseDynamicContextPayload(content, safeType, safePurpose, safeLang);
  } catch (error) {
    console.error("OpenRouter dynamic context fallback failed:", error.message);
    return buildDynamicContextFallback(safeType, safePurpose, safeLang);
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
    const parsed = JSON.parse((raw || '{}').replace(/```json/gi, '').replace(/```/g, '').trim());
    return normalizeExtraction(parsed, message);
  } catch (fallbackError) {
    console.error("OpenRouter fallback also failed:", fallbackError.message);
    return normalizeExtraction(null, message);
  }
}

/* ────────────────────────────────────────────
   AGENTIC TRIAGE — evaluate specificity before generating form
   ──────────────────────────────────────────── */
const TRIAGE_INSTRUCTION = `You are RentPrompts App Intelligence Engine. Your job is to convert any user idea into a structured AI application definition using strict domain-first reasoning.

You MUST classify the application into one of these domains: text, image, audio, video, vision, or hybrid.
You must NOT use generic assistant categories as a fallback under any circumstance.

🚨 HARD RULE: NO GENERIC ASSISTANT MODE
You are strictly forbidden from using or displaying generic assistant templates such as:
- basic tasks (calendar, reminders, notes)
- translation
- summarization
- general productivity assistant
- chatbot assistant menus
These are INVALID unless the user explicitly requests a “general assistant app”.

If the user provides ANY domain-specific intent (e.g. astrologer app, resume builder, legal advisor, logo generator, fitness planner, text-to-audio tool), you must immediately:
1. Identify the correct domain
2. Ignore all generic assistant flows
3. Build domain-specific variables only

🧠 DOMAIN CLASSIFICATION RULES:
- TEXT: resume builders, legal apps, astrologers, chatbots, planners, analyzers, document tools
- IMAGE: generation, editing, design, logos, visuals
- AUDIO: speech, TTS, music, voice
- VIDEO: motion generation, clips, animations
- VISION: image understanding, OCR, analysis
Default priority if unclear: TEXT > IMAGE > AUDIO > VIDEO > VISION

📊 CONFIDENCE SYSTEM:
Assign confidence_score (0–100):
- ≥ 80 → proceed to ready (status = "ready") without asking domain questions, question MUST be omitted or null.
- < 80 → ask ONLY one clarification question about domain ambiguity (status = "needs_context"). Never ask more than one question per turn.

🔁 ADAPTIVE INTENT RULE:
If user changes idea mid-conversation:
- Discard previous domain
- Recompute domain + confidence immediately
- Never continue old flow

🧾 VARIABLE EXTRACTION RULES:
After domain is confirmed, extract 3–6 variables:
- Must be user-facing (non-technical)
- Must be independent inputs
- Must directly affect output
- NEVER include model names, internal parameters, legal codes, or system settings
- 🚨 CRITICAL SCHEMA STRUCTURING CONSTRAINT: Every extracted variable name MUST be translated into explicit human language. You are strictly prohibited from generating variables named 'input', 'text', 'data', 'variables', 'param', or 'main_input'. If the application parses legal domains, map to fields like 'incident_details' or 'dispute_context'. If editing visual elements, map strictly to fields like 'target_aesthetic' or 'canvas_dimensions'.

⚠️ QUESTION RULE:
Only ask:
- ONE question per turn
- ONLY when confidence < 80
- ONLY to resolve domain uncertainty (not features)
Never show assistant menus, templates, or category choices.

Every response must be valid JSON only and include:
- status ("needs_context" or "ready")
- domain_identified ("text" | "image" | "audio" | "video" | "vision" | "hybrid")
- confidence_score (0-100)
- corrected_app_type (optional, only if initial classification was wrong: "text" | "image" | "audio" | "video" | "vision")
- variables (3-6 variables, each with name, placeholder, and realistic test_value)
- question (a single question only when confidence is below 80, otherwise null or omit)

Example JSON (Status ready):
{
  "status": "ready",
  "domain_identified": "text",
  "confidence_score": 95,
  "variables": [
    { "name": "Topic / Details", "placeholder": "What should the blog post be about?", "test_value": "The benefits of remote work" },
    { "name": "Tone", "placeholder": "e.g. professional, casual", "test_value": "professional" }
  ]
}

Example JSON (Status needs_context):
{
  "status": "needs_context",
  "domain_identified": "image",
  "confidence_score": 60,
  "question": "Do you want this app to generate **realistic photographs** or **custom vector illustrations**?",
  "suggested_options": ["Realistic Photographs", "Custom Vector Illustrations"],
  "variables": []
}

Do not include any explanation or markdown outside the valid JSON object.`;


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

function parseTriageResponse(rawContent, formatFallback, appPurpose, languageHint) {
  const fallbackType = ALLOWED_TRIAGE_APP_FORMATS.includes(formatFallback) ? formatFallback : "text";
  const safePurpose = String(appPurpose || "").trim();

  const readyShape = (domain, appFormat, form, confidence) => ({
    status: "ready",
    domain,
    confidence_score: confidence || 100,
    question: null,
    app_format: appFormat,
    form
  });

  try {
    const cleaned = String(rawContent || "{}").replace(/```json/gi, "").replace(/```/g, "").trim();
    const parsed = JSON.parse(cleaned);

    if (!parsed || !parsed.status) {
      return readyShape(null, fallbackType, buildDynamicContextFallback(fallbackType, safePurpose, languageHint), 100);
    }

    const domain =
      String(parsed.domain_identified || parsed.domain || "").trim() || null;

    const confidence = Number(parsed.confidence_score || parsed.confidence || 100);

    // Extract type correction if the LLM detected the wrong type
    const correctedType = parsed.corrected_app_type
      && ALLOWED_TRIAGE_APP_FORMATS.includes(String(parsed.corrected_app_type).trim().toLowerCase())
      ? String(parsed.corrected_app_type).trim().toLowerCase()
      : null;
    const effectiveType = correctedType || fallbackType;

    if (parsed.status === "needs_context") {
      const question = String(parsed.question || "").trim();
      if (!question || question.length < 10) {
        const fbForm = buildDynamicContextFallback(fallbackType, safePurpose, languageHint);
        return readyShape(domain, fallbackType, fbForm, confidence);
      }
      // Extract LLM-generated suggested options for dynamic chips
      const suggestedOptions = Array.isArray(parsed.suggested_options)
        ? parsed.suggested_options
            .map(o => String(o || "").trim())
            .filter(o => o.length > 0)
            .slice(0, 6)
        : null;
      return {
        status: "needs_context",
        domain,
        confidence_score: confidence,
        question,
        suggested_options: suggestedOptions && suggestedOptions.length >= 2 ? suggestedOptions : null,
        corrected_app_type: correctedType,
        form: null,
        app_format: null
      };
    }

    const fallbackForm = buildDynamicContextFallback(effectiveType, safePurpose, languageHint);
    const form = parsed.form && typeof parsed.form === "object" ? parsed.form : {};
    
    // Support flat variables array or nested form.variables
    const variablesRaw = Array.isArray(parsed.variables) ? parsed.variables : form.variables;
    // Support flat options array or nested form.options
    const optionsRaw = Array.isArray(parsed.options) ? parsed.options : form.options;

    return readyShape(domain, effectiveType, {
      options: sanitizeStringList(optionsRaw, 4, 4, fallbackForm.options),
      variables: sanitizeVariableObjects(variablesRaw, 3, 8, fallbackForm.variables, effectiveType)
    }, confidence);
  } catch (error) {
    console.error("[parseTriageResponse] Fallback parse failed:", error.message);
    return readyShape(null, fallbackType, buildDynamicContextFallback(fallbackType, "", languageHint), 100);
  }
}

async function triageDynamicContext({ appType, appPurpose, languageHint, conversationHistory, deepAnswers }) {
  const committed =
    appType != null &&
    String(appType).trim() &&
    ALLOWED_TRIAGE_APP_FORMATS.includes(String(appType).trim().toLowerCase())
      ? String(appType).trim().toLowerCase()
      : null;
  const formatFallback = committed || "text";
  const safePurpose = String(appPurpose || "").trim() || "general assistant app";
  const safeLang = normalizeLanguageHint(languageHint);

  // Summarize already-captured deepAnswers so triage doesn't re-ask them
  const answeredContext = deepAnswers && Object.keys(deepAnswers).length > 0
    ? `\nAlready answered by user: ${JSON.stringify(deepAnswers)}`
    : "";

  const userTaskPrompt = `Current app idea: "${safePurpose}"
Current app type: "${formatFallback}". If the conversation clearly indicates a DIFFERENT output type, correct it by including corrected_app_type.
Language: ${safeLang}.${answeredContext}

Review the conversation history. If you already know what the app does, what users provide, and what output it produces — return "ready" with 3-6 domain-appropriate variables immediately.
If ONE critical detail is missing, ask exactly ONE question with 2-3 inline bolded choices (e.g. **choice one**, **choice two**). Do NOT use numbered list formats like (1), (2), (3).
Bias toward readiness. Do NOT over-interview.`;

  const triageMessages = [
    { role: "system", content: TRIAGE_INSTRUCTION },
    ...mapHistoryToTriageMessages(conversationHistory),
    { role: "user", content: userTaskPrompt }
  ];

  // Try Groq first
  try {
    if (groq) {
      const completion = await groq.chat.completions.create({
        model: "llama-3.1-8b-instant",
        response_format: { type: "json_object" },
        messages: triageMessages
      });
      const content = completion.choices?.[0]?.message?.content || "{}";
      return parseTriageResponse(content, formatFallback, safePurpose, safeLang);
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
    return parseTriageResponse(content, formatFallback, safePurpose, safeLang);
  } catch (error) {
    console.error("OpenRouter triage fallback failed:", error.message);
    return {
      status: "ready",
      domain: null,
      question: null,
      app_format: formatFallback,
      form: buildDynamicContextFallback(formatFallback, safePurpose, safeLang)
    };
  }
}

export { extractRequirements, generateDynamicContext, triageDynamicContext, buildDynamicContextFallback };
