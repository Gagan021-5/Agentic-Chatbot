import Groq from "groq-sdk";
import OpenAI from "openai";
import { normalizeExtraction } from "./gemini.js";

const groqApiKey = process.env.GROQ_API_KEY;

const GROQ_SYSTEM_PROMPT = `You are a strict data extraction engine for RentPrompts — a platform where users CREATE and PUBLISH AI-powered apps.

Users describe an app they want to build.
Your ONLY job: extract what they said. Never invent.
CRITICAL INSTRUCTION: You are a multilingual agent. You MUST detect the language of the user's input. If the user types in Hindi (Devanagari script) or Hinglish (Hindi written in English alphabet), you MUST respond natively in that exact language and tone. All UI options and questions generated must also be translated into the user's detected language.

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
          return { name: item.trim(), placeholder: "Enter details..." };
        }
        if (!item || typeof item !== "object") return null;
        return {
          name: String(item.name || "").trim(),
          placeholder: String(item.placeholder || "Enter details...").trim()
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
CRITICAL INSTRUCTION: You are a multilingual agent. You MUST detect the language of the user's input. If the user types in Hindi (Devanagari script) or Hinglish (Hindi written in English alphabet), you MUST respond natively in that exact language and tone. All UI options and questions generated must also be translated into the user's detected language.
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

export { extractRequirements, generateDynamicContext };
