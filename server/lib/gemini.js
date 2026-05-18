import OpenAI from "openai";
import MODELS from "./models.js";
import { LANGUAGE_MIRROR_DIRECTIVE } from "./languageDirective.js";

// Keep old exports needed by stepRouter and groq
export const EXTRACTION_PROMPT = `You are a strict data extraction engine for RentPrompts — a platform where users CREATE and PUBLISH AI-powered apps.

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

ENTERPRISE SIGNALS — set enterpriseSignals:true if message contains any of:
company, team, employees, scale, API, integrate, bulk, enterprise, SaaS, B2B, workflow, automate our, our company, our team, organization, department, staff

Return ONLY valid JSON. No explanation. No markdown. No invented details.

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

function extractFeatures(message) {
  const lower = message.toLowerCase();
  const features = [];
  if (lower.includes("blog")) features.push("blog generation");
  if (lower.includes("animate")) features.push("animation workflow");
  if (lower.includes("photo") || lower.includes("image")) features.push("image input");
  if (lower.includes("cinematic")) features.push("cinematic styling");
  if (lower.includes("voice")) features.push("voice output");
  if (lower.includes("seo")) features.push("seo metadata");
  return features.slice(0, 5);
}

function detectLanguage(message) {
  const text = String(message || "");
  const lower = text.toLowerCase();
  const hasDevanagari = /[\u0900-\u097F]/.test(text);
  if (hasDevanagari) return "Hindi";
  const hinglishSignals = ["mujhe", "banana", "banani", "hai", "likhta", "likh", "ke liye", "karna", "mera", "meri", "krdo", "chahiye", "jaldi"];
  if (hinglishSignals.some((word) => lower.includes(word))) return "Hinglish";
  return "English";
}

function detectBudget(message) {
  const lower = message.toLowerCase();
  if (lower.includes("free")) return { value: "free", confidence: "HIGH" };
  if (lower.includes("cheap") || lower.includes("low budget") || lower.includes("budget")) return { value: "low", confidence: "MEDIUM" };
  if (lower.includes("premium") || lower.includes("quality")) return { value: "high", confidence: "MEDIUM" };
  if (lower.includes("ultra") || lower.includes("best possible")) return { value: "ultra", confidence: "HIGH" };
  return { value: null, confidence: "LOW" };
}

function inferAppType(message) {
  const lower = message.toLowerCase();
  if (/(video|animate|reel|cinematic|movie)/.test(lower)) return { value: "video", confidence: "HIGH" };
  if (/(blog|write|writer|copy|text|article|caption|planner|workout|meal|diet|itinerary|guide|report|advocate|legal|lawyer|law|document|draft|summary)/.test(lower)) return { value: "text", confidence: "HIGH" };
  if (/(image|photo|picture|poster|thumbnail|logo|design)/.test(lower)) return { value: "image", confidence: "HIGH" };
  if (/(voice|audio|music|podcast|song|speech)/.test(lower)) return { value: "audio", confidence: "HIGH" };
  if (/(vision|analy(s|z)e|ocr|scan|detect|inspect)/.test(lower)) return { value: "vision", confidence: "HIGH" };
  return { value: null, confidence: "LOW" };
}

function inferTone(message) {
  const lower = message.toLowerCase();
  if (/(asap|urgent|quick|jaldi|immediately|fast)/.test(lower)) return "urgent";
  if (/(please|kindly)/.test(lower)) return "formal";
  if (/(maybe|something|not sure|idk)/.test(lower)) return "unsure";
  return "casual";
}

function buildOneLineUnderstanding(extraction) {
  if (extraction.appType === "video" && extraction.wantsImageInput) {
    return "you want a video app that turns photos into cinematic clips";
  }
  if (extraction.appType === "text" && /blog/i.test(extraction.appPurpose)) {
    return "you want a text app that writes blogs";
  }
  if (extraction.appType) {
    return `you want a ${extraction.appType} app for ${extraction.appPurpose || "your use case"}`;
  }
  return extraction.appPurpose || "you want help shaping an app idea";
}

export function normalizeExtraction(raw, fallbackMessage) {
  const message = fallbackMessage || "";
  const inferred = inferAppType(message);
  const budget = detectBudget(message);
  const normalizedBudget =
    budget.value ||
    (raw &&
    ["free", "low", "medium", "high", "ultra"].includes(raw.budget) &&
    raw.confidence &&
    raw.confidence.budget === "HIGH"
      ? raw.budget
      : null);
  const normalizedBudgetConfidence = budget.value
    ? budget.confidence
    : raw && raw.confidence && ["HIGH", "MEDIUM", "LOW"].includes(raw.confidence.budget) && normalizedBudget
      ? raw.confidence.budget
      : "LOW";

  const extraction = {
    appType: raw && ["text", "image", "audio", "video", "vision"].includes(raw.appType) ? raw.appType : inferred.value,
    appPurpose: raw && typeof raw.appPurpose === "string" && raw.appPurpose.trim() ? raw.appPurpose.trim() : null,
    targetUsers: raw && typeof raw.targetUsers === "string" && raw.targetUsers.trim() && raw.targetUsers.trim() !== 'general users' ? raw.targetUsers.trim() : null,
    keyFeatures: raw && Array.isArray(raw.keyFeatures) ? raw.keyFeatures.filter(Boolean).slice(0, 6) : [],
    budget: normalizedBudget,
    wantsImageInput: Boolean(raw && raw.wantsImageInput) || /(photo|image|picture)/.test(message.toLowerCase()),
    detectedLanguage: raw && typeof raw.detectedLanguage === "string" && raw.detectedLanguage.trim() ? raw.detectedLanguage.trim() : detectLanguage(message),
    userTone: raw && ["urgent", "casual", "formal", "unsure"].includes(raw.userTone) ? raw.userTone : inferTone(message),
    userType: raw && ["enterprise", "business", "developer", "normal", "unknown"].includes(raw.userType) ? raw.userType : "unknown",
    enterpriseSignals: Boolean(raw && raw.enterpriseSignals),
    confidence: {
      appType:
        raw && raw.confidence && ["HIGH", "MEDIUM", "LOW"].includes(raw.confidence.appType)
          ? raw.confidence.appType
          : inferred.confidence,
      budget: normalizedBudgetConfidence
    },
    missingFields: raw && Array.isArray(raw.missingFields) ? raw.missingFields.filter(Boolean) : [],
    oneLineUnderstanding: raw && typeof raw.oneLineUnderstanding === "string" && raw.oneLineUnderstanding.trim() ? raw.oneLineUnderstanding.trim() : "",
    suggestedReply: raw && typeof raw.suggestedReply === "string" && raw.suggestedReply.trim() ? raw.suggestedReply.trim() : null
  };

  if (!extraction.oneLineUnderstanding && extraction.appType) {
    extraction.oneLineUnderstanding = `you want a ${extraction.appType} app`;
  } else if (!extraction.oneLineUnderstanding) {
    extraction.oneLineUnderstanding = buildOneLineUnderstanding(extraction);
  }
  if (!extraction.appType) {
    extraction.missingFields = Array.from(new Set([...(extraction.missingFields || []), "appType"]));
  }
  if (!extraction.targetUsers) {
    extraction.missingFields = Array.from(new Set([...(extraction.missingFields || []), "targetUsers"]));
  }
  return extraction;
}

export function applyPromptInstruction(promptData, instruction) {
  const cleanInstruction = (instruction || "").trim();
  const base = promptData || {
    systemPrompt: "",
    userPrompt: "",
    negativePrompt: null,
    acceptImageInput: false,
    variablesUsed: [],
    variableDescriptions: {}
  };
  if (!cleanInstruction) return base;
  const current = String(base.userPrompt || "");
  const suffix = ` Additional instruction: ${cleanInstruction}.`;
  const nextPrompt = current.includes(cleanInstruction) ? current : `${current}${suffix}`;
  return { ...base, userPrompt: nextPrompt };
}

export function buildPromptTemplateFromSession(session) {
  // Dummy implementation for stepRouter's synchronous fallback needs
  return {
    userPrompt: "Write about $$topic in a $$tone tone for $$audience at $$length length, optimized for $$goal.",
    negativePrompt: null,
    acceptImageInput: false,
    variablesUsed: ["topic", "tone", "audience", "length", "goal"],
    advancedSettings: { aspectRatio: null, quality: "balanced" }
  };
}


// ==========================================
// OPENROUTER IMPLEMENTATION
// ==========================================

function hasRealApiKey(value) {
  return Boolean(value && !/^your_.+_here$/i.test(String(value).trim()));
}

let openRouterClient = null;

function getOpenRouterClient() {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!hasRealApiKey(apiKey)) {
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

async function callOpenRouter(systemPrompt, userContent, retries = 2) {
  const client = getOpenRouterClient();
  const models = [
    'google/gemini-1.5-flash',
    'meta-llama/llama-3.3-70b-instruct'
  ];

  for (let i = 0; i < models.length; i++) {
    try {
      const response = await client.chat.completions.create({
        model: models[i],
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userContent }
        ],
        temperature: 0.7,
        max_tokens: 3000,
      });

      const raw = response.choices[0].message.content;
      
      // Robust JSON Extraction using Regex
      const jsonMatch = raw.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error("No JSON object found in response");
      }
      
      // Sanitize control characters that break JSON.parse (newlines/tabs inside string values)
      const sanitized = jsonMatch[0].replace(/[\u0000-\u001F\u007F]/g, (c) => {
        const allowed = { '\n': '\\n', '\r': '\\r', '\t': '\\t' };
        return allowed[c] || '';
      });
      
      try {
        return JSON.parse(sanitized);
      } catch {
        // Last resort: strip ALL control chars and try again
        return JSON.parse(jsonMatch[0].replace(/[\u0000-\u001F\u007F]/g, ' '));
      }

    } catch (err) {
      const is429 = err?.status === 429 || 
                    err?.message?.includes('429') ||
                    err?.message?.includes('quota') ||
                    err?.message?.includes('rate');
      
      if (is429 && i < models.length - 1) {
        console.log(`[OpenRouter] ${models[i]} rate limited, trying ${models[i+1]}...`);
        await new Promise(r => setTimeout(r, 1000));
        continue;
      }
      
      // If it's the last model or not a rate limit, throw to trigger the function's fallback
      if (i === models.length - 1) throw err;
    }
  }
}

export async function runPromptTest({ systemPrompt, userPrompt, testInputs, modelHint }) {
  const startedAt = Date.now();
  const model = modelHint || "google/gemini-1.5-flash";
  const inputs = testInputs && typeof testInputs === "object" ? testInputs : {};
  let resolvedPrompt = String(userPrompt || "");
  Object.entries(inputs).forEach(([key, value]) => {
    const token = `$$${key}`;
    const safeValue = String(value ?? "");
    resolvedPrompt = resolvedPrompt.split(token).join(safeValue);
  });

  const response = await getOpenRouterClient().chat.completions.create({
    model,
    temperature: 0.4,
    max_tokens: 700,
    messages: [
      { role: "system", content: String(systemPrompt || "You are a helpful AI assistant.") },
      { role: "user", content: resolvedPrompt }
    ]
  });

  const output = response.choices?.[0]?.message?.content || "";
  return {
    output: String(output).slice(0, 3000),
    modelUsed: model,
    latencyMs: Date.now() - startedAt,
    tokens: response.usage?.total_tokens || null
  };
}

export async function generatePromptTemplate(session) {
  const editInstruction = session.deepAnswers?.lastEditInstruction
    ? `\n- EDIT INSTRUCTION FROM CREATOR: ${session.deepAnswers.lastEditInstruction}` : '';

  const systemPrompt = `You are an Elite AI Prompt Engineer specializing in production-ready app prompts for the RentPrompts marketplace.
Your job: generate a HIGHLY SPECIFIC, DOMAIN-AWARE prompt configuration for the app described below.
${LANGUAGE_MIRROR_DIRECTIVE}

QUALITY RULES:
1. systemPrompt: Define a tight AI persona. Include: role, domain expertise, tone, output format rules, and constraints. 3-5 sentences.
2. userPrompt: Must be HIGHLY DETAILED (200-400 words). Structure it as:
   a) CONTEXT BLOCK: Set the scenario using the $$variables
   b) PROCESSING LOGIC: Step-by-step instructions on how to analyze/generate
   c) OUTPUT FORMAT: Exactly how the response should look (headings, bullets, JSON, etc.)
   d) CONSTRAINTS: What NOT to do (hallucinate, go off-topic, be generic)
3. Use $$snake_case_variable syntax ONLY for the REQUIRED INPUT VARIABLES listed below. Do not invent extra variables.
4. negativePrompt: For image/video apps, write a detailed negative prompt. For text/audio/vision, set to null.
5. acceptImageInput: true ONLY for vision and image-input apps (background removal, room design, plant disease).
6. NO META-PLATFORM DETAILS: NEVER mention the user's budget, coin cost, or model name (e.g., 'Nano Banana', 'Medium (5 - 20 coins)') in the systemPrompt or userPrompt. The prompt is strictly for the end-user using the app, who doesn't know about models or coins.
7. DOMAIN GROUNDING: If the app is legal/medical/agricultural, explicitly define domain-specific terms in systemPrompt.
8. DO NOT write generic prompts. Every sentence must be specific to THIS app's purpose.
9. NEVER include in prompts: model names (e.g. "Nano Banana", "GPT"), coin costs, budget tiers, platform names ("RentPrompts"), or any internal metadata. Prompts must be self-contained instructions only.

Return ONLY valid JSON:
{
  "reasoning": "Brief explanation of your design choices.",
  "systemPrompt": "Tight 3-5 sentence AI persona with role, domain, tone, format rules.",
  "userPrompt": "Highly detailed 200-400 word prompt with context, processing logic, output format, and constraints.",
  "negativePrompt": "Detailed negative prompt or null",
  "acceptImageInput": true or false,
  "variablesUsed": ["$$var1", "$$var2"],
  "variableDescriptions": { "$$var1": "What the user enters here" }
}`;

  const vars = session.dynamicContext?.variables || [];
  const varList = vars.map(v => typeof v === 'object' ? `$$${v.name}: ${v.placeholder || ''}` : `$$${v}`).join('\n');

  const userContent = `Generate a production-ready prompt for this app:
- App Type: ${session.appType}
- App Purpose: ${session.requirements?.appPurpose || session.extraction?.appPurpose || 'Not specified'}
- Target Users: ${session.requirements?.targetUsers || session.extraction?.targetUsers || 'General Public'}
- Domain Context from Chat: ${JSON.stringify(session.deepAnswers || {})}
- Conversation Summary: ${JSON.stringify((session.history || []).slice(-6).map(h => h.content))}
- REQUIRED INPUT VARIABLES (use EXACTLY these, no more, no less):
${varList || 'Use the most logical 3-4 variables for this app type and purpose.'}${editInstruction}`;

  try {
    const result = await callOpenRouter(systemPrompt, userContent);
    return result;
  } catch (err) {
    console.error('[Sub-agent 2] Error:', err.message);
    const acceptImage = session.appType === 'image' || session.appType === 'vision';
    const mainVar = vars[0]?.name || 'main_input';
    return {
      reasoning: "Fallback triggered.",
      systemPrompt: `You are a highly specialized AI assistant for ${session.appType || 'content'} generation. Focus exclusively on the app's stated purpose. Provide structured, accurate, and domain-specific outputs only.`,
      userPrompt: `Based on the following inputs, perform the requested task precisely:\n\n$$${mainVar}\n\nProvide a detailed, well-structured response that directly addresses the request. Do not add unrelated information.`,
      negativePrompt: acceptImage ? 'blurry, low quality, distorted, watermark, text overlay, pixelated, overexposed, underexposed' : null,
      acceptImageInput: acceptImage,
      variablesUsed: vars.slice(0, 3).map(v => `$$${typeof v === 'object' ? v.name : v}`),
      variableDescriptions: Object.fromEntries(vars.slice(0, 3).map(v => [`$$${typeof v === 'object' ? v.name : v}`, typeof v === 'object' ? v.placeholder : 'Enter details']))
    };
  }
}

export async function generateSEO(session) {
  const systemPrompt = `You are an SEO & Monetization Expert for AI app marketplaces.
Generate metadata that maximizes discoverability.
${LANGUAGE_MIRROR_DIRECTIVE}

Rules:
- App name: catchy, specific, under 55 characters.
- Description: mentions model + main benefit, under 155 characters.
- Tags: exactly 10 tags, lowercase, hyphens not spaces.
- Category must be one of: creative, business, education,
  healthcare, entertainment, productivity, social, other
- Suggested Price: Set this to the exact base model cost + a small 10% to 20% creator markup.
  This is a PER-GENERATION cost. Apps on this platform typically cost between 1 to 50 coins.
  DO NOT suggest thousands of coins. Keep it realistic and competitive.

Return ONLY valid JSON with this exact schema:
{
  "appName": "string max 55 chars",
  "appDescription": "string max 155 chars",
  "tags": ["tag1", "tag2", "tag3", "tag4", "tag5",
           "tag6", "tag7", "tag8", "tag9", "tag10"],
  "category": "creative | business | education | productivity | other",
  "suggestedPrice": 12.5
}`;

  const userContent = `
Generate SEO metadata for this app:
- App Type: ${session.appType}
- Deep Answers: ${JSON.stringify(session.deepAnswers || {})}
- Model Selected: ${session.modelId || 'unknown'}
- Base Model Cost: ${session.modelCost || 0} coins
- Generated System Prompt: ${session.promptData?.systemPrompt || 'N/A'}
  `;

  try {
    const result = await callOpenRouter(systemPrompt, userContent);
    console.log('[Sub-agent 3] SEO generated OK');
    return result;
  } catch (err) {
    console.error('[Sub-agent 3] Error:', err.message);
    return {
      appName: `AI ${session.appType || 'Content'} Generator`,
      appDescription: `Generate ${session.appType} content with AI. Fast, easy, professional results.`,
      tags: ['ai-generated', 'content-creation', 'automation',
             'ai-tool', 'no-code', 'productivity',
             session.appType || 'creative', 'rentprompts',
             'generative-ai', 'easy-to-use'],
      category: 'creative',
      suggestedPrice: (session.modelCost || 5) * 1.2
    };
  }
}
