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
1. systemPrompt: Define a tight AI persona. Include: role, domain expertise, tone, output format rules, and constraints. 3-5 sentences. It MUST be written in the second-person ("You are...") rather than first-person ("I am...").
2. userPrompt — FORMAT DEPENDS ON APP TYPE:
   ▸ For TEXT and AUDIO apps: Must be HIGHLY DETAILED (200-400 words). Structure it as:
     a) A first-person scenario using the $$variables ("I want...", "My topic is...")
     b) Step-by-step processing logic
     c) Output format specification (headings, bullets, structure)
     d) Constraints (what NOT to do)
     DO NOT use markdown headers (##) inside the prompt — write it as flowing prose with labeled sections.
   ▸ For IMAGE and VIDEO apps: Must be a CONCISE VISUAL DESCRIPTION (50-120 words). Write it as:
     - A single flowing visual prompt describing the desired output
     - Incorporate $$variables naturally: "A $$style $$subject with $$details"
     - Include art direction: lighting, camera angle, color palette, mood
     - End with quality keywords: "professional photography, 8K, ultra-detailed"
     - DO NOT use ## headers, numbered steps, or "Processing Logic" — image generators don't read structured prompts
     ❌ WRONG for image: "## Context\nI want to generate an image...\n## Processing Logic\n1. Identify the subject..."
     ✅ RIGHT for image: "A $$style room design featuring $$subject, with $$details. Warm ambient lighting, architectural photography, 8K resolution."
3. Use $$snake_case_variable syntax ONLY for the REQUIRED INPUT VARIABLES listed below. Do not invent extra variables.
4. negativePrompt: For image/video apps, write a detailed negative prompt. For text/audio/vision, set to null.
5. acceptImageInput: SMART DETECTION — do NOT blindly set true for all image apps.
   - SET TRUE (user must upload a photo) ONLY when the app TRANSFORMS or ANALYZES an existing image:
     ✓ Background removal / Background replacer
     ✓ Room redesign / Interior design from photo
     ✓ Face swap / Portrait enhancement
     ✓ Image style transfer / Artistic filter
     ✓ Plant disease detection from photo
     ✓ Product photo editing
     ✓ Any "upload your photo" use case
   - SET FALSE (pure generation, no upload needed) when the app CREATES from text inputs:
     ✗ Logo designer / Brand logo generator
     ✗ Poster / Flyer / Banner creator from text
     ✗ Avatar generator from description
     ✗ Landscape / Fantasy / Sci-fi image generator
     ✗ Album art / Thumbnail creator from description
     ✗ Any app where user describes what they want (not uploads what they have)
   RULE: If the user describes what the OUTPUT should look like → FALSE.
         If the user uploads what they want to TRANSFORM → TRUE.
6. NO META-PLATFORM DETAILS: NEVER mention the user's budget, coin cost, or model name (e.g., 'Nano Banana', 'Medium (5 - 20 coins)') in the systemPrompt or userPrompt. The prompt is strictly for the end-user using the app, who doesn't know about models or coins.
7. DOMAIN GROUNDING: If the app is legal/medical/agricultural, explicitly define domain-specific terms in systemPrompt.
8. DO NOT write generic prompts. Every sentence must be specific to THIS app's purpose.
9. NEVER include in prompts: model names (e.g. "Nano Banana", "GPT"), coin costs, budget tiers, platform names ("RentPrompts"), or any internal metadata. Prompts must be self-contained instructions only.
10. USER PERSPECTIVE PRINCIPLE (CRITICAL): Variables must reflect what a NON-EXPERT end-user can actually provide.
    - WRONG: asking a layperson for "section_number", "diagnosis_code", "article_number", "statute_citation" — they DON'T know these
    - RIGHT: ask for "incident_description" (what happened), "symptoms" (how they feel), "situation_type" (type of problem)
    - The AI in the prompt should DERIVE the technical/legal/medical answer from the user's plain description
    - LEGAL APPS: User describes the incident → AI identifies applicable sections → AI explains them in plain language
    - MEDICAL APPS: User describes symptoms → AI identifies conditions → AI gives advice
    - GOVERNMENT/COMPLIANCE APPS: User describes their situation → AI identifies relevant rules → AI explains them
    - If variables list contains expert-level inputs (section numbers, code references), REPLACE them with plain-language equivalents
11. CONFLICT DETECTION & ANTI-HALLUCINATION (CRITICAL for legal/medical/expert apps):
    - If the app has multiple inputs that could contradict each other (e.g., section number + incident description), the generated systemPrompt MUST include a cross-validation instruction:
      "CROSS-VALIDATION: Before answering, verify that all provided inputs are internally consistent. If $$section_number and $$incident_description describe different types of offenses, explicitly flag the mismatch: (1) state what IPC Section [X] actually covers, (2) identify which section correctly applies to the described incident, (3) explain the discrepancy in plain language. NEVER silently merge contradictory inputs into a fabricated answer."
    - For legal apps: The AI must cite actual IPC/BNS section text and real-world examples. If unsure, it must say so — never invent section content.
    - For medical apps: The AI must note when symptoms don't match stated diagnosis and recommend professional consultation.
    - The CONSTRAINTS block in userPrompt must explicitly say: "Do NOT hallucinate section numbers, case outcomes, or legal precedents. If the input is ambiguous or contradictory, state that clearly."
12. FIRST-PERSON PERSPECTIVE (CRITICAL — #1 mistake to avoid):
    The userPrompt is written FROM the end-user's point of view. The person running the app IS the real user — they are describing what THEY want.
    - ALWAYS use first person: "I want...", "I need...", "My topic is...", "I am looking for..."
    - NEVER use second person: "You want...", "You need...", "Your topic is..."
    - NEVER use third person: "The user wants...", "The user needs..."
    ❌ WRONG: "You want to generate educational audio content on a specific topic."
    ✅ RIGHT: "I want to generate educational audio content on $$topic."
    ❌ WRONG: "You have a preferred format in mind."
    ✅ RIGHT: "My preferred format is $$format."
    The systemPrompt (AI persona) MUST address the AI in the second-person ("You are...") rather than first-person ("I am..."). The userPrompt must ALWAYS be first-person ("I want...", "My sign is...").

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
    // Smart fallback: determine if app needs image upload based on purpose keywords
    const appPurpose = (session.requirements?.appPurpose || session.extraction?.appPurpose || '').toLowerCase();
    const UPLOAD_KEYWORDS = [
      'background remov', 'background replac', 'room design', 'interior design',
      'redesign', 'style transfer', 'face swap', 'portrait', 'enhance photo',
      'edit photo', 'photo editing', 'image editing', 'remove background',
      'plant disease', 'crop disease', 'object detect', 'image analys',
      'my photo', 'my image', 'my room', 'my product', 'photo to', 'image to'
    ];
    const GENERATION_KEYWORDS = [
      'logo', 'poster', 'flyer', 'banner', 'thumbnail creator', 'album art',
      'icon generator', 'brand logo', 'text to image', 'generate image',
      'create image', 'fantasy', 'landscape generator', 'ai art generator'
    ];
    const needsUpload = UPLOAD_KEYWORDS.some(k => appPurpose.includes(k));
    const isGeneration = GENERATION_KEYWORDS.some(k => appPurpose.includes(k));
    const acceptImage =
      session.appType === 'vision' ||
      (session.appType === 'image' && needsUpload && !isGeneration);
    const mainVar = vars[0]?.name || 'main_input';
    return {
      reasoning: "Fallback triggered.",
      systemPrompt: `You are a highly specialized AI assistant for ${session.appType || 'content'} generation. Focus exclusively on the app's stated purpose. Provide structured, accurate, and domain-specific outputs only.`,
      userPrompt: `Based on the following inputs, perform the requested task precisely:\n\n$$${mainVar}\n\nProvide a detailed, well-structured response that directly addresses the request. Do not add unrelated information.`,
      negativePrompt: (session.appType === 'image' || session.appType === 'vision') ? 'blurry, low quality, distorted, watermark, text overlay, pixelated, overexposed, underexposed' : null,
      acceptImageInput: acceptImage,
      variablesUsed: vars.slice(0, 3).map(v => `$$${typeof v === 'object' ? v.name : v}`),
      variableDescriptions: Object.fromEntries(vars.slice(0, 3).map(v => [`$$${typeof v === 'object' ? v.name : v}`, typeof v === 'object' ? v.placeholder : 'Enter details']))
    };
  }
}

export async function generateSEO(session) {
  const systemPrompt = `You are an expert Product Marketer and App Store Optimization (ASO) specialist for a premium AI app marketplace.
Your job: generate irresistible, high-converting marketplace metadata that makes users WANT to try the app instantly.
${LANGUAGE_MIRROR_DIRECTIVE}

STRICT RULES — follow every single one:

1. APP NAME (2-4 words MAXIMUM):
   - Must be catchy, memorable, and sound like a premium SaaS product.
   - Use power words: Forge, Studio, Craft, Spark, Genius, Pulse, Flow, Snap, Boost, Vault, Hive, Lens, Dash, Scope.
   - Format options (pick whichever sounds best):
     a) "BrandName: AI Subtitle" — e.g., "BrandForge: AI Logo Creator"
     b) "ActionNoun + Domain" — e.g., "ScriptSpark AI"
     c) "Catchy Compound" — e.g., "CropGuard Pro"
   - NEVER use generic names like "Logo Maker", "Text Generator", "Image Tool", "Content App".
   - NEVER include model names (lyria, gemini, groq, llama, GPT).
   - Maximum 55 characters including subtitle.

2. DESCRIPTION (under 150 characters, STRICT):
   - Write engaging, benefit-driven copy — NOT a feature list.
   - Lead with the VALUE to the user, not what the app "does".
   - Highlight the "AI-powered" or "AI" aspect naturally.
   - Use action verbs: Transform, Generate, Create, Craft, Design, Analyze, Unlock.
   - GOOD: "Transform any idea into a stunning brand logo in seconds with AI."
   - GOOD: "AI-powered legal guidance — describe your situation, get clear answers instantly."
   - BAD: "Creates logos in PNG format." (too dry)
   - BAD: "This app generates text content using AI." (too generic)
   - NEVER mention model names, coin costs, or platform internals.

3. TAGS (EXACTLY 7 tags):
   - Highly searched, relevant, action-oriented SEO tags.
   - Use hyphens, lowercase, no spaces.
   - Focus on what users SEARCH for: user intent, use case, and benefit.
   - DO NOT prefix tags with the '#' symbol. Return them as plain lowercase string literals.
   - GOOD tags: "brand-design", "ai-logo", "content-creator", "smart-writing", "photo-magic"
   - BAD tags: "#brand-design", "png-output", "text-generation", "api-call"
   - Mix of: 2 broad discovery tags + 3 niche use-case tags + 2 benefit/action tags.

4. CATEGORY: one of: creative, business, education, healthcare, entertainment, productivity, social, other

Return ONLY valid JSON with this exact schema:
{
  "appName": "string, 2-4 words, max 55 chars",
  "appDescription": "string, benefit-driven, max 150 chars",
  "tags": ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7"],
  "category": "creative | business | education | healthcare | entertainment | productivity | social | other"
}`;

  const appPurpose = session.extraction?.appPurpose
    || session.extraction?.oneLineUnderstanding
    || '';
  const deepAnswers = session.deepAnswers || {};
  const triageHistory = (session.history || [])
    .filter(m => m.role === 'user')
    .map(m => m.content)
    .join(' | ')
    .slice(0, 600);

  const userContent = `Generate premium, high-converting marketplace metadata for this AI app:
- What the app does: ${appPurpose}
- App type: ${session.appType}
- User answers during setup: ${JSON.stringify(deepAnswers)}
- Conversation context: ${triageHistory}
- Model used (DO NOT use as app name — this is internal only): ${session.modelId || 'unknown'}

Remember: App name must sound like a premium SaaS product (2-4 words). Description must sell the benefit to the user in under 150 characters. Exactly 7 action-oriented tags.`;


  try {
    const result = await callOpenRouter(systemPrompt, userContent);
    console.log('[Sub-agent 3] SEO generated OK');
    return result;
  } catch (err) {
    console.error('[Sub-agent 3] Error:', err.message);
    // Premium-quality fallback — still sounds marketable even without the LLM
    const purposeClean = (appPurpose || '').trim();
    const typeLabel = session.appType || 'creative';
    const nameMap = {
      image: 'PixelForge AI',
      text: 'CopyFlow AI',
      audio: 'VoiceCraft AI',
      video: 'ReelSpark AI',
      vision: 'InsightLens AI'
    };
    const descMap = {
      image: `Create stunning AI-powered visuals${purposeClean ? ` — ${purposeClean}` : ''} in seconds.`,
      text: `Craft polished, professional content${purposeClean ? ` — ${purposeClean}` : ''} with AI.`,
      audio: `Transform text into natural, professional audio${purposeClean ? ` — ${purposeClean}` : ''} instantly.`,
      video: `Produce scroll-stopping AI videos${purposeClean ? ` — ${purposeClean}` : ''} effortlessly.`,
      vision: `Analyze and extract insights from images${purposeClean ? ` — ${purposeClean}` : ''} with AI.`
    };
    const tagMap = {
      image: ['ai-design', 'visual-creator', 'brand-design', 'image-craft', 'creative-ai', 'instant-design', 'smart-visuals'],
      text: ['ai-writing', 'content-creator', 'smart-copy', 'professional-writing', 'instant-content', 'creative-ai', 'productivity-boost'],
      audio: ['ai-voice', 'text-to-speech', 'audio-creator', 'voice-studio', 'podcast-tool', 'smart-audio', 'creative-ai'],
      video: ['ai-video', 'reel-maker', 'video-creator', 'motion-design', 'content-studio', 'creative-ai', 'instant-video'],
      vision: ['ai-vision', 'image-analysis', 'smart-scan', 'visual-ai', 'insight-tool', 'data-extraction', 'intelligent-scan']
    };
    return {
      appName: nameMap[typeLabel] || 'SparkAI Studio',
      appDescription: (descMap[typeLabel] || `Unlock AI-powered ${typeLabel} creation — fast, polished, professional results.`).slice(0, 150),
      tags: tagMap[typeLabel] || ['ai-powered', 'smart-tool', 'instant-results', 'creative-ai', 'productivity', 'no-code', 'professional'],
      category: 'creative'
    };
  }
}
