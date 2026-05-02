import OpenAI from "openai";
import MODELS from "./models.js";

// Keep old exports needed by stepRouter and groq
export const EXTRACTION_PROMPT = `You are a strict data extraction engine for RentPrompts — a platform where users CREATE and PUBLISH AI-powered apps.

Users describe an app they want to build.
Your ONLY job: extract what they said. Never invent.

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
  const lower = message.toLowerCase();
  const hindiSignals = ["mujhe", "banana", "banani", "hai", "likhta", "likh", "jo", "ke liye", "karna", "ek "];
  if (hindiSignals.some((word) => lower.includes(word))) return "Hindi";
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
  if (/(blog|write|writer|copy|text|article|caption|planner|workout|meal|diet|itinerary|guide|report)/.test(lower)) return { value: "text", confidence: "HIGH" };
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
    // ANTI-HALLUCINATION: null if user didn't describe it
    appPurpose: raw && typeof raw.appPurpose === "string" && raw.appPurpose.trim() ? raw.appPurpose.trim() : null,
    // ANTI-HALLUCINATION: null if user didn't mention them (never default to 'general users')
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
  if (!cleanInstruction) return promptData;
  const suffix = ` Additional instruction: ${cleanInstruction}.`;
  const nextPrompt = promptData.userPrompt.includes(cleanInstruction) ? promptData.userPrompt : `${promptData.userPrompt}${suffix}`;
  return { ...promptData, userPrompt: nextPrompt };
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

const client = new OpenAI({
  baseURL: 'https://openrouter.ai/api/v1',
  apiKey: process.env.OPENROUTER_API_KEY,
  defaultHeaders: {
    'HTTP-Referer': 'http://localhost:5173',
    'X-Title': 'RentPrompts Agent Demo',
  }
});

async function callOpenRouter(systemPrompt, userContent, retries = 2) {
  const models = [
    'google/gemini-1.5-pro',
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
        response_format: { type: 'json_object' },
        temperature: 0.7,
        max_tokens: 1000,
      });

      const raw = response.choices[0].message.content;
      
      const cleaned = raw
        .replace(/```json/gi, '')
        .replace(/```/g, '')
        .trim();

      return JSON.parse(cleaned);

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
      
      throw err;
    }
  }
}

export async function generatePromptTemplate(session) {
  const systemPrompt = `You are an expert AI app prompt engineer for RentPrompts marketplace. 
You have a high level of understanding of human needs.

Create production-ready prompt templates using $$variable syntax for every input the end-user will provide.

Variable rules per app type:
- image: use $$subject $$style $$mood $$lighting $$background
- video: use $$scene $$duration $$camera_movement $$style $$mood
- text:  use $$topic $$tone $$audience $$length $$format
- audio: use $$text $$voice_style $$pace $$emotion
- vision: use $$image_description $$analysis_type $$output_format

Make prompts detailed, professional, and effective.
Negative prompts for image and video types only.

Return ONLY valid JSON with this exact schema:
{
  "userPrompt": "full prompt string with $$variables",
  "negativePrompt": "string or null",
  "acceptImageInput": true or false,
  "variablesUsed": ["$$var1", "$$var2"],
  "variableDescriptions": {
    "$$var1": "what the user should enter here"
  },
  "advancedSettings": {
    "aspectRatio": "16:9 or 1:1 or 9:16 or null",
    "quality": "standard or high or null",
    "duration": "5s or 10s or null"
  },
  "promptExplanation": "one sentence why this prompt works"
}`;

  const userContent = `Generate a prompt template for this app:
App type: ${session.appType}
Purpose: ${session.extraction?.appPurpose || 'not specified'}
Key features: ${JSON.stringify(session.extraction?.keyFeatures || [])}
Target users: ${session.extraction?.targetUsers || 'general users'}
Selected model: ${session.modelId || 'not selected yet'}
Wants image input: ${session.extraction?.wantsImageInput || false}`;

  try {
    const result = await callOpenRouter(systemPrompt, userContent);
    console.log('[Sub-agent 2] Prompt template generated OK');
    return result;
  } catch (err) {
    console.error('[Sub-agent 2] Error:', err.message);
    return {
      userPrompt: `Create content about $$topic in $$style style for $$audience`,
      negativePrompt: null,
      acceptImageInput: false,
      variablesUsed: ['$$topic', '$$style', '$$audience'],
      variableDescriptions: {
        '$$topic': 'Main subject or theme',
        '$$style': 'Desired style or tone',
        '$$audience': 'Target audience'
      },
      advancedSettings: { aspectRatio: null, quality: null, duration: null },
      promptExplanation: 'Generic template — please refine'
    };
  }
}

export async function generateSEO(session) {
  const systemPrompt = `You are an SEO expert for AI app marketplaces with deep understanding of what users search for.

Generate metadata that maximizes discoverability on RentPrompts marketplace.

Rules:
- App name: catchy, specific, under 55 characters
- Description: mentions model + main benefit + target user,
  under 155 characters, no filler words
- Tags: exactly 10 tags, all lowercase, 
  use hyphens not spaces (e.g. image-generation not image generation)
- Category must be one of: creative, business, education,
  healthcare, entertainment, productivity, social, other
- suggestedPrice: slightly above model cost so creator profits
  (minimum model cost + 20% margin, rounded to nearest 0.5)

Return ONLY valid JSON with this exact schema:
{
  "appName": "string max 55 chars",
  "appDescription": "string max 155 chars",
  "tags": ["tag1", "tag2", "tag3", "tag4", "tag5", 
           "tag6", "tag7", "tag8", "tag9", "tag10"],
  "category": "one of the categories listed above",
  "suggestedPrice": 12.5
}`;

  const userContent = `Generate SEO metadata for this AI app:
App type: ${session.appType}
Purpose: ${session.extraction?.appPurpose || 'AI powered app'}
Target users: ${session.extraction?.targetUsers || 'general users'}
Key features: ${JSON.stringify(session.extraction?.keyFeatures || [])}
Selected model: ${session.modelId || 'unknown'}
Cost per run: ${session.modelCost || 0} coins
Prompt template: ${session.promptData?.userPrompt || 'not generated yet'}`;

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

export async function generateScope(session) {
  const systemPrompt = `You are ARIA, expert project scope analyst for RentPrompts bounty platform.

Generate a detailed scope like a senior project manager.
Include realistic hour estimates and AI/human breakdown.

Return ONLY valid JSON:
{
  "scopeSummary": "one sentence total scope description",
  "totalItems": 5,
  "totalHours": 43,
  "aiAcceleratedPercent": 65,
  "humanDirectedPercent": 35,
  "freelancerCostMix": {
    "frontend": { "items": 3, "percent": 60 },
    "backend": { "items": 2, "percent": 40 }
  },
  "items": [
    {
      "title": "Image Upload Functionality",
      "complexity": "simple",
      "priority": "Must Have",
      "aiAssisted": true,
      "estimatedHours": 6,
      "codeGenPercent": 70,
      "aiRole": "AI generates React component code for file input and validation",
      "humanRole": "Human decides UX flow, error handling, and security requirements",
      "description": "Frontend component for image upload with preview"
    }
  ],
  "constraints": {
    "applicantLimit": 3,
    "timeline": "1 week",
    "pricingLane": "Web Application"
  }
}

Rules:
- 4-6 scope items always
- complexity: simple|medium|complex
- priority: Must Have|Should Have|Nice to Have
- simple=4-6h, medium=7-12h, complex=13-20h
- Last item should be Nice to Have
- aiAcceleratedPercent = 60-80 for AI apps
- humanDirectedPercent = 100 - aiAcceleratedPercent
- codeGenPercent per task: simple=70-80, medium=60-70, complex=50-65
- aiRole: what AI tools like Cursor/Copilot will do
- humanRole: what requires human judgment
- pricingLane: Web Application|Mobile App|AI Integration
- Return ONLY valid JSON, no explanation`;

  const userContent = `Generate scope for:
App type: ${session.appType}
Purpose: ${session.extraction?.appPurpose}
Features requested: ${JSON.stringify(session.extraction?.keyFeatures || [])}
Target users: ${session.extraction?.targetUsers}
Deep answers: ${JSON.stringify(session.deepAnswers || {})}
Prompt template: ${session.promptData?.userPrompt}`;

  try {
    const result = await callOpenRouter(systemPrompt, userContent);
    console.log('[Scope] Scope generated OK —', result.totalItems, 'items');
    return result;
  } catch (err) {
    console.error('[Scope] Error:', err.message);
    return {
      scopeSummary: 'Core app features and essential components',
      totalItems: 5,
      totalHours: 32,
      items: [
        { title: 'Core App Interface', complexity: 'complex', priority: 'Must Have', aiAssisted: false, estimatedHours: 12, description: 'Main user interface and primary functionality.' },
        { title: 'AI Model Integration', complexity: 'complex', priority: 'Must Have', aiAssisted: false, estimatedHours: 10, description: 'Connect and configure the selected AI model.' },
        { title: 'Input/Output Handler', complexity: 'medium', priority: 'Must Have', aiAssisted: true, estimatedHours: 6, description: 'Handle user inputs and display AI outputs.' },
        { title: 'Basic Styling', complexity: 'simple', priority: 'Should Have', aiAssisted: true, estimatedHours: 4, description: 'Clean, responsive visual design.' }
      ]
    };
  }
}
