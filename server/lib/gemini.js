import { GoogleGenerativeAI } from "@google/generative-ai";
import MODELS from "./models.js";
const geminiApiKey = process.env.GEMINI_API_KEY;

const EXTRACTION_PROMPT = `You are an AI app creation assistant for RentPrompts.
Extract structured data from the user's message.
Handle any language: English, Hindi, Hinglish, slang, typos.
Return ONLY valid JSON. No explanation. No markdown.

Output this exact schema:
{
  appType: 'text'|'image'|'audio'|'video'|'vision'|null,
  appPurpose: string,
  targetUsers: string,
  keyFeatures: string[],
  budget: 'free'|'low'|'medium'|'high'|'ultra'|null,
  wantsImageInput: boolean,
  detectedLanguage: string,
  userTone: 'urgent'|'casual'|'formal'|'unsure',
  confidence: {
    appType: 'HIGH'|'MEDIUM'|'LOW',
    budget: 'HIGH'|'MEDIUM'|'LOW'
  },
  missingFields: string[],
  oneLineUnderstanding: string
}`;

const PROMPT_TEMPLATE_SYSTEM = `You are an expert AI app prompt engineer for RentPrompts.
Generate a production-ready prompt template using $$variable
syntax for all user inputs.

Rules:
- Every input end-user will provide = $$variableName
- Image apps: include $$subject $$style $$mood
- Video apps: include $$scene $$duration $$camera_movement
- Text apps: include $$topic $$tone $$audience $$length
- Audio apps: include $$text $$voice $$style
- Make negative prompts for image and video types only

Return ONLY valid JSON:
{
  userPrompt: string,
  negativePrompt: string|null,
  acceptImageInput: boolean,
  variablesUsed: string[],
  advancedSettings: {
    aspectRatio: string|null,
    quality: string|null
  }
}`;

const SEO_SYSTEM = `Generate SEO metadata for an AI app on RentPrompts marketplace.
Be specific, creative, keyword-rich.

Return ONLY valid JSON:
{
  appName: string (max 55 chars, catchy),
  appDescription: string (max 155 chars, mentions model + use case),
  tags: string[] (10 tags, lowercase, hyphens not spaces),
  category: string,
  suggestedPrice: number (coins per run, based on model cost)
}`;

function hasRealValue(value) {
  return Boolean(value && !/^your_.+_here$/i.test(value));
}

const genAI = hasRealValue(geminiApiKey) ? new GoogleGenerativeAI(geminiApiKey) : null;

function getGeminiModel(systemInstruction) {
  if (!genAI) {
    return null;
  }

  return genAI.getGenerativeModel({
    model: "gemini-1.5-flash",
    systemInstruction
  });
}

function safeJsonParse(text) {
  try {
    return JSON.parse(text);
  } catch (error) {
    return null;
  }
}

function getTextPart(result) {
  const text = result && result.response && typeof result.response.text === "function" ? result.response.text() : "";
  return text || "";
}

function detectLanguage(message) {
  const lower = message.toLowerCase();
  const hindiSignals = ["mujhe", "banana", "banani", "hai", "likhta", "likh", "jo", "ke liye", "karna", "ek "];

  if (hindiSignals.some((word) => lower.includes(word))) {
    return "Hindi";
  }

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

function inferAppType(message) {
  const lower = message.toLowerCase();

  if (/(video|animate|reel|cinematic|movie)/.test(lower)) return { value: "video", confidence: "HIGH" };
  if (/(blog|write|writer|copy|text|article|caption)/.test(lower)) return { value: "text", confidence: "HIGH" };
  if (/(image|photo|picture|poster|thumbnail|logo|design)/.test(lower)) return { value: "image", confidence: "HIGH" };
  if (/(voice|audio|music|podcast|song|speech)/.test(lower)) return { value: "audio", confidence: "HIGH" };
  if (/(vision|analy(s|z)e|ocr|scan|detect|inspect)/.test(lower)) return { value: "vision", confidence: "HIGH" };
  if (/(clinic|hospital|business|website)/.test(lower)) return { value: null, confidence: "LOW" };

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

function normalizeExtraction(raw, fallbackMessage) {
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
    appPurpose: raw && typeof raw.appPurpose === "string" && raw.appPurpose.trim() ? raw.appPurpose.trim() : message.trim() || "AI app creation",
    targetUsers: raw && typeof raw.targetUsers === "string" && raw.targetUsers.trim() ? raw.targetUsers.trim() : "general users",
    keyFeatures: raw && Array.isArray(raw.keyFeatures) ? raw.keyFeatures.filter(Boolean).slice(0, 6) : extractFeatures(message),
    budget: normalizedBudget,
    wantsImageInput: Boolean(raw && raw.wantsImageInput) || /(photo|image|picture)/.test(message.toLowerCase()),
    detectedLanguage: raw && typeof raw.detectedLanguage === "string" && raw.detectedLanguage.trim() ? raw.detectedLanguage.trim() : detectLanguage(message),
    userTone: raw && ["urgent", "casual", "formal", "unsure"].includes(raw.userTone) ? raw.userTone : inferTone(message),
    confidence: {
      appType:
        raw && raw.confidence && ["HIGH", "MEDIUM", "LOW"].includes(raw.confidence.appType)
          ? raw.confidence.appType
          : inferred.confidence,
      budget: normalizedBudgetConfidence
    },
    missingFields: raw && Array.isArray(raw.missingFields) ? raw.missingFields.filter(Boolean) : [],
    oneLineUnderstanding: raw && typeof raw.oneLineUnderstanding === "string" && raw.oneLineUnderstanding.trim() ? raw.oneLineUnderstanding.trim() : ""
  };

  if (!extraction.oneLineUnderstanding) {
    extraction.oneLineUnderstanding = buildOneLineUnderstanding(extraction);
  }

  if (!extraction.appType) {
    extraction.missingFields = Array.from(new Set([...(extraction.missingFields || []), "appType"]));
  }

  if (!extraction.targetUsers || extraction.targetUsers === "general users") {
    extraction.missingFields = Array.from(new Set([...(extraction.missingFields || []), "targetUsers"]));
  }

  return extraction;
}

function getSelectedModel(session) {
  const list = MODELS[session.appType] || [];
  return list.find((model) => model.id === session.modelId) || null;
}

function buildPromptTemplateFromSession(session) {
  const extraction = session.extraction || {};
  const selectedModel = getSelectedModel(session);
  const acceptsImageInput = Boolean((selectedModel && selectedModel.supports_image_input) || extraction.wantsImageInput);

  if (session.appType === "video") {
    return {
      userPrompt: acceptsImageInput
        ? "Transform $$photo into a polished cinematic video for $$audience. Keep the look $$style, the mood $$mood, the scene focus on $$scene, use $$camera_movement, and keep the final runtime at $$duration."
        : "Create a cinematic video about $$scene for $$audience with a $$style look, $$camera_movement camera movement, $$mood atmosphere, and $$duration runtime.",
      negativePrompt: "blurry frames, warped motion, flicker, distorted faces, bad anatomy, low detail, noisy lighting",
      acceptImageInput: acceptsImageInput,
      variablesUsed: acceptsImageInput
        ? ["photo", "style", "mood", "scene", "duration", "camera_movement", "audience"]
        : ["scene", "style", "mood", "duration", "camera_movement", "audience"],
      advancedSettings: {
        aspectRatio: "16:9",
        quality: "high"
      }
    };
  }

  if (session.appType === "image") {
    return {
      userPrompt: "Generate an image of $$subject in a $$style style with a $$mood mood for $$audience.",
      negativePrompt: "blurry, distorted, low quality, bad composition, extra limbs, watermark, text artifacts",
      acceptImageInput: acceptsImageInput,
      variablesUsed: ["subject", "style", "mood", "audience"],
      advancedSettings: {
        aspectRatio: "1:1",
        quality: "high"
      }
    };
  }

  if (session.appType === "audio") {
    return {
      userPrompt: "Turn $$text into audio using a $$voice voice in a $$style style for $$audience.",
      negativePrompt: null,
      acceptImageInput: false,
      variablesUsed: ["text", "voice", "style", "audience"],
      advancedSettings: {
        aspectRatio: null,
        quality: "studio"
      }
    };
  }

  if (session.appType === "vision") {
    return {
      userPrompt: "Analyze $$image for $$goal and return results suited for $$audience with a $$tone tone.",
      negativePrompt: null,
      acceptImageInput: true,
      variablesUsed: ["image", "goal", "audience", "tone"],
      advancedSettings: {
        aspectRatio: null,
        quality: "detailed"
      }
    };
  }

  return {
    userPrompt: "Write about $$topic in a $$tone tone for $$audience at $$length length, optimized for $$goal.",
    negativePrompt: null,
    acceptImageInput: false,
    variablesUsed: ["topic", "tone", "audience", "length", "goal"],
    advancedSettings: {
      aspectRatio: null,
      quality: "balanced"
    }
  };
}

function applyPromptInstruction(promptData, instruction) {
  const cleanInstruction = (instruction || "").trim();

  if (!cleanInstruction) {
    return promptData;
  }

  const suffix = ` Additional instruction: ${cleanInstruction}.`;
  const nextPrompt = promptData.userPrompt.includes(cleanInstruction) ? promptData.userPrompt : `${promptData.userPrompt}${suffix}`;

  return {
    ...promptData,
    userPrompt: nextPrompt
  };
}

function computeSuggestedPrice(modelCost) {
  if (!modelCost || modelCost <= 0) {
    return 5;
  }

  return Math.max(5, Math.round(modelCost * 1.8 + 2));
}

function buildSeoFromSession(session) {
  const appType = session.appType || "text";
  const modelId = session.modelId || "custom-model";
  const extraction = session.extraction || {};

  let appName = "AI App Builder";
  let category = appType;
  let description = `Built with ${modelId} for a flexible ${appType} workflow.`;
  let tags = [
    "ai-app",
    `${appType}-generator`,
    "rentprompts",
    "automation",
    "workflow",
    "creator-tools",
    "marketplace-app",
    "json-ready",
    "prompt-template",
    "demo-build"
  ];

  if (appType === "video" && extraction.wantsImageInput) {
    appName = "Photo to Cinematic Video AI";
    description = `Turn photos into cinematic videos with ${modelId} using image-guided motion, style controls, and faster prompt-to-publish setup.`;
    tags = [
      "photo-to-video",
      "cinematic-video",
      "image-to-video",
      "ai-video-generator",
      "motion-control",
      "content-creation",
      "rentprompts",
      "video-automation",
      "creator-tools",
      "visual-storytelling"
    ];
  } else if (appType === "text") {
    appName = /blog/i.test(extraction.appPurpose || "") ? "AI Blog Writer Pro" : "Smart Text Generator AI";
    description = `Create polished ${/blog/i.test(extraction.appPurpose || "") ? "blog posts" : "text outputs"} with ${modelId} using prompt variables built for repeatable publishing.`;
    tags = [
      "ai-writer",
      "blog-generator",
      "content-marketing",
      "text-automation",
      "prompt-template",
      "rentprompts",
      "seo-writing",
      "copywriting-ai",
      "creator-tools",
      "writing-workflow"
    ];
  } else if (appType === "image") {
    appName = "Creative Image Prompt Studio";
    description = `Generate consistent image outputs with ${modelId} using variable-driven prompts for style, subject, and mood.`;
    tags = [
      "image-generator",
      "prompt-studio",
      "creative-ai",
      "visual-design",
      "rentprompts",
      "art-direction",
      "image-workflow",
      "style-control",
      "creative-tools",
      "marketing-visuals"
    ];
  }

  return {
    appName: appName.slice(0, 55),
    appDescription: description.slice(0, 155),
    tags: tags.slice(0, 10),
    category,
    suggestedPrice: computeSuggestedPrice(session.modelCost)
  };
}

async function runGeminiJson(systemInstruction, payload) {
  const model = getGeminiModel(systemInstruction);

  if (!model) {
    return null;
  }

  const result = await model.generateContent({
    generationConfig: {
      responseMimeType: "application/json"
    },
    contents: [
      {
        role: "user",
        parts: [{ text: payload }]
      }
    ]
  });

  return safeJsonParse(getTextPart(result));
}

async function extractRequirementsWithGemini(message, history) {
  const payload = JSON.stringify({
    message,
    history: Array.isArray(history) ? history.slice(-8) : []
  });

  try {
    const response = await runGeminiJson(EXTRACTION_PROMPT, payload);
    return normalizeExtraction(response, message);
  } catch (error) {
    return normalizeExtraction(null, message);
  }
}

async function generatePromptTemplate(session, editInstruction) {
  const baseFallback = buildPromptTemplateFromSession(session);
  const fallback = editInstruction ? applyPromptInstruction(baseFallback, editInstruction) : baseFallback;

  const payload = JSON.stringify({
    appType: session.appType,
    modelId: session.modelId,
    modelCost: session.modelCost,
    extraction: session.extraction,
    existingPromptData: session.promptData,
    editInstruction: editInstruction || null
  });

  try {
    const response = await runGeminiJson(PROMPT_TEMPLATE_SYSTEM, payload);

    if (!response) {
      return fallback;
    }

    return {
      userPrompt: typeof response.userPrompt === "string" && response.userPrompt.trim() ? response.userPrompt.trim() : fallback.userPrompt,
      negativePrompt: typeof response.negativePrompt === "string" && response.negativePrompt.trim() ? response.negativePrompt.trim() : fallback.negativePrompt,
      acceptImageInput: typeof response.acceptImageInput === "boolean" ? response.acceptImageInput : fallback.acceptImageInput,
      variablesUsed: Array.isArray(response.variablesUsed) && response.variablesUsed.length ? response.variablesUsed : fallback.variablesUsed,
      advancedSettings: {
        aspectRatio:
          response.advancedSettings && typeof response.advancedSettings.aspectRatio === "string"
            ? response.advancedSettings.aspectRatio
            : fallback.advancedSettings.aspectRatio,
        quality:
          response.advancedSettings && typeof response.advancedSettings.quality === "string"
            ? response.advancedSettings.quality
            : fallback.advancedSettings.quality
      }
    };
  } catch (error) {
    return fallback;
  }
}

async function generateSEO(session) {
  const fallback = buildSeoFromSession(session);

  const payload = JSON.stringify({
    appType: session.appType,
    modelId: session.modelId,
    modelCost: session.modelCost,
    extraction: session.extraction,
    promptData: session.promptData
  });

  try {
    const response = await runGeminiJson(SEO_SYSTEM, payload);

    if (!response) {
      return fallback;
    }

    return {
      appName: typeof response.appName === "string" && response.appName.trim() ? response.appName.trim().slice(0, 55) : fallback.appName,
      appDescription:
        typeof response.appDescription === "string" && response.appDescription.trim()
          ? response.appDescription.trim().slice(0, 155)
          : fallback.appDescription,
      tags: Array.isArray(response.tags) && response.tags.length ? response.tags.slice(0, 10) : fallback.tags,
      category: typeof response.category === "string" && response.category.trim() ? response.category.trim() : fallback.category,
      suggestedPrice: computeSuggestedPrice(session.modelCost)
    };
  } catch (error) {
    return fallback;
  }
}

export {
  EXTRACTION_PROMPT,
  extractRequirementsWithGemini,
  generatePromptTemplate,
  generateSEO,
  normalizeExtraction,
  applyPromptInstruction,
  buildPromptTemplateFromSession,
  buildSeoFromSession
};
