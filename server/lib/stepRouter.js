import MODELS from "./models.js";
import mockData from "./mockData.js";
import { extractRequirements } from "./groq.js";
import { generatePromptTemplate, generateSEO, applyPromptInstruction, buildPromptTemplateFromSession } from "./gemini.js";
import { buildBudgetTiers, getModelCost } from "./costCalculator.js";

const APP_TYPE_OPTIONS = ["Text", "Image", "Audio", "Video", "Vision"];
const COST_WARNING_THRESHOLD = 60;

function normalizeText(message) {
  return String(message || "").trim();
}

function lower(message) {
  return normalizeText(message).toLowerCase();
}

function tierSortValue(tier) {
  return {
    free: 0,
    fast: 1,
    balanced: 2,
    premium: 3,
    ultra: 4
  }[tier] ?? 10;
}

function mapBudgetToTiers(budget) {
  switch (budget) {
    case "free":
      return ["free", "fast"];
    case "low":
      return ["free", "fast"];
    case "medium":
      return ["fast", "balanced"];
    case "high":
      return ["balanced", "premium"];
    case "ultra":
      return ["premium", "ultra"];
    default:
      return null;
  }
}

function getVideoRanking(extraction) {
  if (extraction && extraction.wantsImageInput) {
    return ["seedance-2.0", "kling-v2.6-motion", "seedance-1.5-pro", "seedance-1-pro", "wan-2.2-fast", "kling-v2.6", "gen-4-turbo", "grok-imagine", "pixverse-v5.6", "ray-2-720p", "veo-3-fast", "veo3"];
  }

  return ["wan-2.2-fast", "seedance-1.5-pro", "gen-4-turbo", "kling-v2.6", "gen-4.5", "ray-2-720p", "veo-3-fast", "veo3", "seedance-2.0"];
}

function getRankedModels(appType, extraction) {
  const list = MODELS[appType] || [];

  if (appType === "video") {
    const ranking = getVideoRanking(extraction);
    return [...list].sort((a, b) => {
      const aIndex = ranking.indexOf(a.id);
      const bIndex = ranking.indexOf(b.id);
      const safeA = aIndex === -1 ? ranking.length + tierSortValue(a.tier) : aIndex;
      const safeB = bIndex === -1 ? ranking.length + tierSortValue(b.tier) : bIndex;
      return safeA - safeB;
    });
  }

  return [...list];
}

function pickTopModels(appType, extraction) {
  const ranked = getRankedModels(appType, extraction);
  const allowedTiers = mapBudgetToTiers(extraction && extraction.budget);

  if (!allowedTiers) {
    return ranked.slice(0, 3);
  }

  const filtered = ranked.filter((model) => allowedTiers.includes(model.tier));
  const merged = [...filtered, ...ranked.filter((model) => !filtered.some((selected) => selected.id === model.id))];
  return merged.slice(0, 3);
}

function parseSelectedModelId(message) {
  const match = normalizeText(message).match(/^select\s+([a-z0-9.-]+)$/i);
  return match ? match[1].toLowerCase() : null;
}

function parseSelectedPlan(message) {
  const match = normalizeText(message).match(/^select\s+(lean|recommended|full)$/i);
  return match ? match[1].toLowerCase() : null;
}

function parseChipAppType(message) {
  const value = lower(message);

  if (["text", "image", "audio", "video", "vision"].includes(value)) {
    return value;
  }

  return null;
}

function parsePromptEditInstruction(message) {
  if (!normalizeText(message).toLowerCase().startsWith("edit prompt::")) {
    return null;
  }

  return normalizeText(message).slice("edit prompt::".length).trim();
}

function parseSeoPayload(message) {
  if (!normalizeText(message).toLowerCase().startsWith("confirm seo::")) {
    return null;
  }

  try {
    return JSON.parse(normalizeText(message).slice("confirm seo::".length));
  } catch (error) {
    return null;
  }
}

function shouldConfirmPrompt(message) {
  const value = lower(message);
  return value.includes("looks good") || value.includes("proceed") || value === "yes" || value.includes("confirm");
}

function shouldConfirmSeo(message) {
  const value = lower(message);
  return value.includes("confirm seo") || value.includes("confirm & continue") || value === "continue";
}

function isControlMessage(message) {
  const value = lower(message);
  return Boolean(
    parseSelectedModelId(message) ||
      parseSelectedPlan(message) ||
      parseChipAppType(message) ||
      parsePromptEditInstruction(message) ||
      value.includes("proceed anyway") ||
      value.includes("use cheaper model") ||
      shouldConfirmPrompt(message) ||
      shouldConfirmSeo(message)
  );
}

function shouldWarnForCost(model, appType) {
  if (!model) {
    return false;
  }

  return model.cost > 100 || (appType === "video" && model.cost >= COST_WARNING_THRESHOLD);
}

function findModel(appType, modelId) {
  return (MODELS[appType] || []).find((model) => model.id === modelId) || null;
}

function findCheapestAlternative(appType, selectedModel) {
  if (appType === "video" && selectedModel && !selectedModel.supports_image_input) {
    return findModel(appType, "wan-2.2-fast");
  }

  if (appType === "video" && selectedModel && selectedModel.supports_image_input) {
    return findModel(appType, "seedance-2.0");
  }

  const alternatives = (MODELS[appType] || []).filter((model) => model.id !== (selectedModel && selectedModel.id));
  return [...alternatives].sort((a, b) => a.cost - b.cost)[0] || null;
}

function prependUrgent(reply, extraction) {
  return extraction && extraction.userTone === "urgent" ? `No worries, quick setup - ${reply}` : reply;
}

function computeComplexity(session) {
  const features = session.extraction && Array.isArray(session.extraction.keyFeatures) ? session.extraction.keyFeatures.length : 0;

  if ((session.appType === "video" && (session.extraction && session.extraction.wantsImageInput)) || features >= 4) {
    return "complex";
  }

  if (features >= 2 || ["image", "vision", "audio"].includes(session.appType)) {
    return "medium";
  }

  return "simple";
}

function computeCategory(session) {
  const purpose = lower(session.extraction && session.extraction.appPurpose);

  if (purpose.includes("mobile app") || purpose.includes("android") || purpose.includes("ios")) {
    return "mobile";
  }

  if (["image", "video"].includes(session.appType)) {
    return "design";
  }

  return "website";
}

function getBudgetRow(session) {
  const complexity = computeComplexity(session);
  const category = computeCategory(session);

  return (
    mockData.market_data.find((row) => row.category === category && row.complexity === complexity) ||
    mockData.market_data.find((row) => row.category === "website" && row.complexity === "medium")
  );
}

function buildBudgetUi(session) {
  const row = getBudgetRow(session);
  const tiers = buildBudgetTiers(row);

  return {
    context: {
      category: row.category,
      complexity: row.complexity,
      avgHours: row.avg_hours,
      floorJoules: row.floor_joules,
      marketRange: `${row.floor_joules.toLocaleString()}-${row.market_joules.toLocaleString()}`
    },
    options: {
      lean: tiers.lean,
      recommended: tiers.recommended,
      full: tiers.full
    }
  };
}

function buildCostWarningUi(appType, selectedModel) {
  const alternative = findCheapestAlternative(appType, selectedModel);
  const selectedCost = Number(getModelCost(selectedModel.id, appType).toFixed(2));
  const hundredRunCost = Math.round(selectedCost * 100 * 100) / 100;

  return {
    selectedModel: selectedModel.name,
    selectedModelId: selectedModel.id,
    selectedCost,
    hundredRunCost,
    alternativeModel: alternative ? alternative.name : null,
    alternativeModelId: alternative ? alternative.id : null,
    alternativeCost: alternative ? Number(alternative.cost.toFixed(2)) : null
  };
}

function mergeExtraction(existing, latest, message) {
  if (!existing) {
    return latest;
  }

  const controlMessage = isControlMessage(message);
  const keepExistingAppType = controlMessage || !latest.appType;
  const keepExistingPurpose = controlMessage || !latest.appPurpose || latest.appPurpose.length < 8;
  const keepExistingUsers = controlMessage || !latest.targetUsers || latest.targetUsers === "general users";
  const keepExistingLanguage = controlMessage && existing.detectedLanguage;
  const nextMissingFields = new Set([...(existing.missingFields || []), ...((latest && latest.missingFields) || [])]);

  if (existing.appType) {
    nextMissingFields.delete("appType");
  }

  if (existing.targetUsers && existing.targetUsers !== "general users") {
    nextMissingFields.delete("targetUsers");
  }

  return {
    ...existing,
    ...latest,
    appType: keepExistingAppType ? existing.appType : latest.appType,
    appPurpose: keepExistingPurpose ? existing.appPurpose : latest.appPurpose,
    targetUsers: keepExistingUsers ? existing.targetUsers : latest.targetUsers,
    budget: latest && latest.budget ? latest.budget : existing.budget,
    wantsImageInput: Boolean((existing && existing.wantsImageInput) || (latest && latest.wantsImageInput)),
    detectedLanguage: keepExistingLanguage ? existing.detectedLanguage : latest.detectedLanguage || existing.detectedLanguage,
    userTone: controlMessage && existing.userTone ? existing.userTone : latest.userTone || existing.userTone,
    oneLineUnderstanding: controlMessage ? existing.oneLineUnderstanding || latest.oneLineUnderstanding : latest.oneLineUnderstanding || existing.oneLineUnderstanding,
    confidence: {
      appType: keepExistingAppType ? existing.confidence.appType : latest.confidence.appType,
      budget: latest && latest.budget ? latest.confidence.budget : existing.confidence.budget
    },
    keyFeatures:
      !controlMessage && latest && Array.isArray(latest.keyFeatures) && latest.keyFeatures.length
        ? latest.keyFeatures
        : existing.keyFeatures,
    missingFields: Array.from(nextMissingFields)
  };
}

async function createPromptPreview(session, replyText) {
  if (!session.promptData) {
    session.promptData = await generatePromptTemplate(session);
  }

  return {
    reply: replyText,
    uiType: "prompt_preview",
    uiData: session.promptData,
    nextStep: 3,
    coins: session.modelCost
  };
}

async function route(session, message) {
  const text = normalizeText(message);
  const latestExtraction = await extractRequirements(text, session.history || []);
  session.extraction = mergeExtraction(session.extraction, latestExtraction, text);

  if (parseSelectedPlan(text) && session.seoData) {
    const planId = parseSelectedPlan(text);
    const payload = {
      appType: session.appType,
      modelId: session.modelId,
      costPerRun: session.modelCost,
      userPrompt: session.promptData && session.promptData.userPrompt,
      negativePrompt: session.promptData && session.promptData.negativePrompt,
      acceptImageInput: session.promptData && session.promptData.acceptImageInput,
      appName: session.seoData.appName,
      appDescription: session.seoData.appDescription,
      tags: session.seoData.tags,
      publishedAt: new Date().toISOString()
    };

    console.log("MOCK PUBLISH:", JSON.stringify(payload, null, 2));

    return {
      reply: "App published successfully!",
      uiType: "success",
      uiData: {
        appName: session.seoData.appName,
        modelId: session.modelId,
        costPerRun: session.modelCost,
        tags: session.seoData.tags,
        selectedPlan: planId,
        mockUrl: `https://rentprompts.com/app/demo-${Date.now()}`
      },
      nextStep: 0,
      coins: session.modelCost,
      clearSession: true
    };
  }

  if (session.step === 0 || (!session.appType && !parseChipAppType(text))) {
    if (session.extraction && ["HIGH", "MEDIUM"].includes(session.extraction.confidence && session.extraction.confidence.appType) && session.extraction.appType) {
      session.appType = session.extraction.appType;
      const selectedModels = pickTopModels(session.appType, session.extraction);
      return {
        reply: prependUrgent(`Got it - ${session.extraction.oneLineUnderstanding}. Here are the best models for your app:`, session.extraction),
        uiType: "models",
        uiData: {
          appType: session.appType,
          models: selectedModels
        },
        nextStep: 2,
        coins: null
      };
    }

    return {
      reply: prependUrgent("I can help with that. I understood the rough idea, and the one thing I need next is the output type. What type of output does your app produce?", session.extraction),
      uiType: "chips",
      uiData: {
        options: APP_TYPE_OPTIONS
      },
      nextStep: 1,
      coins: null
    };
  }

  const chipAppType = parseChipAppType(text);

  if (session.step === 1 || chipAppType) {
    session.appType = chipAppType || session.appType || (session.extraction && session.extraction.appType);
    const selectedModels = pickTopModels(session.appType, session.extraction);

    return {
      reply: `Perfect. Here are the top models for ${session.appType} apps:`,
      uiType: "models",
      uiData: {
        appType: session.appType,
        models: selectedModels
      },
      nextStep: 2,
      coins: null
    };
  }

  const selectedModelId = parseSelectedModelId(text);

  if (selectedModelId && !["lean", "recommended", "full"].includes(selectedModelId)) {
    const selectedModel = findModel(session.appType, selectedModelId);

    if (!selectedModel) {
      return {
        reply: "I couldn't match that model. Pick one of the options below and I'll keep moving.",
        uiType: "models",
        uiData: {
          appType: session.appType,
          models: pickTopModels(session.appType, session.extraction)
        },
        nextStep: 2,
        coins: null
      };
    }

    session.modelId = selectedModel.id;
    session.modelCost = selectedModel.cost;
    session.promptData = null;
    session.seoData = null;

    if (shouldWarnForCost(selectedModel, session.appType)) {
      session.costWarning = buildCostWarningUi(session.appType, selectedModel);

      return {
        reply: "Heads up on cost before we continue:",
        uiType: "cost_warning",
        uiData: session.costWarning,
        nextStep: 2,
        coins: selectedModel.cost
      };
    }

    delete session.costWarning;
    return createPromptPreview(session, "Great choice! Here's your auto-generated prompt:");
  }

  if (session.costWarning && session.step === 2) {
    const value = lower(text);

    if (value.includes("use cheaper model")) {
      session.modelId = session.costWarning.alternativeModelId;
      session.modelCost = session.costWarning.alternativeCost;
      delete session.costWarning;
      session.promptData = await generatePromptTemplate(session);

      return {
        reply: "Swapped to the cheaper option. Here's your auto-generated prompt:",
        uiType: "prompt_preview",
        uiData: session.promptData,
        nextStep: 3,
        coins: session.modelCost
      };
    }

    if (value.includes("proceed") || value === "yes") {
      delete session.costWarning;
      return createPromptPreview(session, "Understood. Here's your prompt before we publish anything:");
    }
  }

  const promptEditInstruction = parsePromptEditInstruction(text);

  if (promptEditInstruction && session.modelId) {
    if (!session.promptData) {
      session.promptData = buildPromptTemplateFromSession(session);
    }

    session.promptData = applyPromptInstruction(session.promptData, promptEditInstruction);

    return {
      reply: "Updated! Does this look better?",
      uiType: "prompt_preview",
      uiData: session.promptData,
      nextStep: 3,
      coins: session.modelCost
    };
  }

  if (session.modelId && session.step <= 3 && shouldConfirmPrompt(text)) {
    if (!session.promptData) {
      session.promptData = await generatePromptTemplate(session);
    }

    if (!session.seoData) {
      session.seoData = await generateSEO(session);
    }

    return {
      reply: "Almost done! Here's your app's SEO profile:",
      uiType: "seo_preview",
      uiData: session.seoData,
      nextStep: 4,
      coins: session.modelCost
    };
  }

  const seoPayload = parseSeoPayload(text);

  if (seoPayload && session.seoData) {
    session.seoData = {
      ...session.seoData,
      ...seoPayload,
      tags: Array.isArray(seoPayload.tags) ? seoPayload.tags : session.seoData.tags
    };

    return {
      reply: "Your app is ready. Pick a publishing option:",
      uiType: "budget_cards",
      uiData: buildBudgetUi(session),
      nextStep: 5,
      coins: null
    };
  }

  if (session.seoData && shouldConfirmSeo(text)) {
    return {
      reply: "Your app is ready. Pick a publishing option:",
      uiType: "budget_cards",
      uiData: buildBudgetUi(session),
      nextStep: 5,
      coins: null
    };
  }

  if (session.modelId && !session.promptData) {
    return createPromptPreview(session, "Great choice! Here's your auto-generated prompt:");
  }

  return {
    reply: "I kept the current setup intact. Choose the next card action and I'll continue from there.",
    uiType: "text",
    uiData: {},
    nextStep: session.step || 0,
    coins: session.modelCost
  };
}

export { route };
