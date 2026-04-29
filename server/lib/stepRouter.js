import MODELS from "./models.js";
import mockData from "./mockData.js";
import { extractRequirements } from "./groq.js";
import { generatePromptTemplate, generateSEO, generateScope, applyPromptInstruction, buildPromptTemplateFromSession } from "./gemini.js";
import { buildBudgetTiers, getModelCost } from "./costCalculator.js";

const COST_WARNING_THRESHOLD = 100;

function lower(msg) {
  return String(msg || "").trim().toLowerCase();
}

function normalize(msg) {
  return String(msg || "").trim();
}

/* ────────────────────────────────────────────
   SMART MODEL RANKING (spec-exact)
   ──────────────────────────────────────────── */
function rankModels(models, userMessage, budget) {
  const msg = (userMessage || "").toLowerCase();

  return models
    .map(m => {
      let score = 0;

      // budget signals
      if (budget === "free") score += m.cost === 0 ? 100 : 0;
      if (budget === "low") score += m.tier === "fast" ? 50 : 0;
      if (budget === "high" || budget === "ultra") score += m.tier === "premium" ? 50 : 0;

      // keyword matching against model tags
      (m.tags || []).forEach(tag => {
        if (msg.includes(tag.replace("-", " "))) score += 30;
      });

      // prefer balanced by default
      if (m.tier === "balanced") score += 10;

      // always boost free models
      if (m.cost === 0) score += 20;

      // user said cinematic/motion
      if (msg.includes("cinematic") || msg.includes("motion")) {
        if ((m.tags || []).includes("motion-control")) score += 60;
      }

      // user said cheap/affordable/budget
      if (msg.includes("cheap") || msg.includes("budget") || msg.includes("affordable")) {
        score += (100 - Math.min(m.cost, 100));
      }

      // user said best/professional/high quality
      if (msg.includes("best") || msg.includes("professional") || msg.includes("high quality")) {
        if (m.tier === "premium" || m.tier === "ultra") score += 40;
      }

      return { ...m, score };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, 3);
}

/* ────────────────────────────────────────────
   PARSERS
   ──────────────────────────────────────────── */
function parseSelectedModelId(msg) {
  const match = normalize(msg).match(/^select\s+([a-z0-9.-]+)$/i);
  return match ? match[1].toLowerCase() : null;
}

function parseSelectedPlan(msg) {
  const match = normalize(msg).match(/^select\s+(lean|recommended|full)$/i);
  return match ? match[1].toLowerCase() : null;
}

function parseChipAppType(msg) {
  const v = lower(msg);
  if (["text", "image", "audio", "video", "vision"].includes(v)) return v;
  // Handle chip selections like "Images", "Video tour", "Written content"
  if (v === "images") return "image";
  if (v.includes("video")) return "video";
  if (v.includes("written") || v.includes("content")) return "text";
  return null;
}

function parsePromptEditInstruction(msg) {
  const n = normalize(msg);
  if (!n.toLowerCase().startsWith("edit prompt::")) return null;
  return n.slice("edit prompt::".length).trim();
}

function parseSeoPayload(msg) {
  const n = normalize(msg);
  if (!n.toLowerCase().startsWith("confirm seo::")) return null;
  try { return JSON.parse(n.slice("confirm seo::".length)); } catch { return null; }
}

function isYes(msg) {
  const v = lower(msg);
  return v === "yes" || v === "yes, proceed" || v.includes("looks good") || v.includes("proceed") || v.includes("yes,") || v === "confirm" || v.includes("sahi hai") || v.includes("haan") || v.includes("ha ");
}

function isNo(msg) {
  const v = lower(msg);
  return v.startsWith("no") || v.startsWith("change:") || v === "nahi" || v.includes("let me change");
}

function isChangeMessage(msg) {
  return lower(msg).startsWith("change:");
}

function getChangeText(msg) {
  return normalize(msg).slice("change:".length).trim();
}

/* ────────────────────────────────────────────
   MODEL / COST HELPERS
   ──────────────────────────────────────────── */
function findModel(appType, modelId) {
  return (MODELS[appType] || []).find(m => m.id === modelId) || null;
}

function findCheapestAlternative(appType, selectedModel) {
  const alternatives = (MODELS[appType] || []).filter(m => m.id !== (selectedModel && selectedModel.id));
  return [...alternatives].sort((a, b) => a.cost - b.cost)[0] || null;
}

function shouldWarnForCost(model) {
  return model && model.cost > COST_WARNING_THRESHOLD;
}

function buildCostWarningUi(appType, selectedModel) {
  const alt = findCheapestAlternative(appType, selectedModel);
  const cost = Number(selectedModel.cost.toFixed(2));
  return {
    selectedModel: selectedModel.name,
    selectedModelId: selectedModel.id,
    selectedCost: cost,
    hundredRunCost: Math.round(cost * 100 * 100) / 100,
    alternativeModel: alt ? alt.name : null,
    alternativeModelId: alt ? alt.id : null,
    alternativeCost: alt ? Number(alt.cost.toFixed(2)) : null
  };
}

/* ────────────────────────────────────────────
   TONE-AWARE REPLY PREFIX
   ──────────────────────────────────────────── */
function tonePrefix(extraction) {
  if (!extraction) return "";
  if (extraction.userTone === "urgent") return "No worries, let's set this up quickly! ";
  if (extraction.userTone === "unsure") return "Happy to help figure this out together! ";
  if (extraction.detectedLanguage === "Hindi") return "Samajh gaya — ";
  return "";
}

/* ────────────────────────────────────────────
   BUDGET / COMPLEXITY HELPERS
   ──────────────────────────────────────────── */
function computeComplexity(session) {
  const features = session.extraction && Array.isArray(session.extraction.keyFeatures) ? session.extraction.keyFeatures.length : 0;
  if (features >= 4) return "complex";
  if (features >= 2) return "medium";
  return "simple";
}

function getBudgetRow(session) {
  const complexity = computeComplexity(session);
  return (
    mockData.market_data.find(r => r.category === "ai-app" && r.complexity === complexity) ||
    mockData.market_data.find(r => r.category === "ai-app" && r.complexity === "medium") ||
    mockData.market_data.find(r => r.category === "website" && r.complexity === "medium")
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
    options: { lean: tiers.lean, recommended: tiers.recommended, full: tiers.full }
  };
}

/* ────────────────────────────────────────────
   MERGE EXTRACTION
   ──────────────────────────────────────────── */
function mergeExtraction(existing, latest, message) {
  if (!existing) return latest;

  const isControl = parseSelectedModelId(message) || parseSelectedPlan(message) || parseChipAppType(message) || isYes(message);
  const keepAppType = isControl || !latest.appType;
  const keepPurpose = isControl || !latest.appPurpose || latest.appPurpose.length < 8;

  return {
    ...existing,
    ...latest,
    appType: keepAppType ? existing.appType : latest.appType,
    appPurpose: keepPurpose ? existing.appPurpose : latest.appPurpose,
    targetUsers: (isControl || !latest.targetUsers || latest.targetUsers === "general users") ? existing.targetUsers : latest.targetUsers,
    budget: latest && latest.budget ? latest.budget : existing.budget,
    wantsImageInput: Boolean(existing.wantsImageInput || (latest && latest.wantsImageInput)),
    detectedLanguage: isControl && existing.detectedLanguage ? existing.detectedLanguage : (latest.detectedLanguage || existing.detectedLanguage),
    userTone: isControl && existing.userTone ? existing.userTone : (latest.userTone || existing.userTone),
    oneLineUnderstanding: isControl ? (existing.oneLineUnderstanding || latest.oneLineUnderstanding) : (latest.oneLineUnderstanding || existing.oneLineUnderstanding),
    suggestedReply: isControl ? (existing.suggestedReply || latest.suggestedReply) : (latest.suggestedReply || existing.suggestedReply),
    confidence: {
      appType: keepAppType ? existing.confidence.appType : latest.confidence.appType,
      budget: latest && latest.budget ? latest.confidence.budget : existing.confidence.budget
    },
    keyFeatures: !isControl && latest && Array.isArray(latest.keyFeatures) && latest.keyFeatures.length ? latest.keyFeatures : existing.keyFeatures,
    missingFields: Array.from(new Set([...(existing.missingFields || []), ...((latest && latest.missingFields) || [])]))
  };
}

/* ────────────────────────────────────────────
   BUILD FULL HISTORY STRING for ranking
   ──────────────────────────────────────────── */
function getFullUserText(session) {
  if (!session.history) return "";
  return session.history.filter(h => h.role === "user").map(h => h.content).join(" ");
}

/* ════════════════════════════════════════════
   MAIN ROUTER
   ════════════════════════════════════════════ */
async function route(session, message) {
  const text = normalize(message);
  const latestExtraction = await extractRequirements(text, session.history || []);
  session.extraction = mergeExtraction(session.extraction, latestExtraction, text);

  // ─── STEP 5A: Plan selected → publish ───
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
      selectedPlan: planId,
      publishedAt: new Date().toISOString()
    };

    console.log("\n══════ MOCK PUBLISH ══════");
    console.log(JSON.stringify(payload, null, 2));
    console.log("══════════════════════════\n");

    return {
      reply: `🎉 Your app "${session.seoData.appName}" has been published successfully!`,
      uiType: "success",
      uiData: {
        appName: session.seoData.appName,
        modelId: session.modelId,
        costPerRun: session.modelCost,
        tags: session.seoData.tags,
        selectedPlan: planId,
        mockUrl: `https://rentprompts.ai/app/demo-${Date.now()}`,
        isBounty: false
      },
      nextStep: 0,
      coins: session.modelCost,
      clearSession: true
    };
  }

  // ─── STEP 5B: Bounty publish ───
  if (lower(text) === "post as bounty" && session.seoData) {
    const bountyPayload = {
      title: session.seoData.appName,
      description: session.seoData.appDescription,
      appType: session.appType,
      modelId: session.modelId,
      promptTemplate: session.promptData && session.promptData.userPrompt,
      tags: session.seoData.tags,
      budget_preference: "open_to_bids",
      postedAt: new Date().toISOString()
    };

    console.log("\n══════ MOCK BOUNTY PUBLISH ══════");
    console.log(JSON.stringify(bountyPayload, null, 2));
    console.log("══════════════════════════════════\n");

    return {
      reply: "Your bounty has been posted! Creators will bid on your project within 24-48 hours.",
      uiType: "success",
      uiData: {
        appName: session.seoData.appName,
        modelId: session.modelId,
        costPerRun: session.modelCost,
        tags: session.seoData.tags,
        selectedPlan: "bounty",
        mockUrl: `https://rentprompts.ai/bounty/demo-${Date.now()}`,
        isBounty: true
      },
      nextStep: 0,
      coins: null,
      clearSession: true
    };
  }

  // ─── AWAITING CONFIRMATION HANDLING ───
  if (session.awaitingConfirmation) {
    const confirmStep = session.confirmStep;

    // User said YES
    if (isYes(text)) {
      session.awaitingConfirmation = false;

      // Step 0 confirmed → show models (STEP 1)
      if (confirmStep === 0 && session.appType) {
        const fullText = getFullUserText(session);
        const budget = session.extraction && session.extraction.budget;
        const models = rankModels(MODELS[session.appType] || [], fullText, budget);
        session.step = 1;
        session.awaitingConfirmation = true;
        session.confirmStep = 1;
        return {
          reply: `Great! Here are the top 3 models for your ${session.appType} app:`,
          uiType: "models",
          uiData: { appType: session.appType, models },
          nextStep: 1,
          coins: null,
          confirm: {
            summary: `Here are the top 3 models for your ${session.appType} app. Does one of these fit?`,
            detail: "Click a model card to select it, or say No to see different options."
          }
        };
      }

      // Step 1 confirmed → user needs to pick a model card
      if (confirmStep === 1) {
        return {
          reply: "Click on one of the model cards above to select it.",
          uiType: "text",
          uiData: {},
          nextStep: 1,
          coins: null
        };
      }

      // Step 2 confirmed (prompt) → generate scope (STEP 3)
      if (confirmStep === 2 && session.promptData) {
        if (!session.scopeData) {
          session.scopeData = await generateScope(session);
        }
        session.step = 3;
        session.awaitingConfirmation = true;
        session.confirmStep = 3;
        return {
          reply: `The scope covers ${session.scopeData.totalItems} key items — ` +
                 `${session.scopeData.scopeSummary}\n\n` +
                 `Total estimated effort: ~${session.scopeData.totalHours}h\n\n` +
                 `Want to adjust anything in the scope, or shall we ` +
                 `move on to look at pricing options?`,
          uiType: "scope",
          uiData: session.scopeData,
          nextStep: 3,
          coins: session.modelCost,
          confirm: {
            summary: `Scope: ${session.scopeData.totalItems} items (~${session.scopeData.totalHours}h). Looks good?`,
            detail: "Say Yes to proceed, or let me know what you want to add/remove."
          }
        };
      }

      // Step 3 confirmed (Scope) → generate SEO (STEP 4)
      if (confirmStep === 3 && session.scopeData) {
        if (!session.seoData) {
          session.seoData = await generateSEO(session);
        }
        session.step = 4;
        session.awaitingConfirmation = true;
        session.confirmStep = 4;
        return {
          reply: "Almost done! Here's your app's SEO profile:",
          uiType: "seo_preview",
          uiData: session.seoData,
          nextStep: 4,
          coins: session.modelCost,
          confirm: {
            summary: "Here's your app's name, description and tags. Ready to publish?",
            detail: "You can edit any field inline before confirming."
          }
        };
      }

      // Step 4 confirmed (SEO) → budget/bounty check (STEP 5)
      if (confirmStep === 4 && session.seoData) {
        return buildBudgetStep(session);
      }

      // Step 5 confirmed (bounty) → post as bounty
      if (confirmStep === 5 && session.budgetPath === "bounty") {
        // Re-route to bounty publish
        return route(session, "Post as bounty");
      }
    }

    // User said NO or typed correction
    if (isNo(text) || isChangeMessage(text)) {
      session.awaitingConfirmation = false;
      const correction = isChangeMessage(text) ? getChangeText(text) : "";

      if (confirmStep === 0) {
        if (correction) {
          // Re-extract with correction
          const newExtraction = await extractRequirements(correction, session.history || []);
          session.extraction = mergeExtraction(session.extraction, newExtraction, correction);
          if (session.extraction.appType && ["HIGH", "MEDIUM"].includes(session.extraction.confidence.appType)) {
            session.appType = session.extraction.appType;
          }
        }
        // Show new confirm
        return buildStep0Response(session);
      }

      if (confirmStep === 1) {
        return {
          reply: "No problem! What are you looking for in a model? (e.g. cheaper, higher quality, faster)",
          uiType: "text",
          uiData: {},
          nextStep: 1,
          coins: null
        };
      }

      if (confirmStep === 2) {
        return {
          reply: "Tell me what to change about the prompt, and I'll regenerate it.",
          uiType: "text",
          uiData: {},
          nextStep: 2,
          coins: session.modelCost
        };
      }

      if (confirmStep === 3) {
        const correction = isChangeMessage(text) ? getChangeText(text) : text;
        if (correction) {
          if (!session.extraction.keyFeatures) session.extraction.keyFeatures = [];
          session.extraction.keyFeatures.push(correction);
          session.scopeData = await generateScope(session);
        }
        return {
          reply: `Updated scope — ${session.scopeData.totalItems} items, ~${session.scopeData.totalHours}h total. Does this look better?`,
          uiType: "scope",
          uiData: session.scopeData,
          nextStep: 3,
          coins: session.modelCost,
          confirm: {
            summary: `Scope: ${session.scopeData.totalItems} items (~${session.scopeData.totalHours}h). Looks good?`,
            detail: "Say Yes to proceed, or let me know what you want to add/remove."
          }
        };
      }

      if (confirmStep === 4) {
        return {
          reply: "Which part would you like to change? You can also edit inline above.",
          uiType: "text",
          uiData: {},
          nextStep: 4,
          coins: session.modelCost
        };
      }
    }
  }

  // ─── CHIP APP TYPE SELECTION ───
  const chipType = parseChipAppType(text);
  if (chipType && !session.appType) {
    session.appType = chipType;
    session.extraction.appType = chipType;
    session.extraction.confidence.appType = "HIGH";
    return buildStep0Response(session);
  }

  // ─── MODEL SELECTION ───
  const selectedModelId = parseSelectedModelId(text);
  if (selectedModelId && !["lean", "recommended", "full"].includes(selectedModelId)) {
    const selectedModel = findModel(session.appType, selectedModelId);
    if (!selectedModel) {
      const fullText = getFullUserText(session);
      const budget = session.extraction && session.extraction.budget;
      const models = rankModels(MODELS[session.appType] || [], fullText, budget);
      return {
        reply: "I couldn't match that model. Pick one of the options below:",
        uiType: "models",
        uiData: { appType: session.appType, models },
        nextStep: 1,
        coins: null
      };
    }

    session.modelId = selectedModel.id;
    session.modelCost = selectedModel.cost;
    session.promptData = null;
    session.seoData = null;
    session.awaitingConfirmation = false;

    // Cost warning check
    if (shouldWarnForCost(selectedModel)) {
      session.costWarning = buildCostWarningUi(session.appType, selectedModel);
      session.step = 2;
      return {
        reply: "Heads up — this model is powerful but can get expensive at scale:",
        uiType: "cost_warning",
        uiData: session.costWarning,
        nextStep: 2,
        coins: selectedModel.cost
      };
    }

    // Generate prompt and show with confirm
    delete session.costWarning;
    session.promptData = await generatePromptTemplate(session);
    session.step = 2;
    session.awaitingConfirmation = true;
    session.confirmStep = 2;
    return {
      reply: "Great choice! Here's your auto-generated prompt template:",
      uiType: "prompt_preview",
      uiData: session.promptData,
      nextStep: 2,
      coins: session.modelCost,
      confirm: {
        summary: "Here's the auto-generated prompt for your app. Does this look right?",
        detail: session.promptData.promptExplanation || null
      }
    };
  }

  // ─── COST WARNING RESPONSE ───
  if (session.costWarning && session.step === 2) {
    const v = lower(text);

    if (v.includes("use cheaper") || v.includes("cheaper option") || v.includes("show me cheaper")) {
      session.modelId = session.costWarning.alternativeModelId;
      session.modelCost = session.costWarning.alternativeCost;
      delete session.costWarning;
      session.promptData = await generatePromptTemplate(session);
      session.awaitingConfirmation = true;
      session.confirmStep = 2;
      return {
        reply: "Swapped to the cheaper option. Here's your prompt:",
        uiType: "prompt_preview",
        uiData: session.promptData,
        nextStep: 2,
        coins: session.modelCost,
        confirm: {
          summary: "Here's the auto-generated prompt for your app. Does this look right?",
          detail: session.promptData.promptExplanation || null
        }
      };
    }

    if (v.includes("proceed") || v.includes("understand") || v === "yes") {
      delete session.costWarning;
      session.promptData = await generatePromptTemplate(session);
      session.awaitingConfirmation = true;
      session.confirmStep = 2;
      return {
        reply: "Understood. Here's your prompt template:",
        uiType: "prompt_preview",
        uiData: session.promptData,
        nextStep: 2,
        coins: session.modelCost,
        confirm: {
          summary: "Here's the auto-generated prompt for your app. Does this look right?",
          detail: session.promptData.promptExplanation || null
        }
      };
    }
  }

  // ─── PROMPT EDIT ───
  const promptEdit = parsePromptEditInstruction(text);
  if (promptEdit && session.modelId) {
    if (!session.promptData) {
      session.promptData = buildPromptTemplateFromSession(session);
    }
    session.promptData = applyPromptInstruction(session.promptData, promptEdit);
    session.awaitingConfirmation = true;
    session.confirmStep = 2;
    return {
      reply: "Updated! Does this look better?",
      uiType: "prompt_preview",
      uiData: session.promptData,
      nextStep: 2,
      coins: session.modelCost,
      confirm: {
        summary: "Here's the updated prompt. Does this look right?",
        detail: null
      }
    };
  }

  // ─── SEO INLINE EDIT CONFIRM ───
  const seoPayload = parseSeoPayload(text);
  if (seoPayload && session.seoData) {
    session.seoData = {
      ...session.seoData,
      ...seoPayload,
      tags: Array.isArray(seoPayload.tags) ? seoPayload.tags : session.seoData.tags
    };
    return buildBudgetStep(session);
  }

  // ─── SEO CONFIRM via text ───
  if (session.seoData && (lower(text).includes("confirm seo") || lower(text).includes("confirm & continue") || lower(text) === "continue")) {
    return buildBudgetStep(session);
  }

  // ─── STEP 0: Initial message / no appType yet ───
  if (session.step === 0 || !session.appType) {
    return buildStep0Response(session);
  }

  // ─── FALLBACK: re-generate prompt if model selected but no promptData ───
  if (session.modelId && !session.promptData) {
    session.promptData = await generatePromptTemplate(session);
    session.awaitingConfirmation = true;
    session.confirmStep = 2;
    return {
      reply: "Here's your auto-generated prompt:",
      uiType: "prompt_preview",
      uiData: session.promptData,
      nextStep: 2,
      coins: session.modelCost,
      confirm: {
        summary: "Here's the auto-generated prompt for your app. Does this look right?",
        detail: null
      }
    };
  }

  // ─── CATCH ALL ───
  return {
    reply: "I kept the current setup intact. Use the buttons above to continue, or tell me what you'd like to change.",
    uiType: "text",
    uiData: {},
    nextStep: session.step || 0,
    coins: session.modelCost
  };
}

/* ────────────────────────────────────────────
   STEP 0 RESPONSE BUILDER
   ──────────────────────────────────────────── */
function buildStep0Response(session) {
  const ext = session.extraction;

  // HIGH/MEDIUM confidence → show confirm card for what we understood
  if (ext && ext.appType && ["HIGH", "MEDIUM"].includes(ext.confidence.appType)) {
    session.appType = ext.appType;
    session.step = 0;
    session.awaitingConfirmation = true;
    session.confirmStep = 0;

    const prefix = tonePrefix(ext);
    const reply = ext.suggestedReply
      ? `${prefix}${ext.suggestedReply}`
      : `${prefix}Got it — ${ext.oneLineUnderstanding}.\n\nI'm planning to build a ${ext.appType} app that ${ext.appPurpose}. Is this the right direction?`;

    return {
      reply,
      uiType: "text",
      uiData: {},
      nextStep: 0,
      coins: null,
      confirm: {
        summary: ext.oneLineUnderstanding,
        detail: `App type: ${ext.appType} • Target: ${ext.targetUsers}`
      }
    };
  }

  // LOW confidence → ask for clarification with chips
  session.step = 0;
  const prefix = tonePrefix(ext);
  const options = ext && ext.appPurpose && /clinic|hospital/i.test(ext.appPurpose)
    ? ["Images", "Video tour", "Written content", "Something else"]
    : ["Text", "Image", "Audio", "Video", "Vision"];

  return {
    reply: `${prefix}I'd love to help! What type of output does your app need?`,
    uiType: "chips",
    uiData: { options },
    nextStep: 0,
    coins: null
  };
}

/* ────────────────────────────────────────────
   BUDGET STEP BUILDER (Step 4)
   ──────────────────────────────────────────── */
function buildBudgetStep(session) {
  const budgetUi = buildBudgetUi(session);
  const leanCost = budgetUi.options.lean.joules;
  const userBalance = mockData.userBalance;

  // Check if user has enough joules
  if (userBalance < leanCost) {
    session.step = 5;
    session.budgetPath = "bounty";
    session.awaitingConfirmation = true;
    session.confirmStep = 5;

    const selectedModel = findModel(session.appType, session.modelId);
    return {
      reply: "Your current balance might not cover this. No worries — I can post this as a bounty instead, where creators will bid on your project.",
      uiType: "bounty_fallback",
      uiData: {
        appName: session.seoData.appName,
        promptTemplate: session.promptData && session.promptData.userPrompt,
        modelName: selectedModel ? selectedModel.name : session.modelId,
        userBalance,
        leanCost
      },
      nextStep: 5,
      coins: null,
      confirm: {
        summary: "Would you like to post this as a bounty instead?",
        detail: "Creators on RentPrompts will bid on your project within 24-48 hours."
      }
    };
  }

  // Enough joules → show budget cards
  session.step = 5;
  session.budgetPath = "publish";
  return {
    reply: "Your app is ready! Pick a publishing plan:",
    uiType: "budget_cards",
    uiData: budgetUi,
    nextStep: 5,
    coins: null,
    confirm: {
      summary: "Which plan works best for you?",
      detail: `Your balance: ${userBalance.toLocaleString()} joules`
    }
  };
}

export { route };
