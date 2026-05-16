import MODELS from "./models.js";
import mockData from "./mockData.js";
import { extractRequirements, generateDynamicContext, triageDynamicContext } from "./groq.js";
import { generatePromptTemplate, generateSEO, applyPromptInstruction, buildPromptTemplateFromSession } from "./gemini.js";
import { buildBudgetTiers, getModelCost } from "./costCalculator.js";
import { isOffTopic, OFF_TOPIC_RESPONSE } from "./requirementRouter.js";
import { saveSession, deleteSession } from "./redis.js";

const COST_WARNING_THRESHOLD = 100;

function lower(msg) {
  return String(msg || "").trim().toLowerCase();
}

function normalize(msg) {
  return String(msg || "").trim();
}

function parseMultiSelectPayload(msg) {
  const text = normalize(msg);
  if (!text.toLowerCase().startsWith("multi_select_form::")) return null;
  try {
    const payload = JSON.parse(text.slice("multi_select_form::".length));
    if (!payload || typeof payload !== "object") return null;
    return payload;
  } catch {
    return null;
  }
}

function normalizeSubmittedVariables(variables) {
  if (!Array.isArray(variables)) return [];
  return variables
    .map((variable) => {
      if (typeof variable === "string") {
        return { name: variable.trim(), placeholder: "Enter details...", value: "" };
      }
      if (!variable || typeof variable !== "object") return null;
      return {
        name: String(variable.name || "").trim(),
        placeholder: String(variable.placeholder || "Enter details...").trim(),
        value: String(variable.value || "").trim()
      };
    })
    .filter((variable) => variable && variable.name);
}

function detectLanguageMode(session) {
  const lang = String(session?.extraction?.detectedLanguage || session?.languageMode || "English").toLowerCase();
  if (lang.includes("hinglish")) return "Hinglish";
  if (lang.includes("hindi")) return "Hindi";
  return "English";
}

function localizedText(session, english, hindi, hinglish) {
  const mode = detectLanguageMode(session);
  if (mode === "Hindi") return hindi || english;
  if (mode === "Hinglish") return hinglish || english;
  return english;
}

/* ────────────────────────────────────────────
   SMART MODEL RANKING (spec-exact)
   ──────────────────────────────────────────── */
function rankModels(availableModels, userInput, budgetStr) {
  if (!availableModels || availableModels.length === 0) return [];
  
  let filtered = [...availableModels];

  // 1. STRICT BUDGET FILTERING (Using our established ranges)
  const b = (budgetStr || "").toLowerCase();
  if (b) {
    if (b.includes("medium") || b.includes("5-20") || b.includes("5 - 20")) {
      filtered = filtered.filter(m => m.cost >= 5 && m.cost <= 20);
    } else if (b.includes("low") || b.includes("under 5") || b.includes("< 5")) {
      filtered = filtered.filter(m => m.cost > 0 && m.cost < 5);
    } else if (b.includes("premium") || b.includes("best") || b.includes("> 20")) {
      filtered = filtered.filter(m => m.cost >= 20);
    } else if (b.includes("free") || b.includes("0 coins")) {
      filtered = filtered.filter(m => m.cost === 0);
    } else {
      const numberMatch = b.match(/\d+(\.\d+)?/);
      if (numberMatch) {
        filtered = filtered.filter(m => m.cost <= parseFloat(numberMatch[0]));
      }
    }
  }

  // UNMATCHABLE BUDGET GUARD (If budget is too strict, return absolute cheapest)
  if (filtered.length === 0) {
    return [...availableModels].sort((a, b) => a.cost - b.cost).slice(0, 3);
  }

  // 2. SCORING ENGINE
  const input = (userInput || "").toLowerCase();
  
  const scoredModels = filtered.map(model => {
    let score = 0;
    
    // Boost score if the user's prompt contains the model's tags
    if (model.tags) {
      model.tags.forEach(tag => {
        if (input.includes(tag.toLowerCase())) score += 5;
      });
    }

    // Boost score based on Tiers matching user intent
    if (input.includes("fast") || input.includes("quick") || input.includes("speed")) {
      if (model.tier === "fast") score += 5;
    }
    if (input.includes("quality") || input.includes("best") || input.includes("advanced")) {
      if (model.tier === "premium" || model.tier === "ultra") score += 5;
    }
    if (input.includes("cheap") || input.includes("affordable")) {
      // Reward lower cost models
      score += (20 - model.cost); 
    }

    return { ...model, score };
  });

  // 3. SORT & SELECT
  // Sort primarily by score (highest first). If scores tie, sort by cost (cheapest first).
  scoredModels.sort((a, b) => b.score - a.score || a.cost - b.cost);

  // Return exactly the top 3 (removing the temporary score property to keep data clean)
  return scoredModels.slice(0, 3).map(({ score, ...rest }) => rest);
}

/* ────────────────────────────────────────────
   PARSERS (FIXED BULLETPROOF PARSING)
   ──────────────────────────────────────────── */
function parseSelectedModelId(msg, availableModels) {
  const text = normalize(msg).toLowerCase();
  if (!text.startsWith("select")) return null;

  const query = text.replace(/^select\s+/i, "").trim();
  if (!query) return null;

  const models = availableModels || [];
  const found = models.find((m) => {
    const id = String(m?.id || "").toLowerCase();
    const name = String(m?.name || "").toLowerCase();
    return (id && id === query) || (name && name === query);
  });
  return found ? found.id : query;
}

function parseSelectedPlan(msg) {
  const match = normalize(msg).match(/^select\s+(lean|recommended|full)$/i);
  return match ? match[1].toLowerCase() : null;
}

function parseChipAppType(msg) {
  const v = lower(msg);
  if (["text", "image", "audio", "video", "vision"].includes(v)) return v;
  if (v === "images") return "image";
  if (v.includes("image generator") || v.includes("image app") || v.includes("generate images or photos")) return "image";
  if (v.includes("video creator") || v.includes("video app") || v.includes("create videos or animations")) return "video";
  if (v.includes("text") || v.includes("writing tool") || v.includes("write text") || v.includes("written") || v.includes("content")) return "text";
  if (v.includes("audio generator") || v.includes("audio app") || v.includes("generate voice or music")) return "audio";
  if (v.includes("vision") || v.includes("image analyzer") || v.includes("analyze or understand images")) return "vision";
  if (v.includes("video")) return "video";
  return null;
}

/** User is correcting assumed format after scoping — clear form so triage runs again with full history. */
function shouldRerunTriageAfterFormatCorrection(session, userMessage) {
  if (!session?.dynamicContext || session.step !== 0) return false;
  const t = normalize(userMessage || "");
  if (!t || t.toLowerCase().startsWith("multi_select_form::")) return false;
  const v = lower(t);
  const correctionCue =
    /\b(no|nope|not that|wrong|actually|instead|change it|make it|switch to|i want|i meant|correction|not a text|not text)\b/i.test(
      v
    );
  if (!correctionCue) return false;
  const hasFormatSignal =
    parseChipAppType(t) != null ||
    /\b(text|image|images|picture|pictures|visual|audio|sound|voice|video|vision|tiktok|reel|clip)\b/i.test(v);
  return hasFormatSignal;
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
    mockData.website.find(r => r.category === "website" && r.complexity === "medium")
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
    suggestedReply: isControl ? (existing.suggestedReply || latest.suggestedReply) : (latest.suggestedReply || latest.suggestedReply),
    confidence: {
      appType: keepAppType
        ? (existing.confidence?.appType ?? latest?.confidence?.appType ?? "LOW")
        : (latest?.confidence?.appType ?? existing.confidence?.appType ?? "LOW"),
      budget:
        latest?.budget
          ? (latest.confidence?.budget ?? existing.confidence?.budget ?? "LOW")
          : (existing.confidence?.budget ?? latest?.confidence?.budget ?? "LOW")
    },
    keyFeatures: !isControl && latest && Array.isArray(latest.keyFeatures) && latest.keyFeatures.length ? latest.keyFeatures : existing.keyFeatures,
    missingFields: Array.from(new Set([...(existing.missingFields || []), ...((latest && latest.missingFields) || [])])),
    userType: latest.userType || existing.userType,
    enterpriseSignals: latest.enterpriseSignals !== undefined ? latest.enterpriseSignals : existing.enterpriseSignals
  };
}

/* ────────────────────────────────────────────
   BUILD FULL HISTORY STRING for ranking
   ──────────────────────────────────────────── */
function getFullUserText(session) {
  if (!session.history) return "";
  return session.history.filter(h => h.role === "user").map(h => h.content).join(" ");
}

/* ────────────────────────────────────────────
   DEEP QUESTIONS — app-specific
   ──────────────────────────────────────────── */
async function buildDynamicBundleCard(session) {
  const purpose = session.extraction?.appPurpose || "";
  if (!purpose || purpose.trim().length < 12) return null;

  const lastUserTurn = [...(session.history || [])].reverse().find((h) => h.role === "user");
  const latestUserText = lastUserTurn?.content || "";

  if (session.dynamicContext && shouldRerunTriageAfterFormatCorrection(session, latestUserText)) {
    session.dynamicContext = null;
    session.awaitingTriageAnswer = false;
    const corrected = parseChipAppType(latestUserText);
    if (corrected) {
      session.appType = corrected;
      session.extraction = session.extraction || {};
      session.extraction.appType = corrected;
      session.extraction.confidence = session.extraction.confidence || {};
      session.extraction.confidence.appType = "HIGH";
    }
    await saveSession(session);
  }

  // ─── AGENTIC TRIAGE: Evaluate specificity before generating form ───
  // Skip triage if we already have a ready dynamic context (user answered a clarification)
  if (!session.dynamicContext) {
    const triageResult = await triageDynamicContext({
      appType: session.appType || null,
      appPurpose: purpose,
      languageHint: detectLanguageMode(session),
      conversationHistory: session.history || []
    });

    // ─── NEEDS CONTEXT OR VERIFICATION: Ask a clarifying question ───
    if ((triageResult.status === "needs_context" || triageResult.status === "needs_format") && triageResult.question) {
      session.triageRounds = (session.triageRounds || 0) + 1;
      if (triageResult.domain) session.domainIdentified = triageResult.domain;

      console.log(
        `[Triage] Round ${session.triageRounds} — Status: ${triageResult.status} — Asking: ${triageResult.question.substring(0, 80)}`
      );

      if (session.triageRounds <= 3) {
        session.awaitingTriageAnswer = true;

        // If it's verifying the format, provide the chips so they can click!
        const isFormatCheck = triageResult.status === "needs_format";

        return {
          reply: triageResult.question,
          uiType: isFormatCheck ? "chips" : null,
          uiData: isFormatCheck ? { options: ["Text", "Image", "Audio", "Video"] } : null,
          nextStep: session.step,
          coins: null
        };
      }
    }

    // Same triage outcome but round cap exceeded — still never skip straight to an unconfirmed form
    if (
      (triageResult.status === "needs_context" || triageResult.status === "needs_format") &&
      triageResult.question &&
      session.triageRounds > 3
    ) {
      session.awaitingTriageAnswer = true;
      const isFormatCheck = triageResult.status === "needs_format";
      return {
        reply: triageResult.question,
        uiType: isFormatCheck ? "chips" : null,
        uiData: isFormatCheck ? { options: ["Text", "Image", "Audio", "Video"] } : null,
        nextStep: session.step,
        coins: null
      };
    }

    // Never build the dynamic form while triage still wants domain or format clarification
    if (triageResult.status === "needs_context" || triageResult.status === "needs_format") {
      session.awaitingTriageAnswer = true;
      const isFormatCheck = triageResult.status === "needs_format";
      const fallbackReply =
        triageResult.question && String(triageResult.question).trim().length >= 10
          ? triageResult.question
          : "What type of output should this app generate for users?";
      return {
        reply: fallbackReply,
        uiType: isFormatCheck ? "chips" : null,
        uiData: isFormatCheck ? { options: ["Text", "Image", "Audio", "Video"] } : null,
        nextStep: session.step,
        coins: null
      };
    }

    // ─── READY: AI has enough context, store the form and deduced output format ───
    if (triageResult.status === "ready") {
      console.log(`Successfully scoped app for domain: ${triageResult.domain || "Unknown"}`);
      if (triageResult.domain) session.domainIdentified = triageResult.domain;
      if (triageResult.app_format) {
        session.appType = triageResult.app_format;
        await saveSession(session);
      }
    }
    
    if (triageResult.form) {
      session.dynamicContext = triageResult.form;
    } else {
      // Fallback: generate via the original method
      session.dynamicContext = await generateDynamicContext({
        appType: session.appType || session.extraction?.appType || "text",
        appPurpose: purpose,
        languageHint: detectLanguageMode(session)
      });
    }
  }

  // Reset triage state
  session.triageRounds = 0;
  session.awaitingTriageAnswer = false;

  const displayFormat =
    String(session.appType || "text").charAt(0).toUpperCase() +
    String(session.appType || "text").slice(1).toLowerCase();

  return {
    reply: localizedText(
      session,
      `Perfect. I've scoped out the architecture for your ${displayFormat} app.\n\nNote: If you actually wanted this to generate Images, Audio, or Video instead, just let me know.\n\nPlease confirm these settings.`,
      `बहुत बढ़िया। मैंने इस ऐप को ${displayFormat} प्रारूप में स्कोप किया है।\n\nध्यान दें: अगर आप Image, Audio या Video आउटपुट चाहते हैं तो बता सकते हैं।\n\nकृपया ये सेटिंग्स कन्फर्म करें।`,
      `Perfect! Maine ise ${displayFormat} app samajh ke scope kiya hai.\n\nNote: Agar tumhe Images, Audio ya Video chahiye ho, bas bata dena.\n\nPlease ab settings confirm karo.`
    ),
    uiType: "multi_select_form",
    uiData: {
      appType: session.appType || "text",
      appPurpose: purpose,
      options: session.dynamicContext.options || [],
      variables: session.dynamicContext.variables || []
    },
    nextStep: 0,
    coins: null
  };
}

function getNextDeepQuestion(session) {
  if (!session.deepAnswers) session.deepAnswers = {};

  // Budget is always required last before model selection.
  if (!session.extraction?.budget && !session.deepAnswers?.budgetPreference) {
    return {
      field: "budgetPreference",
      question: "One last thing — what is your target budget per generation for this app?",
      options: [
        "Free models only (0 coins)",
        "Low (< 5 coins)",
        "Medium (5 - 20 coins)",
        "Premium (> 20 coins)"
      ]
    };
  }

  return null;
}

// FIXED: Removed ConfirmCard, set step=1
async function showModels(session) {
  const fullText = [
    session.extraction?.appPurpose || '',
    session.extraction?.oneLineUnderstanding || '',
    JSON.stringify(session.deepAnswers || {})
  ].join(' ');
  // UPDATE: Pull budget from deep answers first, fallback to extraction
  const budget = session.deepAnswers?.budgetPreference || session.extraction?.budget;
  
  const models = rankModels(MODELS[session.appType] || [], fullText, budget);
  
  session.step = 1; 
  session.awaitingConfirmation = false;
  await saveSession(session);
  
  return {
    reply: `Here are the top 3 models for your ${session.appType} app. Click a model card below to select it:`,
    uiType: 'models',
    uiData: { appType: session.appType, models },
    nextStep: 1,
    coins: null
  };
}

async function buildStep0Response(session) {
  const ext = session.extraction;

  // 1. IF APP TYPE IS MISSING: Ask for format (Only happens if Smart Inference fails)
  if (!session.appType) {
    session.step = 0;
    await saveSession(session);

    let replyText = "That sounds like a great idea! ";
    if (ext && ext.appPurpose && ext.appPurpose.length > 5) {
      replyText = `Awesome, an app for ${ext.appPurpose.toLowerCase().replace("i want to make a ", "")} sounds great! `;
    }
    replyText += "To get started, what kind of output should it generate?";

    const options = detectLanguageMode(session) === "Hindi" ? ["टेक्स्ट", "इमेज", "ऑडियो", "वीडियो", "विज़न"] : ["Text", "Image", "Audio", "Video", "Vision"];
    return { reply: replyText, uiType: "chips", uiData: { options }, nextStep: 0, coins: null };
  }

  session.step = 0;
  session.awaitingConfirmation = false;

  // 2. AGENTIC TRIAGE (Deep Context Questions)
  // This loops until the AI has enough specific details to build the app's inputs
  if (!session.dynamicContext) {
    const triageResult = await triageDynamicContext({
      appType: session.appType,
      appPurpose: session.extraction?.appPurpose || "",
      languageHint: detectLanguageMode(session),
      conversationHistory: session.history || []
    });

    // If AI needs more context, ask the specific deep question!
    if (triageResult.status === "needs_context" && (session.triageRounds || 0) < 3) {
      const question = String(triageResult.question || "").trim();
      if (question.length >= 10) {
        session.triageRounds = (session.triageRounds || 0) + 1;
        session.lastQuestion = question;
        await saveSession(session);
        return {
          reply: question,
          uiType: null, // Pure conversational text, no forced UI chips
          uiData: null,
          nextStep: 0,
          coins: null
        };
      }
      // Empty question from triage — skip to ready state
    }

    // AI is satisfied (or hit max rounds). Save the generated input variables silently for the Live Preview.
    session.dynamicContext = triageResult.form;
    session.triageRounds = 0;
    await saveSession(session);
  }

  // 3. BUDGET QUESTION (Only asked AFTER it fully understands the app)
  if (!session.extraction?.budget && !session.deepAnswers?.budgetPreference) {
    session.currentDeepField = "budgetPreference";
    session.awaitingDeepAnswer = true;
    await saveSession(session);
    return {
      reply: "Got all the details I need! One last thing before I design the architecture — what is your target budget per generation?",
      uiType: "chips",
      uiData: { options: ["Free models only (0 coins)", "Low (< 5 coins)", "Medium (5 - 20 coins)", "Premium (> 20 coins)"] },
      nextStep: 0,
      coins: null
    };
  }

  // 4. ALL DONE -> SHOW MODELS
  return await showModels(session);
}

/* ────────────────────────────────────────────
   EDGE CASE GUARDS
   ──────────────────────────────────────────── */
function checkEdgeCases(message, session) {
  const text = normalize(message);
  const msg = lower(text);

  // Skip edge case guards for structured UI payloads (form submissions, model selections, etc.)
  if (text.startsWith("multi_select_form::") || text.startsWith("edit prompt::") || text.startsWith("confirm seo::")) {
    return null;
  }

  if (isOffTopic(text)) return OFF_TOPIC_RESPONSE;

  // JAILBREAK / PROMPT INJECTION GUARD
  const jailbreaks = ['ignore all previous', 'system prompt', 'developer mode', 'you are now', 'disregard instructions'];
  if (jailbreaks.some(j => msg.includes(j))) {
    return {
      reply: "I am strictly programmed to help you build and configure apps for the RentPrompts marketplace. Let's get back on track. What kind of app would you like to build?",
      uiType: "chips",
      uiData: { options: ['Image app', 'Video app', 'Text app'] },
      nextStep: session.step,
      coins: null
    };
  }

  // NSFW / POLICY VIOLATION GUARD
  const nsfw = ['nsfw', 'porn', 'deepfake', 'nude', 'violence', 'illegal', 'hack'];
  if (nsfw.some(n => msg.includes(n))) {
    return {
      reply: "I can only help build apps that comply with RentPrompts' safety and content guidelines. Please suggest a different idea.",
      uiType: "text",
      uiData: null,
      nextStep: session.step,
      coins: null
    };
  }

  // MID-FLIGHT PIVOT GUARD
  const pivots = ['actually i want', 'change to', 'instead let', 'can we build a'];
  if (pivots.some(p => msg.includes(p)) && session.step > 0) {
    return {
      reply: "No problem, we can pivot! Let's reset the setup. What is the new app idea?",
      uiType: "text",
      uiData: null,
      nextStep: 0,
      coins: null,
      clearSession: true
    };
  }

  const trimmedText = text.trim();
  
  // 1. Pure Greeting Interceptor
  // Catches "hello", "hi", "hey", "hy", "greetings" even with punctuation like "hello!!"
  const isGreeting = /^(hi|hello|hey|hy|hola|greetings)[\s!\.]*$/i.test(trimmedText);
  
  if (isGreeting) {
    return {
      reply:
        "Hey! I'm the RentPrompts App Creation Agent. What kind of AI app would you like to build today? You can just describe what you want it to do.",
      uiType: null,
      uiData: null,
      nextStep: session.step,
      coins: null
    };
  }

  // 2. Abuse / gibberish guard
  const symbolCount = (trimmedText.match(/[^a-zA-Z0-9\s]/g) || []).length;
  
  const isGibberish = 
    trimmedText.length < 2 || // Too short
    /(asdf|qwer|zxcv|hjkl|asdasd)/i.test(trimmedText) || // Expanded keyboard smash
    /[a-zA-Z0-9]{20,}/.test(trimmedText) || // Huge 20+ char block with no spaces
    (symbolCount > trimmedText.length / 2 && trimmedText.length > 5) || // Over 50% symbols
    /[bcdfghjklmnpqrstvwxz]{5,}/i.test(trimmedText) || // NEW: 5+ consonants in a row (catches 'abcszs')
    /(.)\1{4,}/i.test(trimmedText) || // 5+ of the exact same character
    (!trimmedText.includes(' ') && trimmedText.length > 8 && /[0-9@#\$\%\^\&\*]/.test(trimmedText)); // NEW: 8+ chars, no spaces, mixed with numbers/symbols (catches 'asdasd@#q31')

  if (isGibberish) {
    return {
      // Changed the reply to be exactly what you want when nonsense is typed
      reply: `Sorry, I didn't quite get what you want to build today. What type of output does your app need?`,
      uiType: 'chips',
      uiData: { options: ['Text', 'Image', 'Audio', 'Video', 'Vision'] },
      nextStep: session.step,
      coins: null
    };
  }

  if (!text || text === '') {
    return {
      reply: `Go ahead — describe what you'd like to build!`,
      uiType: 'text',
      uiData: null,
      nextStep: session.step,
      coins: null
    };
  }

  if (msg.includes('help') && text.trim().split(' ').length <= 3) {
    return {
      reply: `Sure! Here's what I can help you build:\n\n🖼️ Image apps\n🎥 Video apps\n📝 Text apps\n🔊 Audio apps\n👁️ Vision apps\n\nWhich type interests you?`,
      uiType: 'chips',
      uiData: { options: ['Image app', 'Video app', 'Text app', 'Audio app', 'Vision app', 'Help me choose'] },
      nextStep: 0,
      coins: null
    };
  }

  if (msg.includes('start over') || msg.includes('restart') || msg.includes('reset') || msg.includes('new app') || msg.includes('different app')) {
    return {
      reply: `No problem! Let's start fresh. What kind of AI app would you like to build?`,
      uiType: 'chips',
      uiData: { options: ['Image app', 'Video app', 'Text app', 'Audio app', 'Vision app'] },
      nextStep: 0,
      coins: null,
      clearSession: true
    };
  }

  if ((msg.includes('how much') || msg.includes('price') || msg.includes('cost') || msg.includes('joules')) && session.step < 4) {
    return {
      reply: `Great question! The cost depends on which AI model we choose. Prices range from FREE to 318 coins per run.\n\nLet me first understand what you need. What kind of app are you building?`,
      uiType: session.appType ? 'text' : 'chips',
      uiData: session.appType ? null : { options: ['Image app', 'Video app', 'Text app', 'Audio app', 'Vision app'] },
      nextStep: session.step,
      coins: null
    };
  }

  const competitors = ['openai', 'chatgpt', 'midjourney', 'dalle', 'stable diffusion', 'runway', 'sora', 'adobe', 'canva', 'figma'];
  if (competitors.some(c => msg.includes(c))) {
    return {
      reply: `I work specifically with the AI models available on RentPrompts marketplace.\n\nWant me to show you what's available?`,
      uiType: 'chips',
      uiData: { options: ['Yes show me models', 'Tell me more about RentPrompts', 'Start building my app'] },
      nextStep: session.step,
      coins: null
    };
  }

  return null; 
}

/* ════════════════════════════════════════════
   MAIN ROUTER
   ════════════════════════════════════════════ */
export async function route(session, message) {
  // 1. WALL OF TEXT GUARD: Truncate to 1000 characters
  const rawText = String(message || "").substring(0, 1000);
  const text = normalize(rawText);
  const msg = lower(text);

  // ─── 1. EXPLICIT UI CHIP HANDLER ───
  if (session.step === 2 && msg === 'edit app') {
    session.awaitingPromptTweak = true;
    await saveSession(session);
    return {
      reply: "I'm listening! You can describe a completely new app idea, ask to change the model, or just tweak the current instructions.",
      uiType: "text",
      uiData: null,
      nextStep: 2,
      coins: session.modelCost
    };
  }

  // ─── 2. UNIVERSAL PIVOT & EDIT INTERCEPTOR ───
  const editTriggers = ['change ', 'update ', 'switch ', 'actually', 'instead', 'tweak', 'edit '];
  const isEditTrigger = editTriggers.some((t) => msg.includes(t)) || session.awaitingPromptTweak;

  const formatRegex = /(image|video|audio|text|vision)\s+(app|generator|tool|creator)\b/i;
  const isNewFormatMentioned = formatRegex.test(msg);

  const isMajorPivot =
    isNewFormatMentioned ||
    msg.includes('app type') ||
    msg.includes('domain') ||
    msg.includes('instead') ||
    msg.includes('new app') ||
    msg.includes('different app') ||
    /^(i want (to )?(build|make|create)|let'?s (build|make|create)|build an?|make an?|create an?)/i.test(msg) ||
    (session.awaitingPromptTweak && msg.includes('app'));

  if (
    (isEditTrigger || isMajorPivot) &&
    !text.startsWith("select") &&
    !text.toLowerCase().startsWith("multi_select")
  ) {
    // A. Model Change Request
    if (msg.includes('model') || msg.includes('ai engine') || msg.includes('different ai')) {
      if (session.step === 0 && !session.dynamicContext) {
        return {
          reply: `We'll pick the perfect AI model in just a moment! But first, let's finish scoping out the app.\n\n${session.lastQuestion || "What specific details should the app analyze?"}`,
          uiType: null,
          nextStep: 0,
          coins: null
        };
      }
      session.awaitingPromptTweak = false;
      return await showModels(session);
    }

    // B. Major Pivot (Changing App Type or Domain entirely)
    const tweakKeywords = ['prompt', 'instruction', 'description', 'tone', 'make it', 'add a', 'remove', 'change the', 'rewrite'];
    const isMinorTweak = tweakKeywords.some((kw) => msg.includes(kw)) && !isMajorPivot && session.step > 0;

    if (isMajorPivot && !isMinorTweak) {
      let newType = parseChipAppType(text);
      if (!newType) {
        const match = msg.match(/(image|video|audio|text|vision)/i);
        if (match) newType = match[1].toLowerCase();
      }

      let cleanPurpose = text
        .replace(/\b(actually|instead|change it to|switch to|something different)\b/gi, "")
        .replace(/\bi want to (build|make|create) (a|an)\b/gi, "")
        .replace(/\bi want (a|an)\b/gi, "")
        .replace(/refer to the (image|picture|screenshot)/gi, "")
        .trim();

      if (!cleanPurpose || cleanPurpose.length < 3) {
        cleanPurpose = newType ? `${newType} generation app` : "a new idea";
      }

      session.dynamicContext = null;
      session.appType = newType || null;
      session.extraction = { appPurpose: cleanPurpose, confidence: {} };
      session.deepAnswers = {};
      session.step = 0;
      session.triageRounds = 0;
      session.awaitingPromptTweak = false;
      session.awaitingDeepAnswer = false;
      session.currentDeepField = null;
      session.isPivot = true;
      await saveSession(session);

      return await buildStep0Response(session);
    }

    // C. Minor Tweak at the Preview Step
    if (session.step === 2) {
      session.awaitingPromptTweak = false;
      session.promptData = applyPromptInstruction(session.promptData, message);
      await saveSession(session);
      return {
        reply: `Got it! I've updated the app logic based on your request: "${text}".\n\n**Updated Logic:**\n${session.promptData.userPrompt}\n\nReady to publish?`,
        uiType: 'chips',
        uiData: { options: ['Publish to Marketplace', 'Save Draft', 'Edit App'] },
        nextStep: 2,
        coins: session.modelCost
      };
    }

    // D. General mid-flight tweak acknowledgement
    return {
      reply: `Noted! I've updated your app's requirements to include: "${text}". Let's continue.`,
      uiType: null,
      nextStep: session.step,
      coins: null
    };
  }

  // 1. EXTRACTION FIRST (so we never lose the user's context!)
  if (!session.history) session.history = [];
  const latestExtraction = await extractRequirements(text, session.history || []);
  session.extraction = mergeExtraction(session.extraction, latestExtraction, text);
  session.languageMode = detectLanguageMode(session);

  if (session.extraction.enterpriseSignals !== undefined) {
    session.enterpriseSignals = session.extraction.enterpriseSignals;
  }
  if (session.extraction.userType) {
    session.userType = session.extraction.userType;
  }

  // 1. SMART INFERENCE: Auto-assign appType so the AI feels intelligent!
  // If the user says "advocate app", extraction knows it's text. We skip the format question.
  if (!session.appType && session.extraction.appType && session.extraction.confidence?.appType !== 'LOW') {
    session.appType = session.extraction.appType;
  }

  if (lower(text) === 'save draft' || lower(text).includes('save as draft') || lower(text).includes('save to draft')) {
    return {
      reply: `Done! Your progress has been securely saved as a draft. You can access it anytime from your RentPrompts dashboard.`,
      uiType: 'success',
      uiData: {
        appName: session.seoData?.appName || session.extraction?.appPurpose || 'Untitled Draft',
        modelId: session.modelId || 'Draft Mode',
        costPerRun: session.modelCost || 0,
        status: 'Draft'
      },
      nextStep: 0,
      coins: null,
      clearSession: true
    };
  }

  const edgeCaseResponse = checkEdgeCases(message, session);
  if (edgeCaseResponse) {
    if (edgeCaseResponse.clearSession) {
      await deleteSession(session.sessionId);
    }
    return edgeCaseResponse;
  }

  // ─── STEP 0: First message — detect app type ────────
  if (session.step === 0) {
    const greetings = ['hi', 'hello', 'hey', 'hii', 'helo', 'good morning', 'good evening', 'yo', 'sup', 'namaste', 'hola'];
    const isGreeting = greetings.some(g => lower(text) === g || lower(text).startsWith(g + ' '));

    if (isGreeting) {
      await saveSession(session);
      return {
        reply: "Hey! I'm the RentPrompts App Creation Agent. What kind of AI app would you like to build today?",
        uiType: null,
        uiData: null,
        nextStep: 0,
        coins: null
      };
    }

    const isNotSure = lower(text).includes('not sure') || lower(text).includes('help me');
    if (isNotSure) {
      session.step = 0;
      await saveSession(session);
      return {
        reply: "No problem! First — what is the main thing you want your app to CREATE or DO?",
        uiType: 'chips',
        uiData: { options: ['Generate images', 'Create videos', 'Write text', 'Generate audio', 'Analyze images'] },
        nextStep: 0,
        coins: null
      };
    }

    if (session.awaitingDeepAnswer && session.currentDeepField) {
      const answer = text;
      if (!session.deepAnswers) session.deepAnswers = {};
      session.deepAnswers[session.currentDeepField] = answer;
      if (!session.extraction) session.extraction = {};
      if (session.currentDeepField === "budgetPreference") session.extraction.budget = answer;
      session.awaitingDeepAnswer = false;
      session.currentDeepField = null;

      const nextQ = getNextDeepQuestion(session);
      if (nextQ) {
        session.currentDeepField = nextQ.field;
        session.awaitingDeepAnswer = true;
        await saveSession(session);
        return {
          reply: nextQ.question,
          uiType: "chips",
          uiData: { options: nextQ.options || [] },
          nextStep: 0,
          coins: null
        };
      }
      return await showModels(session);
    }

    const chipType = parseChipAppType(text);
    if (chipType && !session.appType) {
      session.appType = chipType;
      session.extraction = session.extraction || {};
      session.extraction.appType = chipType;
    }

    return await buildStep0Response(session);
  }

  // ─── STEP 1: Model selection → generate config ───────────────────────
  if (session.step === 1) {
    const selectedModelId = parseSelectedModelId(text, MODELS[session.appType] || []);

    if (selectedModelId) {
      const selectedModel = findModel(session.appType, selectedModelId);
      if (!selectedModel) {
        return {
          reply: "I couldn't match that model. Please click one of the options above.",
          uiType: "text",
          uiData: null,
          nextStep: 1,
          coins: null
        };
      }

      session.modelId = selectedModel.id;
      session.modelCost = selectedModel.cost;
      session.awaitingConfirmation = false;
      await saveSession(session);

      try {
        const promptData = await generatePromptTemplate(session);
        session.promptData = promptData;

        const seoData = await generateSEO(session);
        session.seoData = seoData;

        session.step = 2;
        await saveSession(session);

        // COMPREHENSIVE FINAL SUMMARY TEXT (SEO, Cost, Retail Price, Tags, Model)
        const summaryText = `🎉 **App Configured Successfully!**

Here is your app's complete setup:

📝 **App Details**
* **Name:** ${seoData.appName}
* **Description:** ${seoData.appDescription}
* **Category:** ${seoData.category}
* **Tags:** ${Array.isArray(seoData.tags) ? seoData.tags.join(", ") : "—"}

⚙️ **Technical Architecture**
* **AI Engine:** ${selectedModel.name}
* **Base API Cost:** ${session.modelCost} coins per run
* **Suggested Retail Price:** ${seoData.suggestedPrice} coins per run

The AI has automatically generated the required input variables and a highly detailed backend prompt for your users based on our chat. Check the **Live Preview** card to test it!

*Want to edit anything? You can say things like "Change the model", "Make the audience kids", or "Tweak the prompt". Otherwise, click Publish!*`;

        return {
          reply: summaryText,
          uiType: 'app_preview',
          uiData: {
            appName: seoData.appName,
            appType: session.appType || session.extraction?.appType || 'text',
            appDescription: seoData.appDescription,
            cost: session.modelCost,
            systemPrompt: promptData.systemPrompt,
            userPrompt: promptData.userPrompt,
            variablesUsed: promptData.variablesUsed,
            acceptImageInput: promptData.acceptImageInput,
            options: ['Publish to Marketplace', 'Save Draft', 'Edit App']
          },
          nextStep: 2,
          coins: session.modelCost
        };
      } catch (error) {
        console.error("Error generating app config:", error);
        return {
          reply: "I hit a snag generating the config. Please try selecting the model again.",
          uiType: 'text',
          uiData: null,
          nextStep: 1,
          coins: null
        };
      }
    }

    return {
      reply: "Please click one of the model cards above to select the engine for your app.",
      uiType: "text",
      uiData: null,
      nextStep: 1,
      coins: null
    };
  }

  // ─── STEP 2: Final Review (Publish, Save Draft, Tweak) ───────────────────
  if (session.step === 2) {
    const msg2 = lower(message);

    if (msg2.includes('publish') || isYes(text)) {
      const payload = {
        appType: session.appType,
        modelId: session.modelId,
        costPerRun: session.modelCost,
        systemPrompt: session.promptData?.systemPrompt,
        userPrompt: session.promptData?.userPrompt,
        negativePrompt: session.promptData?.negativePrompt,
        acceptImageInput: session.promptData?.acceptImageInput,
        appName: session.seoData?.appName,
        appDescription: session.seoData?.appDescription,
        tags: session.seoData?.tags,
        publishedAt: new Date().toISOString()
      };

      console.log("\n══════ MOCK PUBLISH ══════");
      console.log(JSON.stringify(payload, null, 2));
      console.log("══════════════════════════\n");

      return {
        reply: `🎉 Your app "${session.seoData?.appName}" is now live! Users will be charged ${payload.costPerRun} coins per generation.`,
        uiType: "success",
        uiData: {
          appName: session.seoData?.appName,
          modelId: session.modelId,
          costPerRun: payload.costPerRun,
          tags: session.seoData?.tags,
          mockUrl: `https://rentprompts.ai/app/demo-${Date.now()}`
        },
        nextStep: 0,
        coins: session.modelCost,
        clearSession: true
      };
    }

    if (msg2.includes('draft') || msg2.includes('save')) {
      session.status = 'draft';
      await saveSession(session);
      return {
        reply: `Done! "${session.seoData?.appName}" saved as a draft. Publish anytime from your dashboard.`,
        uiType: 'success',
        uiData: { appName: session.seoData?.appName, status: 'Draft' },
        nextStep: 0,
        coins: null
      };
    }

    if (msg2.includes('tweak') || isChangeMessage(text)) {
      const correction = isChangeMessage(text) ? getChangeText(text) : null;
      if (correction) {
        session.promptData = applyPromptInstruction(session.promptData, correction);
        await saveSession(session);
        return {
          reply: `Updated!\n\nUser Prompt: ${session.promptData.userPrompt}\n\nReady to publish?`,
          uiType: 'chips',
          uiData: { options: ['Publish to Marketplace', 'Save Draft', 'Edit App'] },
          nextStep: 2,
          coins: session.modelCost
        };
      }
      session.awaitingPromptTweak = true;
      await saveSession(session);
      return {
        reply: "What would you like to edit? You can change the AI Model, rewrite the prompt, or completely change the app idea (e.g., 'Change this to an image app').",
        uiType: "text",
        uiData: {},
        nextStep: 2,
        coins: session.modelCost
      };
    }
  }

  // ─── CATCH ALL ───
  return {
    reply: "I'm ready to proceed. Let me know if you want to publish, save a draft, or edit the app.",
    uiType: "chips",
    uiData: { options: ['Publish to Marketplace', 'Save Draft', 'Edit App'] },
    nextStep: session.step || 0,
    coins: session.modelCost
  };
}
