import MODELS from "./models.js";
import mockData from "./mockData.js";
import { extractRequirements, generateDynamicContext, triageDynamicContext, buildDynamicContextFallback } from "./groq.js";
import { generatePromptTemplate, generateSEO, applyPromptInstruction, buildPromptTemplateFromSession } from "./gemini.js";
import { buildBudgetTiers, getModelCost } from "./costCalculator.js";
import { isOffTopic, OFF_TOPIC_RESPONSE } from "./requirementRouter.js";
import { saveSession, deleteSession } from "./redis.js";
import { getAgenticIntent, getAgenticDecision } from "./intentEngine.js";

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

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
   SMART MODEL RANKING (spec-exact)
   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
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

  // UNMATCHABLE BUDGET GUARD
  // For premium/medium tiers, show the TOP (most expensive) models when no exact match exists.
  // For free/low, fall back to cheapest.
  if (filtered.length === 0) {
    const isPremiumIntent = b.includes("premium") || b.includes("best") || b.includes("> 20");
    const isMediumIntent  = b.includes("medium") || b.includes("5-20") || b.includes("5 - 20");
    if (isPremiumIntent || isMediumIntent) {
      return [...availableModels].sort((a, bm) => bm.cost - a.cost).slice(0, 3);
    }
    return [...availableModels].sort((a, bm) => a.cost - bm.cost).slice(0, 3);
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

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
   PARSERS (FIXED BULLETPROOF PARSING)
   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
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

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
   MODEL / COST HELPERS
   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

/** Hard guard: Only image and vision apps can ever accept image input.
 *  Audio, text, and video apps should NEVER show an upload UI,
 *  regardless of what the LLM returned. */
function sanitizeAcceptImageInput(rawValue, appType) {
  const type = String(appType || '').toLowerCase();
  if (type === 'image' || type === 'vision') return Boolean(rawValue);
  return false;
}

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

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
   TONE-AWARE REPLY PREFIX
   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
function tonePrefix(extraction) {
  if (!extraction) return "";
  if (extraction.userTone === "urgent") return "No worries, let's set this up quickly! ";
  if (extraction.userTone === "unsure") return "Happy to help figure this out together! ";
  if (extraction.detectedLanguage === "Hindi") return "Samajh gaya — ";
  return "";
}

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
   BUDGET / COMPLEXITY HELPERS
   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
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

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
   MERGE EXTRACTION
   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
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

/* ──────────────────────────────────────────────────────────────────────────
   BUILD FULL HISTORY STRING for ranking
   ────────────────────────────────────────────────────────────────────────── */
function getFullUserText(session) {
  if (!session.history) return "";
  return session.history.filter(h => h.role === "user").map(h => h.content).join(" ");
}

/* ──────────────────────────────────────────────────────────────────────────
   DEEP QUESTIONS — app-specific
   ────────────────────────────────────────────────────────────────────────── */
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
    let triageResult = await triageDynamicContext({
      appType: session.appType || null,
      appPurpose: purpose,
      languageHint: detectLanguageMode(session),
      conversationHistory: session.history || []
    });

    // Enforce question limits (max 3 total questions across the entire conversation)
    if ((session.triageRounds || 0) >= 3 && (triageResult.status === "needs_context" || triageResult.status === "needs_format")) {
      console.log(`[Triage] Limit of 3 questions reached. Forcing status to ready.`);
      triageResult = {
        status: "ready",
        domain: triageResult.domain || session.domainIdentified || null,
        app_format: triageResult.corrected_app_type || session.appType || "text",
        form: null
      };
    }

    // ─── NEEDS CONTEXT OR VERIFICATION: Ask a clarifying question ───
    if ((triageResult.status === "needs_context" || triageResult.status === "needs_format") && triageResult.question) {
      session.triageRounds = (session.triageRounds || 0) + 1;
      if (triageResult.domain) session.domainIdentified = triageResult.domain;

      console.log(
        `[Triage] Round ${session.triageRounds} — Status: ${triageResult.status} — Asking: ${triageResult.question.substring(0, 80)}`
      );

      session.awaitingTriageAnswer = true;

      const isFormatCheck = triageResult.status === "needs_format";
      const hasLLMChips = !isFormatCheck && Array.isArray(triageResult.suggested_options) && triageResult.suggested_options.length >= 2;

      return {
        reply: triageResult.question,
        uiType: (isFormatCheck || hasLLMChips) ? "chips" : null,
        uiData: isFormatCheck
          ? { options: ["Text", "Image", "Audio", "Video"] }
          : hasLLMChips
            ? { options: triageResult.suggested_options }
            : null,
        nextStep: session.step,
        coins: null
      };
    }

    // Never build the dynamic form while triage still wants domain or format clarification
    if (triageResult.status === "needs_context" || triageResult.status === "needs_format") {
      session.awaitingTriageAnswer = true;
      const isFormatCheck = triageResult.status === "needs_format";
      const hasLLMChips = !isFormatCheck && Array.isArray(triageResult.suggested_options) && triageResult.suggested_options.length >= 2;
      const fallbackReply =
        triageResult.question && String(triageResult.question).trim().length >= 10
          ? triageResult.question
          : "What type of output should this app generate for users?";
      return {
        reply: fallbackReply,
        uiType: (isFormatCheck || hasLLMChips) ? "chips" : null,
        uiData: isFormatCheck
          ? { options: ["Text", "Image", "Audio", "Video"] }
          : hasLLMChips
            ? { options: triageResult.suggested_options }
            : null,
        nextStep: session.step,
        coins: null
      };
    }

    // â”€â”€â”€ READY: AI has enough context, store the form and deduced output format â”€â”€â”€
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
      `## ✅ App Architecture Ready\n\nI've analyzed your requirements and scoped out a **${displayFormat}** app.\n\nHere's what I've configured for you — review the features and input fields below, then hit confirm to proceed.\n\n**💡 Tip:** If you actually wanted this to generate a different output type (Images, Audio, Video), just let me know and I'll adjust instantly.`,
      `## ✅ ऐप आर्किटेक्चर तैयार\n\nमैंने आपकी ज़रूरतों का विश्लेषण किया और **${displayFormat}** ऐप स्कोप किया है।\n\nनीचे फीचर्स और इनपुट फील्ड्स देखें, फिर कन्फर्म करें।\n\n**💡 टिप:** अगर आप Image, Audio या Video आउटपुट चाहते हैं तो बता दें।`,
      `## ✅ App Architecture Ready\n\nMaine tumhare requirements analyze karke **${displayFormat}** app scope kiya hai.\n\nNeeche features aur input fields check karo, phir confirm karo.\n\n**💡 Tip:** Agar Images, Audio ya Video chahiye ho, bas bata dena!`
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
      question: "Last step before I pick models — what's your budget per generation?",
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

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
   APPLY EDIT TO SESSION — rewrites session state so generatePromptTemplate
   gets clean, up-to-date context instead of old ghost memory.
   Call this BEFORE generatePromptTemplate on every edit path.
   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
function applyEditToSession(session, editInstruction) {
  const instr = String(editInstruction || '').trim().toLowerCase();
  if (!instr) return;

  // 1. Update appPurpose to include the edit intent
  if (session.extraction) {
    const oldPurpose = session.extraction.appPurpose || '';
    // Avoid duplicating if already in there
    if (!oldPurpose.toLowerCase().includes(instr.slice(0, 20))) {
      session.extraction.appPurpose = `${oldPurpose} (updated: ${editInstruction.trim()})`;
    }
    session.extraction.oneLineUnderstanding = session.extraction.appPurpose;
  }

  // 2. Remove / replace deepAnswers keys that conflict with the edit
  if (session.deepAnswers && typeof session.deepAnswers === 'object') {
    // Detect "transparent" or "no background" edits â†’ kill scene/background answers
    const wantsTransparent = /transparent|no background|remove background|no bg/i.test(editInstruction);
    const wantsNewBackground = /background|backdrop|scene|environment/i.test(editInstruction);
    // Detect "no X" pattern — user explicitly rejecting a value
    const noXMatch = editInstruction.match(/no\s+(\w+)/i);

    const keysToRemove = [];
    for (const [k, v] of Object.entries(session.deepAnswers)) {
      const kl = k.toLowerCase();
      const vl = String(v || '').toLowerCase();

      if (wantsTransparent && /scene|background|forest|nature|backdrop|environment|location|setting/i.test(kl)) {
        keysToRemove.push(k);
      } else if (wantsNewBackground && /scene|background|backdrop|environment/i.test(kl)) {
        // Replace old value with the new instruction
        session.deepAnswers[k] = editInstruction.trim();
      } else if (noXMatch) {
        const rejectedWord = noXMatch[1].toLowerCase();
        if (vl.includes(rejectedWord) || kl.includes(rejectedWord)) {
          keysToRemove.push(k);
        }
      }
    }
    keysToRemove.forEach(k => delete session.deepAnswers[k]);

    // If transparent, also store this as a positive requirement
    if (wantsTransparent) {
      session.deepAnswers.outputType = 'transparent PNG';
      session.deepAnswers.backgroundType = 'transparent';
    }
  }

  // 3. Rebuild dynamicContext variables to strip obsolete ones
  if (session.dynamicContext?.variables) {
    const wantsTransparent = /transparent|no background|remove background|no bg/i.test(editInstruction);
    if (wantsTransparent) {
      // Remove scene/forest/backdrop variables — they conflict with transparent background
      session.dynamicContext.variables = session.dynamicContext.variables.filter(v => {
        const vname = (typeof v === 'object' ? v.name : String(v)).toLowerCase();
        return !/scene|forest|nature|backdrop|environment|location|background_scene|forest_scene/i.test(vname);
      });
    }
  }

  // 4. Store the edit instruction so generatePromptTemplate sees it
  if (!session.deepAnswers) session.deepAnswers = {};
  session.deepAnswers.lastEditInstruction = editInstruction.trim();

  // 5. Append to history so the LLM's context window reflects the change
  if (Array.isArray(session.history)) {
    session.history.push({
      role: 'user',
      content: `[EDIT REQUEST]: ${editInstruction.trim()}`
    });
  }
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
    reply: `## 🤖 AI Model Selection\n\nI've ranked the **top 3 models** for your **${session.appType}** app based on your requirements and budget.\n\nEach card shows the model's strengths, speed, and cost per run — **click any card** to select it.`,
    uiType: 'models',
    uiData: { appType: session.appType, models },
    nextStep: 1,
    coins: null
  };
}

async function buildStep0Response(session, text) {
  const ext = session.extraction;

  // â”€â”€ AMBIGUOUS DOMAIN DETECTION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  // These keywords are inherently ambiguous — they could be text OR image.
  // e.g., "birthday app" â†’ text (written wishes) or image (birthday card).
  // We SKIP local keyword inference for these and let the triage LLM ask the
  // user explicitly what output format they want.
  const AMBIGUOUS_DOMAIN_SIGNALS = [
    'birthday', 'greeting', 'certificate', 'diploma', 'award',
    'wedding', 'anniversary', 'thank you', 'congratulation',
    'wish', 'card app', 'card generat'
  ];

  const purposeLowerForAmbiguity = String(ext?.appPurpose || ext?.oneLineUnderstanding || '').toLowerCase();
  const isAmbiguousDomain = AMBIGUOUS_DOMAIN_SIGNALS.some(sig => purposeLowerForAmbiguity.includes(sig));

  // For ambiguous domains, ONLY skip local inference if the user hasn't already
  // given a strong explicit signal for a specific type.
  const hasExplicitTypeFromUser = Boolean(
    (ext?.confidence?.appType === 'HIGH') ||
    session.formatConfirmedByUser
  );

  // 1. IF APP TYPE IS MISSING: Try local keyword inference FIRST before showing chips
  //    BUT: Skip for ambiguous domains — let triage ask about output format instead.
  if (!session.appType) {
    const LOCAL_TYPE_SIGNALS = {
      image: [
        'background remov', 'remove background', 'bg remov', 'image generat', 'generate image',
        'photo generat', 'interior design', 'room design', 'logo generat', 'logo maker',
        'portrait', 'art generat', 'image creat', 'photo edit', 'image edit',
        'visual generat', 'thumbnail', 'product photo', 'banner maker', 'illustration',
        'greeting card', 'birthday card', 'card maker', 'card generat', 'poster',
        'meme', 'photo frame', 'photo filter', 'in the photo', 'on the image',
        'text on image', 'text overlay', 'image with text', 'invitation',
        'flyer', 'avatar', 'wallpaper', 'sticker', 'with photo', 'with picture',
        'birthday card with photo', 'birthday image', 'birthday poster'
      ],
      video: [
        'video generat', 'generate video', 'video creat', 'animate', 'animation',
        'photo to video', 'image to video', 'text to video', 'reel generat', 'short film',
        'video ad', 'cinematic', 'motion graphic'
      ],
      audio: [
        'audio generat', 'voiceover', 'voice over', 'text to speech', 'tts', 'speech generat',
        'music generat', 'podcast', 'narration', 'voice cloning', 'sound effect',
        'audio app', 'speak text', 'spoken audio', 'verbal briefing', 'audio briefing',
        'spoken summary', 'audio log', 'spoken log', 'verbal report', 'spoken report',
        'convert to audio', 'read aloud', 'audio file', 'voice briefing', 'tts app'
      ],
      vision: [
        'crop disease', 'plant disease', 'detect disease', 'object detect', 'image analys',
        'ocr', 'read text from image', 'invoice scan', 'document scan', 'quality inspect',
        'medical image', 'x-ray', 'analyze image', 'image recognit', 'face detect'
      ],
      text: [
        'advocate', 'legal', 'lawyer', 'law', 'blog', 'article', 'content', 'email',
        'lesson plan', 'quiz', 'teacher', 'educat', 'farm advisor', 'crop advisor',
        'chatbot', 'chat assistant', 'writing', 'summarize', 'translate', 'script',
        'seo', 'description generat', 'report generat', 'story generat',
        'resume', 'cover letter', 'proposal', 'invoice', 'contract', 'newsletter',
        'recipe', 'itinerary', 'planner', 'workout', 'meal plan', 'diet plan',
        'study guide', 'flashcard', 'essay', 'thesis', 'assignment', 'homework',
        'letter', 'memo', 'document', 'template', 'form generat', 'bio generat',
        'caption', 'tagline', 'slogan', 'headline', 'ad copy', 'copywriting',
        'product description', 'review generat', 'feedback generat', 'response generat',
        'text generat', 'write', 'draft', 'compose', 'author',
        'birthday wishes', 'birthday message', 'birthday poem', 'birthday quote'
      ]
    };

    const purposeLower = String(ext?.appPurpose || ext?.oneLineUnderstanding || '').toLowerCase();

    // If ambiguous domain AND user hasn't explicitly confirmed format â†’ skip local inference
    if (isAmbiguousDomain && !hasExplicitTypeFromUser) {
      console.log(`[Ambiguous Domain] "${purposeLower.substring(0, 60)}" matches ambiguous signals — skipping local inference, deferring to triage.`);
    } else {
      for (const [type, signals] of Object.entries(LOCAL_TYPE_SIGNALS)) {
        if (signals.some(sig => purposeLower.includes(sig))) {
          session.appType = type;
          if (!session.extraction) session.extraction = {};
          session.extraction.appType = type;
          break;
        }
      }
    }
  }

  // 2. Still no type after keyword inference?
  // If we have a PURPOSE but no type, let the triage LLM figure it out.
  // Only show bare format chips when we have ZERO context.
  if (!session.appType) {
    const hasPurpose = ext?.appPurpose && ext.appPurpose.length > 5;

    if (hasPurpose) {
      // For ambiguous domains: DON'T auto-infer, let triage ask explicitly.
      if (isAmbiguousDomain && !hasExplicitTypeFromUser) {
        console.log(`[Smart Infer] Ambiguous domain detected — NOT auto-inferring type. Triage will ask.`);
        // Don't set session.appType — fall through to triage which will ask about output format
      } else {
        // AGENTIC APPROACH: Smart default based on purpose keywords.
        // Detect if the purpose is likely visual (image) or textual.
        const purposeL = ext.appPurpose.toLowerCase();
        const imageSignals = ['photo', 'picture', 'image', 'card', 'poster', 'meme', 'frame', 'banner',
          'flyer', 'invitation', 'greeting', 'visual', 'avatar', 'portrait', 'logo', 'thumbnail',
          'in the photo', 'on the image', 'with picture', 'wallpaper', 'sticker'];
        const videoSignals = ['video', 'animation', 'animate', 'reel', 'clip', 'cinematic'];
        const audioSignals = ['audio', 'voice', 'music', 'speech', 'podcast', 'sound', 'tts'];
        const visionSignals = ['detect', 'analyze image', 'scan', 'ocr', 'read from image'];

        let inferredType = 'text'; // default
        if (imageSignals.some(s => purposeL.includes(s))) inferredType = 'image';
        else if (videoSignals.some(s => purposeL.includes(s))) inferredType = 'video';
        else if (audioSignals.some(s => purposeL.includes(s))) inferredType = 'audio';
        else if (visionSignals.some(s => purposeL.includes(s))) inferredType = 'vision';

        session.appType = inferredType;
        if (!session.extraction) session.extraction = {};
        session.extraction.appType = inferredType;
        console.log(`[Smart Infer] No explicit type for "${ext.appPurpose.substring(0, 50)}" — inferred '${inferredType}' from purpose keywords.`);
      }
      // Fall through to triage below — it will ask smart domain questions
    } else {
      // Truly zero context — user said something too vague. Show format chips.
      session.step = 0;
      await saveSession(session);
      const options = detectLanguageMode(session) === "Hindi" ? ["à¤Ÿà¥‡à¤•à¥à¤¸à¥à¤Ÿ", "à¤‡à¤®à¥‡à¤œ", "à¤‘à¤¡à¤¿à¤¯à¥‹", "à¤µà¥€à¤¡à¤¿à¤¯à¥‹", "à¤µà¤¿à¤œà¤¼à¤¨"] : ["Text", "Image", "Audio", "Video", "Vision"];
      return { reply: "What kind of output should your app produce?", uiType: "chips", uiData: { options }, nextStep: 0, coins: null };
    }
  }

  session.step = 0;
  session.awaitingConfirmation = false;

  // 3. AGENTIC TRIAGE (Deep Context Questions)
  if (!session.dynamicContext) {

    // 🟢 YES-AFFIRMATION FAST PATH: user said "yes/sure/ok" during triage → skip API, treat as ready
    const affirmations = ['yes', 'sure', 'ok', 'yep', 'yeah', 'correct', 'sounds good', 'exactly',
      'perfect', 'go ahead', 'proceed', "that's right", 'looks good', 'right', 'agreed', 'great'];
    const msgClean = lower(text).trim().replace(/[!.,?]+$/, '');
    const isAffirmation = affirmations.includes(msgClean);

    if (isAffirmation && (session.triageRounds || 0) > 0) {
      // User confirmed — mark triage complete, fall through to budget
      session.triageRounds = 99;
      await saveSession(session);
    } else {
      let triageResult = await triageDynamicContext({
        appType: session.appType,
        appPurpose: session.extraction?.appPurpose || "",
        languageHint: detectLanguageMode(session),
        conversationHistory: session.history || [],
        deepAnswers: session.deepAnswers || {}
      });

      // Enforce question limits (max 3 total questions across the entire conversation)
      if ((session.triageRounds || 0) >= 3 && (triageResult.status === "needs_context" || triageResult.status === "needs_format")) {
        console.log(`[Triage] Limit of 3 questions reached. Forcing status to ready.`);
        triageResult = {
          status: "ready",
          domain: triageResult.domain || session.domainIdentified || null,
          app_format: triageResult.corrected_app_type || session.appType || "text",
          form: null
        };
      }

      if (triageResult.status === "needs_context") {
        const question = String(triageResult.question || "").trim();
        if (question.length >= 10) {
          session.triageRounds = (session.triageRounds || 0) + 1;
          session.lastQuestion = question;
          await saveSession(session);

          // Use LLM-generated chips if available — fully dynamic requirement gathering
          const hasChips = Array.isArray(triageResult.suggested_options) && triageResult.suggested_options.length >= 2;
          return {
            reply: question,
            uiType: hasChips ? "chips" : null,
            uiData: hasChips ? { options: triageResult.suggested_options } : null,
            nextStep: 0,
            coins: null
          };
        }
      }

      // Apply type correction if triage detected a misclassification
      if (triageResult.corrected_app_type && triageResult.corrected_app_type !== session.appType) {
        console.log(`[Triage] Type correction: ${session.appType} → ${triageResult.corrected_app_type}`);
        session.appType = triageResult.corrected_app_type;
        if (session.extraction) session.extraction.appType = triageResult.corrected_app_type;
      }

      // AI satisfied or hit max rounds — save variables for Live Preview
      if (triageResult.form) {
        session.dynamicContext = triageResult.form;
      } else {
        session.dynamicContext = await generateDynamicContext({
          appType: session.appType || session.extraction?.appType || "text",
          appPurpose: session.extraction?.appPurpose || "",
          languageHint: detectLanguageMode(session)
        });
      }
      session.triageRounds = 0;
      session.formConfirmed = true;
      await saveSession(session);
    } // end else (not affirmation)
  } // end if (!session.dynamicContext)

  // 3. BUDGET QUESTION (Only asked AFTER it fully understands the app and the form has been confirmed)
  if (session.dynamicContext) {
    session.formConfirmed = true;
  }
  const isFormConfirmed = Boolean(session.formConfirmed);
  if (isFormConfirmed) {
    if (!session.extraction?.budget && !session.deepAnswers?.budgetPreference) {
      session.currentDeepField = "budgetPreference";
      session.awaitingDeepAnswer = true;
      await saveSession(session);
      return {
        reply: "Got everything I need to build your app! One last thing — **what's your budget per generation?** This helps me pick the right AI model.",
        uiType: "chips",
        uiData: { options: ["Free models only (0 coins)", "Low (< 5 coins)", "Medium (5 - 20 coins)", "Premium (> 20 coins)"] },
        nextStep: 0,
        coins: null
      };
    }
  } else {
    if (!session.dynamicContext || !Array.isArray(session.dynamicContext.variables) || !Array.isArray(session.dynamicContext.options)) {
      session.dynamicContext = buildDynamicContextFallback(
        session.appType || "text",
        session.extraction?.appPurpose || "",
        detectLanguageMode(session)
      );
    }
    // Return the multi_select_form first!
    return {
      reply: localizedText(
        session,
        "## 📋 Customize Your App Configuration\n\nI've generated a draft of key features and input fields based on our conversation.\n\nVerify or adjust the options below, then click **Confirm options**!",
        "## 📋 à¤†à¤ªà¤•à¥‡ à¤à¤ª à¤•à¤¾ à¤¸à¥‡à¤Ÿà¤…à¤ª\n\nà¤®à¥ˆà¤‚à¤¨à¥‡ à¤¹à¤®à¤¾à¤°à¥€ à¤¬à¤¾à¤¤à¤šà¥€à¤¤ à¤•à¥‡ à¤†à¤§à¤¾à¤° à¤ªà¤° à¤ªà¥à¤°à¤®à¥à¤– à¤µà¤¿à¤¶à¥‡à¤·à¤¤à¤¾à¤“à¤‚ à¤”à¤° à¤‡à¤¨à¤ªà¥à¤Ÿ à¤«à¤¼à¥€à¤²à¥à¤¡à¥à¤¸ à¤•à¤¾ à¤à¤• à¤¡à¥à¤°à¤¾à¤«à¥à¤Ÿ à¤¤à¥ˆà¤¯à¤¾à¤° à¤•à¤¿à¤¯à¤¾ à¤¹à¥ˆà¥¤\n\nà¤•à¥ƒà¤ªà¤¯à¤¾ à¤¨à¥€à¤šà¥‡ à¤¦à¤¿à¤ à¤—à¤ à¤µà¤¿à¤•à¤²à¥à¤ªà¥‹à¤‚ à¤•à¥€ à¤œà¤¾à¤‚à¤š à¤•à¤°à¥‡à¤‚, à¤«à¤¿à¤° **Confirm options** à¤ªà¤° à¤•à¥à¤²à¤¿à¤• à¤•à¤°à¥‡à¤‚!",
        "## 📋 App Configuration Customise Karein\n\nMaine humari conversation ke basis par key features aur input fields ka ek draft generate kiya hai.\n\nNeeche diye options ko check/adjust karein, fir **Confirm options** par click karein!"
      ),
      uiType: "multi_select_form",
      uiData: {
        options: session.dynamicContext.options || [],
        variables: session.dynamicContext.variables || []
      },
      nextStep: 0,
      coins: null
    };
  }

  // 4. ALL DONE -> SHOW MODELS
  return await showModels(session);
}



/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   PIPELINE EXECUTORS  (called from the dispatch switch below)
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

async function execGatherRequirements(session, text) {
  if (!session.history) session.history = [];
  const latestExtraction = await extractRequirements(text, session.history);
  session.extraction = mergeExtraction(session.extraction, latestExtraction, text);
  session.languageMode = detectLanguageMode(session);
  if (session.extraction.enterpriseSignals !== undefined) session.enterpriseSignals = session.extraction.enterpriseSignals;
  if (session.extraction.userType) session.userType = session.extraction.userType;
  if (!session.appType && session.extraction.appType) session.appType = session.extraction.appType;
  return buildStep0Response(session, text);
}

async function execRenderForm(session) {
  if (!session.dynamicContext || !Array.isArray(session.dynamicContext.variables) || !Array.isArray(session.dynamicContext.options)) {
    session.dynamicContext = await generateDynamicContext({
      appType: session.appType || "text",
      appPurpose: session.extraction?.appPurpose || "",
      languageHint: detectLanguageMode(session)
    });
  }
  if (!session.dynamicContext || !Array.isArray(session.dynamicContext.variables) || !Array.isArray(session.dynamicContext.options)) {
    session.dynamicContext = buildDynamicContextFallback(
      session.appType || "text",
      session.extraction?.appPurpose || "",
      detectLanguageMode(session)
    );
  }
  session.formConfirmed = true;
  await saveSession(session);

  const budget = session.deepAnswers?.budgetPreference || session.extraction?.budget;
  if (!budget) {
    session.currentDeepField = "budgetPreference";
    session.awaitingDeepAnswer = true;
    await saveSession(session);
    return {
      reply: "Got everything I need to build your app! One last thing — **what's your budget per generation?** This helps me pick the right AI model.",
      uiType: "chips",
      uiData: { options: ["Free models only (0 coins)", "Low (< 5 coins)", "Medium (5 - 20 coins)", "Premium (> 20 coins)"] },
      nextStep: 0,
      coins: null
    };
  }

  return showModels(session);
}

async function legacy_execRenderForm_unused(session) {
  return {
    reply: localizedText(
      session,
      "## 📋 Customize Your App Configuration\n\nReview the features and input fields below, then click **Confirm options**!",
      "## 📋 à¤†à¤ªà¤•à¥‡ à¤à¤ª à¤•à¤¾ à¤¸à¥‡à¤Ÿà¤…à¤ª\n\nà¤¨à¥€à¤šà¥‡ à¤¦à¤¿à¤ à¤—à¤ à¤µà¤¿à¤•à¤²à¥à¤ªà¥‹à¤‚ à¤•à¥€ à¤œà¤¾à¤‚à¤š à¤•à¤°à¥‡à¤‚, à¤«à¤¿à¤° **Confirm options** à¤ªà¤° à¤•à¥à¤²à¤¿à¤• à¤•à¤°à¥‡à¤‚!",
      "## 📋 App Configuration\n\nNeeche options check karo, fir **Confirm options** par click karo!"
    ),
    uiType: "multi_select_form",
    uiData: {
      appType: session.appType || "text",
      appPurpose: session.extraction?.appPurpose || "",
      options: session.dynamicContext.options || [],
      variables: session.dynamicContext.variables || []
    },
    nextStep: session.step,
    coins: null
  };
}

async function execShowModelCards(session) {
  return showModels(session);
}

async function execGeneratePreview(session, text) {
  const selectedModelId = parseSelectedModelId(text, MODELS[session.appType] || []);
  if (!selectedModelId) {
    return {
      reply: "Please **click one of the model cards** above to select the AI engine. ðŸ‘†",
      uiType: "text", uiData: null, nextStep: 1, coins: null
    };
  }
  const selectedModel = findModel(session.appType, selectedModelId);
  if (!selectedModel) {
    return {
      reply: "I couldn't match that model. Please click one of the options above.",
      uiType: "text", uiData: null, nextStep: 1, coins: null
    };
  }
  session.modelId = selectedModel.id;
  session.modelCost = selectedModel.cost;
  session.modelName = selectedModel.name;
  session.awaitingConfirmation = false;
  await saveSession(session);
  try {
    const [promptData, seoData] = await Promise.all([
      generatePromptTemplate(session),
      generateSEO(session)
    ]);
    session.promptData = promptData;
    session.seoData = seoData;
    session.step = 2;
    await saveSession(session);
    return {
      reply: `## App Preview Ready\n\nI've configured the full AI logic using **${selectedModel.name}**.\n\nTest it in the Live Preview below — click **Approve App** when ready!`,
      uiType: "app_preview",
      uiData: {
        appName: seoData.appName,
        appType: session.appType || "text",
        appDescription: seoData.appDescription,
        cost: session.modelCost,
        systemPrompt: promptData.systemPrompt,
        userPrompt: promptData.userPrompt,
        variablesUsed: promptData.variablesUsed,
        acceptImageInput: sanitizeAcceptImageInput(promptData.acceptImageInput, session.appType),
        options: ["Approve App", "Edit App"]
      },
      nextStep: 2,
      coins: session.modelCost
    };
  } catch (err) {
    console.error("[execGeneratePreview] Error:", err);
    return {
      reply: "Oops! ⚠️ I hit a snag generating the config. Please try selecting the model again.",
      uiType: "text", uiData: null, nextStep: 1, coins: null
    };
  }
}

async function execReviewSEO(session) {
  session.step = 3;
  await saveSession(session);
  const seoData = session.seoData || {};
  return {
    reply: "## 🎉 App Configured — Final Review\n\nReview your **app name, description, and tags** below.\n\nEdit any field before publishing — make it shine! ✨",
    uiType: "seo_preview",
    uiData: {
      appName: seoData.appName || "Your App",
      appDescription: seoData.appDescription || "",
      category: seoData.category || "",
      tags: Array.isArray(seoData.tags) ? seoData.tags : [],
      appType: session.appType || "text",
      modelId: session.modelId,
      costPerRun: session.modelCost
    },
    nextStep: 3,
    coins: session.modelCost
  };
}

async function execPivotApp(session, text, decision) {
  let newType = decision.app_type || parseChipAppType(text);
  if (!newType) { const m = text.toLowerCase().match(/(image|video|audio|text|vision)/i); if (m) newType = m[1].toLowerCase(); }

  // Format-only pivot — keep purpose, just swap type
  const isFormatOnly = newType && newType !== session.appType && session.extraction?.appPurpose?.trim().length > 10 && !decision.is_major_pivot;
  if (isFormatOnly) {
    session.appType = newType;
    if (session.extraction) session.extraction.appType = newType;
    session.dynamicContext = null; session.modelId = null; session.modelCost = null;
    session.step = 0; session.triageRounds = 0;
    session.awaitingPromptTweak = false; session.awaitingDeepAnswer = false; session.currentDeepField = null;
    await saveSession(session);
    const hasBudget = session.deepAnswers?.budgetPreference || session.extraction?.budget;
    if (hasBudget) return showModels(session);
    session.currentDeepField = "budgetPreference"; session.awaitingDeepAnswer = true;
    await saveSession(session);
    return {
      reply: `Got it! Switching to a **${newType}** app — your context is preserved.\n\nWhat budget per generation works for you?`,
      uiType: "chips",
      uiData: { options: ["Free models only (0 coins)", "Low (< 5 coins)", "Medium (5 - 20 coins)", "Premium (> 20 coins)"] },
      nextStep: 0, coins: null
    };
  }

  // Full domain pivot — wipe everything
  let cleanPurpose = text
    .replace(/\b(actually|instead|change it to|switch to|something different)\b/gi, "")
    .replace(/\bi want to (build|make|create) (a|an)\b/gi, "")
    .replace(/\bi want (a|an)\b/gi, "").trim();
  if (!cleanPurpose || cleanPurpose.length < 3) cleanPurpose = newType ? `${newType} generation app` : "a new idea";

  session.dynamicContext = null; session.appType = newType || null;
  session.extraction = { appPurpose: cleanPurpose, confidence: {} };
  session.deepAnswers = {}; session.history = [];
  session.step = 0; session.triageRounds = 0;
  session.awaitingPromptTweak = false; session.awaitingDeepAnswer = false;
  session.currentDeepField = null; session.isPivot = true;
  await saveSession(session);
  return buildStep0Response(session, text);
}

async function execEditApp(session, text, decision) {
  const instruction = decision.extracted_variables?.editInstruction || text;

  if (session.step === 0) {
    applyEditToSession(session, instruction);
    session.formConfirmed = false; session.dynamicContext = null;
    session.triageRounds = 0; session.awaitingDeepAnswer = false; session.currentDeepField = null;
    if (session.deepAnswers) delete session.deepAnswers.budgetPreference;
    if (session.extraction) delete session.extraction.budget;
    await saveSession(session);
    return buildStep0Response(session, text);
  }

  // Preview / SEO step edit
  session.awaitingPromptTweak = false;
  applyEditToSession(session, instruction);
  try {
    const [newPromptData, newSeoData] = await Promise.all([generatePromptTemplate(session), generateSEO(session)]);
    session.promptData = newPromptData; session.seoData = newSeoData;
  } catch (e) {
    console.warn("[execEditApp] Regen failed:", e.message);
    session.promptData = applyPromptInstruction(session.promptData || {}, instruction);
  }
  session.step = 2;
  await saveSession(session);
  return {
    reply: `## ✅ App Updated\n\nApplied: **"${instruction}"**\n\nHere's the refreshed preview — approve when ready!`,
    uiType: "app_preview",
    uiData: {
      appName: session.seoData?.appName || "Your App",
      appType: session.appType || "text",
      appDescription: session.seoData?.appDescription || "",
      cost: session.modelCost,
      systemPrompt: session.promptData.systemPrompt,
      userPrompt: session.promptData.userPrompt,
      variablesUsed: session.promptData.variablesUsed,
      acceptImageInput: sanitizeAcceptImageInput(session.promptData.acceptImageInput, session.appType),
      options: ["Approve App", "Edit App"]
    },
    nextStep: 2,
    coins: session.modelCost
  };
}

async function execHandleBudget(session, text, decision) {
  const tier = decision.budget_tier || decision.extracted_variables?.budget ||
    text.toLowerCase().match(/\b(free|low|medium|premium)\b/i)?.[1]?.toLowerCase();
  if (tier) {
    if (!session.extraction) session.extraction = {};
    session.extraction.budget = tier;
    if (!session.deepAnswers) session.deepAnswers = {};
    session.deepAnswers.budgetPreference = tier;
    session.awaitingDeepAnswer = false; session.currentDeepField = null;
    await saveSession(session);
    return showModels(session);
  }
  return {
    reply: "What budget per generation would you like?",
    uiType: "chips",
    uiData: { options: ["Free models only (0 coins)", "Low (< 5 coins)", "Medium (5 - 20 coins)", "Premium (> 20 coins)"] },
    nextStep: session.step, coins: null
  };
}

async function execChangeModel(session, text, decision) {
  if (!session.dynamicContext) {
    return {
      reply: `We'll pick the perfect model in a moment! First, let's finish scoping the app.\n\n${session.lastQuestion || "What specific details should the app handle?"}`,
      uiType: null, nextStep: 0, coins: null
    };
  }
  const tier = decision.budget_tier || text.toLowerCase().match(/\b(free|low|medium|premium)\b/i)?.[1]?.toLowerCase();
  if (tier) {
    if (!session.extraction) session.extraction = {};
    session.extraction.budget = tier;
    if (!session.deepAnswers) session.deepAnswers = {};
    session.deepAnswers.budgetPreference = tier;
    await saveSession(session);
    return showModels(session);
  }
  if (!session.extraction) session.extraction = {};
  session.extraction.budget = null;
  if (session.deepAnswers) session.deepAnswers.budgetPreference = null;
  session.currentDeepField = "budgetPreference"; session.awaitingDeepAnswer = true; session.step = 0;
  await saveSession(session);
  return {
    reply: "What budget per generation would you like to switch to?",
    uiType: "chips",
    uiData: { options: ["Free models only (0 coins)", "Low (< 5 coins)", "Medium (5 - 20 coins)", "Premium (> 20 coins)"] },
    nextStep: 0, coins: null
  };
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   MAIN ROUTER — Declarative Action Dispatch
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
export async function route(session, message) {
  // â”€â”€ WALL OF TEXT GUARD â”€â”€
  const rawText = String(message || "").substring(0, 1000);
  const text = normalize(rawText);
  const msg = lower(text);

  // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
  // FAST-PATH 1: multi_select_form submission
  // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
  if (text.toLowerCase().startsWith("multi_select_form::")) {
    const payload = parseMultiSelectPayload(message);
    if (payload) {
      if (!session.dynamicContext) session.dynamicContext = {};
      session.dynamicContext.options = payload.selectedOptions || [];
      session.dynamicContext.variables = (payload.variables || []).map(v => ({
        name: v.name, placeholder: v.placeholder || "Enter details...", test_value: v.value || ""
      }));
      session.formConfirmed = true;
      if (!session.extraction) session.extraction = {};
      session.extraction.keyFeatures = payload.selectedOptions || [];
      const budget = session.deepAnswers?.budgetPreference || session.extraction?.budget;
      if (budget) return showModels(session);
      session.currentDeepField = "budgetPreference"; session.awaitingDeepAnswer = true;
      await saveSession(session);
      return {
        reply: "One last thing — **what's your budget per generation?**",
        uiType: "chips",
        uiData: { options: ["Free models only (0 coins)", "Low (< 5 coins)", "Medium (5 - 20 coins)", "Premium (> 20 coins)"] },
        nextStep: 0, coins: null
      };
    }
  }

  // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
  // FAST-PATH 2: SEO_PUBLISH / SEO_DRAFT
  // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
  const isSeoPublish = text.startsWith("SEO_PUBLISH::");
  const isSeoSaveDraft = text.startsWith("SEO_DRAFT::");
  if (isSeoPublish || isSeoSaveDraft) {
    try {
      const jsonStr = text.slice(text.indexOf("::") + 2);
      const cardData = JSON.parse(jsonStr);
      session.seoData = { ...session.seoData, ...cardData };
      await saveSession(session);
    } catch (e) { console.warn("[route] Failed to parse SEO payload:", e.message); }
    if (isSeoPublish) {
      const payload = {
        appType: session.appType, modelId: session.modelId, costPerRun: session.modelCost,
        systemPrompt: session.promptData?.systemPrompt, userPrompt: session.promptData?.userPrompt,
        negativePrompt: session.promptData?.negativePrompt,
        acceptImageInput: sanitizeAcceptImageInput(session.promptData?.acceptImageInput, session.appType),
        appName: session.seoData?.appName, appDescription: session.seoData?.appDescription,
        tags: session.seoData?.tags, publishedAt: new Date().toISOString()
      };
      console.log("\nâ•â• MOCK PUBLISH â•â•\n", JSON.stringify(payload, null, 2), "\nâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•\n");
      await deleteSession(session.sessionId);
      return {
        reply: `## 🎉 Published Successfully!\n\nYour app **"${session.seoData?.appName}"** is now live!\n\n- **Cost per run:** ${payload.costPerRun} coins\n- **Status:** ✅ Live ðŸš€`,
        uiType: "success",
        uiData: {
          appName: session.seoData?.appName, modelId: session.modelId,
          modelName: session.modelName || session.modelId, costPerRun: payload.costPerRun,
          tags: session.seoData?.tags, mockUrl: `https://rentprompts.ai/app/demo-${Date.now()}`
        },
        nextStep: 0, coins: session.modelCost, clearSession: true
      };
    }
    session.status = "draft"; await saveSession(session);
    return {
      reply: `## 📋 Draft Saved\n\n**"${session.seoData?.appName}"** saved. Resume anytime from your dashboard.`,
      uiType: "success",
      uiData: { appName: session.seoData?.appName, status: "Draft" },
      nextStep: 0, coins: null
    };
  }

  // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
  // FAST-PATH 3: Edit App button at preview/SEO step
  // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
  if ((session.step === 2 || session.step === 3) && msg === "edit app") {
    session.awaitingPromptTweak = true; await saveSession(session);
    return {
      reply: "I'm listening! 📝ï¸\n\n- **Tweak the prompt** — tell me what to change\n- **Switch the AI model** — pick a different engine\n- **Start fresh** — describe a completely new app idea\n\nWhat would you like to adjust?",
      uiType: "text", uiData: null, nextStep: session.step, coins: session.modelCost
    };
  }

  // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
  // ORCHESTRATOR BRAIN — get recommended_action
  // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
  const decision = await getAgenticDecision(text, session);
  console.log(`[Router] Action: ${decision.recommended_action} (${decision._source}) | confidence: ${decision.confidence}`);

  // Apply any LLM-corrected app type immediately
  if (decision.app_type && decision.app_type !== session.appType) {
    session.appType = decision.app_type;
    if (!session.extraction) session.extraction = {};
    session.extraction.appType = decision.app_type;
  }

  // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
  // ACTION DISPATCH MATRIX
  // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
  switch (decision.recommended_action) {

    case "HANDLE_GREETING":
      if ((session.history || []).length < 3) {
        return {
          reply: "Hey there! 👋 I'm your **RentPrompts App Architect** — I help you design, configure, and publish AI-powered apps in minutes.\n\n**Just describe your app idea** and I'll handle the rest!\n\nWhat would you like to build today?",
          uiType: null, uiData: null, nextStep: session.step, coins: null
        };
      }
      // Returning user mid-session — fall through to requirements
      return execGatherRequirements(session, text);

    case "HANDLE_OFF_TOPIC":
      if (decision.confidence !== "low") return OFF_TOPIC_RESPONSE;
      return execGatherRequirements(session, text);

    case "HANDLE_VIOLATION":
      return {
        reply: "I can only help build apps that comply with RentPrompts' safety and content guidelines. Please suggest a different idea.",
        uiType: "text", uiData: null, nextStep: session.step, coins: null
      };

    case "HANDLE_GIBBERISH":
      return {
        reply: "Hmm, I didn't quite catch that! 🤔 What type of output should your AI app generate?",
        uiType: "chips",
        uiData: { options: ["Text", "Image", "Audio", "Video", "Vision"] },
        nextStep: session.step, coins: null
      };

    case "HANDLE_BUDGET":
      return execHandleBudget(session, text, decision);

    case "CHANGE_MODEL":
      return execChangeModel(session, text, decision);

    case "PIVOT_APP":
      return execPivotApp(session, text, decision);

    case "EDIT_APP": {
      // Change-prefix fast path: "Change: ..."
      if (/^change\s*:/i.test(msg)) {
        const correction = text.replace(/^change\s*:/i, "").trim();
        if (correction.length > 1) {
          decision.extracted_variables = decision.extracted_variables || {};
          decision.extracted_variables.editInstruction = correction;
        }
      }
      return execEditApp(session, text, decision);
    }

    case "RENDER_FORM":
      if (!session.history) session.history = [];
      if (!session.extraction?.appPurpose) {
        // Need to extract first
        const ext = await extractRequirements(text, session.history);
        session.extraction = mergeExtraction(session.extraction, ext, text);
      }
      await saveSession(session);
      return execRenderForm(session);

    case "SHOW_MODEL_CARDS":
      return execShowModelCards(session);

    case "GENERATE_PREVIEW":
      return execGeneratePreview(session, text);

    case "REVIEW_SEO":
      return execReviewSEO(session);

    case "PUBLISH_APP":
      // Catch-all publish (non-SEO_PUBLISH:: payloads that still mean publish)
      if (msg.includes("save draft") || msg.includes("save to draft")) {
        return {
          reply: "Done! Your progress has been saved as a draft. Access it anytime from your RentPrompts dashboard.",
          uiType: "success",
          uiData: { appName: session.seoData?.appName || session.extraction?.appPurpose || "Untitled Draft", status: "Draft" },
          nextStep: 0, coins: null, clearSession: true
        };
      }
      if (msg.includes("start over") || msg.includes("restart") || msg.includes("reset")) {
        return {
          reply: "No problem! 🔄 Let's start fresh.\n\n**What kind of AI app would you like to build?**",
          uiType: "chips",
          uiData: { options: ["Image app", "Video app", "Text app", "Audio app", "Vision app"] },
          nextStep: 0, coins: null, clearSession: true
        };
      }
      // Generic publish catch-all
      return {
        reply: "Ready to publish? Review the SEO card and hit **Publish to Marketplace**!",
        uiType: "chips",
        uiData: { options: ["Publish to Marketplace", "Save Draft", "Edit App"] },
        nextStep: session.step || 3, coins: session.modelCost
      };

    case "GATHER_REQUIREMENTS":
    default: {
      // Handle chip-based app type selection at the top of triage
      const chipType = parseChipAppType(text);
      if (chipType) {
        if (!session.appType || (session.formatAskedByTriage && !session.formatConfirmedByUser)) {
          session.appType = chipType;
          if (!session.extraction) session.extraction = {};
          session.extraction.appType = chipType;
          if (session.formatAskedByTriage) {
            session.formatConfirmedByUser = true;
            console.log(`[Format Override] Chip confirmed: ${chipType}`);
          }
        }
      }

      // Handle awaitingDeepAnswer (budget or domain field)
      if (session.awaitingDeepAnswer && session.currentDeepField) {
        if (!session.deepAnswers) session.deepAnswers = {};
        session.deepAnswers[session.currentDeepField] = text;
        if (!session.extraction) session.extraction = {};
        if (session.currentDeepField === "budgetPreference") session.extraction.budget = text;
        session.awaitingDeepAnswer = false; session.currentDeepField = null;
        const nextQ = getNextDeepQuestion(session);
        if (nextQ) {
          session.currentDeepField = nextQ.field; session.awaitingDeepAnswer = true;
          await saveSession(session);
          return { reply: nextQ.question, uiType: "chips", uiData: { options: nextQ.options || [] }, nextStep: 0, coins: null };
        }
        return showModels(session);
      }

      return execGatherRequirements(session, text);
    }
  }
}

