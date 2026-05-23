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

        // Format check gets hardcoded format chips; context questions get LLM-generated chips
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
    }

    // Same triage outcome but round cap exceeded — still never skip straight to an unconfirmed form
    if (
      (triageResult.status === "needs_context" || triageResult.status === "needs_format") &&
      triageResult.question &&
      session.triageRounds > 3
    ) {
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
      `## ✅ App Architecture Ready\n\nI've analyzed your requirements and scoped out a **${displayFormat}** app.\n\nHere's what I've configured for you — review the features and input fields below, then hit confirm to proceed.\n\n**💡 Tip:** If you actually wanted this to generate a different output type (Images, Audio, Video), just let me know and I'll adjust instantly.`,
      `## ✅ ऐप आर्किटेक्चर तैयार\n\nमैंने आपकी ज़रूरतों का विश्लेषण किया और **${displayFormat}** ऐप स्कोप किया है।\n\nनीचे फीचर्स और इनपुट फील्ड्स देखें, फिर कन्फर्म करें।\n\n**💡 टिप:** अगर आप Image, Audio या Video आउटपुट चाहते हैं तो बता दें।`,
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

/* ────────────────────────────────────────────
   APPLY EDIT TO SESSION — rewrites session state so generatePromptTemplate
   gets clean, up-to-date context instead of old ghost memory.
   Call this BEFORE generatePromptTemplate on every edit path.
   ──────────────────────────────────────────── */
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
    // Detect "transparent" or "no background" edits → kill scene/background answers
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

  // ── AMBIGUOUS DOMAIN DETECTION ──────────────────────────────────────────
  // These keywords are inherently ambiguous — they could be text OR image.
  // e.g., "birthday app" → text (written wishes) or image (birthday card).
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

    // If ambiguous domain AND user hasn't explicitly confirmed format → skip local inference
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
      const options = detectLanguageMode(session) === "Hindi" ? ["टेक्स्ट", "इमेज", "ऑडियो", "वीडियो", "विज़न"] : ["Text", "Image", "Audio", "Video", "Vision"];
      return { reply: "What kind of output should your app produce?", uiType: "chips", uiData: { options }, nextStep: 0, coins: null };
    }
  }

  session.step = 0;
  session.awaitingConfirmation = false;

  // 3. AGENTIC TRIAGE (Deep Context Questions)
  if (!session.dynamicContext) {

    // ── AMBIGUOUS DOMAIN FORCE-ASK: If no appType yet and domain is ambiguous, inject format question ──
    if (!session.appType && isAmbiguousDomain && !session.formatAskedByTriage) {
      session.formatAskedByTriage = true;
      session.triageRounds = (session.triageRounds || 0) + 1;
      await saveSession(session);
      console.log(`[Ambiguous Domain] Forcing output format question for: "${purposeLowerForAmbiguity.substring(0, 60)}"`);
      return {
        reply: `I noticed your app idea could work as either written text or a visual image. What kind of output should this app generate?`,
        uiType: "chips",
        uiData: { options: ["Written text (wishes, poems, messages)", "Visual image (card, poster, design)"] },
        nextStep: 0,
        coins: null
      };
    }

    // ── Handle the format answer from the ambiguous-domain forced question ──
    if (session.formatAskedByTriage && !session.appType) {
      const answerLower = lower(text);
      if (answerLower.includes('written') || answerLower.includes('text') || answerLower.includes('wishes') ||
          answerLower.includes('poem') || answerLower.includes('message')) {
        session.appType = 'text';
        if (!session.extraction) session.extraction = {};
        session.extraction.appType = 'text';
        session.formatConfirmedByUser = true;
        console.log(`[Ambiguous Domain] User confirmed: TEXT output`);
      } else if (answerLower.includes('visual') || answerLower.includes('image') || answerLower.includes('card') ||
                 answerLower.includes('poster') || answerLower.includes('design') || answerLower.includes('picture') ||
                 answerLower.includes('photo')) {
        session.appType = 'image';
        if (!session.extraction) session.extraction = {};
        session.extraction.appType = 'image';
        session.formatConfirmedByUser = true;
        console.log(`[Ambiguous Domain] User confirmed: IMAGE output`);
      }
      await saveSession(session);
    }

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
      const triageResult = await triageDynamicContext({
        appType: session.appType,
        appPurpose: session.extraction?.appPurpose || "",
        languageHint: detectLanguageMode(session),
        conversationHistory: session.history || [],
        deepAnswers: session.deepAnswers || {}
      });

      if (triageResult.status === "needs_context" && (session.triageRounds || 0) < 6) {
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
      session.dynamicContext = triageResult.form;
      session.triageRounds = 0;
      await saveSession(session);
    } // end else (not affirmation)
  } // end if (!session.dynamicContext)

  // 3. BUDGET QUESTION (Only asked AFTER it fully understands the app)
  if (!session.extraction?.budget && !session.deepAnswers?.budgetPreference) {
    session.currentDeepField = "budgetPreference";
    session.awaitingDeepAnswer = true;
    await saveSession(session);
    return {
      reply: "## 💰 Almost There — Budget Selection\n\nI've gathered all the details I need to architect your app! Just one last thing — **what's your target budget per generation?**\n\nThis helps me recommend the perfect AI model for your use case.",
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
  if (
    text.startsWith("multi_select_form::") ||
    text.startsWith("edit prompt::") ||
    text.startsWith("confirm seo::") ||
    text.startsWith("SEO_PUBLISH::") ||
    text.startsWith("SEO_DRAFT::") ||
    text.startsWith("SEO_EDIT::")
  ) {
    return null;
  }

  if (isOffTopic(text, session)) return OFF_TOPIC_RESPONSE;

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
  // Only fires at step 0 with no history — prevents "hey" inside a real word/sentence from triggering
  const isGreeting = /^(hi|hello|hey|hy|hola|greetings)[\s!\.]*$/i.test(trimmedText);

  if (isGreeting && session.step === 0 && (session.history || []).length < 3) {
    return {
      reply:
        "Hey there! 👋 I'm your **RentPrompts App Architect** — I help you design, configure, and publish AI-powered apps in minutes.\n\n**Just describe your app idea** and I'll handle the rest — from picking the right AI model to crafting the perfect prompt.\n\nWhat would you like to build today?",
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

  // Only run gibberish detection at step 0 (no established session)
  // Mid-conversation, unusual short inputs like "ipc" or domain jargon should never be rejected
  if (isGibberish && session.step === 0) {
    return {
      // Changed the reply to be exactly what you want when nonsense is typed
      reply: `Hmm, I didn't quite catch that! 🤔 Let me help — **what type of output** should your AI app generate?`,
      uiType: 'chips',
      uiData: { options: ['Text', 'Image', 'Audio', 'Video', 'Vision'] },
      nextStep: session.step,
      coins: null
    };
  }

  if (!text || text === '') {
    return {
      reply: `I'm all ears! 🎧 Go ahead — **describe what you'd like to build** and I'll start architecting it for you.`,
      uiType: 'text',
      uiData: null,
      nextStep: session.step,
      coins: null
    };
  }

  if (msg.includes('help') && text.trim().split(' ').length <= 3) {
    return {
      reply: `## 🚀 What Can I Build For You?\n\nHere's every type of AI app I can help you create:\n\n- 🖼️ **Image apps** — logos, posters, photo editing, avatars\n- 🎥 **Video apps** — reels, animations, talking avatars\n- 📝 **Text apps** — blogs, legal docs, planners, scripts\n- 🔊 **Audio apps** — voiceovers, podcasts, text-to-speech\n- 👁️ **Vision apps** — image analysis, OCR, object detection\n\nWhich type interests you?`,
      uiType: 'chips',
      uiData: { options: ['Image app', 'Video app', 'Text app', 'Audio app', 'Vision app', 'Help me choose'] },
      nextStep: 0,
      coins: null
    };
  }

  if (msg.includes('start over') || msg.includes('restart') || msg.includes('reset') || msg.includes('new app') || msg.includes('different app')) {
    return {
      reply: `No problem! 🔄 Let's start fresh.\n\n**What kind of AI app would you like to build?** Just describe your idea and I'll take it from there.`,
      uiType: 'chips',
      uiData: { options: ['Image app', 'Video app', 'Text app', 'Audio app', 'Vision app'] },
      nextStep: 0,
      coins: null,
      clearSession: true
    };
  }

  if ((msg.includes('how much') || msg.includes('price') || msg.includes('cost') || msg.includes('joules')) && session.step < 4) {
    return {
      reply: `Great question! 💡 The cost depends on which AI model we select.\n\n**Price range:** FREE → 318 coins per run\n\nBut first, let me understand your app idea so I can recommend the best value model. **What kind of app are you building?**`,
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

  // ─── 0. SEO PUBLISH / DRAFT — MUST be first, before ANY extraction or keyword matching ───
  // These payloads come from the SEOPreviewCard UI and contain JSON. They MUST NOT be
  // processed as regular messages (keyword matching sees "cost" → fires "Great question!").
  const isSeoPublish = text.startsWith('SEO_PUBLISH::');
  const isSeoSaveDraft = text.startsWith('SEO_DRAFT::');
  if (isSeoPublish || isSeoSaveDraft) {
    // Ensure we're at step 3 — if session was cleared, still handle gracefully
    try {
      const jsonStr  = text.slice(text.indexOf('::') + 2);
      const cardData = JSON.parse(jsonStr);
      session.seoData = { ...session.seoData, ...cardData };
      await saveSession(session);
      console.log('[stepRouter] SEO card merged:', session.seoData?.appName);
    } catch (e) {
      console.warn('[stepRouter] Failed to parse SEO card payload:', e.message);
    }
    if (isSeoPublish) {
      const payload = {
        appType:          session.appType,
        modelId:          session.modelId,
        costPerRun:       session.modelCost,
        systemPrompt:     session.promptData?.systemPrompt,
        userPrompt:       session.promptData?.userPrompt,
        negativePrompt:   session.promptData?.negativePrompt,
        acceptImageInput: sanitizeAcceptImageInput(session.promptData?.acceptImageInput, session.appType),
        appName:          session.seoData?.appName,
        appDescription:   session.seoData?.appDescription,
        tags:             session.seoData?.tags,
        publishedAt:      new Date().toISOString()
      };
      console.log('\n══ MOCK PUBLISH ══');
      console.log(JSON.stringify(payload, null, 2));
      console.log('══════════════════\n');
      await deleteSession(session.sessionId);
      return {
        reply: `## 🎉 Published Successfully!\n\nYour app **"${session.seoData?.appName}"** is now live on the marketplace!\n\n- **Cost per run:** ${payload.costPerRun} coins\n- **Status:** ✅ Live\n\nUsers can now discover and use your app. Great work! 🚀`,
        uiType: "success",
        uiData: {
          appName:    session.seoData?.appName,
          modelId:    session.modelId,
          modelName:  session.modelName || session.modelId,
          costPerRun: payload.costPerRun,
          tags:       session.seoData?.tags,
          mockUrl:    `https://rentprompts.ai/app/demo-${Date.now()}`
        },
        nextStep: 0,
        coins: session.modelCost,
        clearSession: true
      };
    } else {
      // SEO_DRAFT
      session.status = 'draft';
      await saveSession(session);
      return {
        reply: `## 📋 Draft Saved\n\n**"${session.seoData?.appName}"** has been securely saved as a draft.\n\nYou can resume editing or publish it anytime from your **RentPrompts dashboard**.`,
        uiType: 'success',
        uiData: { appName: session.seoData?.appName, status: 'Draft' },
        nextStep: 0,
        coins: null
      };
    }
  }

  // ─── 1. EXPLICIT UI CHIP HANDLER ───
  if ((session.step === 2 || session.step === 3) && msg === 'edit app') {
    session.awaitingPromptTweak = true;
    await saveSession(session);
    return {
      reply: "I'm listening! ✏️ Here's what you can do:\n\n- **Tweak the prompt** — tell me what to change\n- **Switch the AI model** — pick a different engine\n- **Start fresh** — describe a completely new app idea\n\nWhat would you like to adjust?",
      uiType: "text",
      uiData: null,
      nextStep: session.step,
      coins: session.modelCost
    };
  }

  // ─── 2a. CHANGE-PREFIX FAST PATH (step 2 or 3) ───────────────────────────
  // If the user explicitly prefixes with "Change:" while at the preview or publish step,
  // treat it as a direct app edit — UNLESS it's a completely different app idea (major pivot).
  if (
    (session.step === 2 || session.step === 3) &&
    /^change\s*:/i.test(msg)
  ) {
    const correction = text.replace(/^change\s*:/i, '').trim();
    if (correction.length > 1) {
      const corrLower = correction.toLowerCase();

      // ── PIVOT GUARD: Detect if this "change" is actually a COMPLETELY different app ──
      // If the correction mentions a different app type, or describes an entirely new app idea,
      // DON'T treat it as a minor edit — fall through to the Universal Pivot Interceptor.
      const mentionsDifferentType = /(image|video|audio|text|vision)\s+(app|generator|tool|creator)\b/i.test(corrLower) &&
        !corrLower.includes(session.appType);
      const isNewAppIdea = /\b(i want|build|create|make)\b.*\b(app|tool|generator)\b/i.test(corrLower) ||
        /\b(room designer|logo maker|background remov|recipe|resume|birthday|meme|podcast)\b/i.test(corrLower);
      const mentionsNewDomain = isNewAppIdea && !corrLower.includes(session.extraction?.appPurpose?.toLowerCase()?.slice(0, 15) || '____');

      if (mentionsDifferentType || mentionsNewDomain) {
        // This is a major pivot disguised as a "Change:" — let it fall through
        // to the Universal Pivot Interceptor below by NOT returning here.
        console.log(`[Change-prefix] Detected major pivot in "Change:" message — routing to pivot handler. correction="${correction.slice(0, 60)}"`);
      } else {
        // Genuine minor edit — apply it directly
        applyEditToSession(session, correction);
        let promptData = session.promptData;
        let seoData = session.seoData;
        try {
          [promptData, seoData] = await Promise.all([
            generatePromptTemplate(session),
            generateSEO(session)
          ]);
          session.promptData = promptData;
          session.seoData = seoData;
        } catch (e) {
          console.warn('[Change-prefix fast path] Regen failed:', e.message);
          session.promptData = applyPromptInstruction(session.promptData || {}, correction);
        }
        session.step = 2;
        session.awaitingPromptTweak = false;
        await saveSession(session);
        return {
          reply: `## ✅ App Updated\n\nI've applied your change: **"${correction}"**\n\nHere's the refreshed preview below — give it a test run and hit **Approve** when you're happy with the results!`,
          uiType: 'app_preview',
          uiData: {
            appName: session.seoData?.appName || 'Your App',
            appType: session.appType || session.extraction?.appType || 'text',
            appDescription: session.seoData?.appDescription || '',
            cost: session.modelCost,
            systemPrompt: session.promptData?.systemPrompt || '',
            userPrompt: session.promptData?.userPrompt || '',
            variablesUsed: session.promptData?.variablesUsed || [],
            acceptImageInput: sanitizeAcceptImageInput(session.promptData?.acceptImageInput, session.appType),
            options: ['Approve App', 'Edit App']
          },
          nextStep: 2,
          coins: session.modelCost
        };
      }
    }
  }

  // ─── 2. UNIVERSAL PIVOT & EDIT INTERCEPTOR ───
  const editTriggers = ['change ', 'update ', 'switch ', 'actually', 'instead', 'tweak', 'edit '];
  const isEditTrigger = editTriggers.some((t) => msg.includes(t)) || session.awaitingPromptTweak;

  const formatRegex = /(image|video|audio|text|vision)\s+(app|generator|tool|creator)\b/i;
  const isNewFormatMentioned = formatRegex.test(msg);

  // TRIAGE GUARD: If the user is answering a triage/clarification question at step 0,
  // do NOT treat their reply as a major pivot — they're giving us the info we asked for.
  const isAnsweringTriage = session.step === 0 &&
    (session.awaitingTriageAnswer === true || (session.triageRounds || 0) > 0) &&
    !isNewFormatMentioned; // Only allow format correction (e.g. "actually image") to still pivot

  const isMajorPivot =
    !isAnsweringTriage && (
      isNewFormatMentioned ||
      msg.includes('app type') ||
      msg.includes('domain') ||
      msg.includes('instead') ||
      msg.includes('new app') ||
      msg.includes('different app') ||
      /^(i want (to )?(build|make|create)|let'?s (build|make|create)|build an?|make an?|create an?)/i.test(msg) ||
      // Catch "i want [noun] app" — e.g. "i want room designer app", "i want recipe app"
      /\bi want\b.{2,40}\bapp\b/i.test(msg) ||
      (session.awaitingPromptTweak && msg.includes('app'))
    );

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

      // SMART BUDGET DETECTION
      const budgetMatch = msg.match(/(free|low|medium|premium)/i);
      
      if (budgetMatch) {
          if (!session.extraction) session.extraction = {};
          session.extraction.budget = budgetMatch[1].toLowerCase();
          if (session.deepAnswers) session.deepAnswers.budgetPreference = budgetMatch[1].toLowerCase();
          await saveSession(session);
          return await showModels(session);
      } else {
          // They just said "change the model". Wipe the old budget and ask again!
          if (!session.extraction) session.extraction = {};
          session.extraction.budget = null; 
          if (session.deepAnswers) session.deepAnswers.budgetPreference = null;
          session.currentDeepField = "budgetPreference";
          session.awaitingDeepAnswer = true;
          session.step = 0; 
          await saveSession(session);
          
          return {
            reply: "No problem! What target budget per generation would you like to switch to?",
            uiType: "chips",
            uiData: { options: ["Free models only (0 coins)", "Low (< 5 coins)", "Medium (5 - 20 coins)", "Premium (> 20 coins)"] },
            nextStep: 0,
            coins: null
          };
      }
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

      // ── FORMAT-ONLY PIVOT: User is changing the OUTPUT TYPE of the SAME app ──
      // e.g., "change this to an audio app" / "make it generate audio instead"
      // Keep ALL context (purpose, triage answers, model), just swap appType + regenerate.
      // Do NOT wipe history — the user described their app already, no need to re-ask.
      const isFormatOnlyPivot =
        newType &&                                      // a valid new format was found
        newType !== session.appType &&                  // it's actually different
        session.extraction?.appPurpose &&               // we already know the app purpose
        session.extraction.appPurpose.trim().length > 10 && // purpose is meaningful
        !msg.includes('new app') &&
        !msg.includes('different app') &&
        !/^(i want (to )?(build|make|create)|let'?s (build|make|create)|build an?|make an?|create an?)/i.test(msg);

      if (isFormatOnlyPivot) {
        // Just swap the app type and regenerate — no triage restart!
        session.appType = newType;
        if (session.extraction) session.extraction.appType = newType;
        // Reset dynamic context so the preview variables rebuild for the new type
        session.dynamicContext = null;
        session.modelId = null;
        session.modelCost = null;
        session.step = 0;
        session.triageRounds = 0;
        session.awaitingPromptTweak = false;
        session.awaitingDeepAnswer = false;
        session.currentDeepField = null;
        // Keep: session.history, session.extraction.appPurpose, session.deepAnswers
        await saveSession(session);

        // Skip straight to model selection if we already have budget
        const hasBudget = session.deepAnswers?.budgetPreference || session.extraction?.budget;
        if (hasBudget) {
          return await showModels(session);
        }
        // Otherwise just ask for budget (skip triage — purpose is already known)
        session.currentDeepField = 'budgetPreference';
        session.awaitingDeepAnswer = true;
        await saveSession(session);
        return {
          reply: `Got it! Switching to an **${newType}** app — your incident briefing context is preserved.\n\nWhat budget per generation works for you?`,
          uiType: 'chips',
          uiData: { options: ['Free models only (0 coins)', 'Low (< 5 coins)', 'Medium (5 - 20 coins)', 'Premium (> 20 coins)'] },
          nextStep: 0,
          coins: null
        };
      }

      // ── FULL PIVOT: Completely new app idea ──
      let cleanPurpose = text
        .replace(/\b(actually|instead|change it to|switch to|something different)\b/gi, "")
        .replace(/\bi want to (build|make|create) (a|an)\b/gi, "")
        .replace(/\bi want (a|an)\b/gi, "")
        .replace(/refer to the (image|picture|screenshot)/gi, "")
        .trim();

      if (!cleanPurpose || cleanPurpose.length < 3) {
        cleanPurpose = newType ? `${newType} generation app` : "a new idea";
      }

      // Wipe everything — completely new app idea
      session.dynamicContext = null;
      session.appType = newType || null;
      session.extraction = { appPurpose: cleanPurpose, confidence: {} };
      session.deepAnswers = {};
      session.history = []; // Kill ghost memory for full pivots only
      session.step = 0;
      session.triageRounds = 0;
      session.awaitingPromptTweak = false;
      session.awaitingDeepAnswer = false;
      session.currentDeepField = null;
      session.isPivot = true;
      await saveSession(session);

      return await buildStep0Response(session, text);
    }

    // C. Minor Tweak at the Preview Step
    if (session.step === 2) {
      session.awaitingPromptTweak = false;
      // Apply the edit to session context FIRST (kills ghost memory), then regenerate
      applyEditToSession(session, message);
      try {
        const [newPromptData, newSeoData] = await Promise.all([
          generatePromptTemplate(session),
          generateSEO(session)
        ]);
        session.promptData = newPromptData;
        session.seoData = newSeoData;
      } catch (e) {
        // Fallback: just patch the existing prompt text
        console.warn('[Edit] Regen failed, keeping patched promptData:', e.message);
        session.promptData = applyPromptInstruction(session.promptData, message);
      }
      await saveSession(session);
      return {
        reply: `Got it! I've updated the app based on your request: "${text}".\n\nCheck the refreshed preview below, test it out, then approve when ready.`,
        uiType: 'app_preview',
        uiData: {
          appName: session.seoData?.appName || 'Your App',
          appType: session.appType || session.extraction?.appType || 'text',
          appDescription: session.seoData?.appDescription || '',
          cost: session.modelCost,
          systemPrompt: session.promptData.systemPrompt,
          userPrompt: session.promptData.userPrompt,
          variablesUsed: session.promptData.variablesUsed,
          acceptImageInput: sanitizeAcceptImageInput(session.promptData.acceptImageInput, session.appType),
          options: ['Approve App', 'Edit App']
        },
        nextStep: 2,
        coins: session.modelCost
      };
    }

    // D. General mid-flight tweak acknowledgement
    if (session.step > 0) {
      return {
        reply: `Noted! I've updated your app's requirements to include: "${text}". Let's continue.`,
        uiType: null,
        nextStep: session.step,
        coins: null
      };
    }
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

  // 1. SMART INFERENCE: Auto-assign appType — TRUST the LLM.
  // The extraction LLM is smart enough to know "resume generator" = text.
  // Never reject its judgment — if it returned an appType, use it.
  if (!session.appType && session.extraction.appType) {
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
    // When we explicitly asked the user about output format, their chip answer
    // OVERRIDES any LLM-inferred type (even if session.appType was already set).
    if (chipType && session.formatAskedByTriage && !session.formatConfirmedByUser) {
      session.appType = chipType;
      session.extraction = session.extraction || {};
      session.extraction.appType = chipType;
      session.formatConfirmedByUser = true;
      console.log(`[Format Override] User responded to format question with chip: ${chipType}`);
    }

    return await buildStep0Response(session, text);
  }

  // ─── STEP 1: Model selection → generate config ───────────────────────
  if (session.step === 1) {
    // ── BUDGET RE-ROLL DETECTOR ──────────────────────────────────────────
    // Catches: "I want medium models", "show me low models", "switch to premium", etc.
    // Fires when a budget tier keyword appears in ANY model-change intent message.
    const budgetTierMatch = msg.match(/\b(free|low|medium|premium)\b/i);
    const hasModelIntent = /\b(model|models|different|change|switch|show|want|use|prefer|budget)\b/i.test(msg);

    if (budgetTierMatch && hasModelIntent && !text.toLowerCase().startsWith("select")) {
      const newBudget = budgetTierMatch[1].toLowerCase();
      if (!session.extraction) session.extraction = {};
      session.extraction.budget = newBudget;
      if (!session.deepAnswers) session.deepAnswers = {};
      session.deepAnswers.budgetPreference = newBudget;
      await saveSession(session);
      return await showModels(session);
    }
    // ────────────────────────────────────────────────────────────────────

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
        // Run both LLM calls in parallel to halve latency (~15s vs ~30s)
        const [promptData, seoData] = await Promise.all([
          generatePromptTemplate(session),
          generateSEO(session)
        ]);
        session.promptData = promptData;
        session.seoData = seoData;

        session.step = 2;
        await saveSession(session);

        const summaryText = `## 🎯 App Preview Ready\n\nI've configured the full AI logic for your app using **${selectedModel.name}**.\n\nCheck the **Live Preview** card below — give it a test run with sample inputs. If it works perfectly, click **Approve App** to finalize!`;

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
            acceptImageInput: sanitizeAcceptImageInput(promptData.acceptImageInput, session.appType),
            options: ['Approve App', 'Edit App']
          },
          nextStep: 2,
          coins: session.modelCost
        };
      } catch (error) {
        console.error("Error generating app config:", error);
        return {
          reply: "Oops! ⚠️ I hit a snag while generating the config. **Please try selecting the model again** — it should work on the second attempt.",
          uiType: 'text',
          uiData: null,
          nextStep: 1,
          coins: null
        };
      }
    }

    return {
      reply: "Please **click one of the model cards** above to select the AI engine for your app. 👆",
      uiType: "text",
      uiData: null,
      nextStep: 1,
      coins: null
    };
  }

  // ─── STEP 2: Live Preview Approval & Tweaks ───────────────────────────────
  if (session.step === 2) {
    const msg2 = lower(message);

    if (msg2.includes('approve') || msg2 === 'approve app' || isYes(text)) {
      session.step = 3;
      await saveSession(session);
      
      const seoData = session.seoData || {};

      return {
        reply: `## 🎉 App Configured — Final Review\n\nEverything is set! Review your **app name, description, and tags** below.\n\nYou can edit any field before publishing — make it shine! ✨`,
        uiType: "seo_preview",
        uiData: {
          appName: seoData.appName || 'Your App',
          appDescription: seoData.appDescription || '',
          category: seoData.category || '',
          tags: Array.isArray(seoData.tags) ? seoData.tags : [],
          appType: session.appType || 'text',
          modelId: session.modelId,
          costPerRun: session.modelCost
        },
        nextStep: 3,
        coins: session.modelCost
      };
    }

    if (msg2.includes('tweak') || isChangeMessage(text)) {
      const correction = isChangeMessage(text) ? getChangeText(text) : null;
      if (correction) {
        // Apply edit to session context FIRST (kills ghost memory), then regenerate
        applyEditToSession(session, correction);
        try {
          const [newPromptData, newSeoData] = await Promise.all([
            generatePromptTemplate(session),
            generateSEO(session)
          ]);
          session.promptData = newPromptData;
          session.seoData = newSeoData;
        } catch (e) {
          console.warn('[Step2 Edit] Regen failed, keeping patched promptData:', e.message);
          session.promptData = applyPromptInstruction(session.promptData, correction);
        }
        await saveSession(session);
        return {
          reply: `Updated! Check the refreshed preview below — test it out, then approve when ready.`,
          uiType: 'app_preview',
          uiData: {
            appName: session.seoData?.appName || 'Your App',
            appType: session.appType || session.extraction?.appType || 'text',
            appDescription: session.seoData?.appDescription || '',
            cost: session.modelCost,
            systemPrompt: session.promptData.systemPrompt,
            userPrompt: session.promptData.userPrompt,
            variablesUsed: session.promptData.variablesUsed,
            acceptImageInput: sanitizeAcceptImageInput(session.promptData.acceptImageInput, session.appType),
            options: ['Approve App', 'Edit App']
          },
          nextStep: 2,
          coins: session.modelCost
        };
      }
      session.awaitingPromptTweak = true;
      await saveSession(session);
      return {
        reply: "What would you like to edit? You can change the AI Model, rewrite the prompt, or completely change the app idea.",
        uiType: "text",
        uiData: {},
        nextStep: 2,
        coins: session.modelCost
      };
    }
    
    // Catch-all for step 2
    return {
      reply: "Please test the app in the Live Preview. Let me know if you want to approve it or edit it.",
      uiType: "chips",
      uiData: { options: ['Approve App', 'Edit App'] },
      nextStep: 2,
      coins: session.modelCost
    };
  }

  // ─── STEP 3: Final Publish & Draft ───────────────────────────────────────
  if (session.step === 3) {
    const msg3 = lower(message);

    // ── SEO_PUBLISH / SEO_DRAFT are now handled at the TOP of route() before extraction.
    // This block is a no-op fallback in case step===3 and somehow the top handler missed it.
    const _isSeoP = message.startsWith('SEO_PUBLISH::') || message.startsWith('SEO_DRAFT::');
    if (_isSeoP) {
      // Already handled above — shouldn't reach here, but just in case:
      return {
        reply: 'Processing your publish request...',
        uiType: 'text',
        uiData: null,
        nextStep: 3,
        coins: session.modelCost
      };
    }



    if (msg3.includes('edit app') || msg3.includes('tweak') || isChangeMessage(text)) {
      const correction = isChangeMessage(text) ? getChangeText(text) : text;
      // Apply edit to session context FIRST (kills ghost memory), then regenerate
      if (correction && correction.trim().length > 2) {
        applyEditToSession(session, correction);
      }
      try {
        const [newPromptData, newSeoData] = await Promise.all([
          generatePromptTemplate(session),
          generateSEO(session)
        ]);
        session.promptData = newPromptData;
        session.seoData = newSeoData;
      } catch (e) {
        console.warn('[Step3 Edit] Regen failed, keeping patched promptData:', e.message);
        if (session.promptData) session.promptData = applyPromptInstruction(session.promptData, correction);
      }
      session.step = 2;
      session.awaitingPromptTweak = false;
      await saveSession(session);
      return {
        reply: isChangeMessage(text)
          ? `Done! I've updated the app based on: "${correction}".\n\nHere's the refreshed preview — test it out and approve when you're happy!`
          : "No problem! I've re-opened the app preview so you can make changes. Test it below, then approve to continue.",
        uiType: 'app_preview',
        uiData: {
          appName: session.seoData?.appName || 'Your App',
          appType: session.appType || session.extraction?.appType || 'text',
          appDescription: session.seoData?.appDescription || '',
          cost: session.modelCost,
          systemPrompt: session.promptData.systemPrompt,
          userPrompt: session.promptData.userPrompt,
          variablesUsed: session.promptData.variablesUsed,
          acceptImageInput: sanitizeAcceptImageInput(session.promptData.acceptImageInput, session.appType),
          options: ['Approve App', 'Edit App']
        },
        nextStep: 2,
        coins: session.modelCost
      };
    }
    
    return {
      reply: "Ready to publish?",
      uiType: "chips",
      uiData: { options: ['Publish to Marketplace', 'Save Draft', 'Edit App'] },
      nextStep: 3,
      coins: session.modelCost
    };
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
