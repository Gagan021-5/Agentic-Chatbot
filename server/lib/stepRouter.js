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
function parseSelectedModelId(msg) {
  const text = normalize(msg).toLowerCase();
  if (text.startsWith("select")) {
    return text.replace("select", "").trim();
  }
  return null;
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
      appType: keepAppType ? existing.confidence.appType : latest.confidence.appType,
      budget: latest && latest.budget ? latest.confidence.budget : existing.confidence.budget
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

  // ─── AGENTIC TRIAGE: Evaluate specificity before generating form ───
  // Skip triage if we already have a ready dynamic context (user answered a clarification)
  if (!session.dynamicContext) {
    const triageResult = await triageDynamicContext({
      appType: session.appType || session.extraction?.appType || "text",
      appPurpose: purpose,
      languageHint: detectLanguageMode(session),
      conversationHistory: session.history || []
    });

    // ─── NEEDS CONTEXT: Ask a domain-specific clarifying question ───
    if (triageResult.status === "needs_context" && triageResult.question) {
      // Cap at 2 triage rounds to prevent infinite loops
      session.triageRounds = (session.triageRounds || 0) + 1;
      if (triageResult.domain) session.domainIdentified = triageResult.domain;
      console.log(`[Triage] Round ${session.triageRounds} — Domain: ${triageResult.domain || "unknown"} — Asking: ${triageResult.question.substring(0, 80)}`);
      if (session.triageRounds <= 2) {
        session.awaitingTriageAnswer = true;
        return {
          reply: triageResult.question,
          uiType: null,
          uiData: null,
          nextStep: session.step, // Stay on same step
          coins: null
        };
      }
      // If we've asked 2 clarifications already, force generate with what we have
    }

    // ─── READY: AI has enough context, store the form ───
    if (triageResult.status === "ready") {
      console.log(`Successfully scoped app for domain: ${triageResult.domain || "Unknown"}`);
      if (triageResult.domain) session.domainIdentified = triageResult.domain;
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

  return {
    reply: localizedText(
      session,
      "Perfect. I've scoped out the architecture for your app. Please confirm these settings.",
      "बहुत बढ़िया। मैंने आपके ऐप के लिए ज़रूरी फीचर्स और इनपुट तैयार किए हैं। कृपया इन्हें कंफर्म करें।",
      "Perfect. Maine aapke app ka architecture scope kar liya hai. Please in settings ko confirm karo."
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
  
  // HIGH/MEDIUM CONFIDENCE -> Skip confirmation, jump straight into deep questions
  if (ext && ext.appType && ['HIGH','MEDIUM'].includes(ext.confidence.appType)) {
    session.appType = ext.appType;
    session.step = 0;
    session.awaitingConfirmation = false; 

    const bundleCard = await buildDynamicBundleCard(session);
    if (bundleCard) {
      await saveSession(session);
      return bundleCard;
    }

    if (!ext?.appPurpose || ext.appPurpose.trim().length < 12) {
      await saveSession(session);
      return {
        reply: localizedText(
          session,
          `Alright. What kind of ${session.appType || 'AI'} app are you looking to create?`,
          `ठीक है। आप किस तरह का ${session.appType || 'AI'} ऐप बनाना चाहते हैं?`,
          `Alright. Aap kis type ka ${session.appType || 'AI'} app banana chahte ho?`
        ),
        uiType: "text",
        uiData: null,
        nextStep: 0,
        coins: null
      };
    }

    const nextQ = getNextDeepQuestion(session);
    if (nextQ) {
      session.currentDeepField = nextQ.field;
      session.awaitingDeepAnswer = true;
      await saveSession(session);
      return {
        reply: nextQ.question,
        uiType: "chips",
        uiData: { options: nextQ.options },
        nextStep: 0,
        coins: null
      };
    }

    return await showModels(session);
  }
  
  session.step = 0;
  await saveSession(session);
  const prefix = tonePrefix(ext);
  const options = detectLanguageMode(session) === "Hindi"
    ? ['टेक्स्ट','इमेज','ऑडियो','वीडियो','विज़न']
    : detectLanguageMode(session) === "Hinglish"
      ? ['Text','Image','Audio','Video','Vision']
      : ['Text','Image','Audio','Video','Vision'];
    
  return {
    reply: `${prefix}${localizedText(session, "I'd love to help! What type of output does your app need?", "मैं मदद करना चाहूंगा! आपके ऐप को किस प्रकार का आउटपुट चाहिए?", "Main help karunga! Aapke app ko kis type ka output chahiye?")}`,
    uiType: 'chips',
    uiData: { options },
    nextStep: 0,
    coins: null
  };
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
      reply: "Hello! I am ready to help you build your AI application today. To get started, what type of output does your app need?",
      uiType: 'chips',
      uiData: { options: ['Text', 'Image', 'Audio', 'Video', 'Vision'] },
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
  const trimmedText = text ? text.trim().toLowerCase() : "";

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

  // If extraction figured out the appType with decent confidence, save it so we don't ask again
  if (!session.appType && session.extraction.appType && session.extraction.confidence?.appType !== 'LOW') {
    session.appType = session.extraction.appType;
  }

  // 2. ABSOLUTE GATEKEEPER
  if (!session.appType) {
    const validTypes = ['text', 'image', 'audio', 'video'];
    const chipType = parseChipAppType(text);

    // Check if the user clicked a chip or typed a valid type
    if (chipType || validTypes.includes(trimmedText)) {
      session.appType = chipType || trimmedText;
      await saveSession(session);

      // SMART CHECK: If we already extracted their app purpose, skip asking them to describe it again!
      if (session.extraction?.appPurpose && session.extraction.appPurpose.length > 5) {
        // Do nothing here. Let it fall through to step 0 / triage below!
      } else {
        return {
          reply: `Awesome. You want to build a ${session.appType} app. Describe what it should do, and I'll scope out the architecture!`,
          uiType: null,
          nextStep: 0,
          coins: null
        };
      }
    } else {
      const isInitial = trimmedText === "" || /^(hi|hello|hey|hy|hola|greetings)[\s!\.]*$/i.test(trimmedText);
      return {
        reply: isInitial
          ? "Hey! I'm the RentPrompts App Creation Agent. To get started, what type of output will your AI app generate?"
          : "Before we design the architecture, I need to know the format. Will this app generate Text, Images, Audio, or Video?",
        uiType: 'chips',
        uiData: { options: ['Text', 'Image', 'Audio', 'Video'] },
        nextStep: 0,
        coins: null
      };
    }
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
        reply: `Hey! 👋 I'm RentPrompts Agent.\n\nI help you create and publish AI-powered apps on the RentPrompts marketplace — no coding needed.\n\nWhat kind of AI app are you thinking of building?`,
        uiType: 'chips',
        uiData: { options: ['Image generator', 'Video creator', 'Text / writing tool', 'Audio generator', 'Vision / image analyzer', 'Not sure yet'] },
        nextStep: 0,
        coins: null
      };
    }

    const isNotSure = lower(text).includes('not sure') || lower(text).includes('help me') || lower(text).includes('dont know');
    if (isNotSure) {
      session.step = 0;
      await saveSession(session);
      return {
        reply: `No problem! Answer a few quick questions and I'll recommend the best fit. First — what is the main thing you want your app to CREATE or DO?`,
        uiType: 'chips',
        uiData: { options: ['Generate images', 'Create videos', 'Write text', 'Generate audio', 'Analyze images'] },
        nextStep: 0,
        coins: null
      };
    }

    // Handle Triage Clarification Answer
    if (session.awaitingTriageAnswer) {
      session.awaitingTriageAnswer = false;
      // Enrich the appPurpose with the user's clarification
      const existingPurpose = session.extraction?.appPurpose || "";
      session.extraction = session.extraction || {};
      session.extraction.appPurpose = `${existingPurpose}. User clarified: ${text}`;
      await saveSession(session);

      // Re-run triage with enriched context — should now return "ready"
      const bundleCard = await buildDynamicBundleCard(session);
      if (bundleCard) {
        await saveSession(session);
        return bundleCard;
      }
    }

    // Handle Multi-Select Form Submission
    if (text.startsWith("multi_select_form::")) {
      try {
        const formData = JSON.parse(text.replace("multi_select_form::", ""));

        session.extraction = session.extraction || {};
        session.extraction.features = Array.isArray(formData.selectedOptions) ? formData.selectedOptions : [];
        session.extraction.variables = normalizeSubmittedVariables(formData.variables);
        session.extraction.keyFeatures = session.extraction.features;
        session.deepAnswers = session.deepAnswers || {};
        session.deepAnswers.dynamicFeatures = session.extraction.features;
        session.deepAnswers.dynamicVariables = session.extraction.variables;

        await saveSession(session);

        const nextQ = getNextDeepQuestion(session);

        if (nextQ) {
          session.currentDeepField = nextQ.field;
          session.awaitingDeepAnswer = true;
          await saveSession(session);
          return {
            reply: `Got it! I've saved those features. ${nextQ.question}`,
            uiType: "chips",
            uiData: { options: nextQ.options || [] },
            nextStep: session.step,
            coins: null
          };
        } else {
          return await showModels(session);
        }
      } catch (e) {
        console.error("Failed to parse form data:", e);
        return {
          reply: "Sorry, I had trouble reading that form. Let's try again. What features do you need?",
          uiType: "text",
          uiData: null,
          nextStep: session.step,
          coins: null
        };
      }
    }

    if (session.awaitingDeepAnswer && session.currentDeepField) {
      const answer = text;
      if (!session.deepAnswers) session.deepAnswers = {};
      session.deepAnswers[session.currentDeepField] = answer;
      if (!session.extraction) session.extraction = {};
      if (session.currentDeepField === "budgetPreference") {
        session.extraction.budget = answer;
      }
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
      session.extraction.appType = chipType;
      session.extraction.confidence.appType = "HIGH";
    } else if (session.extraction.appType && session.extraction.confidence.appType !== 'LOW') {
      session.appType = session.extraction.appType;
    }

    return await buildStep0Response(session);
  }

  // ─── STEP 1: Model selection → generate config ───────────────────────
  if (session.step === 1) {
    const selectedModelId = parseSelectedModelId(text);

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

        return {
          reply: `I understand your requirement perfectly. You want an app that ${session.extraction?.appPurpose || 'does exactly that'}. Let me confirm the setup so we can proceed.`,
          uiType: 'app_preview',
          uiData: { 
            appName: seoData.appName,
            appType: session.appType || session.extraction?.appType || 'text', // Prevent state-loss bug
            appDescription: seoData.appDescription,
            cost: session.modelCost, // Strictly overrides any LLM hallucinations or math errors
            systemPrompt: promptData.systemPrompt,
            userPrompt: promptData.userPrompt,
            variablesUsed: promptData.variablesUsed,
            acceptImageInput: promptData.acceptImageInput
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

    if (session.awaitingPromptTweak) {
      session.awaitingPromptTweak = false;
      session.promptData = applyPromptInstruction(session.promptData, message);
      await saveSession(session);
      return {
        reply: `Updated! Here's the revised setup:\n\n**User Prompt:** ${session.promptData.userPrompt}\n\nReady to publish?`,
        uiType: 'chips',
        uiData: { options: ['Publish App', 'Save Draft', 'Tweak Prompts'] },
        nextStep: 2,
        coins: session.modelCost
      };
    }

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
          reply: `Updated!\n\n**User Prompt:** ${session.promptData.userPrompt}\n\nReady to publish?`,
          uiType: 'chips',
          uiData: { options: ['Publish App', 'Save Draft', 'Tweak Prompts'] },
          nextStep: 2,
          coins: session.modelCost
        };
      }
      session.awaitingPromptTweak = true;
      await saveSession(session);
      return {
        reply: "What would you like to change about the instructions or description?",
        uiType: "text",
        uiData: {},
        nextStep: 2,
        coins: session.modelCost
      };
    }
  }

  // ─── CATCH ALL ───
  return {
    reply: "I'm ready to proceed. Let me know if you want to 'Publish' this app, 'Save Draft', or change something.",
    uiType: "chips",
    uiData: { options: ['Publish App', 'Save Draft'] },
    nextStep: session.step || 0,
    coins: session.modelCost
  };
}
