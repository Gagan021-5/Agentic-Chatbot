import MODELS from "./models.js";
import mockData from "./mockData.js";
import { extractRequirements } from "./groq.js";
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

/* ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
   SMART MODEL RANKING (spec-exact)
   ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ */
function rankModels(models, userMessage, budgetStr) {
  const msg = (userMessage || "").toLowerCase();
  
  let filteredModels = models;

  // STRICT BUDGET FILTERING
  if (budgetStr) {
    const b = budgetStr.toLowerCase();
    
    // 1. Check for explicit numbers first (e.g., "I have 6 coins", "max 10")
    const numberMatch = b.match(/\d+(\.\d+)?/);
    
    if (numberMatch) {
      const maxCost = parseFloat(numberMatch[0]);
      filteredModels = models.filter(m => m.cost <= maxCost);
    } 
    // 2. Fall back to keyword matching if no number is found
    else if (b.includes("free")) {
      filteredModels = models.filter(m => m.cost === 0);
    } else if (b.includes("low")) {
      filteredModels = models.filter(m => m.cost <= 5);
    } else if (b.includes("medium")) {
      filteredModels = models.filter(m => m.cost <= 20);
    }
    // If premium/high, don't filter out anything, just let the scoring boost the expensive ones
  }

  // Fallback: If filtering leaves us with 0 options (e.g., no free video models exist), show the cheapest available
  if (filteredModels.length === 0) {
    filteredModels = [...models].sort((a, b) => a.cost - b.cost).slice(0, 3);
  }

  return filteredModels
    .map(m => {
      let score = 0;

      // keyword matching against model tags
      (m.tags || []).forEach(tag => {
        if (msg.includes(tag.replace("-", " "))) score += 30;
      });

      // Scoring logic based on tier
      if (m.tier === "balanced") score += 10;
      if (m.cost === 0) score += 20;

      if (msg.includes("cinematic") || msg.includes("motion")) {
        if ((m.tags || []).includes("motion-control")) score += 60;
      }

      // Boost specific tiers based on requested budget
      if (budgetStr) {
        const b = budgetStr.toLowerCase();
        if (b.includes("premium") || b.includes("high") || b.includes("best")) {
          if (m.tier === "premium" || m.tier === "ultra") score += 50;
        }
      }

      return { ...m, score };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, 3);
}

/* ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
   PARSERS (FIXED BULLETPROOF PARSING)
   ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ */
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

/* ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
   MODEL / COST HELPERS
   ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ */
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

/* ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
   TONE-AWARE REPLY PREFIX
   ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ */
function tonePrefix(extraction) {
  if (!extraction) return "";
  if (extraction.userTone === "urgent") return "No worries, let's set this up quickly! ";
  if (extraction.userTone === "unsure") return "Happy to help figure this out together! ";
  if (extraction.detectedLanguage === "Hindi") return "Samajh gaya ΓÇö ";
  return "";
}

/* ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
   BUDGET / COMPLEXITY HELPERS
   ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ */
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

/* ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
   MERGE EXTRACTION
   ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ */
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

/* ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
   BUILD FULL HISTORY STRING for ranking
   ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ */
function getFullUserText(session) {
  if (!session.history) return "";
  return session.history.filter(h => h.role === "user").map(h => h.content).join(" ");
}

/* ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
   DEEP QUESTIONS ΓÇö app-specific
   ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ */
const DEEP_QUESTIONS = {
  image: [
    { field: 'imageStyle', question: 'What visual style should the output have?', options: ['Photorealistic / photography','Comic book / superhero style','Anime / manga','Oil painting / artistic','Cinematic / dramatic','Cartoon / illustrated','User chooses the style'] },
    { field: 'imageInputType', question: 'What does the user provide to generate the image?', options: ['Just a text description','Upload a photo of themselves','Upload any reference image','Both text and photo upload'] },
    { field: 'imageUseCase', question: 'What will people mainly use this for?', options: ['Social media profile pictures','Fun personal use','Marketing and branding','Gaming avatars','Gifts and merchandise','Professional headshots','Something else'] }
  ],
  video: [
    { field: 'videoType', question: 'What kind of video should this app create?', options: ['Animate a still photo into video','Text description to video','Cinematic scenes','Short social media reels','Product showcase videos','Talking avatar / presenter'] },
    { field: 'videoEffects', question: 'What motion or visual effect does it need?', options: ['Smooth cinematic camera movement','Dynamic action sequences','Slow motion dramatic effect','Natural realistic motion','User chooses the effect'] },
    { field: 'videoDuration', question: 'How long should each generated video be?', options: ['3-5 seconds','5-10 seconds','10-30 seconds','User sets the duration'] }
  ],
  text: [
    { field: 'textPurpose', question: 'What exactly should this app generate or plan?', options: ['Workout / fitness plans','Meal / diet plans','Blog posts and articles','Social media captions','Email and newsletters','Product descriptions','Study / learning plans','Travel itineraries','Scripts and screenplays','Something else'] },
    { field: 'textTone', question: 'What tone should the generated content have?', options: ['Professional and formal','Casual and friendly','Motivational and energetic','Educational and clear','Creative and expressive','User controls the tone'] },
    { field: 'textPersonalization', question: 'Should the app personalize content per user?', options: ['Yes ΓÇö based on user goals and preferences','Yes ΓÇö based on user input each time','No ΓÇö fixed template with variables','Not sure yet'] }
  ],
  audio: [
    { field: 'audioType', question: 'What kind of audio should this app generate?', options: ['Voice narration / text to speech','AI music generation','Sound effects','Podcast production','Voice cloning','Speech to text transcription'] },
    { field: 'audioStyle', question: 'What voice or audio style is needed?', options: ['Professional narrator voice','Warm conversational tone','Energetic / motivational voice','Multiple language support','User picks from voice options'] }
  ],
  vision: [
    { field: 'visionTask', question: 'What should this app do when it sees an image?', options: ['Describe what is in the image','Detect and label objects','Read text from image (OCR)','Analyze medical images','Inspect product quality','Answer questions about an image'] },
    { field: 'visionOutput', question: 'What format should the analysis result be in?', options: ['Plain text description','Structured report','JSON for developers','Simple yes/no answer','User-friendly summary'] }
  ]
};

function getNextDeepQuestion(session) {
  if (!session.deepAnswers) session.deepAnswers = {};

  const purpose = session.extraction?.appPurpose || session.deepAnswers?.appPurpose || "";

  // 1. Force the user to provide actual details if the purpose is empty or just a generic short phrase (under 12 chars)
  if (!purpose || purpose.trim().length < 12) {
    return {
      field: 'appPurpose',
      question: `Got it. Before we configure the settings, what exactly do you want this ${session.appType || 'AI'} app to generate or do? Describe your specific idea.`,
      options: null // Forces a free-text input box
    };
  }

  // 2. If we have a detailed purpose, proceed with the type-specific questions
  const questions = DEEP_QUESTIONS[session.appType] || [];
  for (const q of questions) {
    if (!session.deepAnswers[q.field]) return q;
  }
  
  // NEW BUDGET CHECK: Ask for budget if it wasn't extracted initially and hasn't been answered yet
  if (!session.extraction?.budget && !session.deepAnswers.budgetPreference) {
    return {
      field: 'budgetPreference',
      question: 'One last thing ΓÇö what is your target budget per generation for this app?',
      options: ['Free models only', 'Low (Under 5 coins)', 'Medium (5-20 coins)', 'Premium / Best Quality']
    };
  }
  
  return null; // All done, ready for models
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

    const nextQ = getNextDeepQuestion(session);

    if (nextQ) {
      session.currentDeepField = nextQ.field;
      session.awaitingDeepAnswer = true;
      await saveSession(session);
      
      const prefix = tonePrefix(ext);
      const questionText = ext.suggestedReply 
        ? `${prefix}${ext.suggestedReply}` 
        : `${prefix}Got it. ${nextQ.question}`;

      return {
        reply: questionText,
        uiType: nextQ.options ? 'chips' : 'text',
        uiData: nextQ.options ? { options: nextQ.options } : null,
        nextStep: 0,
        coins: null
      };
    }

    return await showModels(session);
  }
  
  session.step = 0;
  await saveSession(session);
  const prefix = tonePrefix(ext);
  const options = ext && ext.appPurpose && /clinic|hospital/i.test(ext.appPurpose)
    ? ['Images','Video tour','Written content','Something else']
    : ['Text','Image','Audio','Video','Vision'];
    
  return {
    reply: `${prefix}I'd love to help! What type of output does your app need?`,
    uiType: 'chips',
    uiData: { options },
    nextStep: 0,
    coins: null
  };
}

/* ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
   EDGE CASE GUARDS
   ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ */
function checkEdgeCases(message, session) {
  const text = normalize(message);
  const msg = lower(text);

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

  // 2. Abuse / gibberish guard
  const trimmedText = text.trim();
  const symbolCount = (trimmedText.match(/[^a-zA-Z0-9\s]/g) || []).length;
  
  const isGibberish = 
    trimmedText.length < 2 || // Too short
    /(asdf|qwer|zxcv|hjkl)/i.test(trimmedText) || // Classic keyboard smash
    /[a-zA-Z0-9]{20,}/.test(trimmedText) || // Huge 20+ char block with no spaces
    (symbolCount > trimmedText.length / 2 && trimmedText.length > 5); // Over 50% symbols

  if (isGibberish) {
    return {
      reply: `I didn't quite catch that. Could you describe what kind of AI app you'd like to build?`,
      uiType: 'chips',
      uiData: { options: ['Image app', 'Video app', 'Text app', 'Audio app', 'Vision app'] },
      nextStep: session.step,
      coins: null
    };
  }

  if (!text || text === '') {
    return {
      reply: `Go ahead ΓÇö describe what you'd like to build!`,
      uiType: 'text',
      uiData: null,
      nextStep: session.step,
      coins: null
    };
  }

  if (msg.includes('help') && text.trim().split(' ').length <= 3) {
    return {
      reply: `Sure! Here's what I can help you build:\n\n≡ƒû╝∩╕Å Image apps\n≡ƒÄÑ Video apps\n≡ƒô¥ Text apps\n≡ƒöè Audio apps\n≡ƒæü∩╕Å Vision apps\n\nWhich type interests you?`,
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

/* ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
   MAIN ROUTER
   ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ */
export async function route(session, message) {
  // 1. WALL OF TEXT GUARD: Truncate to 1000 characters
  const rawText = String(message || "").substring(0, 1000);
  const text = normalize(rawText);
  const msg = lower(text);

  // 2. GHOST TOWN GUARD: If session step is > 0 but appType is missing, Redis wiped the session
  if (session.step > 0 && !session.appType && !parseChipAppType(text)) {
    return {
      reply: "It looks like you've been away for a while and your session expired! Let's start fresh. What kind of AI app are you building today?",
      uiType: 'chips',
      uiData: { options: ['Image app', 'Video app', 'Text app', 'Audio app', 'Vision app'] },
      nextStep: 0,
      coins: null,
      clearSession: true
    };
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

  if (!session.history) session.history = [];

  const latestExtraction = await extractRequirements(text, session.history || []);
  session.extraction = mergeExtraction(session.extraction, latestExtraction, text);

  if (session.extraction.enterpriseSignals !== undefined) {
    session.enterpriseSignals = session.extraction.enterpriseSignals;
  }
  if (session.extraction.userType) {
    session.userType = session.extraction.userType;
  }

  // ΓöÇΓöÇΓöÇ STEP 0: First message ΓÇö detect app type ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
  if (session.step === 0) {
    const greetings = ['hi', 'hello', 'hey', 'hii', 'helo', 'good morning', 'good evening', 'yo', 'sup', 'namaste', 'hola'];
    const isGreeting = greetings.some(g => lower(text) === g || lower(text).startsWith(g + ' '));

    if (isGreeting) {
      await saveSession(session);
      return {
        reply: `Hey! ≡ƒæï I'm RentPrompts Agent.\n\nI help you create and publish AI-powered apps on the RentPrompts marketplace ΓÇö no coding needed.\n\nWhat kind of AI app are you thinking of building?`,
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
        reply: `No problem! Answer a few quick questions and I'll recommend the best fit. First ΓÇö what is the main thing you want your app to CREATE or DO?`,
        uiType: 'chips',
        uiData: { options: ['Generate images', 'Create videos', 'Write text', 'Generate audio', 'Analyze images'] },
        nextStep: 0,
        coins: null
      };
    }

    if (session.awaitingDeepAnswer && session.currentDeepField) {
      
      // NEW: Intercept "User chooses" to ask for the specific dropdown options
      const isUserChoice = ['user chooses', 'user sets', 'user picks', 'user controls'];
      if (isUserChoice.some(u => lower(text).includes(u))) {
        return {
          reply: "Got it! Since the end-user will decide, what specific options should we give them to choose from? (e.g., 'Realistic, Anime, or 3D')",
          uiType: 'text',
          uiData: null,
          nextStep: 0,
          coins: null
        };
      }

      // NEW: INTERCEPT "Something else"
      const isSomethingElse = ['something else', 'other', 'none of these'];
      if (isSomethingElse.some(s => lower(text) === s || lower(text) === 'other')) {
        return {
          reply: "No problem. Please type out exactly what you have in mind so I can configure the prompt correctly.",
          uiType: 'text',
          uiData: null,
          nextStep: 0,
          coins: null
        };
      }

      let finalAnswer = text;
      
      // I DON'T KNOW GUARD: If user is unsure, grab the most generic option
      const unsureSignals = ['idk', 'i dont know', 'not sure', 'you decide', 'whatever', 'doesnt matter'];
      if (unsureSignals.some(s => lower(text).includes(s))) {
        const currentQ = getNextDeepQuestion(session);
        // Default to "User chooses" or the last option in the list
        finalAnswer = currentQ?.options?.find(opt => lower(opt).includes('user')) || 
                      currentQ?.options?.[currentQ.options.length - 1] || 'Not specified';
      }

      if (!session.deepAnswers) session.deepAnswers = {};
      session.deepAnswers[session.currentDeepField] = finalAnswer;
      if (!session.extraction) session.extraction = {};
      Object.assign(session.extraction, session.deepAnswers);
      session.awaitingDeepAnswer = false;
      session.currentDeepField = null;

      const nextQ = getNextDeepQuestion(session);

      if (nextQ) {
        session.currentDeepField = nextQ.field;
        session.awaitingDeepAnswer = true;
        await saveSession(session);

        const acks = {
          'appPurpose': 'Great idea.',
          'Free models only': 'Got it. Keeping it free.',
          'Low (Under 5 coins)': 'Got it. Budget-friendly models coming right up.',
          'Medium (5-20 coins)': 'Understood. Finding the best mid-range models.',
          'Premium / Best Quality': 'Perfect. Giving you the top-tier flagship models.',
          'Workout / fitness plans': 'Perfect.',
          'Meal / diet plans': 'Got it.',
          'Blog posts and articles': 'Got it.',
          'Photorealistic / photography': 'Nice choice.',
          'Comic book / superhero style': 'Love it.',
          'Animate a still photo into video': 'Got it.',
          'Text description to video': 'Understood.',
        };
        const lastAnswer = text;
        const ack = session.currentDeepField === 'appPurpose' ? acks['appPurpose'] : (acks[lastAnswer] || 'Got it.');

        return {
          reply: `${ack} ${nextQ.question}`,
          uiType: nextQ.options ? 'chips' : 'text',
          uiData: nextQ.options ? { options: nextQ.options } : null,
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

  // ΓöÇΓöÇΓöÇ STEP 1: Model selection ΓåÆ generate config ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
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
            appDescription: seoData.appDescription,
            cost: seoData.suggestedPrice || session.modelCost,
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

  // ΓöÇΓöÇΓöÇ STEP 2: Final Review (Publish, Save Draft, Tweak) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
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
        costPerRun: session.seoData?.suggestedPrice || session.modelCost,
        systemPrompt: session.promptData?.systemPrompt,
        userPrompt: session.promptData?.userPrompt,
        negativePrompt: session.promptData?.negativePrompt,
        acceptImageInput: session.promptData?.acceptImageInput,
        appName: session.seoData?.appName,
        appDescription: session.seoData?.appDescription,
        tags: session.seoData?.tags,
        publishedAt: new Date().toISOString()
      };

      console.log("\nΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ MOCK PUBLISH ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ");
      console.log(JSON.stringify(payload, null, 2));
      console.log("ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ\n");

      return {
        reply: `≡ƒÄë Your app "${session.seoData?.appName}" is now live! Users will be charged ${payload.costPerRun} coins per generation.`,
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

  // ΓöÇΓöÇΓöÇ CATCH ALL ΓöÇΓöÇΓöÇ
  return {
    reply: "I'm ready to proceed. Let me know if you want to 'Publish' this app, 'Save Draft', or change something.",
    uiType: "chips",
    uiData: { options: ['Publish App', 'Save Draft'] },
    nextStep: session.step || 0,
    coins: session.modelCost
  };
}
