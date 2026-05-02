import MODELS from "./models.js";
import mockData from "./mockData.js";
import { extractRequirements } from "./groq.js";
import { generatePromptTemplate, generateSEO, generateScope, applyPromptInstruction, buildPromptTemplateFromSession } from "./gemini.js";
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
    { field: 'textPersonalization', question: 'Should the app personalize content per user?', options: ['Yes — based on user goals and preferences','Yes — based on user input each time','No — fixed template with variables','Not sure yet'] }
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
  const questions = DEEP_QUESTIONS[session.appType] || [];
  if (!session.deepAnswers) session.deepAnswers = {};
  for (const q of questions) {
    if (!session.deepAnswers[q.field]) return q;
  }
  return null;
}

async function showModels(session) {
  const fullText = [
    session.extraction?.appPurpose || '',
    session.extraction?.oneLineUnderstanding || '',
    JSON.stringify(session.deepAnswers || {})
  ].join(' ');
  const budget = session.extraction?.budget;
  const models = rankModels(MODELS[session.appType] || [], fullText, budget);
  session.step = 1;
  session.awaitingConfirmation = true;
  session.confirmStep = 1;
  await saveSession(session);
  return {
    reply: `Here are the top 3 models for your ${session.appType} app:`,
    uiType: 'models',
    uiData: { appType: session.appType, models },
    nextStep: 1,
    coins: null,
    confirm: {
      summary: `Top 3 models for your ${session.appType} app. Does one fit?`,
      detail: 'Click a model card to select it.'
    }
  };
}

async function buildStep0Response(session) {
  const ext = session.extraction;
  if (ext && ext.appType && ['HIGH','MEDIUM'].includes(ext.confidence.appType)) {
    session.appType = ext.appType;
    session.step = 0;
    session.awaitingConfirmation = true;
    session.confirmStep = 0;
    await saveSession(session);
    const prefix = tonePrefix(ext);
    const reply = ext.suggestedReply
      ? `${prefix}${ext.suggestedReply}`
      : `${prefix}Got it — ${ext.oneLineUnderstanding}.\n\nI'm planning to build a ${ext.appType} app that ${ext.appPurpose || 'does what you described'}. Is this the right direction?`;
    return {
      reply,
      uiType: 'confirm',
      uiData: {},
      nextStep: 0,
      coins: null,
      confirm: {
        summary: ext.oneLineUnderstanding,
        detail: `App type: ${ext.appType} • Target: ${ext.targetUsers || 'not specified yet'}`
      }
    };
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

/* ────────────────────────────────────────────
   EDGE CASE GUARDS — must be checked FIRST
   ──────────────────────────────────────────── */
function checkEdgeCases(message, session) {
  const text = normalize(message);
  const msg = lower(text);

  // 1. Off-topic guard
  if (isOffTopic(text)) {
    return OFF_TOPIC_RESPONSE;
  }

  // 2. Abuse / gibberish guard
  const isGibberish = text.length < 2 || /^[^a-zA-Z0-9ऀ-ॿ\s]{3,}$/.test(text);
  if (isGibberish) {
    return {
      reply: `I didn't quite catch that. Could you describe what kind of AI app you'd like to build?`,
      uiType: 'chips',
      uiData: { options: ['Image app', 'Video app', 'Text app', 'Audio app', 'Vision app'] },
      nextStep: session.step,
      coins: null
    };
  }

  // 3. Empty message guard
  if (!text || text === '') {
    return {
      reply: `Go ahead — describe what you'd like to build!`,
      uiType: 'text',
      uiData: null,
      nextStep: session.step,
      coins: null
    };
  }

  // 4. Help request guard
  if (msg.includes('help') && text.trim().split(' ').length <= 3) {
    return {
      reply: `Sure! Here's what I can help you build:

🖼️ Image apps — generate photos, art, portraits
🎥 Video apps — create cinematic clips, animations
📝 Text apps — write blogs, emails, scripts
🔊 Audio apps — voiceovers, music, speech
👁️ Vision apps — analyze and understand images

Which type interests you?`,
      uiType: 'chips',
      uiData: { options: ['Image app', 'Video app', 'Text app', 'Audio app', 'Vision app', 'Help me choose'] },
      nextStep: 0,
      coins: null
    };
  }

  // 5. "Start over" / "restart" guard
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

  // 6. Pricing question without context guard
  if ((msg.includes('how much') || msg.includes('price') || msg.includes('cost') || msg.includes('joules')) && session.step < 4) {
    return {
      reply: `Great question! The cost depends on which AI model we choose for your app. Prices range from FREE all the way to 318 coins per run depending on the model's capability.

Let me first understand what you need, then I'll show you the best options with their exact costs.

What kind of app are you building?`,
      uiType: session.appType ? 'text' : 'chips',
      uiData: session.appType ? null : { options: ['Image app', 'Video app', 'Text app', 'Audio app', 'Vision app'] },
      nextStep: session.step,
      coins: null
    };
  }

  // 7. Competitor mention guard
  const competitors = ['openai', 'chatgpt', 'midjourney', 'dalle', 'stable diffusion', 'runway', 'sora', 'adobe', 'canva', 'figma'];
  if (competitors.some(c => msg.includes(c))) {
    return {
      reply: `I work specifically with the AI models available on RentPrompts marketplace.

I can help you build apps powered by our top models like Flux, Kling, Veo, ElevenLabs and more — all publishable directly to our marketplace.

Want me to show you what's available?`,
      uiType: 'chips',
      uiData: { options: ['Yes show me models', 'Tell me more about RentPrompts', 'Start building my app'] },
      nextStep: session.step,
      coins: null
    };
  }

  return null; // No edge case matched
}

/* ════════════════════════════════════════════
   MAIN ROUTER
   ════════════════════════════════════════════ */
async function route(session, message) {
  const text = normalize(message);

  // Check edge cases FIRST
  const edgeCaseResponse = checkEdgeCases(message, session);
  if (edgeCaseResponse) {
    if (edgeCaseResponse.clearSession) {
      await deleteSession(session.sessionId);
    }
    return edgeCaseResponse;
  }

  // History is managed by server.js — do NOT push again here
  if (!session.history) session.history = [];

  // Run extraction
  const latestExtraction = await extractRequirements(text, session.history || []);
  session.extraction = mergeExtraction(session.extraction, latestExtraction, text);

  // Update enterprise signals from extraction
  if (session.extraction.enterpriseSignals !== undefined) {
    session.enterpriseSignals = session.extraction.enterpriseSignals;
  }
  if (session.extraction.userType) {
    session.userType = session.extraction.userType;
  }

  // ─── STEP 0: First message — detect app type ────────
  if (session.step === 0) {
    // Handle greetings
    const greetings = ['hi', 'hello', 'hey', 'hii', 'helo', 'good morning', 'good evening', 'yo', 'sup', 'namaste', 'hola'];
    const isGreeting = greetings.some(g => lower(text) === g || lower(text).startsWith(g + ' '));

    if (isGreeting) {
      await saveSession(session);
      return {
        reply: `Hey! 👋 I'm RentPrompts Agent.

I help you create and publish AI-powered apps on the RentPrompts marketplace — no coding needed.

What kind of AI app are you thinking of building?`,
        uiType: 'chips',
        uiData: {
          options: ['Image generator', 'Video creator', 'Text / writing tool', 'Audio generator', 'Vision / image analyzer', 'Not sure yet — help me decide']
        },
        nextStep: 0,
        coins: null
      };
    }

    // "Not sure" handler
    const isNotSure = lower(text).includes('not sure') || lower(text).includes('help me') || lower(text).includes('dont know') || lower(text).includes("don't know") || lower(text).includes('suggest');
    if (isNotSure) {
      session.step = 0;
      await saveSession(session);
      return {
        reply: `No problem! Let me help you figure out the right type of app.

Answer a few quick questions and I'll recommend the best fit. First — what is the main thing you want your app to CREATE or DO?`,
        uiType: 'chips',
        uiData: {
          options: [
            'Generate images or photos',
            'Create videos or animations',
            'Write text, blogs, or emails',
            'Generate voice or music',
            'Analyze or understand images',
            'I have something different in mind'
          ]
        },
        nextStep: 0,
        coins: null
      };
    }

    // Handle deep question answers (if in deep Q&A mode)
    if (session.awaitingDeepAnswer && session.currentDeepField) {
      if (!session.deepAnswers) session.deepAnswers = {};
      session.deepAnswers[session.currentDeepField] = text;
      if (!session.extraction) session.extraction = {};
      Object.assign(session.extraction, session.deepAnswers);
      session.awaitingDeepAnswer = false;
      session.currentDeepField = null;

      const nextQ = getNextDeepQuestion(session);

      if (nextQ) {
        session.currentDeepField = nextQ.field;
        session.awaitingDeepAnswer = true;
        await saveSession(session);

        const lastAnswer = text;
        const acks = {
          'Workout / fitness plans': 'Perfect.',
          'Meal / diet plans': 'Got it.',
          'Blog posts and articles': 'Got it.',
          'Photorealistic / photography': 'Nice choice.',
          'Comic book / superhero style': 'Love it.',
          'Animate a still photo into video': 'Got it.',
          'Text description to video': 'Understood.',
        };
        const ack = acks[lastAnswer] || 'Got it.';

        return {
          reply: `${ack} ${nextQ.question}`,
          uiType: 'chips',
          uiData: { options: nextQ.options },
          nextStep: 0,
          coins: null
        };
      }

      // All deep questions answered — show models
      return await showModels(session);
    }

    // Handle confirmation of app type understanding (confirmStep === 0)
    if (session.awaitingConfirmation && session.confirmStep === 0) {
      if (isYes(text)) {
        session.awaitingConfirmation = false;
        session.deepAnswers = session.deepAnswers || {};

        const nextQ = getNextDeepQuestion(session);

        if (nextQ) {
          session.currentDeepField = nextQ.field;
          session.awaitingDeepAnswer = true;
          await saveSession(session);
          return {
            reply: nextQ.question,
            uiType: 'chips',
            uiData: { options: nextQ.options },
            nextStep: 0,
            coins: null
          };
        }

        // No deep questions needed — go straight to models
        return await showModels(session);
      }

      if (isNo(text)) {
        session.awaitingConfirmation = false;
        session.appType = null;
        session.step = 0;
        await saveSession(session);
        return {
          reply: `No problem! What type of AI app would you like to build?`,
          uiType: 'chips',
          uiData: { options: ['Image app', 'Video app', 'Text app', 'Audio app', 'Vision app'] },
          nextStep: 0,
          coins: null
        };
      }
    }

    // Detect app type from extraction or chip selection
    const chipType = parseChipAppType(text);
    if (chipType && !session.appType) {
      session.appType = chipType;
      session.extraction.appType = chipType;
      session.extraction.confidence.appType = "HIGH";
    } else if (session.extraction.appType && session.extraction.confidence.appType !== 'LOW') {
      session.appType = session.extraction.appType;
    }

    // If app type still unknown — use buildStep0Response
    if (!session.appType) {
      return await buildStep0Response(session);
    }

    // App type known — show confirm card via buildStep0Response
    return await buildStep0Response(session);
  }

  // ─── STEP 1/3: Model selection ───────────────────────
  if (session.step === 1 || session.step === 3) {

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
          nextStep: 3,
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
        session.step = 4;
        await saveSession(session);
        return {
          reply: "Heads up — this model is powerful but can get expensive at scale:",
          uiType: "cost_warning",
          uiData: session.costWarning,
          nextStep: 4,
          coins: selectedModel.cost
        };
      }

      // Generate prompt and show with confirm
      delete session.costWarning;
      session.promptData = await generatePromptTemplate(session);
      session.step = 4;
      session.awaitingConfirmation = true;
      session.confirmStep = 4;
      await saveSession(session);
      return {
        reply: "Great choice! Here's your auto-generated prompt template:",
        uiType: "prompt_preview",
        uiData: session.promptData,
        nextStep: 4,
        coins: session.modelCost,
        confirm: {
          summary: "Here's the auto-generated prompt for your app. Does this look right?",
          detail: session.promptData.promptExplanation || null
        }
      };
    }

    // Handle "No" response to model selection
    if (isNo(text)) {
      const fullText = getFullUserText(session);
      const budget = session.extraction && session.extraction.budget;
      const models = rankModels(MODELS[session.appType] || [], fullText, budget);
      return {
        reply: "No problem! What are you looking for in a model? (e.g. cheaper, higher quality, faster)",
        uiType: "models",
        uiData: { appType: session.appType, models },
        nextStep: 3,
        coins: null
      };
    }
  }

  // ─── STEP 4: Cost warning handling ─────────────────
  if (session.step === 4 && session.costWarning) {
    const v = lower(text);

    if (v.includes("use cheaper") || v.includes("cheaper option") || v.includes("show me cheaper")) {
      session.modelId = session.costWarning.alternativeModelId;
      session.modelCost = session.costWarning.alternativeCost;
      delete session.costWarning;
      session.promptData = await generatePromptTemplate(session);
      session.awaitingConfirmation = true;
      session.confirmStep = 4;
      await saveSession(session);
      return {
        reply: "Swapped to the cheaper option. Here's your prompt:",
        uiType: "prompt_preview",
        uiData: session.promptData,
        nextStep: 4,
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
      session.confirmStep = 4;
      await saveSession(session);
      return {
        reply: "Understood. Here's your prompt template:",
        uiType: "prompt_preview",
        uiData: session.promptData,
        nextStep: 4,
        coins: session.modelCost,
        confirm: {
          summary: "Here's the auto-generated prompt for your app. Does this look right?",
          detail: session.promptData.promptExplanation || null
        }
      };
    }
  }

  // ─── STEP 4: Prompt confirmation ────────────────────
  if (session.step === 4 && session.awaitingConfirmation && session.confirmStep === 4) {
    if (isYes(text)) {
      // Prompt confirmed → generate scope
      if (!session.scopeData) {
        session.scopeData = await generateScope(session);
      }
      session.step = 5;
      session.awaitingConfirmation = true;
      session.confirmStep = 5;
      await saveSession(session);
      return {
        reply: `The scope covers ${session.scopeData.totalItems} key items — ` +
               `${session.scopeData.scopeSummary}\n\n` +
               `Total estimated effort: ~${session.scopeData.totalHours}h\n\n` +
               `Want to adjust anything in the scope, or shall we ` +
               `move on to look at pricing options?`,
        uiType: "scope",
        uiData: session.scopeData,
        nextStep: 5,
        coins: session.modelCost,
        confirm: {
          summary: `Scope: ${session.scopeData.totalItems} items (~${session.scopeData.totalHours}h). Looks good?`,
          detail: "Say Yes to proceed, or let me know what you want to add/remove."
        }
      };
    }

    if (isNo(text) || isChangeMessage(text)) {
      const correction = isChangeMessage(text) ? getChangeText(text) : "";
      if (correction) {
        session.promptData = applyPromptInstruction(session.promptData, correction);
        await saveSession(session);
        return {
          reply: "Updated! Does this look better?",
          uiType: "prompt_preview",
          uiData: session.promptData,
          nextStep: 4,
          coins: session.modelCost,
          confirm: {
            summary: "Here's the updated prompt. Does this look right?",
            detail: null
          }
        };
      }

      return {
        reply: "Tell me what to change about the prompt, and I'll regenerate it.",
        uiType: "text",
        uiData: {},
        nextStep: 4,
        coins: session.modelCost
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
    session.confirmStep = 4;
    await saveSession(session);
    return {
      reply: "Updated! Does this look better?",
      uiType: "prompt_preview",
      uiData: session.promptData,
      nextStep: 4,
      coins: session.modelCost,
      confirm: {
        summary: "Here's the updated prompt. Does this look right?",
        detail: null
      }
    };
  }

  // ─── STEP 5: Scope confirmation ────────────────────
  if (session.step === 5 && session.awaitingConfirmation && session.confirmStep === 5) {
    if (isYes(text)) {
      // Scope confirmed → generate SEO
      if (!session.seoData) {
        session.seoData = await generateSEO(session);
      }
      session.step = 6;
      session.awaitingConfirmation = true;
      session.confirmStep = 6;
      await saveSession(session);
      return {
        reply: "Almost done! Here's your app's SEO profile:",
        uiType: "seo_preview",
        uiData: session.seoData,
        nextStep: 6,
        coins: session.modelCost,
        confirm: {
          summary: "Here's your app's name, description and tags. Ready to publish?",
          detail: "You can edit any field inline before confirming."
        }
      };
    }

    if (isNo(text) || isChangeMessage(text)) {
      const correction = isChangeMessage(text) ? getChangeText(text) : text;
      if (correction) {
        if (!session.extraction.keyFeatures) session.extraction.keyFeatures = [];
        session.extraction.keyFeatures.push(correction);
        session.scopeData = await generateScope(session);
        await saveSession(session);
      }
      return {
        reply: `Updated scope — ${session.scopeData.totalItems} items, ~${session.scopeData.totalHours}h total. Does this look better?`,
        uiType: "scope",
        uiData: session.scopeData,
        nextStep: 5,
        coins: session.modelCost,
        confirm: {
          summary: `Scope: ${session.scopeData.totalItems} items (~${session.scopeData.totalHours}h). Looks good?`,
          detail: "Say Yes to proceed, or let me know what you want to add/remove."
        }
      };
    }
  }

  // ─── STEP 6: SEO confirmation ──────────────────────
  if (session.step === 6 && session.awaitingConfirmation && session.confirmStep === 6) {
    if (isYes(text)) {
      // SEO confirmed → budget/bounty check
      return await buildBudgetStep(session);
    }

    if (isNo(text) || isChangeMessage(text)) {
      return {
        reply: "Which part would you like to change? You can also edit inline above.",
        uiType: "text",
        uiData: {},
        nextStep: 6,
        coins: session.modelCost
      };
    }
  }

  // ─── SEO INLINE EDIT CONFIRM ───
  const seoPayload = parseSeoPayload(text);
  if (seoPayload && session.seoData) {
    session.seoData = {
      ...session.seoData,
      ...seoPayload,
      tags: Array.isArray(seoPayload.tags) ? seoPayload.tags : session.seoData.tags
    };
    await saveSession(session);
    return await buildBudgetStep(session);
  }

  // ─── SEO CONFIRM via text ───
  if (session.seoData && (lower(text).includes("confirm seo") || lower(text).includes("confirm & continue") || lower(text) === "continue")) {
    return await buildBudgetStep(session);
  }

  // ─── STEP 7: Plan selected → publish ───
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

  // ─── STEP 7: Bounty publish ───
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

  // ─── FALLBACK: re-generate prompt if model selected but no promptData ───
  if (session.modelId && !session.promptData) {
    session.promptData = await generatePromptTemplate(session);
    session.awaitingConfirmation = true;
    session.confirmStep = 4;
    await saveSession(session);
    return {
      reply: "Here's your auto-generated prompt:",
      uiType: "prompt_preview",
      uiData: session.promptData,
      nextStep: 4,
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
   BUDGET STEP BUILDER (Step 7)
   ──────────────────────────────────────────── */
async function buildBudgetStep(session) {
  const budgetUi = buildBudgetUi(session);
  const leanCost = budgetUi.options.lean.joules;
  const userBalance = mockData.userBalance;

  // Check if user has enough joules
  if (userBalance < leanCost) {
    session.step = 7;
    session.budgetPath = "bounty";
    session.awaitingConfirmation = true;
    session.confirmStep = 7;
    await saveSession(session);

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
      nextStep: 7,
      coins: null,
      confirm: {
        summary: "Would you like to post this as a bounty instead?",
        detail: "Creators on RentPrompts will bid on your project within 24-48 hours."
      }
    };
  }

  // Enough joules → show budget cards
  session.step = 7;
  session.budgetPath = "publish";
  await saveSession(session);
  return {
    reply: "Your app is ready! Pick a publishing plan:",
    uiType: "budget_cards",
    uiData: budgetUi,
    nextStep: 7,
    coins: null,
    confirm: {
      summary: "Which plan works best for you?",
      detail: `Your balance: ${userBalance.toLocaleString()} joules`
    }
  };
}

export { route };
