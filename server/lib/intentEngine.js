/**
 * ═══════════════════════════════════════════════════════════════════════════
 *  ORCHESTRATOR BRAIN — LLM Function Calling (Full Agentic Decision Engine)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  This module is the single point of intelligence for the entire pipeline.
 *  It replaces both the old intent classifier AND the step-based routing logic
 *  with a single LLM tool call: `orchestrate_pipeline`.
 *
 *  Design:
 *  ─ One tool: `orchestrate_pipeline`
 *  ─ Returns a `recommended_action` enum that directly drives the dispatcher
 *  ─ Injects a full stateless session context snapshot on every turn
 *  ─ Graceful regex fallback when Groq is unavailable (rate limit / no key)
 *  ─ Fast-path for deterministic UI payloads (zero LLM tokens spent)
 *
 *  Agentic score improvement:
 *  ─ No hardcoded step numbers leak into routing decisions
 *  ─ LLM sees the FULL session state map and makes holistic decisions
 *  ─ extracted_variables bubble up domain data without a separate extractor call
 *  ─ app_type correction is embedded in every turn (self-healing classification)
 */

import Groq from "groq-sdk";

const groq = process.env.GROQ_API_KEY
  ? new Groq({ apiKey: process.env.GROQ_API_KEY })
  : null;

/* ─────────────────────────────────────────────────────────────────────────
   CAPABILITY TOOL SCHEMA — orchestrate_pipeline
   ───────────────────────────────────────────────────────────────────────── */
const ORCHESTRATOR_TOOL = {
  type: "function",
  function: {
    name: "orchestrate_pipeline",
    description:
      "Given the user's message and the full session state, decide the single next pipeline " +
      "action the orchestrator should execute. Extract any runtime variables the user mentioned " +
      "and correct the app type if the classification is clearly wrong.",
    parameters: {
      type: "object",
      properties: {

        recommended_action: {
          type: "string",
          enum: [
            "GATHER_REQUIREMENTS", // User is describing/scoping their app — run triage
            "RENDER_FORM",         // Enough info collected — surface the multi_select_form
            "SHOW_MODEL_CARDS",    // Form confirmed + budget set — rank and show AI models
            "GENERATE_PREVIEW",   // Model selected — build prompt template + SEO, show live preview
            "REVIEW_SEO",          // Preview approved — show final SEO / publish card
            "PUBLISH_APP",         // User is publishing or saving draft
            "PIVOT_APP",           // User wants a completely different app (domain-level change)
            "EDIT_APP",            // User wants to tweak the current app (prompt, tone, style)
            "CHANGE_MODEL",        // User wants to swap the AI engine
            "HANDLE_BUDGET",       // User is answering / changing budget preference
            "HANDLE_GREETING",     // User said hi with no app idea
            "HANDLE_OFF_TOPIC",    // Message is unrelated to app building
            "HANDLE_VIOLATION",    // Policy / jailbreak / NSFW
            "HANDLE_GIBBERISH"     // Random characters / keyboard smash
          ],
          description:
            "The single pipeline action the orchestrator should fire next. " +
            "GATHER_REQUIREMENTS = user still describing idea or answering triage questions. " +
            "RENDER_FORM = enough domain context gathered, ready to show multi_select_form. " +
            "SHOW_MODEL_CARDS = form confirmed AND budget known, show ranked model cards. " +
            "GENERATE_PREVIEW = model selected, produce prompt + SEO, render live preview. " +
            "REVIEW_SEO = live preview approved, transition to SEO publish card. " +
            "PUBLISH_APP = user clicking publish / save draft. " +
            "PIVOT_APP = user is describing a COMPLETELY different app (kills session context). " +
            "EDIT_APP = user wants to tweak current app without changing the domain. " +
            "CHANGE_MODEL = user wants to switch AI engine. " +
            "HANDLE_BUDGET = user is selecting or changing their budget tier. " +
            "HANDLE_GREETING = just a greeting, no app idea yet. " +
            "HANDLE_OFF_TOPIC = unrelated topic. " +
            "HANDLE_VIOLATION = harmful / NSFW / jailbreak. " +
            "HANDLE_GIBBERISH = random or meaningless input."
        },

        extracted_variables: {
          type: "object",
          description:
            "Key-value map of any clean runtime configuration properties the user mentioned. " +
            "Examples: { domain: 'Indian criminal law', tone: 'formal', targetUsers: 'lawyers', " +
            "budget: 'medium', editInstruction: 'make it more concise' }. " +
            "Only include fields the user explicitly stated — never invent values.",
          additionalProperties: { type: "string" }
        },

        app_type: {
          type: ["string", "null"],
          enum: ["text", "image", "video", "audio", "vision", null],
          description:
            "The correct output type for this app. " +
            "Set this if the user explicitly states a type OR if the current session type is clearly wrong. " +
            "image = visual output (logos, cards, posters, room designs). " +
            "video = animation, reels, cinematic clips. " +
            "audio = voiceover, podcast, TTS, music. " +
            "text = written content (blogs, legal, recipes, emails). " +
            "vision = image ANALYSIS (OCR, object detection). " +
            "null = no change from current session type."
        },

        is_major_pivot: {
          type: "boolean",
          description:
            "True ONLY when the user abandons the current domain for a completely different one. " +
            "Example: switching from 'legal advisor' to 'birthday card generator' = true. " +
            "Changing tone or model = false."
        },

        budget_tier: {
          type: ["string", "null"],
          enum: ["free", "low", "medium", "premium", null],
          description:
            "Budget tier if stated. free=0 coins. low=<5 coins. medium=5-20 coins. premium=>20 coins. " +
            "null = not mentioned."
        },

        confidence: {
          type: "string",
          enum: ["high", "medium", "low"],
          description: "Confidence in this classification."
        }
      },
      required: ["recommended_action", "is_major_pivot", "confidence"]
    }
  }
};

/* ─────────────────────────────────────────────────────────────────────────
   SYSTEM PROMPT — stateless orchestration brain
   ───────────────────────────────────────────────────────────────────────── */
const ORCHESTRATOR_SYSTEM_PROMPT = `You are the Orchestration Brain for RentPrompts — a platform where users CREATE and PUBLISH AI-powered apps.

On every turn you receive: the user's latest message, the conversation history, and a SESSION STATE SNAPSHOT.
Your ONLY job: call orchestrate_pipeline with the single correct next action.

═══════════════════════════════════════
PIPELINE STAGE LOGIC
═══════════════════════════════════════

Use the SESSION STATE SNAPSHOT to determine what stage the pipeline is at:

STAGE 0 — REQUIREMENTS GATHERING (session.hasPurpose=false OR session.triageComplete=false):
  → Default action: GATHER_REQUIREMENTS
  → If user supplies budget: HANDLE_BUDGET
  → If user says hi only: HANDLE_GREETING

STAGE 1 — FORM READY (session.triageComplete=true AND session.formConfirmed=false):
  → RENDER_FORM unless user just answered a budget question → SHOW_MODEL_CARDS

STAGE 2 — MODEL SELECTION (session.formConfirmed=true AND session.modelSelected=false):
  → SHOW_MODEL_CARDS or HANDLE_BUDGET

STAGE 3 — PREVIEW (session.modelSelected=true AND session.previewApproved=false):
  → GENERATE_PREVIEW (model was just selected)
  → EDIT_APP (user wants tweaks)
  → PIVOT_APP (user describes completely different app)

STAGE 4 — SEO REVIEW (session.previewApproved=true AND session.seoReviewed=false):
  → REVIEW_SEO

STAGE 5 — PUBLISH (session.seoReviewed=true):
  → PUBLISH_APP

═══════════════════════════════════════
CLASSIFICATION RULES
═══════════════════════════════════════

PIVOT vs EDIT:
  - PIVOT_APP: user describes a DIFFERENT domain entirely. Set is_major_pivot=true.
  - EDIT_APP: user wants to refine the SAME app (tone, model, prompt, variable).

APP TYPE (most common mistake — read carefully):
  - "image": output is a PICTURE — logos, cards, posters, room designs, memes, avatars
  - "text": output is WRITTEN WORDS — blogs, legal docs, recipes, emails, wishes
  - "audio": output is SOUND — TTS, voiceover, music, podcast
  - "video": output is VIDEO — reels, animations, clips
  - "vision": INPUT is an image to ANALYZE — OCR, defect detection, medical imaging
  - "birthday card with photo" = IMAGE. "birthday wishes text" = TEXT. This is the #1 error.

AFFIRMATIONS ("yes", "sure", "ok", "sounds good", "go ahead", "proceed"):
  - During triage → GATHER_REQUIREMENTS (the yes is an answer to a triage question)
  - After form shown but not confirmed → RENDER_FORM
  - After model shown → interpret as model selection if a model was displayed, else SHOW_MODEL_CARDS
  - After preview shown → REVIEW_SEO

BUDGET CHIPS ("Free models only (0 coins)", "Low (< 5 coins)", etc.):
  → HANDLE_BUDGET always

MODEL SELECTION ("Select model-id"):
  → GENERATE_PREVIEW

APPROVE ("Approve App", "approve"):
  → REVIEW_SEO

You MUST call orchestrate_pipeline. Never respond with plain text.`;

/* ─────────────────────────────────────────────────────────────────────────
   SESSION STATE SNAPSHOT — injected into every LLM call
   ───────────────────────────────────────────────────────────────────────── */
function buildSessionSnapshot(session) {
  const s = session || {};
  return {
    hasPurpose:           Boolean(s.extraction?.appPurpose && s.extraction.appPurpose.length > 5),
    appPurposePreview:    (s.extraction?.appPurpose || "").slice(0, 80) || null,
    currentAppType:       s.appType || null,
    appTypeConfidence:    s.extraction?.confidence?.appType || "LOW",
    triageComplete:       Boolean(s.dynamicContext),
    formConfirmed:        Boolean(s.formConfirmed),
    modelSelected:        Boolean(s.modelId),
    selectedModel:        s.modelId || null,
    modelCost:            s.modelCost ?? null,
    previewApproved:      Boolean(s.step >= 2 && s.promptData),
    seoReviewed:          Boolean(s.step >= 3),
    budgetSet:            Boolean(s.extraction?.budget || s.deepAnswers?.budgetPreference),
    currentBudget:        s.deepAnswers?.budgetPreference || s.extraction?.budget || null,
    awaitingTriageAnswer: Boolean(s.awaitingTriageAnswer),
    awaitingDeepAnswer:   Boolean(s.awaitingDeepAnswer),
    awaitingPromptTweak:  Boolean(s.awaitingPromptTweak),
    triageRounds:         s.triageRounds || 0,
    currentDeepField:     s.currentDeepField || null,
    domainIdentified:     s.domainIdentified || null,
    languageMode:         s.languageMode || "English",
    isPivot:              Boolean(s.isPivot),
    // Legacy step kept for reference but NOT used for routing decisions
    _legacyStep:          s.step ?? 0
  };
}

/* ─────────────────────────────────────────────────────────────────────────
   FAST-PATH DETERMINISTIC INTERCEPTORS
   These never spend LLM tokens — they are exact string patterns from UI buttons.
   ───────────────────────────────────────────────────────────────────────── */
function tryFastPath(text, session) {
  const t = String(text || "").trim();
  const lower = t.toLowerCase();

  // ── Structured UI payloads ──
  if (
    t.startsWith("multi_select_form::") ||
    t.startsWith("SEO_PUBLISH::") ||
    t.startsWith("SEO_DRAFT::") ||
    t.startsWith("SEO_EDIT::") ||
    t.startsWith("confirm seo::")
  ) {
    return { recommended_action: "PUBLISH_APP", is_major_pivot: false, confidence: "high", _source: "fast_path" };
  }

  if (t.startsWith("edit prompt::")) {
    return { recommended_action: "EDIT_APP", is_major_pivot: false, confidence: "high", _source: "fast_path" };
  }

  // ── Approve button ──
  if (lower === "approve app" || lower === "approve") {
    return { recommended_action: "REVIEW_SEO", is_major_pivot: false, confidence: "high", _source: "fast_path" };
  }

  // ── Edit App button ──
  if (lower === "edit app" || lower === "edit") {
    return { recommended_action: "EDIT_APP", is_major_pivot: false, confidence: "high", _source: "fast_path" };
  }

  // ── Publish / Save Draft ──
  if (lower === "publish to marketplace" || lower === "save draft") {
    return { recommended_action: "PUBLISH_APP", is_major_pivot: false, confidence: "high", _source: "fast_path" };
  }

  // ── Model selection chip ("Select model-id") ──
  if (/^select\s+\S/i.test(t) && !["select lean", "select recommended", "select full"].some(p => lower === p)) {
    return { recommended_action: "GENERATE_PREVIEW", is_major_pivot: false, confidence: "high", _source: "fast_path" };
  }

  // ── Budget chips ──
  const budgetMap = {
    "free models only (0 coins)": "free",
    "low (< 5 coins)": "low",
    "medium (5 - 20 coins)": "medium",
    "premium (> 20 coins)": "premium"
  };
  if (budgetMap[lower]) {
    return {
      recommended_action: "HANDLE_BUDGET",
      budget_tier: budgetMap[lower],
      is_major_pivot: false,
      confidence: "high",
      extracted_variables: { budget: budgetMap[lower] },
      _source: "fast_path"
    };
  }

  // ── App type chips ──
  const chipTypes = {
    text: "text", image: "image", audio: "audio", video: "video", vision: "vision",
    "text app": "text", "image app": "image", "audio app": "audio", "video app": "video", "vision app": "vision",
    "generate images or photos": "image", "image generator": "image",
    "create videos or animations": "video", "video creator": "video",
    "write text": "text", "writing tool": "text",
    "generate voice or music": "audio", "audio generator": "audio",
    "analyze or understand images": "vision", "image analyzer": "vision"
  };
  if (chipTypes[lower]) {
    return {
      recommended_action: "GATHER_REQUIREMENTS",
      app_type: chipTypes[lower],
      is_major_pivot: false,
      confidence: "high",
      _source: "fast_path"
    };
  }

  return null; // No fast-path match — call LLM
}

/* ─────────────────────────────────────────────────────────────────────────
   getAgenticDecision() — main export
   ───────────────────────────────────────────────────────────────────────── */

/**
 * Orchestration brain. Replaces getAgenticIntent() + step-based routing.
 *
 * @param {string} message - Raw user message
 * @param {object} session - Full session state object
 * @returns {Promise<object>} Decision object with recommended_action, extracted_variables, app_type, etc.
 */
export async function getAgenticDecision(message, session) {
  const text = String(message || "").trim();

  // ── 1. FAST-PATH: deterministic UI payloads ──
  const fast = tryFastPath(text, session);
  if (fast) {
    console.log(`[Orchestrator] Fast-path: ${fast.recommended_action}`);
    return fast;
  }

  // ── 2. LLM ORCHESTRATION CALL ──
  if (!groq) {
    console.warn("[Orchestrator] GROQ_API_KEY not set — falling back to regex dispatcher");
    return buildFallbackDecision(text, session);
  }

  try {
    const snapshot = buildSessionSnapshot(session);

    // Build lean conversation context (last 8 turns, 400 chars each)
    const historySlice = (session?.history || []).slice(-8).map(h => ({
      role: h.role === "agent" ? "assistant" : "user",
      content: typeof h.content === "string"
        ? h.content.slice(0, 400)
        : JSON.stringify(h.content).slice(0, 400)
    }));

    const messages = [
      { role: "system", content: ORCHESTRATOR_SYSTEM_PROMPT },
      {
        role: "system",
        content: `SESSION STATE SNAPSHOT:\n${JSON.stringify(snapshot, null, 2)}`
      },
      ...historySlice,
      { role: "user", content: text }
    ];

    const completion = await groq.chat.completions.create({
      model: "llama-3.1-8b-instant",
      messages,
      tools: [ORCHESTRATOR_TOOL],
      tool_choice: { type: "function", function: { name: "orchestrate_pipeline" } },
      max_tokens: 300,
      temperature: 0.1
    });

    const toolCall = completion.choices?.[0]?.message?.tool_calls?.[0];
    if (toolCall?.function?.arguments) {
      const parsed = JSON.parse(toolCall.function.arguments);

      // ── SAFEGUARD: prevent premature pipeline jumps ──
      let action = parsed.recommended_action || "GATHER_REQUIREMENTS";

      // If we don't have a purpose yet, never skip past GATHER_REQUIREMENTS
      if (!snapshot.hasPurpose && !["HANDLE_GREETING", "HANDLE_OFF_TOPIC", "HANDLE_VIOLATION", "HANDLE_GIBBERISH", "GATHER_REQUIREMENTS"].includes(action)) {
        action = "GATHER_REQUIREMENTS";
      }

      // If model not selected yet, never fire GENERATE_PREVIEW from free text
      if (!snapshot.modelSelected && action === "GENERATE_PREVIEW" && !/^select\s+/i.test(text)) {
        action = snapshot.formConfirmed ? "SHOW_MODEL_CARDS" : "GATHER_REQUIREMENTS";
      }

      const decision = {
        recommended_action: action,
        extracted_variables: parsed.extracted_variables || {},
        app_type: parsed.app_type || null,
        is_major_pivot: Boolean(parsed.is_major_pivot),
        budget_tier: parsed.budget_tier || null,
        confidence: parsed.confidence || "medium",
        _source: "llm"
      };

      console.log(
        `[Orchestrator] Action: ${decision.recommended_action} | AppType: ${decision.app_type} | ` +
        `Pivot: ${decision.is_major_pivot} | Budget: ${decision.budget_tier} | Confidence: ${decision.confidence}`
      );

      return decision;
    }

    console.warn("[Orchestrator] No tool call in LLM response — falling back");
    return buildFallbackDecision(text, session);

  } catch (err) {
    console.error("[Orchestrator] LLM call failed:", err.message);
    return buildFallbackDecision(text, session);
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   FALLBACK — Regex-based decision when LLM is unavailable
   Mirrors the action enum so the dispatcher is never blind.
   ───────────────────────────────────────────────────────────────────────── */
function buildFallbackDecision(message, session) {
  const msg = String(message || "").trim().toLowerCase();
  const snapshot = buildSessionSnapshot(session);

  let recommended_action = "GATHER_REQUIREMENTS";
  let app_type = null;
  let budget_tier = null;
  let is_major_pivot = false;
  const extracted_variables = {};

  // Greeting
  if (/^(hi|hello|hey|hy|hola|greetings)[\s!.]*$/i.test(msg)) {
    recommended_action = "HANDLE_GREETING";
  }
  // Off-topic / help (very short)
  else if (/^(help)[\s!.]*$/i.test(msg)) {
    recommended_action = "HANDLE_OFF_TOPIC";
  }
  // Policy
  else if (/\b(jailbreak|nsfw|nude|hack|exploit|bomb|weapon|illegal)\b/i.test(msg)) {
    recommended_action = "HANDLE_VIOLATION";
  }
  // Budget chip
  else if (/\b(free|low|medium|premium)\b/i.test(msg) && /\b(coin|budget|model)\b/i.test(msg)) {
    recommended_action = "HANDLE_BUDGET";
    const bm = msg.match(/\b(free|low|medium|premium)\b/i);
    budget_tier = bm ? bm[1].toLowerCase() : null;
    if (budget_tier) extracted_variables.budget = budget_tier;
  }
  // Approve
  else if (/\b(approve|approved|looks good|proceed|confirm|yes proceed)\b/i.test(msg)) {
    recommended_action = snapshot.previewApproved ? "PUBLISH_APP" : "REVIEW_SEO";
  }
  // Publish
  else if (/\b(publish|save draft|go live)\b/i.test(msg)) {
    recommended_action = "PUBLISH_APP";
  }
  // Change model
  else if (/\b(change|switch|different)\b.{0,20}\b(model|ai|engine)\b/i.test(msg)) {
    recommended_action = "CHANGE_MODEL";
  }
  // Major pivot
  else if (/\b(i want|build|create|make)\b.{2,50}\b(app|tool|generator)\b/i.test(msg) && snapshot.hasPurpose) {
    recommended_action = "PIVOT_APP";
    is_major_pivot = true;
  }
  // Edit
  else if (/\b(change|edit|tweak|update|modify|make it|add|remove)\b/i.test(msg) && snapshot.hasPurpose) {
    recommended_action = "EDIT_APP";
    extracted_variables.editInstruction = message;
  }
  // Form confirmed → show models
  else if (snapshot.formConfirmed && snapshot.budgetSet && !snapshot.modelSelected) {
    recommended_action = "SHOW_MODEL_CARDS";
  }
  // Triage complete → render form
  else if (snapshot.triageComplete && !snapshot.formConfirmed) {
    recommended_action = "RENDER_FORM";
  }

  // App type signals
  const typeSignals = {
    image: /\b(image|photo|picture|logo|poster|card|avatar|portrait|room design|banner|meme|flyer|sticker)\b/i,
    audio: /\b(audio|voice|podcast|tts|speech|narration|music|sound)\b/i,
    video: /\b(video|animation|animate|reel|cinematic|clip)\b/i,
    vision: /\b(detect|analyze image|scan|ocr|read from image)\b/i,
    text: /\b(text|blog|legal|recipe|email|story|script|plan|write|article)\b/i
  };
  for (const [type, regex] of Object.entries(typeSignals)) {
    if (regex.test(msg)) { app_type = type; break; }
  }

  console.log(`[Orchestrator:Fallback] Action: ${recommended_action} | Confidence: low`);

  return {
    recommended_action,
    extracted_variables,
    app_type,
    is_major_pivot,
    budget_tier,
    confidence: "low",
    _source: "fallback_regex"
  };
}

/* ─────────────────────────────────────────────────────────────────────────
   LEGACY SHIM — keeps any callers of getAgenticIntent() working
   Maps old action labels → new recommended_action tokens
   ───────────────────────────────────────────────────────────────────────── */
const ACTION_MAP = {
  start_app:       "GATHER_REQUIREMENTS",
  pivot_app:       "PIVOT_APP",
  edit_app:        "EDIT_APP",
  select_budget:   "HANDLE_BUDGET",
  select_model:    "CHANGE_MODEL",
  affirmation:     "GATHER_REQUIREMENTS",
  greeting:        "HANDLE_GREETING",
  off_topic:       "HANDLE_OFF_TOPIC",
  policy_violation:"HANDLE_VIOLATION",
  gibberish:       "HANDLE_GIBBERISH",
  answer_question: "GATHER_REQUIREMENTS",
  ui_action:       "PUBLISH_APP"
};

export async function getAgenticIntent(message, session) {
  const decision = await getAgenticDecision(message, session);
  // Return in the old shape so stepRouter's legacy branches still work
  return {
    action: Object.entries(ACTION_MAP).find(([, v]) => v === decision.recommended_action)?.[0] || "answer_question",
    recommended_action: decision.recommended_action,
    app_type: decision.app_type,
    budget_tier: decision.budget_tier,
    is_major_pivot: decision.is_major_pivot,
    edit_instruction: decision.extracted_variables?.editInstruction || null,
    extracted_details: decision.extracted_variables || {},
    confidence: decision.confidence,
    _source: decision._source,
    // Full decision always accessible
    _decision: decision
  };
}
