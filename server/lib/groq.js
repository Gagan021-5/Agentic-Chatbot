import Groq from "groq-sdk";
import OpenAI from "openai";
import { normalizeExtraction } from "./gemini.js";
import { LANGUAGE_MIRROR_DIRECTIVE } from "./languageDirective.js";

const groqApiKey = process.env.GROQ_API_KEY;

const GROQ_SYSTEM_PROMPT = `You are a strict data extraction engine for RentPrompts — a platform where users CREATE and PUBLISH AI-powered apps.

Users describe an app they want to build.
Your ONLY job: extract what they said. Never invent.
${LANGUAGE_MIRROR_DIRECTIVE}

APP TYPE RULES — read every word carefully:
- "image" app: generates images, photos, portraits, transforms photos, superhero filter, avatar maker, logo maker, any visual output
- "video" app: creates videos, animations, reels, cinematic clips, animates photos, talking avatars
- "text" app: generates written content — blogs, emails, captions, scripts, stories, reports, product descriptions, workout PLANS, meal PLANS, diet plans, study guides, itineraries, any document or written plan output
- "audio" app: voice, music, speech, podcast, sound effects, text to speech, transcription
- "vision" app: analyzes images, reads text from images, detects objects, medical image analysis

CRITICAL: A "planner" app (workout planner, meal planner, travel planner) = appType "text" BUT appPurpose must describe WHAT it plans, not just say "text generation"

ANTI-HALLUCINATION:
1. If user said nothing about target users → null
2. If user said nothing about budget → null
3. oneLineUnderstanding = rephrase ONLY what user said
4. suggestedReply = one warm question about the MOST important unknown detail. Always a question.
5. Never say "building my app" or repeat vague phrases
6. Never invent features not mentioned by user

Enterprise detection rules:
- enterpriseSignals = true if message contains ANY of: company, team, employees, scale, API, integrate, bulk, enterprise, SaaS, B2B, workflow, automate our, our company, our team, organization, department, staff
- userType = "enterprise" if enterpriseSignals is true and message mentions large scale or enterprise context
- userType = "business" if enterpriseSignals is true but no enterprise-specific language
- userType = "developer" if message mentions developers, API, SDK, code, or integration work
- userType = "normal" if message describes personal or creator use case with no business/developer signals
- userType = "unknown" if no clear signals detected

Return ONLY valid JSON. No markdown. No explanation.
{
  "appType": "text|image|audio|video|vision|null",
  "appPurpose": "describe what this app generates/does",
  "targetUsers": "string or null",
  "keyFeatures": [],
  "budget": "free|low|medium|high|ultra|null",
  "wantsImageInput": false,
  "detectedLanguage": "english",
  "userType": "enterprise|business|developer|normal|unknown",
  "enterpriseSignals": false,
  "userTone": "urgent|casual|formal|unsure",
  "confidence": {
    "appType": "HIGH|MEDIUM|LOW",
    "budget": "HIGH|MEDIUM|LOW"
  },
  "missingFields": [],
  "oneLineUnderstanding": "only rephrase what user said",
  "suggestedReply": "one warm follow-up question"
}`;

function hasRealValue(value) {
  return Boolean(value && !/^your_.+_here$/i.test(value));
}

const groq = hasRealValue(groqApiKey) ? new Groq({ apiKey: groqApiKey }) : null;

let openRouterClient = null;

function getOpenRouterClient() {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!hasRealValue(apiKey)) {
    throw new Error("OPENROUTER_API_KEY is not configured");
  }
  if (!openRouterClient) {
    openRouterClient = new OpenAI({
      baseURL: "https://openrouter.ai/api/v1",
      apiKey,
      defaultHeaders: {
        "HTTP-Referer": "http://localhost:5173",
        "X-Title": "RentPrompts Agent Demo"
      }
    });
  }
  return openRouterClient;
}

function isRateLimitError(error) {
  return error && (error.status === 429 || error.statusCode === 429 || error.code === 429 || (error.message && error.message.includes('429')));
}

function normalizeLanguageHint(languageHint) {
  const normalized = String(languageHint || "").toLowerCase();
  if (normalized.includes("hinglish")) return "Hinglish";
  if (normalized.includes("hindi")) return "Hindi";
  return "English";
}

function sanitizeStringList(list, minLen, maxLen, fallback) {
  const cleaned = Array.isArray(list)
    ? list
      .map((item) => String(item || "").trim())
      .filter(Boolean)
      .filter((item, idx, arr) => arr.findIndex((x) => x.toLowerCase() === item.toLowerCase()) === idx)
      .slice(0, maxLen)
    : [];

  if (cleaned.length >= minLen) return cleaned;
  return fallback.slice(0, maxLen);
}

function sanitizeVariableObjects(list, minLen, maxLen, fallback) {
  const normalized = Array.isArray(list)
    ? list
      .map((item) => {
        if (typeof item === "string") {
          return { name: item.trim(), placeholder: "Enter details...", test_value: "" };
        }
        if (!item || typeof item !== "object") return null;
        return {
          name: String(item.name || "").trim(),
          placeholder: String(item.placeholder || "Enter details...").trim(),
          test_value: String(item.test_value || "").trim()
        };
      })
      .filter((item) => item && item.name)
      .filter((item, idx, arr) => arr.findIndex((x) => x.name.toLowerCase() === item.name.toLowerCase()) === idx)
      .slice(0, maxLen)
    : [];

  if (normalized.length >= minLen) return normalized;
  return fallback.slice(0, maxLen);
}

function buildDynamicContextFallback(appType, appPurpose, languageHint) {
  const lang = normalizeLanguageHint(languageHint);
  const isHindi = lang === "Hindi";
  const isHinglish = lang === "Hinglish";
  const p = String(appPurpose || "").toLowerCase();

  if (isHindi) {
    return {
      options: ["उपयोगकर्ता के लिए पर्सनल परिणाम", "स्पष्ट और संरचित आउटपुट", "तेज और विश्वसनीय प्रतिक्रिया", "कस्टम इनपुट आधारित जनरेशन"],
      variables: [
        { name: "मुख्य इनपुट", placeholder: "अपनी आवश्यकता लिखें...", test_value: "फसल की बीमारी की पहचान" },
        { name: "संदर्भ", placeholder: "कॉन्टेक्स्ट या पृष्ठभूमि जोड़ें...", test_value: "महाराष्ट्र, खरीफ मौसम" },
        { name: "पसंदीदा स्टाइल", placeholder: "जैसे: प्रोफेशनल, फ्रेंडली...", test_value: "सरल हिंदी में" }
      ]
    };
  }
  if (isHinglish) {
    return {
      options: ["Personalized output for user", "Structured and clear result", "Fast and reliable response", "Custom input based generation"],
      variables: [
        { name: "Main input", placeholder: "Aap kya generate karna chahte ho?", test_value: "professional resume" },
        { name: "Context", placeholder: "Background ya extra details", test_value: "5 years experience in software" },
        { name: "Preferred style", placeholder: "Jaise: Formal, Friendly", test_value: "Professional" }
      ]
    };
  }

  // ── Smart fallbacks based on both appType AND appPurpose keywords ──
  const typeSpecific = {
    // IMAGE type — differentiate by sub-purpose
    image: (() => {
      if (p.includes('background remov') || p.includes('bg remov'))
        return { options: ["Precise edge detection", "Transparent PNG output", "Batch processing support", "Custom background replacement"], variables: [
          { name: "image_source", placeholder: "Upload or paste image URL", test_value: "portrait photo of a person" },
          { name: "background_type", placeholder: "transparent / white / custom color / new background", test_value: "transparent" },
          { name: "output_quality", placeholder: "standard / high / ultra", test_value: "high" }
        ]};
      if (p.includes('interior') || p.includes('room') || p.includes('design'))
        return { options: ["Style consistency", "Photorealistic rendering", "Multi-room support", "Furniture detection"], variables: [
          { name: "room_type", placeholder: "Living room, bedroom, office...", test_value: "living room" },
          { name: "design_palette", placeholder: "Modern minimalist, Boho, Industrial...", test_value: "modern minimalist" },
          { name: "square_feet", placeholder: "e.g., 250", test_value: "300" },
          { name: "objects", placeholder: "Sofa, table, wardrobe...", test_value: "sofa, coffee table, bookshelf" }
        ]};
      if (p.includes('logo') || p.includes('brand'))
        return { options: ["Brand color enforcement", "Vector-ready output", "Multiple variants", "Style lock"], variables: [
          { name: "brand_name", placeholder: "Your brand or company name", test_value: "TechNova" },
          { name: "industry", placeholder: "Tech, food, fashion...", test_value: "technology" },
          { name: "style", placeholder: "Minimal, bold, playful, corporate...", test_value: "minimal and modern" },
          { name: "colors", placeholder: "Preferred brand colors", test_value: "blue and white" }
        ]};
      // generic image
      return { options: ["Style consistency", "Composition control", "High-detail output", "Negative prompt support"], variables: [
        { name: "subject", placeholder: "What should appear in the image?", test_value: "a futuristic city at night" },
        { name: "style", placeholder: "Anime, realistic, cinematic, watercolor...", test_value: "cinematic realism" },
        { name: "mood", placeholder: "Colors, lighting, atmosphere", test_value: "dark, moody, neon lights" }
      ]};
    })(),
    text: (() => {
      if (p.includes('legal') || p.includes('law') || p.includes('advocate'))
        return { options: ["Indian law coverage (BNS/IPC)", "Section-specific citation", "Layman explanation mode", "Case analysis"], variables: [
          { name: "legal_query", placeholder: "Describe your legal situation or question", test_value: "My landlord is refusing to return my security deposit" },
          { name: "jurisdiction", placeholder: "State or country", test_value: "Maharashtra, India" },
          { name: "party_role", placeholder: "Are you the plaintiff or defendant?", test_value: "Plaintiff (tenant)" }
        ]};
      if (p.includes('lesson') || p.includes('teach') || p.includes('educat') || p.includes('quiz'))
        return { options: ["Grade-level adaptation", "Interactive quiz generation", "Curriculum alignment", "Multiple learning styles"], variables: [
          { name: "subject", placeholder: "e.g., Mathematics, Science, History", test_value: "Grade 8 Mathematics" },
          { name: "topic", placeholder: "Specific chapter or concept", test_value: "Pythagoras theorem" },
          { name: "grade_level", placeholder: "e.g., Grade 8, Class 10, University", test_value: "Grade 8" },
          { name: "learning_objective", placeholder: "What should students learn?", test_value: "Understand and apply the Pythagorean theorem" }
        ]};
      if (p.includes('farm') || p.includes('crop') || p.includes('agricultur'))
        return { options: ["Region-specific advice", "Season-aware recommendations", "Pest & disease guidance", "Multi-language support"], variables: [
          { name: "crop_type", placeholder: "e.g., wheat, rice, tomato", test_value: "wheat" },
          { name: "problem", placeholder: "Describe the issue you're facing", test_value: "yellow spots on leaves, plants drying up" },
          { name: "region", placeholder: "State and season", test_value: "Punjab, Rabi season" }
        ]};
      if (p.includes('blog') || p.includes('content') || p.includes('article') || p.includes('seo'))
        return { options: ["SEO optimization", "Tone control", "Keyword integration", "CTA generation"], variables: [
          { name: "topic", placeholder: "What is the blog about?", test_value: "10 benefits of daily meditation" },
          { name: "target_audience", placeholder: "Who is reading this?", test_value: "working professionals aged 25-40" },
          { name: "tone", placeholder: "Professional, conversational, motivational...", test_value: "friendly and informative" },
          { name: "word_count", placeholder: "e.g., 800", test_value: "800" }
        ]};
      if (p.includes('dispatch') || p.includes('rescue') || p.includes('emergency') || p.includes('animal') || p.includes('ngo') || p.includes('urban'))
        return { options: ["Real-time location tracking", "Priority triage scoring", "Multi-responder coordination", "Case status updates"], variables: [
          { name: "incident_type", placeholder: "Animal rescue, accident, fire, flood...", test_value: "injured stray dog" },
          { name: "location", placeholder: "Street address or landmark", test_value: "MG Road, Bengaluru" },
          { name: "urgency_level", placeholder: "Critical / High / Medium / Low", test_value: "High" },
          { name: "injury_description", placeholder: "Describe the situation", test_value: "Animal is bleeding, unable to move" }
        ]};
      if (p.includes('factory') || p.includes('safety') || p.includes('industrial') || p.includes('hazard') || p.includes('manufactur') || p.includes('report'))
        return { options: ["Incident auto-categorization", "Shift-based logging", "Machine-specific tracking", "Regulatory compliance output"], variables: [
          { name: "machine_id", placeholder: "Machine ID or name (e.g., CNC-04)", test_value: "CNC-04" },
          { name: "hazard_type", placeholder: "e.g., electrical, chemical, mechanical", test_value: "mechanical pinch point" },
          { name: "shift_time", placeholder: "Morning / Afternoon / Night", test_value: "Morning shift" },
          { name: "incident_description", placeholder: "What happened?", test_value: "Worker hand caught in conveyor belt" }
        ]};
      if (p.includes('workout') || p.includes('fitness') || p.includes('exercise') || p.includes('gym') || p.includes('health') || p.includes('diet') || p.includes('meal'))
        return { options: ["Goal-based personalization", "Progressive overload tracking", "Rest day optimization", "Nutrition sync"], variables: [
          { name: "fitness_goal", placeholder: "Weight loss, muscle gain, endurance...", test_value: "muscle gain" },
          { name: "current_level", placeholder: "Beginner / Intermediate / Advanced", test_value: "Intermediate" },
          { name: "duration_weeks", placeholder: "How many weeks?", test_value: "12" },
          { name: "available_equipment", placeholder: "Gym, home, no equipment...", test_value: "Full gym access" }
        ]};
      if (p.includes('food') || p.includes('nutrit') || p.includes('restaurant') || p.includes('recipe') || p.includes('menu'))
        return { options: ["Allergen detection", "Calorie calculation", "Dietary compliance check", "Ingredient substitution"], variables: [
          { name: "food_item", placeholder: "Dish or food product name", test_value: "Butter Chicken with Naan" },
          { name: "dietary_restrictions", placeholder: "Vegan, gluten-free, diabetic...", test_value: "Diabetic-friendly" },
          { name: "serving_size", placeholder: "e.g., 1 cup, 200g, 1 plate", test_value: "1 plate" }
        ]};
      // generic text
      return { options: ["Tone control", "Structured output", "Goal-focused generation", "Context awareness"], variables: [
        { name: "main_input", placeholder: "Describe what you want generated", test_value: "a professional email requesting a meeting" },
        { name: "audience", placeholder: "Who is this for?", test_value: "HR manager at a tech company" },
        { name: "tone", placeholder: "Professional, friendly, formal...", test_value: "professional" }
      ]};
    })(),
    video: { options: ["Scene pacing control", "Shot/style consistency", "Platform-ready output", "Voiceover sync"], variables: [
      { name: "concept", placeholder: "What story or scene to create?", test_value: "a 15-second ad for a new coffee brand" },
      { name: "visual_style", placeholder: "Cinematic, vlog, animation, ad...", test_value: "cinematic with warm tones" },
      { name: "platform", placeholder: "YouTube, Instagram Reels, TikTok...", test_value: "Instagram Reels" }
    ]},
    audio: { options: ["Voice style selection", "Language support", "Pacing control", "Emotion control"], variables: [
      { name: "script", placeholder: "Paste the text to convert to speech", test_value: "Welcome to RentPrompts, the AI app marketplace." },
      { name: "voice_style", placeholder: "Male/Female, energetic/calm, accent...", test_value: "female, warm and professional" },
      { name: "language", placeholder: "e.g., English, Hindi, Spanish", test_value: "English" }
    ]},
    vision: { options: ["Accurate extraction", "Structured JSON response", "Confidence scoring", "Use-case specific analysis"], variables: [
      { name: "task_type", placeholder: "OCR, object detection, image QA, disease detection...", test_value: "OCR - extract all text from invoice" },
      { name: "output_format", placeholder: "JSON, plain text, bullet points...", test_value: "structured JSON" },
      { name: "special_instructions", placeholder: "Any focus areas or constraints?", test_value: "focus on line items and total amount" }
    ]}
  };
  return typeSpecific[appType] || typeSpecific.text;
}

function parseDynamicContextPayload(rawContent, appType, appPurpose, languageHint) {
  const fallback = buildDynamicContextFallback(appType, appPurpose, languageHint);
  try {
    const parsed = JSON.parse(String(rawContent || "{}").replace(/```json/gi, "").replace(/```/g, "").trim());
    return {
      options: sanitizeStringList(parsed.options, 4, 4, fallback.options),
      variables: sanitizeVariableObjects(parsed.variables, 3, 4, fallback.variables)
    };
  } catch {
    return fallback;
  }
}

async function extractRequirements(message, history) {
  // If Groq is not configured, go straight to OpenRouter fallback
  if (!groq) {
    return extractWithOpenRouterFallback(message, history);
  }

  try {
    const completion = await groq.chat.completions.create({
      model: "llama-3.3-70b-versatile",
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content: GROQ_SYSTEM_PROMPT
        },
        {
          role: "user",
          content: JSON.stringify({
            message,
            history: Array.isArray(history) ? history.slice(-8) : []
          })
        }
      ]
    });

    const content = completion.choices && completion.choices[0] && completion.choices[0].message ? completion.choices[0].message.content : "{}";
    return normalizeExtraction(JSON.parse(content), message);
  } catch (error) {
    if (isRateLimitError(error)) {
      console.log("[Groq] 429 hit, falling back to OpenRouter...");
      return extractWithOpenRouterFallback(message, history);
    }

    console.error("Groq extraction error, falling back to OpenRouter:", error.message);
    return extractWithOpenRouterFallback(message, history);
  }
}

async function generateDynamicContext({ appType, appPurpose, languageHint }) {
  const safeType = ["text", "image", "video", "audio", "vision"].includes(appType) ? appType : "text";
  const safePurpose = String(appPurpose || "").trim() || "general assistant app";
  const safeLang = normalizeLanguageHint(languageHint);
  const systemPrompt = `You are a strict JSON generator.
Generate compact, practical setup suggestions for an AI app idea.
${LANGUAGE_MIRROR_DIRECTIVE}
When generating 'variables', you MUST include a 'placeholder' key.
- If the variable name contains 'Date', placeholder MUST be 'DD/MM/YYYY'.
- If the variable name contains 'Time', placeholder MUST be 'HH:MM AM/PM'.
- If the variable name contains 'Place' or 'Location', placeholder MUST be 'City, Country'.
- For everything else, use a relevant example.
NEVER use 'Enter details...' as a placeholder for date, time, or location fields.
Output must be strict JSON with this exact shape:
{"options":["4 concise feature options"],"variables":[{"name":"Date of Birth","placeholder":"DD/MM/YYYY"},{"name":"Location","placeholder":"City, Country"}]}
No markdown. No prose.`;
  const userPrompt = `The user wants to build a ${safeType} app for: ${safePurpose}.
Language mode: ${safeLang}.
Generate 4 highly relevant specific features and 3-4 input variables needed for the app.
For each variable include name and helpful placeholder.`;

  try {
    if (groq) {
      const completion = await groq.chat.completions.create({
        model: "llama-3.3-70b-versatile",
        response_format: { type: "json_object" },
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt }
        ]
      });
      const content = completion.choices?.[0]?.message?.content || "{}";
      return parseDynamicContextPayload(content, safeType, safePurpose, safeLang);
    }
  } catch (error) {
    if (!isRateLimitError(error)) {
      console.error("Groq dynamic context error, falling back:", error.message);
    }
  }

  try {
    const fallback = await getOpenRouterClient().chat.completions.create({
      model: "meta-llama/llama-3.3-70b-instruct",
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt }
      ]
    });
    const content = fallback.choices?.[0]?.message?.content || "{}";
    return parseDynamicContextPayload(content, safeType, safePurpose, safeLang);
  } catch (error) {
    console.error("OpenRouter dynamic context fallback failed:", error.message);
    return buildDynamicContextFallback(safeType, safePurpose, safeLang);
  }
}

async function extractWithOpenRouterFallback(message, history) {
  try {
    const fallback = await getOpenRouterClient().chat.completions.create({
      model: 'meta-llama/llama-3.3-70b-instruct',
      messages: [
        { role: 'system', content: GROQ_SYSTEM_PROMPT },
        { role: 'user', content: JSON.stringify({ message, history: Array.isArray(history) ? history.slice(-8) : [] }) }
      ],
      response_format: { type: 'json_object' },
    });

    const raw = fallback.choices[0].message.content;
    const parsed = JSON.parse((raw || '{}').replace(/```json/gi, '').replace(/```/g, '').trim());
    return normalizeExtraction(parsed, message);
  } catch (fallbackError) {
    console.error("OpenRouter fallback also failed:", fallbackError.message);
    return normalizeExtraction(null, message);
  }
}

/* ────────────────────────────────────────────
   AGENTIC TRIAGE — evaluate specificity before generating form
   ──────────────────────────────────────────── */
const TRIAGE_INSTRUCTION = `
You are an expert AI Product Architect conducting a thorough requirements interview before building an AI-powered app.
Your job: have a REAL conversation to deeply understand what the app does, who uses it, what inputs it needs, and what output it produces — BEFORE declaring you're ready to build.

Behave like a senior product manager doing a requirements discovery call. Ask one clear, specific question per turn.
Do NOT rush to "ready". It is better to ask one more question than to build the wrong app.

📌 FUNDAMENTAL FRAMING RULE:
The person you are talking to IS the real user — they are describing an app for their OWN use case or problem.
Always address them directly: "What do YOU need?", "What would YOU provide?", "What do YOU want the output to look like?"
NEVER ask about "the users of this app" as if they are a separate abstract group.
Treat this like helping a friend describe their idea — personal, direct, and specific to their situation.

⚠️ ABSOLUTE RULES:
1. PURPOSE ANCHOR: The original app idea is LOCKED. User answers narrow it — they do NOT change what the app is.
   Example: original = "crop disease detection" → all questions must be about crop disease. Never drift.
2. APP TYPE LOCK: App type (image/text/audio/video/vision) is already set. NEVER re-ask it.
3. Ask questions ONE AT A TIME. Never ask two questions in one message.
4. NEVER ask: "what variables do you need?" — that is YOUR job to figure out.
5. NEVER declare "ready" until you have covered the 5 REQUIRED DIMENSIONS below.

🟢 YES-AFFIRMATION RULE (HIGHEST PRIORITY):
If the user's last message is a short affirmation — "yes", "sure", "ok", "yep", "correct", "sounds good", "exactly", "perfect", "that's right", "go ahead", "proceed" — you MUST immediately return status "ready".
Do NOT ask another question after a "yes". Accept it as confirmation of everything discussed so far.
This overrides ALL other rules including the 5-dimension requirement.

✅ THE 5 REQUIRED DIMENSIONS (must cover ALL before "ready"):
D1. SPECIFIC USE CASE: What exact task will users run this app for? (not just the domain, the specific action — draft, explain, analyze, detect, generate?)
D2. DOMAIN SCOPE: What specific subject area, jurisdiction, or specialization should the app cover? (e.g., Indian criminal law, Grade 8 math, wheat disease in Punjab, Instagram captions for fitness brands)
D3. KEY INPUT FACTORS: What specific information does the user provide each time they run the app? (the actual data they type or upload)
D4. DESIRED OUTPUT FORMAT: What does the output look like? (step-by-step plan, plain-language explanation, rendered image, JSON, bullet list, ready-to-post text?)
D5. DOMAIN-SPECIFIC RULES: Any constraints, regulations, tone requirements, or edge cases the app must respect?

⛔ NEVER ASK:
- "What skill level are the users?" — infer it from context (advocate app → non-lawyers need plain language)
- "Who will use this?" in a demographic sense — not useful for building prompts
- "What is the target audience?" as a standalone question — already implied by the app type
- Two questions in one message
- Anything about model, budget, or coins — that comes later

STRATEGY:
- Check the conversation history and the "Already answered" context. Identify which of D1-D5 are MISSING.
- Ask about the HIGHEST PRIORITY missing dimension first.
- If D1 is missing: ask them what specific task they want the app to do for them (with 2-3 concrete examples).
- If D2 is missing: ask what specific domain, subject area, or jurisdiction they need (e.g., "Indian criminal law or civil disputes?", "wheat/rice/cotton?").
- If D3 is missing: ask what details THEY would type in each time they use the app.
- If D4 is missing: ask what they want the output to look like for them.
- If D5 is missing: ask about any special rules, language, or constraints important to their situation.

DOMAIN INTELLIGENCE — Use these to ask SMART, SPECIFIC questions:
- TEACHER/EDUCATOR: D1=lesson type(quiz/plan/explanation), D2=what subject+grade they teach, D3=topic+learning objective+duration, D4=lesson plan format, D5=curriculum board
- FARMER/AGRICULTURAL: D1=specific problem(disease/yield/pest/irrigation), D2=what crop+region they work with, D3=crop+symptoms+season+location, D4=recommendation+steps, D5=local language+seasonal constraints
- LAWYER/LEGAL: D1=legal task(explain concept/draft/analyze/advise), D2=what area of law+jurisdiction they deal with, D3=what the user would describe in their own words (incident_description, their_role, state/city) — NEVER ask for section numbers since non-experts don't know them, D4=plain-language explanation+relevant laws cited by AI, D5=Indian law(BNS/IPC/CrPC)+state
- CONTENT CREATOR: D1=content type(blog/caption/script/reel), D2=what niche+platform they post on, D3=topic+tone+word count+keywords, D4=ready-to-post format, D5=brand voice+SEO rules
- INDUSTRIAL: D1=inspection task(defect/quality/measurement), D2=what product+industry they work in, D3=product details+defect type+criteria, D4=pass/fail report, D5=tolerance standards
- BACKGROUND REMOVAL (image): D1=use case(e-commerce/portrait/social), D2=what type of images they work with, D3=source_image+output_background, D4=transparent PNG/white/custom, D5=batch needs+edge cases
- INTERIOR DESIGN (image): D1=design task(visualize/suggest/render), D2=what type of space they have, D3=room_type+square_feet+design_palette+objects, D4=rendered image+description, D5=budget constraints+style preferences
- HEALTH/FITNESS: D1=specific task(workout plan/diet/symptom check), D2=their fitness level+goal, D3=goal+current status+constraints, D4=structured plan, D5=safety disclaimers+medical limits

WHEN TO RETURN "ready" (ALL must be true):
✓ At least 4 of the 5 dimensions (D1-D5) have clear answers from the conversation history
✓ You know the specific inputs the end-user will provide each run
✓ You know what the output looks like
✓ You can confidently generate 3-4 precise, domain-specific input variables

NEVER return "ready" if:
✗ You only know the general domain (e.g., "legal app") but not the specific task they want
✗ You don't know what format the output should be in

VARIABLE QUALITY (only in "ready" response):
- Exactly 3-4 variables, snake_case, domain-specific names
- Each has: name, a descriptive placeholder, a realistic test_value
- Variables = the EXACT data the end-user types in when running the app
- ⚠️ USER PERSPECTIVE RULE: NEVER create variables requiring domain expertise the end-user doesn't have
  BAD: section_number (user doesn't know IPC sections), diagnosis_code, article_citation, statute_reference
  GOOD: incident_description ("what happened"), situation_type ("type of dispute"), location, their_role
  For LEGAL apps: user describes what happened → AI identifies the law. Never ask user to provide the law.
  For MEDICAL apps: user describes symptoms → AI identifies condition. Never ask for diagnosis codes.
- For image/vision apps requiring file upload: include {"name": "source_image", "placeholder": "Upload your image", "test_value": "photo of product"}
- options = 4 specific, compelling features of THIS exact app (not generic)

Return STRICTLY as JSON (no markdown, no explanation outside JSON):
{
  "status": "needs_context" | "ready",
  "domain_identified": "Specific domain, e.g.: Agricultural - Crop Disease Detection",
  "dimensions_covered": ["D1", "D3"],
  "question": "ONLY if needs_context: one focused question targeting the most important missing dimension. Include 2-3 inline examples to guide the user — write examples as plain text inside parentheses like (e.g., episode title, speaker name, script notes). NEVER wrap examples in single quotes or double quotes.",
  "form": {
    "options": ["4 specific features of this exact app"],
    "variables": [{"name": "snake_case_name", "placeholder": "specific helpful hint", "test_value": "realistic domain example"}]
  }
}
`;

const ALLOWED_TRIAGE_APP_FORMATS = ["text", "image", "audio", "video", "vision"];

function normalizeTriageAppFormat(raw, fallbackType) {
  const fb = ALLOWED_TRIAGE_APP_FORMATS.includes(fallbackType) ? fallbackType : "text";
  const v = String(raw == null ? "" : raw).trim().toLowerCase();
  if (ALLOWED_TRIAGE_APP_FORMATS.includes(v)) return v;
  if (String(raw || "").trim()) console.warn("[Triage] Invalid or unknown app_format:", raw, "→ using", fb);
  return fb;
}

function mapHistoryToTriageMessages(conversationHistory) {
  // Last 8 turns so chip / format clicks in the previous user turn stay visible to the model
  const recent = Array.isArray(conversationHistory) ? conversationHistory.slice(-8) : [];
  return recent
    .map((m) => {
      if (!m) return null;
      const raw = String(m.role || "").toLowerCase();
      const role = raw === "user" ? "user" : "assistant";
      const content = String(m.content ?? m.text ?? "").trim();
      if (!content) return null;
      return { role, content };
    })
    .filter(Boolean);
}

function parseTriageResponse(rawContent, formatFallback, appPurpose, languageHint) {
  const fallbackType = ALLOWED_TRIAGE_APP_FORMATS.includes(formatFallback) ? formatFallback : "text";
  const safePurpose = String(appPurpose || "").trim();

  const readyShape = (domain, appFormat, form) => ({
    status: "ready",
    domain,
    question: null,
    app_format: appFormat,
    form
  });

  try {
    const cleaned = String(rawContent || "{}").replace(/```json/gi, "").replace(/```/g, "").trim();
    const parsed = JSON.parse(cleaned);

    if (!parsed || !parsed.status) {
      return readyShape(null, fallbackType, buildDynamicContextFallback(fallbackType, safePurpose, languageHint));
    }

    const domain =
      String(parsed.domain_identified || parsed.domain || "").trim() || null;

    if (parsed.status === "needs_context") {
      const question = String(parsed.question || "").trim();
      if (!question || question.length < 10) {
        const fbForm = buildDynamicContextFallback(fallbackType, safePurpose, languageHint);
        return readyShape(domain, fallbackType, fbForm);
      }
      return {
        status: "needs_context",
        domain,
        question,
        form: null,
        app_format: null
      };
    }

    if (parsed.status !== "ready") {
      return readyShape(domain, fallbackType, buildDynamicContextFallback(fallbackType, safePurpose, languageHint));
    }

    const fallbackForm = buildDynamicContextFallback(fallbackType, safePurpose, languageHint);
    const form = parsed.form && typeof parsed.form === "object" ? parsed.form : {};
    return readyShape(domain, fallbackType, {
      options: sanitizeStringList(form.options, 4, 4, fallbackForm.options),
      variables: sanitizeVariableObjects(form.variables, 3, 4, fallbackForm.variables)
    });
  } catch {
    return readyShape(null, fallbackType, buildDynamicContextFallback(fallbackType, languageHint));
  }
}

async function triageDynamicContext({ appType, appPurpose, languageHint, conversationHistory, deepAnswers }) {
  const committed =
    appType != null &&
    String(appType).trim() &&
    ALLOWED_TRIAGE_APP_FORMATS.includes(String(appType).trim().toLowerCase())
      ? String(appType).trim().toLowerCase()
      : null;
  const formatFallback = committed || "text";
  const safePurpose = String(appPurpose || "").trim() || "general assistant app";
  const safeLang = normalizeLanguageHint(languageHint);

  // Summarize already-captured deepAnswers so triage doesn't re-ask them
  const answeredContext = deepAnswers && Object.keys(deepAnswers).length > 0
    ? `\nAlready answered by user: ${JSON.stringify(deepAnswers)}`
    : "";

  const userTaskPrompt = `ORIGINAL APP IDEA (LOCKED): "${safePurpose}"
App type is LOCKED to: "${formatFallback}" — do NOT ask the user to choose an app type.
Language mode: ${safeLang}.${answeredContext}

Look at the conversation history. If it already answers the key domain questions (use case, vertical, output format), return "ready" with domain-appropriate variables immediately.
If still unclear on ONE key dimension, ask exactly ONE focused question — suggest 2-3 specific examples inline to guide the user, formatted as plain text inside parentheses like (e.g., example one, example two) — NEVER use single quotes or double quotes around example phrases.
Do NOT ask generic questions. Do NOT re-ask what has already been answered above.`;

  const triageMessages = [
    { role: "system", content: TRIAGE_INSTRUCTION },
    ...mapHistoryToTriageMessages(conversationHistory),
    { role: "user", content: userTaskPrompt }
  ];

  // Try Groq first
  try {
    if (groq) {
      const completion = await groq.chat.completions.create({
        model: "llama-3.3-70b-versatile",
        response_format: { type: "json_object" },
        messages: triageMessages
      });
      const content = completion.choices?.[0]?.message?.content || "{}";
      return parseTriageResponse(content, formatFallback, safePurpose, safeLang);
    }
  } catch (error) {
    if (!isRateLimitError(error)) {
      console.error("Groq triage error, falling back:", error.message);
    }
  }

  // OpenRouter fallback
  try {
    const fallback = await getOpenRouterClient().chat.completions.create({
      model: "meta-llama/llama-3.3-70b-instruct",
      response_format: { type: "json_object" },
      messages: triageMessages
    });
    const content = fallback.choices?.[0]?.message?.content || "{}";
    return parseTriageResponse(content, formatFallback, safePurpose, safeLang);
  } catch (error) {
    console.error("OpenRouter triage fallback failed:", error.message);
    return {
      status: "ready",
      domain: null,
      question: null,
      app_format: formatFallback,
      form: buildDynamicContextFallback(formatFallback, safePurpose, safeLang)
    };
  }
}

export { extractRequirements, generateDynamicContext, triageDynamicContext };
