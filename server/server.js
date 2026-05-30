import "dotenv/config";
import cors from "cors";
import express from "express";
import Groq from "groq-sdk";
import { createSession, getSession, saveSession, deleteSession } from "./lib/redis.js";
import { formatUserPayloadForHistory } from "./lib/formatUserHistoryDisplay.js";
import { route } from "./lib/stepRouter.js";
import { runPromptTest } from "./lib/gemini.js";

const app = express();
const PORT = Number(process.env.PORT) || 3001;

const groq = process.env.GROQ_API_KEY ? new Groq({ apiKey: process.env.GROQ_API_KEY }) : null;

const POLLINATIONS_PREVIEW_SIZE = 768;

/**
 * Image preview prompts used to lead with a truncated system prompt ("studio lighting", "3D render"),
 * which drowned out user fields like backgroundStyle / colorScheme. Put user-driven constraints first.
 */
function buildUserDrivenVisualClauses(variables) {
  const entries = Object.entries(variables || {}).filter(([, v]) => v != null && String(v).trim() !== "");
  if (!entries.length) return "";

  const parts = [];

  const bg = entries.find(([k]) =>
    /background|backdrop|environment|setting|scene|location|surround/i.test(k)
  );
  if (bg) {
    const val = String(bg[1]).trim();
    parts.push(
      `The visible background and environment must clearly show: ${val}. Do not replace this with a plain solid studio backdrop, empty gradient, or flat void unless the user explicitly asked for a minimal or solid background.`
    );
  }

  const col = entries.find(([k]) => /color|palette|scheme|hue|tint|tone/i.test(k));
  if (col) {
    const val = String(col[1]).trim();
    parts.push(
      `Color direction: "${val}" should read as the dominant color story — saturated, vivid, and clearly visible on the subject and in the scene (not pale gray, not mostly white/off-white unless the user asked for that).`
    );
  }

  const subject = entries.find(
    ([k]) =>
      /shape|subject|object|form|creature|motif|theme|type|design/i.test(k) &&
      !/background|backdrop/i.test(k)
  );
  if (subject) {
    parts.push(`Primary subject or form to feature: ${String(subject[1]).trim()}.`);
  }

  const varsCompact = entries.map(([k, v]) => `${k}: ${String(v).trim()}`).join("; ");
  parts.push(`Honor every user field together in one coherent image: ${varsCompact}.`);

  return parts.join(" ");
}

function buildPollinationsImageAppPrompt(systemPrompt, variables) {
  const userBlock = buildUserDrivenVisualClauses(variables);
  const styleHint = String(systemPrompt || "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 240);

  const core = [
    userBlock,
    styleHint ? `Art direction (secondary to user fields above): ${styleHint}` : ""
  ]
    .filter(Boolean)
    .join(" ");

  return core.trim() || "High quality detailed creative render.";
}

function truncatePollinationsPrompt(text) {
  let t = String(text || "").trim();

  // CRITICAL FIX: Strip out markdown, asterisks, brackets, and quotes that cause 500 errors in Pollinations' URL path
  t = t.replace(/[^a-zA-Z0-9\s,.-]/g, " ").replace(/\s+/g, " ").trim();

  if (!t.length) return "High quality detailed illustration, professional lighting.";

  // Reduce max length from 640 to 350. Pollinations often throws 500s on very long URLs.
  if (t.length <= 350) return t;
  return t.slice(0, 350).trimEnd();
}

function previewImageUnavailableDataUrl() {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512"><rect fill="#1a1525" width="512" height="512"/><text x="256" y="248" fill="#a77bf3" font-family="system-ui,sans-serif" font-size="18" font-weight="600" text-anchor="middle">Could not load preview image</text><text x="256" y="278" fill="#9ca3af" font-family="system-ui,sans-serif" font-size="13" text-anchor="middle">Pollinations may be busy — try Run again</text></svg>`;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

async function fetchPollinationsPreviewDataUrl(plainPrompt) {
  const truncated = truncatePollinationsPrompt(plainPrompt);
  const imageUrl = `https://image.pollinations.ai/prompt/${encodeURIComponent(
    truncated
  )}?width=${POLLINATIONS_PREVIEW_SIZE}&height=${POLLINATIONS_PREVIEW_SIZE}&nologo=true`;

  const imgRes = await fetch(imageUrl, {
    headers: {
      Accept: "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
      "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    },
    signal: AbortSignal.timeout(120000)
  });

  if (!imgRes.ok) {
    throw new Error(`Pollinations HTTP ${imgRes.status}`);
  }

  const mime = imgRes.headers.get("content-type") || "image/jpeg";
  const buf = Buffer.from(await imgRes.arrayBuffer());
  if (!mime.startsWith("image/") || buf.length < 800) {
    throw new Error(`Pollinations returned non-image or empty body (${mime}, ${buf.length}b)`);
  }

  return `data:${mime};base64,${buf.toString("base64")}`;
}

async function fetchPollinationsWithFallback(primaryPrompt, fallbackPrompt) {
  try {
    return await fetchPollinationsPreviewDataUrl(primaryPrompt);
  } catch (err) {
    console.warn("[test-preview] Pollinations primary failed:", err.message);
    const fb = String(fallbackPrompt || "").trim();

    try {
      if (fb && fb !== String(primaryPrompt || "").trim()) {
        return await fetchPollinationsPreviewDataUrl(fb);
      }
      throw err;
    } catch (fallbackErr) {
      console.warn("[test-preview] Pollinations secondary failed:", fallbackErr.message);
      // ULTIMATE SAFETY NET: If the prompt is still breaking the server, send a guaranteed clean string
      return await fetchPollinationsPreviewDataUrl(
        "A beautiful creative illustration, high quality rendering, detailed"
      );
    }
  }
}

/** Groq often wraps voiceover in "Here's the script … : \"…\"" — keep only text safe for TTS. */
function normalizeSpokenScriptForTTS(raw) {
  let s = String(raw ?? "")
    .replace(/\r\n/g, "\n")
    .replace(/\*\*/g, "")
    .replace(/\*/g, "")
    .replace(/#/g, "")
    .trim();

  const tailQuoted = s.match(/:\s*["“”']([\s\S]+)["”']\s*$/);
  if (tailQuoted && tailQuoted[1].trim().length > 24) {
    return tailQuoted[1].trim();
  }

  const colonQuoteMarkers = [': "', ": \"", ": '", ": '"];
  for (const m of colonQuoteMarkers) {
    const at = s.lastIndexOf(m);
    if (at !== -1) {
      const rest = s.slice(at + m.length).trim().replace(/["”']\s*$/g, "").trim();
      if (rest.length > 24) return rest;
    }
  }

  s = s
    .replace(/^(here'?s|here is|this is)\s+[\s\S]{0,480}?[:.;]\s*/i, "")
    .replace(/^(the\s+)?(script|voiceover|intro|copy|audio)\s*(is|:)\s*/i, "")
    .trim();

  const lines = s.split("\n");
  if (
    lines.length > 1 &&
    lines[0].length < 220 &&
    /here'?s|here is|this is|below is|script for/i.test(lines[0])
  ) {
    s = lines.slice(1).join("\n").trim();
  }

  s = s.replace(/^["“'`]+|["”'`]+$/g, "").trim();

  s = s.replace(/^here'?s\s+(?:a\s+)?(?:the\s+)?script[^:]{0,400}:\s*/i, "").trim();

  return s;
}

app.use(
  cors({
    origin: "http://localhost:5173"
  })
);
app.use(express.json({ limit: '15mb' }));
app.use(express.urlencoded({ extended: true, limit: '15mb' }));

// Per-request timeout: send a clean 504 after 115s rather than letting the
// TCP socket reset (which Vite's proxy reports as ECONNRESET).
app.use((req, res, next) => {
  res.setTimeout(115000, () => {
    if (!res.headersSent) {
      res.status(504).json({ error: "Request timed out. The AI model is taking too long — please try again." });
    }
  });
  next();
});

app.post("/api/agent/chat", async (req, res) => {
  const { sessionId, message } = req.body || {};

  if (!sessionId || typeof sessionId !== "string" || !message || typeof message !== "string") {
    return res.status(400).json({
      error: "sessionId and message are required"
    });
  }

  try {
    const existingSession = await getSession(sessionId);
    const session = existingSession || createSession(sessionId);

    session.history = Array.isArray(session.history) ? session.history : [];
    const userDisplay = formatUserPayloadForHistory(message);
    session.history.push({
      role: "user",
      content: message,
      ...(userDisplay ? { displayContent: userDisplay } : {})
    });

    const result = await route(session, message);

    session.step = result.nextStep;
    session.history.push({
      role: "agent",
      content: result.reply,
      uiType: result.uiType ?? null,
      uiData: result.uiData ?? null
    });

    if (result.clearSession) {
      await deleteSession(sessionId);
    } else {
      await saveSession(session);
    }

    return res.json({
      reply: result.reply,
      uiType: result.uiType,
      uiData: result.uiData,
      step: result.nextStep,
      coins: result.coins ?? null,
      confirm: result.confirm ?? null
    });
  } catch (error) {
    console.error("Agent chat error:", error?.message || error);
    return res.status(500).json({
      error: error?.message || "Internal server error"
    });
  }
});

app.get("/api/agent/history", async (req, res) => {
  const sessionId = req.query.sessionId;
  if (!sessionId) {
    return res.status(400).json({ error: "sessionId is required" });
  }
  try {
    const session = await getSession(sessionId);
    if (!session || !session.history) {
      return res.json({ history: [] });
    }
    return res.json({ history: session.history });
  } catch (error) {
    console.error("Agent history error:", error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

app.post("/api/test-prompt", async (req, res) => {
  const { systemPrompt, userPrompt, testInputs, modelHint } = req.body || {};
  if (!systemPrompt || !userPrompt) {
    return res.status(400).json({ error: "systemPrompt and userPrompt are required" });
  }

  try {
    const result = await runPromptTest({
      systemPrompt: String(systemPrompt),
      userPrompt: String(userPrompt),
      testInputs: testInputs && typeof testInputs === "object" ? testInputs : {},
      modelHint: typeof modelHint === "string" ? modelHint : undefined
    });
    return res.json(result);
  } catch (error) {
    console.error("Prompt test error:", error);
    return res.status(500).json({ error: "Unable to run test prompt" });
  }
});

// ─── Live Preview Route (Free-tier APIs for testing) ───────────────────────
app.post('/api/test-preview', async (req, res) => {
  // 1. Extract variables outside the try block just in case req.body is malformed
  const appType = req.body?.appType || 'text';
  const variables = req.body?.variables || {};
  const systemPrompt = req.body?.systemPrompt || '';
  const testImageBase64 = req.body?.testImageBase64 || null;
  
  try {
    let previewResult = null;
    const type = (appType || 'text').toLowerCase();

    // ---------------------------------------------------------
    // 1. TEXT APP (Groq + optional Pollinations image for visual domains)
    // ---------------------------------------------------------
    if (type === 'text') {
      const groqRes = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${process.env.GROQ_API_KEY}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: "llama-3.1-8b-instant", 
          messages: [
            {
              role: "system",
              content:
                "You are the backend engine for an application. Output EXACTLY what the app is supposed to output. NEVER apologize. NEVER mention that you are an AI, a text-based model, or that you cannot generate images. CRITICAL DIRECTIVE: You must output ONLY pure, raw plain text. DO NOT use any Markdown formatting whatsoever. No asterisks (**), no hashes (#) for headers, and no markdown bullet points. Just clean, readable plain text."
            },
            {
              role: "user",
              content: `${systemPrompt}\n\nUser Inputs: ${JSON.stringify(variables)}`
            }
          ],
          max_tokens: 300 
        })
      });
      const groqData = await groqRes.json();
      if (!groqRes.ok) throw new Error(groqData.error?.message || "Groq Error");
      
      const rawText = groqData.choices[0].message.content;
      const textContent = String(rawText)
        .replace(/\*\*\*/g, "")
        .replace(/\*\*/g, "")
        .replace(/\*/g, "")
        .replace(/^#{1,6}\s?/gm, "")
        .replace(/#/g, "")
        .trim();
      let imageUrl = null;

      // Surprise & Delight: If text app is visual (Astrology, Story), add an image (Pollinations — clean prompt, no raw JSON)
      if (systemPrompt.toLowerCase().match(/(astrology|horoscope|story|character|design)/)) {
        const combinedPrompt = buildPollinationsImageAppPrompt(
          `Illustration supporting the app's narrative or design. ${systemPrompt}`,
          variables
        );
        try {
          imageUrl = await fetchPollinationsWithFallback(
            combinedPrompt,
            Object.entries(variables || {})
              .filter(([, v]) => v && String(v).trim())
              .map(([k, v]) => `${k}: ${v}`)
              .join("; ")
              .slice(0, 320) || combinedPrompt.slice(0, 200)
          );
        } catch (e) {
          console.warn("[test-preview] Multimodal Pollinations failed:", e.message);
          imageUrl = previewImageUnavailableDataUrl();
        }
      }
      previewResult = { type: imageUrl ? 'multimodal' : 'text', content: textContent, url: imageUrl };
    }

    // ---------------------------------------------------------
    // 2. IMAGE APP — transform uploaded image OR generate from text
    // ---------------------------------------------------------
    else if (type === 'image') {

      if (testImageBase64) {
        // ── PATH A: Uploaded image → 2-Sub-agent pipeline ──
        // Sub-agent 1: Groq Vision — deeply analyzes the photo + writes a precise render prompt
        // Sub-agent 2: Pollinations — renders the transformed image from that prompt
        console.log("[SubAgent-1] Analyzing uploaded image via Groq Vision...");

        const groqKey = process.env.GROQ_API_KEY;
        let renderPrompt = "";

        // Summarize what transformation the user wants
        const transformGoal = Object.entries(variables || {})
          .filter(([, v]) => v && String(v).trim())
          .map(([k, v]) => `${k.replace(/_/g, " ")}: ${String(v).trim()}`)
          .join("; ") || systemPrompt.slice(0, 200);

        if (groqKey) {
          try {
            const visionBody = JSON.stringify({
              model: "meta-llama/llama-4-scout-17b-16e-instruct",
              max_tokens: 280,
              messages: [{
                role: "user",
                content: [
                  { type: "image_url", image_url: { url: testImageBase64 } },
                  {
                    type: "text",
                    text: `You are a professional image-to-prompt writer for an AI image generator.\n\nSTEP 1 — Identify subject details: gender, approximate age, hair (color+length+style), skin tone, expression, clothing (style+color), body pose, lighting.\n\nSTEP 2 — Write a single render prompt (80-120 words) showing EXACTLY the same person AFTER this transformation:\n"${transformGoal}"\n\nRules:\n- Keep ALL subject details identical (face, hair, clothes, pose)\n- Only change what the transformation requires\n- If transformation is "transparent background" or "no background": write "isolated subject on pure white background, no background elements, clean cutout"\n- End with: "photorealistic, ultra-detailed, professional photography, 8K resolution"\n- Output ONLY the prompt text. No explanation, no prefix like "here is".`
                  }
                ]
              }]
            });

            const groqVisionRes = await fetch("https://api.groq.com/openai/v1/chat/completions", {
              method: "POST",
              headers: { "Authorization": `Bearer ${groqKey}`, "Content-Type": "application/json" },
              body: visionBody,
              signal: AbortSignal.timeout(25000)
            });

            const groqVisionData = await groqVisionRes.json();
            if (groqVisionRes.ok) {
              renderPrompt = groqVisionData.choices?.[0]?.message?.content?.trim() || "";
              console.log("[SubAgent-1] Render prompt:", renderPrompt.slice(0, 120));
            } else {
              console.warn("[SubAgent-1] Groq Vision error:", groqVisionData.error?.message || groqVisionRes.status);
            }
          } catch (e) {
            console.warn("[SubAgent-1] Groq Vision failed:", e.message);
          }
        }

        // Fallback if Groq Vision is unavailable
        if (!renderPrompt || renderPrompt.length < 20) {
          renderPrompt = `${transformGoal}, photorealistic, high resolution, professional photography, 8K`;
          console.log("[SubAgent-1] Using fallback prompt.");
        }

        // Sub-agent 2: Pollinations renders the final image
        console.log("[SubAgent-2] Rendering via Pollinations...");
        let imageDataUrl;
        try {
          imageDataUrl = await fetchPollinationsWithFallback(renderPrompt, renderPrompt.slice(0, 200));
        } catch (e) {
          console.error("[SubAgent-2] Pollinations failed:", e.message);
          imageDataUrl = previewImageUnavailableDataUrl();
        }

        previewResult = { type: "image", url: imageDataUrl };

      } else {
        // ── PATH B: Check if any variable holds a source image URL ──
        const SOURCE_IMAGE_FIELDS = [
          'source_image', 'sourceImage', 'image_url', 'imageUrl', 'input_image',
          'photo_url', 'portrait', 'person_image', 'subject_image', 'photo'
        ];
        // Match URLs with image extensions anywhere, OR from known image CDNs (Unsplash, Cloudinary, etc.)
        const isImageUrl = (v) => {
          const s = String(v || '').trim();
          if (!/^https?:\/\//i.test(s)) return false;
          return /\.(jpg|jpeg|png|webp|gif|avif)(\?|$|&|#)/i.test(s) ||
                 /^https?:\/\/(images\.unsplash\.com|cdn\.pixabay\.com|images\.pexels\.com|res\.cloudinary\.com|i\.imgur\.com|raw\.githubusercontent\.com|media\.giphy\.com)/i.test(s) ||
                 /auto=format/i.test(s); // Unsplash's format param signals an image
        };
        const urlField = SOURCE_IMAGE_FIELDS.find(f => variables?.[f] && isImageUrl(variables[f]));
        const anyUrlField = !urlField && Object.entries(variables || {}).find(([, v]) => isImageUrl(v));
        const resolvedUrlField = urlField || (anyUrlField && anyUrlField[0]);
        const sourceImageUrl = resolvedUrlField ? String(variables[resolvedUrlField]).trim() : null;

        if (sourceImageUrl) {
          // ── PATH B1: Variable contains an image URL → fetch it and run the same pipeline as PATH A ──
          console.log(`[Image] Variable "${resolvedUrlField}" has an image URL — fetching for Groq Vision pipeline:`, sourceImageUrl.slice(0, 80));
          let fetchedBase64 = null;
          let fetchedMime = 'image/jpeg';
          try {
            const imgFetch = await fetch(sourceImageUrl, {
              headers: { 'User-Agent': 'Mozilla/5.0' },
              signal: AbortSignal.timeout(20000)
            });
            if (imgFetch.ok) {
              const ct = imgFetch.headers.get('content-type') || 'image/jpeg';
              fetchedMime = ct.split(';')[0].trim();
              const buf = Buffer.from(await imgFetch.arrayBuffer());
              fetchedBase64 = `data:${fetchedMime};base64,${buf.toString('base64')}`;
              console.log(`[Image] Fetched source image: ${buf.length} bytes, mime: ${fetchedMime}`);
            } else {
              console.warn(`[Image] Failed to fetch source image URL: HTTP ${imgFetch.status}`);
            }
          } catch (fetchErr) {
            console.warn('[Image] Source image URL fetch error:', fetchErr.message);
          }

          if (fetchedBase64) {
            // Run Groq Vision to describe the transformation
            let roomDescription = '';
            const groqKey = process.env.GROQ_API_KEY;
            if (groqKey) {
              try {
                const varStr = Object.entries(variables || {})
                  .filter(([k, v]) => v && String(v).trim() && k !== resolvedUrlField)
                  .map(([k, v]) => `${k.replace(/_/g, ' ')}: ${v}`)
                  .join(', ');
                const groqVisionRes = await fetch('https://api.groq.com/openai/v1/chat/completions', {
                  method: 'POST',
                  headers: { 'Authorization': `Bearer ${groqKey}`, 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    model: 'meta-llama/llama-4-scout-17b-16e-instruct',
                    max_tokens: 220,
                    messages: [{
                      role: 'user',
                      content: [
                        { type: 'image_url', image_url: { url: fetchedBase64 } },
                        {
                          type: 'text',
                          text: `This is a portrait/subject image. Describe in 2-3 sentences how it looks AFTER these transformations are applied: ${varStr || systemPrompt}. Focus on the person staying the same but with the background replaced/removed as requested. Write only the visual description for an image generator — no preamble.`
                        }
                      ]
                    }]
                  })
                });
                const groqVisionData = await groqVisionRes.json();
                if (groqVisionRes.ok) {
                  roomDescription = groqVisionData.choices?.[0]?.message?.content?.trim() || '';
                  console.log('[Image] Groq vision (URL path):', roomDescription.slice(0, 150));
                }
              } catch (e) {
                console.warn('[Image] Groq vision (URL path) failed:', e.message);
              }
            }

            const varStr = Object.entries(variables || {})
              .filter(([k, v]) => v && String(v).trim() && k !== resolvedUrlField)
              .map(([k, v]) => `${k.replace(/_/g, ' ')}: ${v}`)
              .join(', ');
            const renderPrompt = roomDescription || `${varStr}, photorealistic, high resolution, professional photography`;
            console.log('[Image] Render prompt (URL path):', renderPrompt.slice(0, 150));

            let imageDataUrl;
            try {
              imageDataUrl = await fetchPollinationsWithFallback(renderPrompt, renderPrompt.slice(0, 200));
            } catch (e) {
              imageDataUrl = previewImageUnavailableDataUrl();
            }
            previewResult = { type: 'image', url: imageDataUrl };

          } else {
            // URL fetch failed — fall through to pure text-to-image
            const combinedPrompt = buildPollinationsImageAppPrompt(systemPrompt, variables);
            let imageDataUrl;
            try { imageDataUrl = await fetchPollinationsWithFallback(combinedPrompt, combinedPrompt.slice(0, 200)); }
            catch (e) { imageDataUrl = previewImageUnavailableDataUrl(); }
            previewResult = { type: 'image', url: imageDataUrl };
          }

        } else {
          // ── PATH B2: No image at all → pure text-to-image via Pollinations ──
          const combinedPrompt = buildPollinationsImageAppPrompt(systemPrompt, variables);
          console.log('[Image] Text-to-image via Pollinations');
          let imageDataUrl;
          try {
            imageDataUrl = await fetchPollinationsWithFallback(combinedPrompt, combinedPrompt.slice(0, 200));
          } catch (e) {
            console.warn('[Image] Pollinations failed:', e.message);
            imageDataUrl = previewImageUnavailableDataUrl();
          }
          previewResult = { type: 'image', url: imageDataUrl };
        }
      }
    }

    // ---------------------------------------------------------
    // 3. AUDIO APP — speak user's script directly via Murf AI TTS
    // ---------------------------------------------------------
    else if (type === "audio") {

      // ── Detect if the user already provided the script text in their inputs ──
      // Common field names for script/text content in audio apps
      const SCRIPT_FIELDS = [
        'scripted_conversation', 'script', 'content', 'text', 'dialogue',
        'narration', 'transcript', 'message', 'body', 'story', 'article', 'podcast_script'
      ];
      const userScriptField = SCRIPT_FIELDS.find(f =>
        variables?.[f] && String(variables[f]).trim().length > 30
      );
      const userProvidedScript = userScriptField ? String(variables[userScriptField]).trim() : null;

      let scriptContent;

      if (userProvidedScript) {
        // ✅ User typed the script — speak it directly, no AI rewriting
        console.log(`[Audio] Using user-provided script from field: "${userScriptField}" (${userProvidedScript.length} chars)`);
        scriptContent = userProvidedScript;
      } else {
        // ⚙️ No script provided — generate one via Groq based on app inputs
        if (!groq) throw new Error("GROQ_API_KEY is not configured");
        console.log("[Audio] No script in variables — generating via Groq...");
        const chatCompletion = await groq.chat.completions.create({
          messages: [
            {
              role: "system",
              content:
                "You write ONLY the exact words a voice actor will read aloud for the final audio. " +
                "Never add titles, labels, or meta lines (no \"Here is\", \"Here's the script\", \"Below is\"). " +
                "No stage directions, sound effects, or speaker names. No markdown. Output plain spoken text only."
            },
            { role: "user", content: `${systemPrompt}\n\nInputs: ${JSON.stringify(variables)}` }
          ],
          model: "llama-3.1-8b-instant",
          max_tokens: 500
        });
        const rawScript = chatCompletion.choices[0]?.message?.content ?? "";
        scriptContent = normalizeSpokenScriptForTTS(rawScript);
      }

      if (!scriptContent.trim()) throw new Error("Audio preview: empty script");

      // ── Murf AI TTS ──────────────────────────────────────────────────
      const murfKey = process.env.MURF_API_KEY;

      if (!murfKey) {
        console.warn("[Murf] MURF_API_KEY not set — falling back to browser TTS");
        previewResult = { type: "audio", url: null, data: scriptContent.slice(0, 4500) };
      } else {
        // ── Murf voice selection: explicit UI button → speaker name heuristic → default female ──
        const explicitGender = (variables?.voice_gender || '').toLowerCase();

        let wantsMale;
        if (explicitGender === 'male') {
          wantsMale = true;
        } else if (explicitGender === 'female') {
          wantsMale = false;
        } else {
          // Fallback: infer from speaker name
          const speakerName = (variables?.speaker_name || variables?.Speaker_Name || '').toLowerCase();
          const femaleNames = ['alice','anna','emma','olivia','sophia','emily','sarah','lisa','mary','jessica','jennifer','natalie','rachel','grace','priya','ananya'];
          const maleNames   = ['john','james','david','michael','robert','marcus','daniel','alex','raj','arjun','sam','adam','chris','kevin','tyler'];
          const speakerIsMale   = maleNames.some(n => speakerName.includes(n));
          const speakerIsFemale = femaleNames.some(n => speakerName.includes(n));
          wantsMale = speakerIsMale && !speakerIsFemale;
        }

        // Murf voice IDs — use simple actor names for broadest compatibility
        const MURF_VOICES = {
          female: 'natalie',
          male:   'terrell'
        };
        const voiceId     = wantsMale ? MURF_VOICES.male : MURF_VOICES.female;
        const genderLabel = wantsMale ? 'Male (Terrell)' : 'Female (Natalie)';
        console.log(`[Murf] Voice: ${genderLabel} → voiceId: ${voiceId} (source: ${explicitGender || 'inferred'})`);

        // Murf free plan: max ~3000 chars per request
        const ttsText = scriptContent.slice(0, 3000);

        const voiceRate = variables?.voice_speed != null ? Number(variables.voice_speed) : 1.0;
        const voicePitch = variables?.voice_pitch != null ? Number(variables.voice_pitch) : 0;

        try {
          // Use the REST /v1/speech/generate endpoint (non-streaming) with base64 response
          // The streaming /v1/speech/stream endpoint migrated to WebSocket-only and returns 500 on HTTP POST
          const murfRes = await fetch("https://api.murf.ai/v1/speech/generate", {
            method: "POST",
            headers: {
              "api-key": murfKey,
              "Content-Type": "application/json"
            },
            body: JSON.stringify({
              text: ttsText,
              voiceId: voiceId,
              modelVersion: "GEN2",
              locale: "en-US",
              format: "MP3",
              encodeAsBase64: true,
              rate: voiceRate,
              pitch: voicePitch
            }),
            signal: AbortSignal.timeout(30000)
          });

          const contentType = murfRes.headers.get("content-type") || "";

          if (!murfRes.ok) {
            const errText = await murfRes.text();
            console.error("[Murf] TTS error:", murfRes.status, errText);
            previewResult = { type: "audio", url: null, data: scriptContent.slice(0, 4500) };
          } else if (contentType.includes("application/json")) {
            // REST endpoint returns JSON with audioFile (URL) or encodedAudio (base64)
            const murfJson = await murfRes.json();
            const audioBase64 = murfJson.encodedAudio || murfJson.audioFile || null;

            if (audioBase64 && audioBase64.startsWith("data:")) {
              // Already a full data URI
              previewResult = {
                type: "audio",
                url: audioBase64,
                data: scriptContent.slice(0, 4500),
                voiceLabel: genderLabel
              };
              console.log(`[Murf] Audio generated (data URI)`);
            } else if (audioBase64 && !audioBase64.startsWith("http")) {
              // Raw base64 string — wrap as data URI
              previewResult = {
                type: "audio",
                url: `data:audio/mpeg;base64,${audioBase64}`,
                data: scriptContent.slice(0, 4500),
                voiceLabel: genderLabel
              };
              console.log(`[Murf] Audio generated (base64, ${audioBase64.length} chars)`);
            } else if (audioBase64 && audioBase64.startsWith("http")) {
              // URL to hosted audio file
              previewResult = {
                type: "audio",
                url: audioBase64,
                data: scriptContent.slice(0, 4500),
                voiceLabel: genderLabel
              };
              console.log(`[Murf] Audio generated (URL): ${audioBase64.slice(0, 80)}`);
            } else {
              console.warn("[Murf] No audio in response:", JSON.stringify(murfJson).slice(0, 200));
              previewResult = { type: "audio", url: null, data: scriptContent.slice(0, 4500) };
            }
          } else {
            // Binary audio response (unlikely for /generate but handle gracefully)
            const audioBuffer = await murfRes.arrayBuffer();
            if (!audioBuffer.byteLength) {
              console.error("[Murf] Empty audio response");
              previewResult = { type: "audio", url: null, data: scriptContent.slice(0, 4500) };
            } else {
              const base64Audio = Buffer.from(audioBuffer).toString("base64");
              previewResult = {
                type: "audio",
                url: `data:audio/mpeg;base64,${base64Audio}`,
                data: scriptContent.slice(0, 4500),
                voiceLabel: genderLabel
              };
              console.log(`[Murf] Audio generated: ${audioBuffer.byteLength} bytes`);
            }
          }
        } catch (murfErr) {
          console.error("[Murf] Fetch error:", murfErr.message);
          previewResult = { type: "audio", url: null, data: scriptContent.slice(0, 4500) };
        }
      }
    }

    // ---------------------------------------------------------
    // 4. VIDEO APP (Pollinations Thumbnail + Groq Screenplay)
    // ---------------------------------------------------------
    else if (type === 'video') {
      if (!groq) throw new Error("GROQ_API_KEY is not configured");
      // Step 1: Generate the Screenplay
      const chatCompletion = await groq.chat.completions.create({
        messages: [
          { role: "system", content: "You are an AI video director. Generate a short, engaging video concept based on the inputs. Include 'Scene 1: [Visual Description]' followed by the Voiceover narration. Keep it under 150 words. Pure plain text, no markdown." },
          { role: "user", content: `${systemPrompt}\n\nInputs: ${JSON.stringify(variables)}` }
        ],
        model: "llama-3.1-8b-instant",
      });
      const videoScript = chatCompletion.choices[0].message.content.replace(/\*\*/g, '').replace(/\*/g, '');

      // Step 2: Generate a Cinematic Thumbnail
      const cleanVars = Object.entries(variables || {}).filter(([k, v]) => v && String(v).trim() !== '').map(([k, v]) => `${k}: ${v}`).join(', ');
      const safeContext = (systemPrompt || '').substring(0, 120).replace(/[^a-zA-Z0-9 \.,]/g, '');
      const combinedPrompt = `Cinematic high quality video still shot, 8k resolution. Subject: ${safeContext}. ${cleanVars.substring(0, 150)}`;
      const encodedPrompt = encodeURIComponent(combinedPrompt);
      const thumbnailUrl = `https://image.pollinations.ai/prompt/${encodedPrompt}?width=1024&height=576&nologo=true`;

      previewResult = { type: 'video', data: videoScript, url: thumbnailUrl };
    }

    // ---------------------------------------------------------
    // 5. VISION APP (Image Analysis)
    // ---------------------------------------------------------
    else if (type === 'vision') {
      // Simulate Vision analysis using Groq Text (unless you have a groq vision model available)
      const visionResponse = `👁️ **Vision Analysis Complete**\n\nBased on the uploaded image and your parameters (${JSON.stringify(variables)}), the AI detects elements perfectly matching your custom ${systemPrompt.substring(0,20)}... logic.`;
      
      previewResult = { 
        type: 'multimodal', 
        url: testImageBase64 || "https://via.placeholder.com/400x200.png?text=No+Image+Uploaded",
        content: visionResponse
      };
    }

    // If everything succeeds:
    return res.json({ success: true, preview: previewResult });

  } catch (error) {
    // 2. THE SAFETY NET: Catch the error and send it to the frontend cleanly
    console.error("🚨 [SERVER CAUGHT ERROR] Preview Generation Failed:", error.message);
    
    // Ensure we only send headers once
    if (!res.headersSent) {
      return res.status(500).json({ 
        success: false, 
        error: "Preview failed: " + error.message 
      });
    }
  }
});

const server = app.listen(PORT, () => {
  console.log(`RentPrompts agent server listening on http://localhost:${PORT}`);
});

// Prevent ECONNRESET on slow LLM calls:
// Node's default keepAliveTimeout (5s) is shorter than a typical LLM round-trip,
// which causes the proxy to see a connection reset mid-request.
server.keepAliveTimeout = 125000; // 125 s
server.headersTimeout   = 130000; // must be > keepAliveTimeout

server.on("error", (err) => {
  if (err.code === "EADDRINUSE") {
    console.log(`Port ${PORT} is busy. Trying port ${PORT + 1}...`);
    app.listen(PORT + 1, () => {
      console.log(`RentPrompts agent server listening on http://localhost:${PORT + 1}`);
    });
  } else {
    throw err;
  }
});
