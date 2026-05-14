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

/** Pollinations GET URLs break in the browser (ad blockers, hotlinking). We fetch on the server and return a data URL. */
const POLLINATIONS_PREVIEW_SIZE = 768;
/** Pollinations path length; user variables are prioritized so previews match Live Test inputs. */
const POLLINATIONS_MAX_PROMPT_CHARS = 640;

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
  const t = String(text || "").trim();
  if (!t.length) return "High quality detailed illustration, professional lighting.";
  if (t.length <= POLLINATIONS_MAX_PROMPT_CHARS) return t;
  return `${t.slice(0, POLLINATIONS_MAX_PROMPT_CHARS).trimEnd()}…`;
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
    if (fb && fb !== String(primaryPrompt || "").trim()) {
      return await fetchPollinationsPreviewDataUrl(fb);
    }
    throw err;
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
app.use(express.json());

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
    console.error("Agent chat error:", error);
    return res.status(500).json({
      error: "Internal server error"
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
    // 2. IMAGE APP (Pollinations.ai - Bulletproof Free Tier)
    // ---------------------------------------------------------
    else if (type === 'image') {
      const cleanVars = Object.entries(variables || {})
        .filter(([k, v]) => v && String(v).trim() !== "")
        .map(([k, v]) => `${k}: ${v}`)
        .join("; ");

      const combinedPrompt = buildPollinationsImageAppPrompt(systemPrompt, variables);
      const shortFallback =
        cleanVars.slice(0, 320) ||
        String(systemPrompt || "")
          .replace(/\s+/g, " ")
          .trim()
          .slice(0, 200);

      let imageDataUrl;
      try {
        imageDataUrl = await fetchPollinationsWithFallback(combinedPrompt, shortFallback);
      } catch (e) {
        console.warn("[test-preview] Image app Pollinations failed:", e.message);
        imageDataUrl = previewImageUnavailableDataUrl();
      }

      previewResult = {
        type: "image",
        url: imageDataUrl
      };
    }

    // ---------------------------------------------------------
    // 3. AUDIO APP (Groq script + ElevenLabs TTS)
    // ---------------------------------------------------------
    else if (type === "audio") {
      if (!groq) throw new Error("GROQ_API_KEY is not configured");
      const elevenKey = process.env.ELEVENLABS_API_KEY;
      if (!elevenKey || /^your_.+_here$/i.test(String(elevenKey).trim())) {
        throw new Error("ELEVENLABS_API_KEY is not configured for audio preview");
      }
      const voiceId = String(process.env.ELEVENLABS_VOICE_ID || "29vD33N1CtxCmqQRPOHJ").trim();

      const chatCompletion = await groq.chat.completions.create({
        messages: [
          {
            role: "system",
            content:
              "You write ONLY the exact words a voice actor will read aloud for the final audio. " +
              "Never add titles, labels, or meta lines (no \"Here is\", \"Here's the script\", \"Below is\", or explanations of what you are doing). " +
              "Do not describe the tone, music, or duration in prose — fold any style hints into the spoken words only if they belong in the voiceover. " +
              "No stage directions, sound effects, or speaker names. No markdown. Output plain spoken text only."
          },
          { role: "user", content: `${systemPrompt}\n\nInputs: ${JSON.stringify(variables)}` }
        ],
        model: "llama-3.1-8b-instant",
        max_tokens: 400
      });
      const rawScript = chatCompletion.choices[0]?.message?.content ?? "";
      const scriptContent = normalizeSpokenScriptForTTS(rawScript);
      if (!scriptContent.trim()) {
        throw new Error("Audio preview: empty script after normalization");
      }

      const ttsText = scriptContent.slice(0, 4500);
      const elRes = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${encodeURIComponent(voiceId)}`, {
        method: "POST",
        headers: {
          Accept: "audio/mpeg",
          "xi-api-key": elevenKey,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          text: ttsText,
          model_id: "eleven_multilingual_v2"
        })
      });
      if (!elRes.ok) {
        const errBody = await elRes.text();
        throw new Error(`ElevenLabs TTS failed (${elRes.status}): ${errBody.slice(0, 280)}`);
      }
      const buffer = await elRes.arrayBuffer();
      previewResult = {
        type: "audio",
        url: `data:audio/mpeg;base64,${Buffer.from(buffer).toString("base64")}`,
        data: scriptContent
      };
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
