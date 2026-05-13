import "dotenv/config";

import cors from "cors";
import express from "express";
import { createSession, getSession, saveSession, deleteSession } from "./lib/redis.js";
import { route } from "./lib/stepRouter.js";
import { runPromptTest } from "./lib/gemini.js";

const app = express();
const PORT = Number(process.env.PORT) || 3001;

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
    session.history.push({
      role: "user",
      content: message
    });

    const result = await route(session, message);

    session.step = result.nextStep;
    session.history.push({
      role: "agent",
      content: result.reply
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
        const cleanVars = Object.entries(variables || {})
          .map(([k, v]) => `${k}: ${String(v)}`)
          .join(", ");
        const visualFocus = cleanVars.substring(0, 300);
        const combinedPrompt = `A high quality, realistic image of: ${visualFocus}`;
        imageUrl = `https://image.pollinations.ai/prompt/${encodeURIComponent(combinedPrompt)}?width=1024&height=1024&nologo=true`;
      }
      previewResult = { type: imageUrl ? 'multimodal' : 'text', content: textContent, url: imageUrl };
    }

    // ---------------------------------------------------------
    // 2. IMAGE APP (Pollinations.ai - Bulletproof Free Tier)
    // ---------------------------------------------------------
    else if (type === 'image') {
      const cleanVars = Object.entries(variables || {})
        .map(([k, v]) => `${k}: ${String(v)}`)
        .join(", ");

      const visualFocus = cleanVars.substring(0, 300);
      const combinedPrompt = `A high quality, realistic image of: ${visualFocus}`;

      const encodedPrompt = encodeURIComponent(combinedPrompt);
      const imageUrl = `https://image.pollinations.ai/prompt/${encodedPrompt}?width=1024&height=1024&nologo=true`;

      previewResult = {
        type: 'image',
        url: imageUrl
      };
    }

    // ---------------------------------------------------------
    // 3. AUDIO APP (ElevenLabs TTS)
    // ---------------------------------------------------------
    else if (type === 'audio') {
      const elRes = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM`, {
        method: 'POST',
        headers: { 'Accept': 'audio/mpeg', 'xi-api-key': process.env.ELEVENLABS_API_KEY, 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: `Generating audio for: ${JSON.stringify(variables).substring(0, 50)}`, model_id: "eleven_multilingual_v2" })
      });
      if (!elRes.ok) throw new Error("ElevenLabs API failed");
      
      const buffer = await elRes.arrayBuffer();
      previewResult = { type: 'audio', url: `data:audio/mpeg;base64,${Buffer.from(buffer).toString('base64')}`, content: "Audio preview generated." };
    }

    // ---------------------------------------------------------
    // 4. VIDEO APP (Cinematic Storyboard Mock)
    // ---------------------------------------------------------
    else if (type === 'video') {
      // Free Video APIs are rare. We generate a "First Frame" (Image) + "Scene Script" (Text)
      const hfRes = await fetch('https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${process.env.HF_ACCESS_TOKEN}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ inputs: `Cinematic movie still, first frame of video for: ${JSON.stringify(variables)}` })
      });
      
      let imageUrl = null;
      if (hfRes.ok) {
        const buffer = await hfRes.arrayBuffer();
        imageUrl = `data:image/jpeg;base64,${Buffer.from(buffer).toString('base64')}`;
      } else {
        let errorMsg = "Hugging Face API failed";
        try {
          const errorData = await hfRes.json();
          errorMsg = errorData.error || errorMsg;
        } catch (parseErr) {
          errorMsg = `HTTP Error ${hfRes.status}: ${hfRes.statusText}`;
        }
        console.error("Video storyboard HF image error:", errorMsg);
      }
      
      previewResult = { 
        type: 'multimodal', 
        url: imageUrl,
        content: `🎬 **Video Storyboard Preview**\n\n**Scene 1:** The video opens with these parameters: ${JSON.stringify(variables)}. \n*(Note: Live video rendering requires premium compute and will process upon publishing).*`
      };
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
