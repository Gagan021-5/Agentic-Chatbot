import "dotenv/config";

import cors from "cors";
import express from "express";
import { createSession, getSession, saveSession, deleteSession } from "./lib/redis.js";
import { route } from "./lib/stepRouter.js";

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
