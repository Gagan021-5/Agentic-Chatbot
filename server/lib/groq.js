import Groq from "groq-sdk";
import OpenAI from "openai";
import { EXTRACTION_PROMPT, normalizeExtraction } from "./gemini.js";

const groqApiKey = process.env.GROQ_API_KEY;

function hasRealValue(value) {
  return Boolean(value && !/^your_.+_here$/i.test(value));
}

const groq = hasRealValue(groqApiKey) ? new Groq({ apiKey: groqApiKey }) : null;

const openRouterClient = new OpenAI({
  baseURL: 'https://openrouter.ai/api/v1',
  apiKey: process.env.OPENROUTER_API_KEY,
  defaultHeaders: {
    'HTTP-Referer': 'http://localhost:5173',
    'X-Title': 'RentPrompts Agent Demo',
  }
});

function isRateLimitError(error) {
  return error && (error.status === 429 || error.statusCode === 429 || error.code === 429 || (error.message && error.message.includes('429')));
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
          content: EXTRACTION_PROMPT
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

async function extractWithOpenRouterFallback(message, history) {
  try {
    const fallback = await openRouterClient.chat.completions.create({
      model: 'meta-llama/llama-3.3-70b-instruct',
      messages: [
        { role: 'system', content: EXTRACTION_PROMPT },
        { role: 'user', content: JSON.stringify({ message, history: Array.isArray(history) ? history.slice(-8) : [] }) }
      ],
      response_format: { type: 'json_object' },
    });
    
    const raw = fallback.choices[0].message.content;
    const parsed = JSON.parse(raw.replace(/```json/gi, '').replace(/```/g, '').trim());
    return normalizeExtraction(parsed, message);
  } catch (fallbackError) {
    console.error("OpenRouter fallback also failed:", fallbackError.message);
    return normalizeExtraction(null, message);
  }
}

export { extractRequirements };
