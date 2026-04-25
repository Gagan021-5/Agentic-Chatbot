import Groq from "groq-sdk";
import { EXTRACTION_PROMPT, extractRequirementsWithGemini, normalizeExtraction } from "./gemini.js";
const groqApiKey = process.env.GROQ_API_KEY;

function hasRealValue(value) {
  return Boolean(value && !/^your_.+_here$/i.test(value));
}

const groq = hasRealValue(groqApiKey) ? new Groq({ apiKey: groqApiKey }) : null;

function isRateLimitError(error) {
  return error && (error.status === 429 || error.statusCode === 429 || error.code === 429);
}

async function extractRequirements(message, history) {
  if (!groq) {
    return extractRequirementsWithGemini(message, history);
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
      return extractRequirementsWithGemini(message, history);
    }

    return extractRequirementsWithGemini(message, history);
  }
}

export { extractRequirements };
