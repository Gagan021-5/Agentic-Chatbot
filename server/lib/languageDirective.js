/* ──────────────────────────────────────────────
   STRICT LANGUAGE MIRRORING DIRECTIVE
   Shared across all LLM prompt injection points.
   ────────────────────────────────────────────── */
export const LANGUAGE_MIRROR_DIRECTIVE = `
CRITICAL LANGUAGE DIRECTIVE:
You must analyze the user's input language and mirror it exactly.
1. If the user writes in English, reply in English.
2. If the user writes in pure Hindi (Devanagari script, e.g., "मुझे एक ऐप बनाना है"), you MUST reply entirely in Hindi (Devanagari script).
3. If the user writes in Hinglish (Hindi written in English alphabet, e.g., "Mujhe ek app banana hai"), you MUST reply entirely in Hinglish.
Do not mix scripts unless the user does. All UI options, questions, forms, feature names, and variable names generated must also be translated into this matched language.
`.trim();
