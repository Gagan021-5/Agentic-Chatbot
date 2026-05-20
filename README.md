# 🧠 RentPrompts — Agentic AI App Builder

> **A conversational, AI-powered platform where creators build, configure, and publish AI apps through natural chat — no coding required.**

RentPrompts Agent is a full-stack agentic chatbot that guides users from a raw idea to a fully published AI app on the marketplace. The agent handles requirement gathering, model selection, prompt engineering, live preview testing, SEO metadata generation, and final publishing — all within a single chat interface.

---

## ✨ What It Does

1. **Triage & Understand** — The agent asks targeted clarifying questions to understand the creator's app idea
2. **Auto-configure** — Detects app type (text / image / audio / video / vision), selects the right model, builds dynamic variables
3. **Generate Prompts** — Uses an LLM sub-agent to write production-quality system + user prompts with `$$variable` injection
4. **Live Preview** — Creator can test the app in real-time before approving (text output, generated images, Murf AI audio)
5. **SEO Metadata** — Generates marketplace-ready app name, description, category, and tags
6. **Publish** — One-click publish to the RentPrompts marketplace with coin-based pricing

---

## 🏗️ Architecture

```
rentprompts-agent/
├── client/                          # React + Vite frontend
│   └── src/
│       ├── components/              # All UI components
│       │   ├── ChatWindow.jsx       # Main chat orchestrator
│       │   ├── MessageBubble.jsx    # Per-message renderer (filters internal payloads)
│       │   ├── AppPreviewCard.jsx   # Prompt preview + Live Preview tester
│       │   ├── SEOPreviewCard.jsx   # SEO metadata editor before publish
│       │   ├── ModelCard.jsx        # AI model selection UI
│       │   ├── BudgetCards.jsx      # Budget tier selection chips
│       │   ├── OptionChips.jsx      # Inline chip button renderer
│       │   ├── PublishSuccessCard.jsx  # Post-publish celebration UI
│       │   ├── ConfirmCard.jsx      # User confirmation flow
│       │   ├── ScopeCard.jsx        # Requirement scope display
│       │   └── ...                  # Other UI cards
│       ├── hooks/
│       │   └── useChat.js           # Chat state, session management, WebSocket-like polling
│       └── utils/
│           ├── api.js               # API call helpers
│           └── livePreviewStorage.js # Persists Live Preview state to localStorage
│
└── server/                          # Node.js + Express backend
    ├── server.js                    # Express API, test-preview endpoint, Murf TTS integration
    └── lib/
        ├── stepRouter.js            # 🧠 CORE: Agentic state machine (Steps 0→1→2→3)
        ├── requirementRouter.js     # Triage question engine, deep-answer extractor
        ├── gemini.js                # LLM sub-agents: extractRequirements, generatePromptTemplate, generateSEO
        ├── groq.js                  # Groq + vision pipeline integration
        ├── models.js                # AI model catalog (text/image/audio/video/vision)
        ├── redis.js                 # Upstash Redis session store (create/get/save/delete)
        ├── costCalculator.js        # Per-run coin cost calculations
        ├── languageDirective.js     # Language mirroring system prompt directive
        └── formatUserHistoryDisplay.js  # Chat history formatter
```

---

## 🔁 Agentic State Machine

The agent runs a **4-step lifecycle** managed in `stepRouter.js`:

```
Step 0: TRIAGE
  └─ requirementRouter.js asks domain-specific clarifying questions
  └─ LLM extracts: appPurpose, appType, targetUsers, tone, format...
  └─ Budget selection → Model selection → GOTO Step 1

Step 1: PROMPT GENERATION
  └─ gemini.js generates: systemPrompt, userPrompt, $$variables, acceptImageInput
  └─ SEO sub-agent generates: appName, appDescription, category, tags
  └─ Returns AppPreviewCard to creator → GOTO Step 2

Step 2: PREVIEW & APPROVE
  └─ Creator tests app in Live Preview (text/image/audio output)
  └─ Creator can edit → prompt regenerates (format-only pivot preserves context)
  └─ "Approve & Continue" → Returns SEOPreviewCard → GOTO Step 3

Step 3: PUBLISH / DRAFT
  └─ SEO_PUBLISH:: payload intercepted at route() top — never hits triage
  └─ App published with coin pricing, model ID, prompts, SEO metadata
  └─ Session cleared → PublishSuccessCard shown
```

### Pivot Intelligence

- **Format-only pivot** (e.g. "change this to an audio app") → Preserves all app context, swaps `appType`, goes straight to model selection — no re-triage
- **Full pivot** (brand new app idea) → Clears history, starts triage fresh
- **Minor tweak** (e.g. "make it more formal") → Regenerates prompt in-place, stays at Step 2

---

## 🤖 AI Model Catalog

### Text Models
| Model | Cost/run | Tier |
|---|---|---|
| GPT-4.1 Nano | 0.63 coins | Fast |
| MiniMax M2.7 | 1.12 coins | Fast |
| GPT-4o Mini | 0.5 coins | Fast |
| LLaMA 3.3 70B | 0.35 coins | Fast |
| GPT-4.1 Mini | 1.69 coins | Balanced |
| GPT-4o | 1.09 coins | Balanced |
| GPT-5.2 | 2.0 coins | Balanced |
| Kimi K2 Thinking | 3.5 coins | Premium |
| GPT-5.1 | 8.0 coins | Premium |
| Grok 4 | 20.4 coins | Ultra |

### Image Models
| Model | Cost/run | Notes |
|---|---|---|
| Flux Schnell | 0.5 coins | Fastest |
| SDXL | 1.0 coins | Open-source |
| Flux 2 Pro | 2.5 coins | Realistic |
| Recraft v4 Pro | 4.6 coins | Design/text |
| Imagen 4 | 3.69 coins | Google photorealistic |
| vGPT Image 2 | 18.0 coins | Advanced editing |

### Audio Models
| Model | Cost/run | Notes |
|---|---|---|
| Kokoro 82M | 0.79 coins | Multilingual TTS |
| Stable Audio | 2.0 coins | Music from prompts |
| Orpheus TTS | 4.49 coins | Expressive speech |
| TTS 1.5 Max | 5.7 coins | Human-like |
| Lyria 3 Pro | 8.5 coins | Music generation |

### Video Models
| Model | Cost/run | Notes |
|---|---|---|
| Wan 2.2 Fast | 6.3 coins | Fastest video |
| Seedance 1.5 Pro | 24.0 coins | Balanced |
| Veo 3 Fast | 273.0 coins | Google (with audio) |
| Veo 3 | 318.29 coins | Best cinematic |

### Vision Models
| Model | Cost/run | Notes |
|---|---|---|
| GPT-4.1 Vision | 7.09 coins | Image analysis, OCR |

---

## 🖼️ Live Preview Pipeline

### Text Apps
→ Groq (`llama-3.1-8b-instant`) generates output from system + user prompts with filled `$$variables`

### Image Apps (Smart Upload Detection)
The agent intelligently determines whether image upload is required:

| App Purpose | Upload Required? | Reason |
|---|---|---|
| Background remover | ✅ YES | Transforms user's uploaded photo |
| Room redesign from photo | ✅ YES | Analyzes and redesigns user's room |
| Logo designer | ❌ NO | Creates from text description only |
| Poster / Banner creator | ❌ NO | Generates from user's text inputs |

**With upload:** Groq Vision (`llama-4-scout-17b`) analyzes photo → writes render prompt → Pollinations renders  
**Without upload:** Variables + system prompt → Pollinations generates directly

### Audio Apps
→ Groq generates a polished script → **Murf AI** TTS synthesizes real audio (FALCON model)
- Female voice: `en-US-natalie`
- Male voice: `en-US-terrell` (authoritative, FALCON-compatible)

### Video Apps
→ Groq generates screenplay → Pollinations generates cinematic thumbnail → video player UI

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 18, Vite, Vanilla CSS |
| **Backend** | Node.js, Express (ESM) |
| **Session Store** | Upstash Redis (REST) |
| **LLM Routing** | OpenRouter API |
| **Fast Inference** | Groq SDK (`llama-3.1-8b`, `llama-4-scout-17b`) |
| **Image Generation** | Pollinations.ai |
| **Text-to-Speech** | Murf AI (FALCON model, `v1/speech/stream`) |
| **Hot Reload** | `node --watch` (server), Vite HMR (client) |

---

## 🚀 Getting Started

### Prerequisites
- Node.js v18+
- An Upstash Redis database (free tier works)
- API keys (see below)

### 1. Clone & Install

```bash
git clone <repo-url>
cd rentprompts-agent

# Install server deps
cd server && npm install

# Install client deps
cd ../client && npm install
```

### 2. Configure Environment

Create `server/.env`:

```env
# Required
GROQ_API_KEY=gsk_...          # https://console.groq.com
GEMINI_API_KEY=AIza...        # https://aistudio.google.com
OPENROUTER_API_KEY=sk-or-...  # https://openrouter.ai
UPSTASH_REDIS_REST_URL=https://...upstash.io
UPSTASH_REDIS_REST_TOKEN=...

# Optional (enables real audio generation)
MURF_API_KEY=ap2_...          # https://murf.ai/api

# Optional
PORT=3001
HF_ACCESS_TOKEN=hf_...        # Hugging Face (if needed)
```

### 3. Run Development Servers

```bash
# Terminal 1 — Backend (auto-restarts on file changes)
cd server
node --watch server.js

# Terminal 2 — Frontend (Vite HMR)
cd client
npm run dev
```

- **Client:** http://localhost:5173
- **Server:** http://localhost:3001

---

## 🔌 API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/agent/chat` | Main chat route — processes message, returns structured response |
| `GET` | `/api/agent/history` | Fetch chat history for current session |
| `POST` | `/api/agent/test-preview` | Live Preview — runs app with test inputs (text/image/audio/video) |
| `POST` | `/api/agent/reset` | Clear session and start fresh |

### `/api/agent/chat` Response Shape

```json
{
  "reply": "string — message to display in chat",
  "uiType": "text | chips | budget | models | app_preview | seo_preview | success",
  "uiData": { ... },
  "nextStep": 0,
  "coins": 5.7,
  "clearSession": false
}
```

---

## 🧩 UI Component Map

| Component | Purpose |
|---|---|
| `ChatWindow.jsx` | Root chat orchestrator, renders messages + UI cards |
| `MessageBubble.jsx` | Per-message bubble (auto-hides `SEO_PUBLISH::` internal payloads) |
| `AppPreviewCard.jsx` | Shows system/user prompt + Live Preview tester with `$$variable` inputs |
| `SEOPreviewCard.jsx` | Editable SEO metadata (app name, description, tags, category) before publish |
| `ModelCard.jsx` | AI model chooser with tier badges, cost, and tags |
| `BudgetCards.jsx` | Budget tier chips (Free / Low / Medium / Premium) |
| `OptionChips.jsx` | Inline option buttons (e.g. "Approve App", "Edit App") |
| `PublishSuccessCard.jsx` | Post-publish card with mock marketplace URL |
| `ConfirmCard.jsx` | Confirmation prompts (Yes/No, approve flows) |
| `ScopeCard.jsx` | Displays extracted app requirements for creator review |
| `SEOPreviewCard.jsx` | SEO metadata card — sends `SEO_PUBLISH::` or `SEO_DRAFT::` payloads |

---

## 🔐 Session & State Management

Sessions are stored in **Upstash Redis** with the following shape:

```json
{
  "sessionId": "uuid",
  "step": 0,
  "appType": "audio",
  "modelId": "orpheus-tts",
  "modelCost": 4.49,
  "extraction": {
    "appPurpose": "Convert incident notes to audio briefing",
    "targetUsers": "Field officers, police"
  },
  "deepAnswers": {
    "tone": "formal and authoritative",
    "budgetPreference": "medium"
  },
  "promptData": {
    "systemPrompt": "...",
    "userPrompt": "...",
    "variablesUsed": ["$$incident_location", "$$timestamp"],
    "acceptImageInput": false
  },
  "seoData": {
    "appName": "Field Incident Briefing Generator",
    "appDescription": "...",
    "tags": ["police", "audio", "incident-log"],
    "category": "Productivity"
  },
  "history": []
}
```

---

## 🛡️ Key Hardening Details

### SEO Payload Interception
`SEO_PUBLISH::` and `SEO_DRAFT::` messages from `SEOPreviewCard` are caught **at the very top of `route()`** before any keyword extraction runs. This prevents the `"cost"` keyword in the JSON payload from triggering the price FAQ handler ("Great question! The cost depends on...").

### Payload Visibility Fix
`MessageBubble.jsx` filters all structured payloads with a `HIDDEN_PAYLOAD_PREFIXES` list — they return `null` and are invisible in the chat thread.

### Format-Only Pivot
When a creator says "change this to an audio app" mid-flow, the agent detects it's a **format-only pivot** (same app purpose, different output type). It preserves all triage context, swaps `appType`, and jumps straight to model selection — no re-triage.

### Smart Image Upload Detection
`acceptImageInput` is determined by the LLM based on app purpose, not by app type. Background removers and room redesigners show an upload button. Logo designers and poster creators don't.

---

## 📁 Scripts

```
scripts/
├── update_questions.js   # Seed/update triage question bank
└── patch_router.js       # One-off router migration utility
```

---

## 📄 License

Private — RentPrompts internal tooling. Not for public distribution.

---

*Built with ❤️ by the RentPrompts team. Powered by Groq, OpenRouter, Murf AI, and Pollinations.*
