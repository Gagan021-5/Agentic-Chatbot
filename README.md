# RentPrompts Agent

RentPrompts Agent is a conversational AI app builder. Creators describe an app idea, the system gathers requirements, selects an AI model, generates production prompts, supports live preview, prepares SEO metadata, and publishes the app to the marketplace.

This repo now uses a Python backend only:

- Frontend: React + Vite
- Backend: FastAPI
- Agent workflow: LangGraph plus the ported RentPrompts state router
- RAG: ChromaDB with Gemini `text-embedding-004`
- Session/cache: Upstash Redis
- LLM providers: Gemini, Groq, OpenRouter

The old JavaScript backend has been removed.

## Architecture

```text
User
  -> React Chat UI
  -> Vite proxy / deployed API origin
  -> FastAPI backend
  -> Agent router + LangGraph workflows
  -> Redis session store
  -> ChromaDB knowledge base
  -> Gemini / Groq / OpenRouter
```

## Project Structure

```text
rentprompts-agent/
├── ai-service/
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── config/
│   ├── data/
│   ├── graphs/
│   ├── knowledge/
│   │   ├── examples/
│   │   ├── marketplace/
│   │   ├── models/
│   │   ├── prompting/
│   │   └── seo/
│   ├── middleware/
│   ├── rag/
│   ├── routers/
│   ├── schemas/
│   ├── scripts/
│   ├── services/
│   └── tools/
├── client/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── docker-compose.yml
└── README.md
```

## FastAPI Endpoints

React-facing compatibility endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/agent/chat` | Main chat route |
| `GET` | `/api/agent/history` | Fetch session chat history |
| `POST` | `/api/test-preview` | Live preview run |
| `POST` | `/api/test-prompt` | Prompt test run |
| `GET` | `/health` | Health check |

AI/RAG endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/chat` | LangGraph chat workflow |
| `POST` | `/retrieve-context` | RAG retrieval |
| `POST` | `/optimize-prompt` | Prompt optimization |
| `POST` | `/web-research` | Web research summary |

## LangGraph Flow

```text
START
  -> RequirementAnalysisNode
  -> ModelSelectionNode
  -> RetrievalNode
       -> Internal RAG
       -> Historical app search
       -> Web search when needed
  -> PromptEngineeringNode
  -> VariableExtractionNode
  -> OutputNode
END
```

## Knowledge Base

```text
ai-service/knowledge/
├── models/
├── prompting/
├── examples/
├── seo/
└── marketplace/
```

Run ingestion after adding or changing knowledge files:

```bash
cd ai-service
python scripts/ingest_knowledge.py
```

## Environment

Create `ai-service/.env` from `ai-service/.env.example`.

Required for full production behavior:

```env
GEMINI_API_KEY=
GROQ_API_KEY=
OPENROUTER_API_KEY=
UPSTASH_REDIS_REST_URL=
UPSTASH_REDIS_REST_TOKEN=
```

Optional:

```env
PAYLOAD_CMS_URL=http://localhost:3000
PAYLOAD_CMS_API_KEY=
MURF_API_KEY=
AUTH_MODE=none
AUTH_SECRET_KEY=rp-backend-secret-change-me-in-prod
CHROMA_PERSIST_DIR=./data/chromadb
```

## Development

Backend:

```bash
cd ai-service
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd client
npm install
npm run dev
```

Local URLs:

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

## Docker

```bash
docker compose up --build
```

The compose file builds only the Python backend and persists ChromaDB data in the `chromadb-data` volume.

## Notes

- `client/vite.config.js` proxies `/api` to FastAPI on port `8000` during development.
- `VITE_API_ORIGIN` can point the frontend at a deployed FastAPI backend in production.
- Generated folders such as `node_modules`, Python virtual environments, `__pycache__`, build output, logs, and ChromaDB persistence are ignored.
