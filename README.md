# Money Forward India — AI Chat Assistant

An AI-powered bilingual chat assistant for **Money Forward India (MFI)** built with **LangChain**, **ChromaDB**, and **OpenAI GPT-4o-mini**. It answers questions about the company, its privacy & security policies, and live job openings — grounding every response in verified source documents, with real-time **English ↔ Japanese** translation.

---

## Architecture

```
User Query (Streaming via SSE)
    │
    ▼
FastAPI (api.py)
    ├── Session Memory (ConversationBufferMemory per session)
    ├── Query Reformulation (follow-up questions rewritten using chat history)
    ├── QueryCache (LRU in-memory cache for redundant queries)
    ├── Analytics Agent (Analyse.py: Latency, Tokens, Cost)
    └── SmartRetriever (chatbot.py)
         ├── Query expansion (acronyms, policy keywords)
         ├── MMR vector search over ChromaDB (k=8, fetch_k=24)
         ├── Priority re-ranking for privacy/security topics
         └── Injects live jobs document for job-related queries
    │
    ▼
GPT-4o-mini (native language output + streaming)
    └── SSE Stream → Web UI

Real-time Translation (on toggle)
    Frontend collects bubble HTML → POST /translate
    └── LLM translates preserving all HTML structure → DOM swap
```

---

## Project Structure

```
├── data/                  # Markdown knowledge base (company, policies, jobs)
├── vector_db/             # Persisted ChromaDB embeddings (auto-created by ingest.py)
├── ingest.py              # Ingests data/ → vector_db/
├── chatbot.py             # Core RAG logic (SmartRetriever, job feed)
├── api.py                 # FastAPI backend (Streaming SSE, /translate, /refresh-jobs)
├── Analyse.py             # Performance analytics & cost tracking
├── mfi_chatbot.html       # Premium Frontend UI (served by FastAPI)
├── requirements.txt       # Python dependencies
├── .env                   # API keys (not committed)
├── SETUP_GUIDE.md         # Local setup instructions
└── SUMMARY.md             # Project summary & design decisions
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
```

### 3. Ingest knowledge base

```bash
python ingest.py
```

Chunks all `.md` files in `data/`, embeds them with OpenAI, and persists to `vector_db/`.

### 4. Start the server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

> Live job data is fetched dynamically from the MFI Zoho Recruit RSS feed — no re-ingestion needed for job queries.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Send a query, get a streaming SSE grounded response |
| `POST` | `/translate` | Translate an array of HTML fragments to English or Japanese |
| `POST` | `/refresh-jobs` | Re-fetch live job listings from RSS feed |
| `GET` | `/health` | Health check + job count |
| `GET` | `/` | Serve the frontend HTML |

---

## Key Features

### Bilingual Support (English ↔ Japanese)
Toggle the `EN/JA` button in the header to instantly translate all visible chat messages. A dedicated `/translate` endpoint uses the LLM to translate HTML fragments while preserving all tags and structure. Translations are cached in the DOM so toggling back is instant.

### Smart Memory & Follow-up Questions
Each session has its own `ConversationBufferMemory`. Follow-up questions (e.g., *"what are the qualifications for this role?"*) are automatically reformulated into standalone queries using chat history before hitting the retriever — ensuring accurate context-aware retrieval.

### SmartRetriever
Custom `BaseRetriever` that:
- Expands short/ambiguous queries (`"MFI"` → `"Money Forward India"`, `"PII"` → `"Personal Information Protection Policy"`)
- Prioritises privacy/security policy documents for relevant queries
- Injects the live jobs document as the first context chunk for career-related queries

### Real-time Agent Analytics
Track query latency, prompt/completion token counts, and session cost via the analytics modal (📊 icon in the header).

### Live Job Listings
Job data is fetched fresh from the MFI Zoho Recruit RSS feed on startup and via the `/refresh-jobs` endpoint, rendered as interactive job cards in the UI.

---

## Retrieval Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `search_type` | `mmr` | Maximal Marginal Relevance — reduces redundant chunks |
| `k` | 8 | Chunks returned to the LLM |
| `fetch_k` | 24 | Candidate pool before MMR re-ranking |
| `lambda_mult` | 0.55 | Balance relevance vs. diversity |
| `Embeddings` | `text-embedding-3-small` | High-efficiency OpenAI embeddings |

---

## Adding Knowledge

1. Add or edit `.md` files in `data/`
2. Run `python ingest.py` to rebuild the vector store
3. Restart the server (picks up the new `vector_db/` on next start)

---

## Dependencies

See `requirements.txt`. Key packages:

- `langchain`, `langchain-community`, `langchain-openai`, `langchain-classic`
- `langchain-chroma`, `chromadb`
- `fastapi`, `uvicorn`
- `python-dotenv`, `requests`, `tiktoken`
