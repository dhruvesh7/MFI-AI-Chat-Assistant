# Money Forward India — RAG Chatbot

An AI-powered chatbot for Money Forward India (MFI) built with **LangChain**, **ChromaDB**, and **OpenAI GPT-4o-mini**. It answers questions about the company, its privacy & security policies, and live job openings — grounding every response in verified source documents.

---

## Architecture

```
User Query
    │
    ▼
SmartRetriever (chatbot.py)
    ├── Query expansion (expand acronyms, policy keywords)
    ├── MMR vector search over ChromaDB (k=8, fetch_k=24)
    ├── Priority re-ranking for privacy/security topics
    └── Injects live jobs document for job-related queries
    │
    ▼
RetrievalQA (LangChain)
    └── GPT-4o-mini → grounded answer
```

---

## Project Structure

```
├── data/                  # Markdown knowledge base (company, policies, jobs)
├── vector_db/             # Persisted ChromaDB embeddings (auto-created by ingest.py)
├── ingest.py              # Ingests data/ → vector_db/
├── chatbot.py             # Core RAG logic + CLI
├── api.py                 # FastAPI HTTP wrapper
├── mfi_chatbot.html       # Frontend UI (served by FastAPI)
├── requirements.txt       # Python dependencies
└── .env                   # API keys (not committed)
```

---

## Setup

### 1. Prerequisites

- Python 3.10+
- OpenAI API key

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
```

### 4. Prepare knowledge base

Add your Markdown documents to the `data/` directory, then ingest them:

```bash
python ingest.py
```

This chunks all `.md` files in `data/`, embeds them with OpenAI, and persists them to `vector_db/`.

---

## Running

### CLI mode

```bash
python chatbot.py
```

Commands inside the CLI:
- Type any question to chat
- `refresh jobs` — re-fetch live MFI job listings from the careers RSS feed
- `exit` — quit

### API + Web UI mode

```bash
uvicorn api:app --reload
```

Open `http://localhost:8000` in your browser to access the chat UI.

#### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Send a query, get a grounded answer |
| `POST` | `/refresh-jobs` | Re-fetch live job listings |
| `GET` | `/health` | Health check + job count |
| `GET` | `/` | Serve the frontend HTML |

**Example `/chat` request:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Money Forward India privacy policy?"}'
```

---

## Key Components

### `ingest.py`
Loads all `.md` files from `data/`, splits them into 800-token chunks (200 overlap), embeds with `OpenAIEmbeddings`, and persists to ChromaDB.

### `chatbot.py`
- **`fetch_live_jobs()`** — Fetches and parses the MFI Zoho Recruit RSS feed, extracts title, department, location, and description for each listing.
- **`jobs_to_doc()`** — Converts job listings into a single LangChain `Document` for injection into the retriever.
- **`SmartRetriever`** — Custom `BaseRetriever` that:
  - Expands queries (e.g., "MFI" → "Money Forward India") before vector search
  - Prioritises privacy/security policy documents for relevant queries
  - Prepends the live jobs document for career/role-related queries

### `api.py`
Thin FastAPI wrapper exposing the chatbot as an HTTP service. Loads the vector DB and jobs once on startup, reuses across requests.

---

## Retrieval Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `search_type` | `mmr` | Maximal Marginal Relevance — reduces redundant chunks |
| `k` | 8 | Chunks returned to the LLM |
| `fetch_k` | 24 | Candidate pool before MMR re-ranking |
| `lambda_mult` | 0.55 | Balance relevance vs. diversity |

---

## Adding Knowledge

1. Add or edit `.md` files in `data/`
2. Re-run `python ingest.py` to rebuild the vector store
3. Restart the server (or it will pick up the new `vector_db/` on next start)

> Live job data is fetched dynamically from the MFI RSS feed — no re-ingestion needed for job queries.

---

## Dependencies

See `requirements.txt`. Key packages:

- `langchain`, `langchain-community`, `langchain-openai`
- `langchain-chroma`, `chromadb`
- `fastapi`, `uvicorn`
- `python-dotenv`, `requests`
