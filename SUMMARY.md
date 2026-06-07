# MFI AI Chat Assistant — Project Summary

## What It Does

A production-ready bilingual AI chat assistant for **Money Forward India**. It answers natural-language questions in **English or Japanese** about the company, its Privacy Policy, Security Policy, and live job openings — using Retrieval-Augmented Generation (RAG) grounded in verified source documents.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | ChromaDB (local, persistent) |
| RAG Framework | LangChain (`BaseRetriever`, streaming via LLM) |
| Memory | `ConversationBufferMemory` (per-session) |
| API | FastAPI (SSE streaming) |
| Translation | `/translate` endpoint — LLM-powered HTML-safe translation |
| Analytics | Custom `Analyse.py` (Latency, Tokens, Cost) |
| Live Data | Zoho Recruit RSS Feed |
| Config | python-dotenv |

---

## Core Features

### Bilingual Support (English ↔ Japanese)
Real-time translation of all visible chat messages via the `EN/JA` header toggle. The `/translate` endpoint sends raw bubble HTML to the LLM with a strict prompt to preserve all tags, classes, and structure. Translations are cached in DOM attributes — toggling back and forth is instant with no additional API calls.

### Context-Aware Memory
Each browser session maintains its own `ConversationBufferMemory`. When a follow-up question is detected (e.g., *"what experience is needed for this role?"*), the backend first reformulates it into a self-contained standalone question using the LLM and the stored chat history. This reformulated query is then sent to the retriever, ensuring accurate and relevant results.

### Hybrid Knowledge Retrieval
Static Markdown knowledge base combined with a dynamically fetched live jobs feed. The custom `SmartRetriever` decides at query-time which sources to prioritise.

### Streaming & Performance
- Real-time token streaming via Server-Sent Events (SSE).
- In-memory LRU `QueryCache` for instant responses to repeated queries (language-aware).
- Query reformulation ensures follow-up questions are resolved correctly even for a cold cache.

### Query Intelligence
- Expands short/ambiguous queries before vector search (`"MFI"` → `"Money Forward India"`, `"PII"` → `"Personal Information Protection Policy"`)
- Re-ranks retrieved chunks to surface privacy/security policy documents for relevant queries
- Injects the live jobs document as the first context chunk for any career-related query

### Real-time Agent Analytics
- Tracks **Latency (ms)**, **Prompt/Completion Tokens**, and **Session Cost (USD)**
- Accessible via a glassmorphic analytics modal in the UI (📊 icon)

---

## Data Flow

```
Query → API layer (Session Memory + LRU Cache check)
           ↓
        Query Reformulation (if history exists → LLM rewrites to standalone)
           ↓
        SmartRetriever
           ├─ Expand query terms
           ├─ MMR search on ChromaDB (static KB)
           ├─ Re-rank by topic (privacy / security / jobs)
           └─ Inject live jobs doc if job-related
                          ↓
                GPT-4o-mini (streaming, native language output)
                          ↓
                Save to session memory + update cache
                          ↓
                SSE Stream (tokens + stats) → Web UI

On Language Toggle:
Frontend DOM → POST /translate (HTML array)
                          ↓
                LLM translates, preserves HTML structure
                          ↓
                DOM swap + cache in data attributes
```

---

## Design Decisions

**Why a custom retriever instead of standard LangChain?**
Standard retrievers treat all queries equally. MFI queries fall into clear topic buckets (company info, privacy, security, jobs), each requiring different retrieval strategies. `SmartRetriever` encodes this domain knowledge directly.

**Why query reformulation before retrieval?**
`ConversationBufferMemory` stores conversation history, but the retriever only sees the raw latest question. A follow-up like *"what is the salary for that role?"* would retrieve nothing useful. The reformulation step turns it into *"What is the salary for the AI Engineer role?"* before hitting the vector store.

**Why MMR over similarity search?**
Policy documents often have many near-duplicate chunks. MMR ensures the LLM receives a diverse, non-redundant context window, improving answer quality.

**Why inject jobs as a Document instead of ingesting into ChromaDB?**
Job listings change frequently. Injecting the live-fetched document directly at query time avoids stale embeddings and eliminates the need to re-ingest on every job update.

**Why GPT-4o-mini?**
Sufficient capability for grounded Q&A over structured documents, at significantly lower cost than GPT-4o.

---

## Limitations & Next Steps

- Vector store is local; production would use a managed vector DB (Pinecone, Weaviate)
- No authentication on API endpoints
- Job descriptions are truncated at 12,000 characters to control context size
- Could add hybrid search (BM25 + dense) for improved recall on exact-match queries
- Translation could be parallelised per-bubble for faster large-conversation toggles
