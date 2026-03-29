# MFI RAG Chatbot — Project Summary

## What It Does

A production-ready Retrieval-Augmented Generation (RAG) chatbot built for **Money Forward India**. It answers natural-language questions about the company, its Privacy Policy, Security Policy, and live job openings — using only verified source documents, never hallucinating.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI `text-embedding-ada-002` |
| Vector Store | ChromaDB (local, persistent) |
| RAG Framework | LangChain (`RetrievalQA`, `BaseRetriever`) |
| API | FastAPI |
| Live Data | Zoho Recruit RSS Feed (HTTP) |
| Config | python-dotenv |

---

## Core Features

**Hybrid Knowledge Retrieval**
Static knowledge base (Markdown documents) combined with a dynamically fetched live jobs feed. The custom `SmartRetriever` decides at query time which sources to prioritise.

**Query Intelligence**
- Expands short/ambiguous queries before vector search (e.g. "MFI" → "Money Forward India", "PII" → "Personal Information Protection Policy")
- Re-ranks retrieved chunks to surface privacy/security policy documents for relevant queries
- Injects the live jobs document as the first context chunk for any career-related query

**MMR Retrieval**
Uses Maximal Marginal Relevance (fetch_k=24, k=8, λ=0.55) to balance relevance with diversity, avoiding redundant chunks in the LLM context.

**Dual Interfaces**
- Interactive CLI for local testing
- FastAPI HTTP API (`/chat`, `/refresh-jobs`, `/health`) with a bundled HTML frontend

**Live Job Sync**
Jobs are fetched from the MFI Zoho Recruit RSS feed on startup and on demand via `/refresh-jobs` — no re-ingestion required for career queries.

---

## Data Flow

```
Query → SmartRetriever
           ├─ Expand query terms
           ├─ MMR search on ChromaDB (static KB)
           ├─ Re-rank by topic (privacy / security / jobs)
           └─ Inject live jobs doc if job-related
                        ↓
              LangChain RetrievalQA
                        ↓
              GPT-4o-mini (grounded, no hallucination)
                        ↓
                    Answer
```

---

## Design Decisions

**Why a custom retriever instead of standard LangChain?**
Standard retrievers treat all queries equally. MFI queries fall into clear topic buckets (company info, privacy, security, jobs), each with different optimal retrieval strategies. `SmartRetriever` encodes this domain knowledge directly.

**Why MMR over similarity search?**
Policy documents often have many near-duplicate chunks. MMR ensures the LLM receives a diverse, non-redundant context window, improving answer quality.

**Why inject jobs as a Document instead of ingesting into ChromaDB?**
Job listings change frequently. Injecting the live-fetched document directly at query time avoids stale embeddings and eliminates the need to re-ingest on every update.

**Why GPT-4o-mini?**
Sufficient capability for grounded Q&A over structured documents, at significantly lower cost than GPT-4o — appropriate for a demo/interview asset.

---

## Limitations & Next Steps

- Vector store is local; production would use a managed vector DB (Pinecone, Weaviate)
- No authentication on the API endpoints
- Job descriptions are truncated at 12,000 characters to control context size
- Could add hybrid search (BM25 + dense) for improved recall on exact-match queries
- Conversation history / multi-turn support not yet implemented
