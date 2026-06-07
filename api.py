"""
api.py — thin FastAPI wrapper around chatbot.py
Run: uvicorn api:app --reload
"""

from __future__ import annotations
from collections import OrderedDict
from threading import Lock

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import warnings
from langchain_core._api import LangChainDeprecationWarning

# Suppress LangChain deprecation warnings for cleaner terminal output
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

from chatbot import (
    fetch_live_jobs,
    jobs_to_doc,
    SmartRetriever,
)

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "vector_db"

RAG_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant for Money Forward India. Answer using ONLY the "
            "information in the context below. If the context does not contain enough "
            "information, tell the user you don't know but offer to help with other company-related topics.\n\n"
            "IMPORTANT: When returning a live job opening, you MUST format it exactly like this template and do NOT use standard markdown for it:\n"
            "||JOB|Title|Department|Location|Type|Experience|Top 3 Skills (CSV)|ApplyLink|Salary||\n"
            "For example:\n"
            "||JOB|Data Engineer|Data|Chennai, India|Full-time|2-5 yrs exp|Python, Spark, GCP|https://link...|Not disclosed||\n"
            "For the Salary field: extract it from the job description if mentioned (e.g. '₹12–18 LPA', '$80k–$100k'). If not found, write 'Not disclosed'.\n"
            "Put 'Not specified' for any other missing field.\n\n"
            "IMPORTANT: You MUST respond in the {language} language.\n\n"
            "CHAT HISTORY:\n{chat_history}\n\n"
            "CONTEXT:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

app = FastAPI(title="Money Forward India Chat API")


# =========================
# Query Cache (LRU)
# =========================
class QueryCache:
    """Simple LRU cache for repeated queries."""

    def __init__(self, maxsize: int = 100):
        self.cache: OrderedDict[str, str] = OrderedDict()
        self.maxsize = maxsize

    def get(self, query: str) -> str | None:
        key = query.lower().strip()
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, query: str, answer: str):
        key = query.lower().strip()
        self.cache[key] = answer
        self.cache.move_to_end(key)
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)


query_cache = QueryCache(maxsize=100)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve local static assets used by the frontend (icons/images).
app.mount("/assets", StaticFiles(directory="."), name="assets")

# ── Load once on startup ──
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 24, "lambda_mult": 0.55},
)
jobs = fetch_live_jobs()
jobs_doc = jobs_to_doc(jobs)
smart_retriever = SmartRetriever(base_retriever=retriever, jobs_doc=jobs_doc)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
session_memories: dict[str, ConversationBufferMemory] = {}
session_memories_lock = Lock()


def get_session_memory(session_id: str) -> ConversationBufferMemory:
    """Return per-session memory so follow-up questions keep context."""
    with session_memories_lock:
        if session_id not in session_memories:
            session_memories[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                input_key="question",
                output_key="answer",
            )
        return session_memories[session_id]


from Analyse import analytics_agent

class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None
    language: str | None = "English"


class ChatResponse(BaseModel):
    result: str
    session_id: str
    cached: bool = False


class RefreshResponse(BaseModel):
    count: int
    message: str


class TranslateRequest(BaseModel):
    texts: list[str]
    target_language: str


class TranslateResponse(BaseModel):
    translated_texts: list[str]


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    session_id = (req.session_id or "default").strip() or "default"
    start_time = analytics_agent.start_timer()
    lang = req.language or "English"

    # Check cache first
    cache_key = f"{lang}:{req.query}"
    cached_answer = query_cache.get(cache_key)
    if cached_answer:
        async def cached_stream():
            latency_ms = analytics_agent.end_timer(start_time)
            import json
            yield f"data: {json.dumps({'content': cached_answer})}\n\n"
            stats = analytics_agent.generate_stats(latency_ms, req.query, cached_answer)
            yield f"data: {json.dumps({'stats': stats})}\n\n"
            yield "data: [CACHED]\n\n"

        return StreamingResponse(
            cached_stream(),
            media_type="text/event-stream",
            headers={"X-Cached": "true"}
        )

    memory = get_session_memory(session_id)
    chat_history_messages = memory.chat_memory.messages
    chat_history_text = "\n".join([f"{'User' if i%2==0 else 'Assistant'}: {msg.content}" for i, msg in enumerate(chat_history_messages)])

    search_query = req.query
    if chat_history_messages:
        from langchain_core.prompts import ChatPromptTemplate
        reformulate_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the following chat history and the user's latest question, formulate a standalone question that can be understood without the chat history. Do NOT answer the question, just reformulate it if needed, otherwise return it as is."),
            ("human", "Chat History:\n{chat_history}\n\nLatest Question: {question}")
        ])
        reformulate_chain = reformulate_prompt | llm
        res = await reformulate_chain.ainvoke({"chat_history": chat_history_text, "question": req.query})
        search_query = res.content.strip()

    # Get context from retriever for streaming
    docs = smart_retriever.invoke(search_query)
    context = "\n\n".join(d.page_content for d in docs)

    # Build prompt with context and history
    prompt = RAG_CHAT_PROMPT.format_messages(
        question=req.query, 
        context=context,
        chat_history=chat_history_text,
        language=lang
    )
    prompt_text = "\n".join([msg.content for msg in prompt])

    async def event_generator():
        import json
        full_response = ""
        latency_ms = 0.0
        first_token_received = False
        
        async for chunk in llm.astream(prompt):
            if chunk.content:
                if not first_token_received:
                    latency_ms = analytics_agent.end_timer(start_time)
                    first_token_received = True
                full_response += chunk.content
                data_str = json.dumps({"content": chunk.content})
                yield f"data: {data_str}\n\n"

        # Update memory
        memory.save_context({"question": req.query}, {"answer": full_response})

        # Cache the response
        query_cache.put(cache_key, full_response)

        # Generate and send stats
        stats = analytics_agent.generate_stats(latency_ms, prompt_text, full_response)
        yield f"data: {json.dumps({'stats': stats})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/translate", response_model=TranslateResponse)
async def translate_endpoint(req: TranslateRequest):
    import json
    from langchain_core.prompts import ChatPromptTemplate
    
    if not req.texts:
        return TranslateResponse(translated_texts=[])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert translator. Translate the following list of HTML/text fragments into {language}. Return a valid JSON array of strings containing ONLY the translated texts, in the exact same order. Preserve all HTML tags, CSS classes, structure, and placeholders exactly as they are. Do NOT wrap in markdown code blocks. Output only the JSON array."),
        ("human", "{texts}")
    ])
    
    texts_json = json.dumps(req.texts)
    chain = prompt | llm
    res = await chain.ainvoke({"language": req.target_language, "texts": texts_json})
    
    # Clean up markdown code block if present
    content = res.content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    
    try:
        translated = json.loads(content)
        if isinstance(translated, list):
            return TranslateResponse(translated_texts=translated)
    except Exception as e:
        print("Translation parse error:", e, content)
    
    return TranslateResponse(translated_texts=req.texts) # fallback


@app.post("/refresh-jobs", response_model=RefreshResponse)
async def refresh_jobs() -> RefreshResponse:
    global jobs, jobs_doc
    jobs = fetch_live_jobs()
    jobs_doc = jobs_to_doc(jobs)
    smart_retriever.jobs_doc = jobs_doc
    return RefreshResponse(count=len(jobs), message="Jobs refreshed")


@app.get("/health")
async def health():
    return {"status": "ok", "jobs_loaded": len(jobs)}


@app.get("/")
async def get_frontend():
    with open("mfi_chatbot.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())