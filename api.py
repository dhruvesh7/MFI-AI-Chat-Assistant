"""
api.py — thin FastAPI wrapper around chatbot.py
Run: uvicorn api:app --reload
"""

from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chatbot import (
    fetch_live_jobs,
    jobs_to_doc,
    SmartRetriever,
)

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.chains import RetrievalQA
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
            "information, say you don't know—do not invent details.\n\n{context}",
        ),
        ("human", "{question}"),
    ]
)

app = FastAPI(title="Money Forward India Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load once on startup ──
embeddings = OpenAIEmbeddings()
db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 24, "lambda_mult": 0.55},
)
jobs = fetch_live_jobs()
jobs_doc = jobs_to_doc(jobs)
smart_retriever = SmartRetriever(base_retriever=retriever, jobs_doc=jobs_doc)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=smart_retriever,
    chain_type_kwargs={"prompt": RAG_CHAT_PROMPT},
)


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    result: str


class RefreshResponse(BaseModel):
    count: int
    message: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    result = qa.invoke({"query": req.query})
    return ChatResponse(result=result["result"])


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
