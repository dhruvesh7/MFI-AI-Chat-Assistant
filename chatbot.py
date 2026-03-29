from __future__ import annotations

from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma  # type: ignore[import-not-found]
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

import html
import re
import xml.etree.ElementTree as ET
import requests

load_dotenv()

DB_PATH = "vector_db"

JOBS_URL = "https://moneyforward.zohorecruit.in/jobs"
JOBS_RSS_URL = "https://moneyforward.zohorecruit.in/jobs/Careers/rss"

# Chat models use ChatPromptTemplate (see langchain_classic stuff_prompt CHAT_PROMPT).
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

# RSS <description> is HTML; cap size so the live-jobs doc stays reasonable.
_MAX_JOB_DESC_CHARS = 12000


def _rss_html_to_plain(raw: str) -> str:
    """Turn RSS item HTML into readable plain text for RAG."""
    if not raw:
        return ""
    t = html.unescape(raw)
    t = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", t)
    t = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", t)
    t = re.sub(r"(?i)<br\s*/?>", "\n", t)
    t = re.sub(r"(?i)</p>", "\n\n", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n\n", t)
    return t.strip()


def _truncate_desc(text: str) -> str:
    if len(text) <= _MAX_JOB_DESC_CHARS:
        return text
    cut = text[: _MAX_JOB_DESC_CHARS].rsplit(" ", 1)[0]
    return cut + "\n\n[Description truncated…]"


# =========================
# Fetch Live Jobs
# =========================
def fetch_live_jobs():

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    jobs = []

    try:
        r = requests.get(JOBS_RSS_URL, headers=headers, timeout=15)
        r.raise_for_status()

        root = ET.fromstring(r.content)

        for item in root.findall(".//item"):
            t_el, l_el, d_el = item.find("title"), item.find("link"), item.find("description")
            if t_el is None or not (t_el.text or "").strip():
                continue
            title = html.unescape(t_el.text.strip())
            link = (l_el.text or "").strip() if l_el is not None else JOBS_URL
            raw_desc = d_el.text if d_el is not None and d_el.text else ""
            desc = html.unescape(raw_desc)

            dept = "N/A"
            loc = "N/A"

            m1 = re.search(r"Category:\s*([^<]+)", desc)
            m2 = re.search(r"Location:\s*([^<]+)", desc)

            if m1:
                dept = m1.group(1).strip()

            if m2:
                loc = m2.group(1).strip()

            description = _truncate_desc(_rss_html_to_plain(raw_desc))

            jobs.append({
                "title": title,
                "department": dept,
                "location": loc,
                "link": link,
                "description": description,
            })

    except Exception as e:
        print("Job fetch error:", e)

    return jobs


# =========================
# Convert Jobs → Document
# =========================
def jobs_to_doc(jobs):

    if not jobs:
        text = (
            "Money Forward India Job Openings\n\n"
            "No listings could be loaded from the careers RSS feed. "
            f"Check {JOBS_URL} for the latest openings.\n"
        )
        return Document(page_content=text, metadata={"source": "live_jobs"})

    text = "Money Forward India Job Openings\n\n"

    for i, job in enumerate(jobs, 1):
        desc = job.get("description") or ""
        text += f"""
{i}. {job['title']}
Department: {job['department']}
Location: {job['location']}
Apply: {job['link']}

Role description (from careers RSS):
{desc}

"""

    return Document(
        page_content=text,
        metadata={"source": "live_jobs"}
    )


# =========================
# Custom Retriever
# =========================
class SmartRetriever(BaseRetriever):
    """Must use Pydantic-declared fields (BaseRetriever is a BaseModel)."""

    base_retriever: Any
    jobs_doc: Document

    _JOB_WORDS = frozenset({
        "job", "opening", "hiring", "career", "intern", "apply",
        "position", "vacancy", "recruit", "role", "description",
        "innovation", "productivity", "engineer", "salary", "requirement",
    })

    @staticmethod
    def _expand_query_for_vector_search(query: str) -> str:
        """
        Embeddings often miss acronym / short queries vs. formal titles in `data/*.md`.
        Add aligned phrases so chunks from the knowledge base surface (e.g. privacy policy).
        """
        q = query.strip()
        ql = q.lower()
        hints: list[str] = []
        if "mfi" in ql:
            hints.append("Money Forward India")
        if any(
            w in ql
            for w in (
                "privacy",
                "personal information",
                "personal data",
                "data protection",
                "pii",
            )
        ):
            hints.append("Personal Information Protection Policy privacy")
        if "security" in ql and "policy" in ql:
            hints.append("security policy information")
        if any(w in ql for w in ("regulation", "compliance", "rbi", "sebi")):
            hints.append("India regulations compliance")
        if "interview" in ql or "faq" in ql:
            hints.append("interview hiring FAQ")
        if "hiring" in ql or "recruitment process" in ql:
            hints.append("hiring process recruitment")
        if not hints:
            return q
        return q + " " + " ".join(hints)

    def _get_relevant_documents(self, query: str) -> list[Document]:
        retrieval_q = self._expand_query_for_vector_search(query)
        docs: list[Document] = self.base_retriever.invoke(retrieval_q)
        ql = query.lower()

        # Prefer the dedicated policy file when the question is clearly about privacy,
        # so other knowledge-base chunks do not dilute or contradict the answer.
        if any(
            w in ql
            for w in (
                "privacy",
                "personal information",
                "personal data",
                "data protection",
                "pii",
            )
        ):
            priv = [
                d
                for d in docs
                if "privacy" in (d.metadata.get("source") or "").lower()
            ]
            if priv:
                priv_sources = {d.metadata.get("source") for d in priv}
                rest = [
                    d for d in docs
                    if d.metadata.get("source") not in priv_sources
                ][:2]
                docs = priv + rest

        if any(w in ql for w in self._JOB_WORDS):
            return [self.jobs_doc] + [
                d for d in docs
                if d.metadata.get("source") != "live_jobs"
            ]
        return docs


# =========================
# Main
# =========================
def main():

    print("Loading vector DB...")

    embeddings = OpenAIEmbeddings()

    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    try:
        n_chunks = db._collection.count()
        print(f"Knowledge base: {n_chunks} chunks in vector store (run `python ingest.py` after editing data/)")
    except Exception:
        pass

    # MMR + higher fetch_k improves recall when queries don't match doc wording exactly.
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 24, "lambda_mult": 0.55},
    )

    print("Fetching live jobs...")
    jobs = fetch_live_jobs()

    print("Jobs found:", len(jobs))

    jobs_doc = jobs_to_doc(jobs)

    smart_retriever = SmartRetriever(
        base_retriever=retriever,
        jobs_doc=jobs_doc
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=smart_retriever,
        chain_type_kwargs={"prompt": RAG_CHAT_PROMPT},
    )

    print("\nMoney Forward India Chat Assistant")
    print("Type 'exit' to quit")
    print("Type 'refresh jobs' to reload jobs")

    while True:

        query = input("\nYou: ")

        if query.lower() == "exit":
            break

        if query.lower() == "refresh jobs":
            print("Refreshing jobs...")
            jobs = fetch_live_jobs()
            jobs_doc = jobs_to_doc(jobs)
            smart_retriever.jobs_doc = jobs_doc
            print("Jobs updated:", len(jobs))
            continue

        result = qa.invoke({"query": query})

        print("\nBot:", result["result"])


if __name__ == "__main__":
    main()