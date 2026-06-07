# MFI AI Chat Assistant — Setup Guide

This guide walks you through setting up and running the MFI AI Chat Assistant on your local machine.

---

## 1. Prerequisites

Before starting, ensure you have:

- **Python 3.10 or higher** — verify with `python --version`
- **OpenAI API Key** — get one at [platform.openai.com](https://platform.openai.com)

---

## 2. Create a Virtual Environment

It is strongly recommended to isolate dependencies in a virtual environment.

**Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

With your virtual environment activated:

```bash
pip install -r requirements.txt
```

This installs LangChain, ChromaDB, FastAPI, OpenAI SDK, and all other required packages.

---

## 4. Configure Environment Variables

Create a `.env` file in the project root and add your OpenAI API key:

```env
OPENAI_API_KEY=sk-your-actual-api-key-here
```

> **Security:** The `.env` file is listed in `.gitignore` and will not be committed to version control.

---

## 5. Ingest the Knowledge Base

The chatbot searches a local ChromaDB vector store. You must process the documents in `data/` before running the server for the first time.

```bash
python ingest.py
```

- Chunks all `.md` files in `data/` into 800-token segments (200-token overlap)
- Generates vector embeddings using OpenAI `text-embedding-3-small`
- Persists the index to the `vector_db/` directory

> **Note:** Whenever you add or update files in `data/`, stop the server, rerun `python ingest.py`, and restart the server. Live job listings are fetched dynamically from the Zoho Recruit RSS feed and **do not require ingestion**.

---

## 6. Start the Server

**Local access only:**
```bash
uvicorn api:app --port 8000
```

**Network access (e.g., test on your phone):**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Then open your browser and navigate to:
👉 [http://localhost:8000](http://localhost:8000)

For network access, find your local IP with `ipconfig` (Windows) or `ifconfig` (macOS/Linux) and open `http://<YOUR_IP>:8000` on any device on the same Wi-Fi network.

> **Hot Reload (development):** Add the `--reload` flag to automatically pick up code changes without restarting manually:
> ```bash
> uvicorn api:app --host 0.0.0.0 --port 8000 --reload
> ```

---

## 7. Using the Chat Assistant

| Feature | How to use |
|---------|-----------|
| **Ask a question** | Type in the input box and press Enter or click Send |
| **Language toggle** | Click `EN` / `JA` in the header to translate all messages instantly |
| **Analytics** | Click the 📊 icon to view latency, tokens, and session cost |
| **Refresh jobs** | Click "Refresh job listings" in the sidebar to reload live job data |
| **Quick prompts** | Click any item in the sidebar to send a pre-set query |

---

## 8. Updating the Knowledge Base

1. Add or edit `.md` files in the `data/` directory
2. Stop the server (`Ctrl+C`)
3. Run `python ingest.py` to rebuild the vector index
4. Restart the server

---

## 9. Verifying the Setup

Once the server is running, try asking:

> *"What is Money Forward India's privacy policy?"*

The assistant should retrieve the relevant policy document and provide a grounded response. You can then click `JA` to translate the entire conversation to Japanese instantly.

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Streaming SSE response to a query |
| `POST` | `/translate` | Translate an array of HTML text fragments |
| `POST` | `/refresh-jobs` | Re-fetch live job listings |
| `GET` | `/health` | Server health check |
| `GET` | `/` | Serve the frontend UI |
