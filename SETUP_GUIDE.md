# MFI Chatbot Setup Guide

This comprehensive guide will walk you through setting up and running the MFI RAG Chatbot on your local machine.

## 1. Prerequisites

Before starting, ensure you have the following installed and obtained:

- **Python 3.10 or higher**: Verify your installation by running `python --version` in your terminal.
- **OpenAI API Key**: The chatbot relies on OpenAI's `text-embedding-ada-002` for vector embeddings and `GPT-4o-mini` for generation.
  - Obtain an API key by signing up at the [OpenAI Platform](https://platform.openai.com).
  - Navigate to the **API keys** section in your dashboard and generate a new secret key.

## 2. Environment Setup

It is strongly recommended to use a Python virtual environment to keep dependencies isolated.

### On Windows:
```cmd

python -m venv .venv
.venv\Scripts\activate

```

### On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

With your virtual environment activated, install the required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

This will install packages like LangChain, ChromaDB, FastAPI, and OpenAI SDK.

## 4. Environment Variables Configuration

Create a file named `.env` in the root directory of the project (the same folder as this guide). Open it and add your OpenAI API key:

```env
OPENAI_API_KEY=sk-your-actual-api-key-here
```
> **Security Aspect:** Ensure this `.env` file is excluded from version control (it is usually in the `.gitignore` by default).

## 5. Ingesting the Knowledge Base

The chatbot requires a local ChromaDB vector store to search for answers efficiently. You must process and ingest the documents situated in the `data/` directory.

Run the ingestion script:

```bash
python ingest.py
```

- This script chunks the `.md` documents in the `data/` folder and generates vector embeddings.
- The results are persistently stored in the `vector_db/` directory.
- *Note:* Whenever you add or update files in `data/`, you should rerun this script. The live job listings are fetched dynamically and don't require ingestion.

## 6. Running the Chatbot Application

The application comes with two interfaces to interact with.

### Option A: Web API & UI (Recommended)
This launches the FastAPI backend and serves the interactive HTML frontend.

**For local access (this computer only):**
```bash
uvicorn api:app --port 8000
```
Once the server has started, open your web browser and navigate to:
👉 [http://localhost:8000](http://localhost:8000)

> **Pro Tip:** The new UI supports **Real-time Streaming**. You will see the chatbot's response appear chunk-by-chunk as it is generated, providing a much more interactive experience.

**For network access (e.g., testing on your phone):**
To allow other devices on your local Wi-Fi network to connect to the chatbot, bind the server to `0.0.0.0`:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
Find your computer's local network IP address (e.g., `192.168.x.x` by running `ipconfig` on Windows or `ifconfig` on macOS/Linux). Then, on your phone's browser, navigate to:
👉 `http://<YOUR_IP_ADDRESS>:8000`

> **Note:** Ensure your phone is on the exact same Wi-Fi network. If the page times out, check that your computer's firewall (like Windows Defender) allows inbound connections for port `8000`.

---

## 7. Managing the Knowledge Base

If you add new Markdown files to the `data/` folder or modify existing ones, the system won't see them automatically. You MUST:
1. Stop the server (`Ctrl+C`).
2. Run `python ingest.py` to update the vector database.
3. Start the server again.

The **Live Jobs Feed** does NOT require ingestion; it is fetched fresh every time the server starts or when you click "Refresh job listings" in the UI.

---

## 8. Verifying the Setup

### Option B: Command Line Interface (CLI)
You can directly interact with the chatbot in your terminal for quick testing:

```bash
python chatbot.py
```
- Type any question to interact with the LLM.
- Type `refresh jobs` to re-fetch live listings manually.
- Type `exit` to quit the CLI.

---

## 7. Verifying the Setup

To make sure everything is working correctly, ask a question such as:
> *"What is Money Forward India's privacy policy?"*

The chatbot should retrieve the context from your vectorized documents and provide a grounded response, completely free of generic AI hallucinations.
