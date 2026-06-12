# DocMind AI Demo

An AI-powered document Q&A system that lets users upload PDFs or text files and ask questions in natural language. Answers come with cited source excerpts — never hallucinated.

---

## What is RAG?

RAG (Retrieval-Augmented Generation) is a technique where an AI first **searches your documents** for relevant passages, then uses those passages as context to generate an accurate answer. This ensures the AI answers only from your content, not from guesswork.

---

## How It Prevents Hallucination

1. Every question is converted to a vector (a mathematical representation of meaning)
2. FAISS searches the document index for the top 5 most similar passages
3. Only those passages are sent to Claude — the system prompt explicitly forbids answering from general knowledge
4. If the answer isn't in the documents, Claude says so

Claude **never invents facts**. Every answer is traceable to a specific document excerpt shown in the UI.

---

## Features

- Upload PDF or TXT documents (up to 10 MB)
- FAISS vector search — no external API required for search
- Claude claude-haiku for fast, accurate answers
- Source citations with relevance scores
- Filter Q&A by specific document
- Clean two-panel web UI
- Call history via SQLite
- Pre-loaded demo document (ACME HR policy)

---

## Quick Start

### 1. Install dependencies

```bash
cd docmind_demo
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~90 MB) on first run. This is a one-time download.

### 2. Configure environment

```bash
cp .env.example .env
# Open .env and add your ANTHROPIC_API_KEY
```

### 3. Run the server

```bash
uvicorn main:app --reload --port 8000
```

Open your browser at **http://localhost:8000**

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Upload a PDF or TXT document |
| `POST` | `/ask` | Ask a question (JSON body) |
| `GET` | `/documents` | List all uploaded documents |
| `DELETE` | `/documents/{doc_id}` | Delete a document |
| `GET` | `/health` | Health check + stats |

### Example: Upload a document

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@your_document.pdf"
```

Response:
```json
{ "doc_id": "a3f9c1b2", "filename": "your_document.pdf", "chunk_count": 42, "status": "ready" }
```

### Example: Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the vacation policy?", "doc_id": "a3f9c1b2"}'
```

Response:
```json
{
  "answer": "According to Section 2, employees in their first two years receive 10 vacation days per year...",
  "sources": [
    { "chunk_text": "Full-time employees accrue paid vacation...", "page_num": 1, "relevance_score": 0.89 }
  ],
  "doc_used": "company_policy.txt",
  "model": "claude-haiku-20240307"
}
```

---

## Sample Questions (Demo Document)

The pre-loaded `company_policy.txt` is a fictional ACME Corporation HR policy. Try these:

- *"How many vacation days do new employees get?"*
- *"What is the parental leave policy?"*
- *"What health insurance plans does ACME offer?"*
- *"Can I carry over unused vacation days?"*
- *"What is the remote work policy?"*
- *"What happens to equipment when I leave the company?"*
- *"How do performance reviews work?"*
- *"What is the mental health benefit?"*

---

## Deploy to Railway

```bash
npm install -g @railway/cli
railway login && railway init && railway up
```

Set `ANTHROPIC_API_KEY` in the Railway dashboard environment variables.

---

## File Structure

```
docmind_demo/
├── main.py              # FastAPI app — all routes
├── rag_engine.py        # FAISS index + embeddings + text chunking
├── claude_qa.py         # Claude Q&A with RAG context
├── document_store.py    # SQLite persistence layer
├── static/
│   └── index.html       # Two-panel web UI (Tailwind CSS)
├── demo_docs/
│   └── company_policy.txt   # Pre-loaded sample document
├── requirements.txt
├── Procfile
└── .env.example
```

---

## Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.12** | Runtime |
| **FastAPI + Uvicorn** | Async web server |
| **Anthropic Claude API** | AI answer generation (claude-haiku-20240307) |
| **FAISS (faiss-cpu)** | Vector similarity search |
| **sentence-transformers** | Local text embeddings (all-MiniLM-L6-v2) |
| **PyMuPDF** | PDF text extraction |
| **SQLite** | Document metadata + chunk storage |
| **Railway** | Cloud deployment |

---

Built by **Edward Kim** — AI Developer
