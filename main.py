import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import document_store
import rag_engine
import claude_qa

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
DEMO_DOC_PATH = Path(__file__).parent / "demo_docs" / "company_policy.txt"


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    document_store.init_db()
    rag_engine.init_rag()

    # Auto-load demo document on first run
    if not document_store.filename_exists("company_policy.txt"):
        if DEMO_DOC_PATH.exists():
            try:
                content = DEMO_DOC_PATH.read_bytes()
                result = rag_engine.process_document(
                    file_bytes=content,
                    filename="company_policy.txt",
                    file_size=len(content),
                )
                logger.info("Demo document loaded: %s", result)
            except Exception as exc:
                logger.warning("Could not auto-load demo document: %s", exc)

    logger.info("DocMind AI is ready.")
    yield


app = FastAPI(
    title="DocMind AI",
    description="RAG Document Q&A powered by Claude + FAISS",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    doc_id: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: .pdf, .txt",
        )

    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // 1024 // 1024} MB.",
        )

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = rag_engine.process_document(
            file_bytes=file_bytes,
            filename=file.filename,
            file_size=len(file_bytes),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Upload error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")

    return JSONResponse(result)


@app.post("/ask")
async def ask_question(body: AskRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="Question is too long (max 1000 chars).")

    doc_name: str | None = None
    if body.doc_id:
        doc = document_store.get_document(body.doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document '{body.doc_id}' not found.")
        doc_name = doc["filename"]

    chunks = rag_engine.search(question=question, doc_id=body.doc_id)

    result = claude_qa.answer_question(
        question=question,
        chunks=chunks,
        doc_name=doc_name,
    )
    return JSONResponse(result)


@app.get("/documents")
async def list_documents():
    docs = document_store.get_all_documents()
    return JSONResponse({"total": len(docs), "documents": docs})


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    deleted = rag_engine.remove_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return JSONResponse({"deleted": doc_id, "status": "ok"})


@app.get("/health")
async def health():
    index_size = rag_engine._index.ntotal if rag_engine._index else 0
    docs = document_store.get_all_documents()
    return JSONResponse({
        "status": "ok",
        "service": "DocMind AI",
        "documents_indexed": len(docs),
        "vectors_in_index": index_size,
        "embedding_model": rag_engine.EMBEDDING_MODEL,
        "llm_model": claude_qa.MODEL,
        "anthropic_key_set": bool(os.getenv("ANTHROPIC_API_KEY")),
    })


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
