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

# 5 MB, down from 10. The seed document is 7 KB and a 5 MB PDF already runs to
# hundreds of pages, so this refuses nothing a visitor would try while halving
# what a single request can put on the volume.
MAX_FILE_SIZE = 5 * 1024 * 1024
DEMO_DOC_PATH = Path(__file__).parent / "demo_docs" / "company_policy.txt"
SEED_FILENAME = "company_policy.txt"


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    document_store.init_db()
    rag_engine.init_rag()

    # The seed document, and the guarantee that there is always exactly one.
    #
    # Keyed on the is_seed flag rather than the filename: a visitor may upload
    # their own company_policy.txt, and a name check would then mistake it for
    # ours -- protecting their file from deletion and leaving the real seed
    # unprotected.
    #
    # Three cases, in order:
    #   * a seed row exists          -> nothing to do, and no model is loaded,
    #                                   which is the whole point of the volume
    #   * a row from before the flag -> adopt it, still no model
    #   * nothing                    -> ingest, which loads the model once ever
    seed = document_store.get_seed()
    if seed is None:
        legacy = document_store.get_document_by_filename(SEED_FILENAME)
        if legacy:
            document_store.mark_seed(legacy["doc_id"])
            logger.info("Adopted existing %s as the seed document (%s).",
                        SEED_FILENAME, legacy["doc_id"])
        elif DEMO_DOC_PATH.exists():
            try:
                content = DEMO_DOC_PATH.read_bytes()
                result = rag_engine.process_document(
                    file_bytes=content,
                    filename=SEED_FILENAME,
                    file_size=len(content),
                )
                document_store.mark_seed(result["doc_id"])
                logger.info("Seed document loaded: %s", result)
            except Exception as exc:
                logger.warning("Could not auto-load seed document: %s", exc)
    else:
        logger.info("Seed document present (%s); not re-ingesting.",
                    seed["doc_id"])

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
            detail=(
                f"That file is {len(file_bytes) / 1024 / 1024:.1f} MB. This demo "
                f"accepts up to {MAX_FILE_SIZE // 1024 // 1024} MB per file — "
                f"try a shorter document, or a few pages of this one."
            ),
        )

    # Sweep before accepting, so a visitor's own upload is never the one
    # evicted to make room for it.
    rag_engine.enforce_retention()

    count, total = document_store.visitor_totals()
    if total + len(file_bytes) > rag_engine.MAX_VISITOR_BYTES:
        raise HTTPException(
            status_code=507,
            detail=(
                "This demo is temporarily full — a few people are trying it at "
                f"once. Uploads are cleared after {rag_engine.UPLOAD_TTL_HOURS} "
                "hours, so this frees up shortly. The pre-loaded HR policy "
                "document is still there to ask questions about."
            ),
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

    # SCOPE. An unscoped question used to search every chunk of every
    # document, which meant one visitor's upload was answerable by the next
    # one to open the page. It now falls back to the seed document.
    #
    # This costs the demo nothing: the frontend sends no doc_id only before
    # anything has been uploaded, and at that moment the seed is the only
    # thing worth searching. After an upload the browser holds the doc_id and
    # sends it, so a visitor still asks about their own file.
    #
    # It is not access control -- someone who learns another doc_id can still
    # pass it -- and it is not meant to be. It closes the path where a
    # visitor reads a stranger's document without trying to.
    scope_id = body.doc_id
    if not scope_id:
        seed = document_store.get_seed()
        if seed:
            scope_id = seed["doc_id"]

    doc_name: str | None = None
    if body.doc_id:
        doc = document_store.get_document(body.doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document '{body.doc_id}' not found.")
        doc_name = doc["filename"]

    chunks = rag_engine.search(question=question, doc_id=scope_id)

    result = claude_qa.answer_question(
        question=question,
        chunks=chunks,
        doc_name=doc_name,
    )
    return JSONResponse(result)


@app.get("/documents")
async def list_documents():
    # The list is what one visitor sees of another, so it is also where the
    # sweep is most useful: an expired upload is gone before it is shown.
    rag_engine.enforce_retention()
    docs = document_store.get_all_documents()
    return JSONResponse({"total": len(docs), "documents": docs})


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    try:
        deleted = rag_engine.remove_document(doc_id)
    except document_store.SeedDocumentProtected:
        raise HTTPException(
            status_code=403,
            detail=(
                "The pre-loaded sample document cannot be removed — it is what "
                "this demo has to answer questions about. Documents you upload "
                "can be deleted, and clear themselves after "
                f"{rag_engine.UPLOAD_TTL_HOURS} hours."
            ),
        )
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
