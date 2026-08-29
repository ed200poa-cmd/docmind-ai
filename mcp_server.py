"""MCP server exposing DocMind's RAG retrieval as standard MCP tools.

Wraps the existing FAISS + SQLite retrieval layer (rag_engine, document_store) so any
MCP client can search the indexed documents and get back cited excerpts.

Deliberately read-only: rag_engine.remove_document() is NOT exposed. An MCP client is
a language model, and document deletion is irreversible.

Retrieval only, no answer synthesis: claude_qa.answer_question() is not exposed either.
Under MCP the client already is the model, so the server's job ends at supplying
excerpts and citations.

Run:  python mcp_server.py
"""

import contextvars
import functools
import json
import os
import sys
import logging
import contextlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

# stdio transport speaks JSON-RPC over stdout. Anything else written there corrupts the
# stream, so logs go to stderr before any project import can call logging.basicConfig().
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("docmind-mcp")

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

import document_store  # noqa: E402
import rag_engine  # noqa: E402

from mcp.server.fastmcp import FastMCP  # noqa: E402

# The project modules resolve the database relative to the working directory. An MCP
# client spawns this server from an arbitrary cwd, so pin it to the project.
document_store.DB_PATH = PROJECT_DIR / "docmind.db"

mcp = FastMCP("docmind")

_ready = False

# Per call tracing. A client sees only its own wall clock, which includes the
# model's thinking time, so the server has to say how long its own work took if
# the two are ever to be told apart. One JSON object per call, appended.
#
# DOCMIND_MCP_TRACE overrides the path. Set it to "off" to disable.
TRACE_PATH = Path(os.getenv("DOCMIND_MCP_TRACE") or (PROJECT_DIR / "mcp-trace.jsonl"))

_annotations: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "docmind_mcp_annotations", default=None)


def _annotate(**fields: Any) -> None:
    """Let a tool body add identifying detail to its own trace record."""
    current = _annotations.get()
    if current is not None:
        current.update(fields)


def _write_trace(record: dict) -> None:
    if str(TRACE_PATH) == "off":
        return
    try:
        TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TRACE_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        # Tracing must never take the server down.
        logger.warning("could not write trace: %s", exc)


def _instrumented(fn: Callable) -> Callable:
    """Time one tool call and record it. Never records the response body.

    What is kept is what identifies the call and lets it be matched against a
    client side record: the tool, its arguments, how long the server spent, how
    many results went back, and for a retrieval the chunk ids. The excerpt text
    itself is not written, because the trace would otherwise become a second
    copy of the corpus.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        token = _annotations.set({})
        started = time.perf_counter()
        record = {
            "ts_start": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "tool": fn.__name__,
            "arguments": {k: v for k, v in kwargs.items()},
        }
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            record.update(ok=False, error=type(exc).__name__, error_message=str(exc)[:200])
            raise
        else:
            record["ok"] = True
            record["result_count"] = len(result) if isinstance(result, (list, tuple)) else 1
            return result
        finally:
            record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
            record["ts_end"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            record.update(_annotations.get() or {})
            _annotations.reset(token)
            _write_trace(record)
    return wrapper


@contextlib.contextmanager
def _stdout_to_stderr() -> Iterator[None]:
    """Keep stdout clean for JSON-RPC.

    fastembed prints model-download progress and faiss can emit to stdout on first use.
    """
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _ensure_ready() -> None:
    """Build the FAISS index once, on first use rather than at import."""
    global _ready
    if _ready:
        return
    with _stdout_to_stderr():
        document_store.init_db()
        rag_engine.init_rag()
    _ready = True


@mcp.tool()
@_instrumented
def search_documents(
    question: str,
    doc_id: Optional[str] = None,
    top_k: int = 5,
) -> list[dict]:
    """Search the indexed documents and return the most relevant excerpts.

    Each excerpt carries its source document so the answer can be cited. `page` is the
    page number for PDFs and null for plain-text documents, which have no pagination.
    Returns an empty list when nothing is indexed or nothing is relevant.

    Args:
        question: Natural-language query to search for.
        doc_id: Restrict the search to a single document. Omit to search everything.
        top_k: Maximum number of excerpts to return (1-20).
    """
    _ensure_ready()

    if not question.strip():
        raise ValueError("question must not be empty")
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")

    if doc_id and document_store.get_document(doc_id) is None:
        raise ValueError(f"No document with doc_id '{doc_id}'. Call list_documents first.")

    with _stdout_to_stderr():
        chunks = rag_engine.search(question, doc_id=doc_id, top_k=top_k)

    _annotate(chunk_ids=[c["chunk_id"] for c in chunks])

    # Resolve filenames so the client can cite by name, not opaque id.
    docs: dict[str, dict | None] = {}
    for chunk in chunks:
        cid = chunk["doc_id"]
        if cid not in docs:
            docs[cid] = document_store.get_document(cid)

    results = []
    for c in chunks:
        doc = docs[c["doc_id"]]
        # Only PDFs are paginated. parse_txt() stores every chunk as page 1, which a
        # client would otherwise cite as a real page.
        is_pdf = doc is not None and doc["file_type"] == "pdf"
        results.append({
            "text": c["chunk_text"],
            "filename": doc["filename"] if doc else "unknown",
            "doc_id": c["doc_id"],
            "page": c["page_num"] if is_pdf else None,
            "relevance_score": c["relevance_score"],
        })
    return results


@mcp.tool()
@_instrumented
def list_documents() -> list[dict]:
    """List every indexed document available to search."""
    _ensure_ready()
    return [
        {
            "doc_id": d["doc_id"],
            "filename": d["filename"],
            "file_type": d["file_type"],
            "chunk_count": d["chunk_count"],
            "uploaded_at": d["upload_time"],
        }
        for d in document_store.get_all_documents()
    ]


@mcp.tool()
@_instrumented
def get_document_info(doc_id: str) -> dict:
    """Return metadata for one indexed document."""
    _ensure_ready()
    doc = document_store.get_document(doc_id)
    if doc is None:
        raise ValueError(f"No document with doc_id '{doc_id}'. Call list_documents first.")
    return {
        "doc_id": doc["doc_id"],
        "filename": doc["filename"],
        "file_type": doc["file_type"],
        "chunk_count": doc["chunk_count"],
        "file_size_bytes": doc["file_size"],
        "uploaded_at": doc["upload_time"],
    }


@mcp.resource("docmind://documents")
def documents_resource() -> str:
    """The indexed document collection, as a readable summary."""
    _ensure_ready()
    docs = document_store.get_all_documents()
    if not docs:
        return "No documents are indexed."
    lines = [f"{len(docs)} indexed document(s):", ""]
    lines += [
        f"- {d['filename']} (doc_id={d['doc_id']}, {d['chunk_count']} chunks)"
        for d in docs
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    logger.info("Starting DocMind MCP server (stdio) from %s", PROJECT_DIR)
    mcp.run()
