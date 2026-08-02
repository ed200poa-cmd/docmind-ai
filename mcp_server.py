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

import sys
import logging
import contextlib
from pathlib import Path
from typing import Iterator, Optional

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
