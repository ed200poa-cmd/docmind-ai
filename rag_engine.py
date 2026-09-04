from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

import document_store

# `faiss` and `fastembed` are imported inside the functions that use them.
# fastembed drags in onnxruntime and tokenizers, faiss its own native library,
# and together they were most of a 335MB resident set held to answer a handful
# of requests a week.
#
# THIS ONLY PAYS BECAUSE THE BOOT PATH NO LONGER NEEDS THE MODEL. An earlier
# attempt at exactly this change was reverted on 2026-09-03 after measuring
# that it saved nothing: main.py's lifespan re-ingested the demo document on
# every boot, because the SQLite file sat on the container's writable layer
# and did not survive a deploy, and ingestion loads the model. The database is
# on a volume now (see document_store.DB_PATH), the demo document is ingested
# once ever, and deferring the import finally reaches something.
#
# Rebuilding the index does NOT need the model: the embeddings are already in
# SQLite as BLOBs and _rebuild_index only reads them back. Only NEW text -- an
# upload, or a question -- needs it.
#
# `from __future__ import annotations` makes the hints below strings at
# runtime, so `Optional[faiss.Index]` no longer requires faiss to be loaded.
# numpy stays eager: document_store imports it anyway.
if TYPE_CHECKING:  # pragma: no cover
    import faiss
    from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CHUNK_SIZE = 500
TOP_K = 5

_model: Optional[TextEmbedding] = None
_index: Optional[faiss.Index] = None
_chunk_id_map: list[int] = []  # faiss_position -> sqlite chunk.id


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def get_model() -> TextEmbedding:
    global _model
    if _model is None:
        # The import sits with the construction: the first caller that needs
        # an embedding pays for onnxruntime and the weights, and no boot does.
        from fastembed import TextEmbedding

        logger.info("Loading embedding model '%s'…", EMBEDDING_MODEL)
        _model = TextEmbedding(model_name=EMBEDDING_MODEL)
        logger.info("Embedding model ready.")
    return _model


def init_rag() -> None:
    """Rebuild FAISS index from persisted SQLite chunks on startup."""
    _rebuild_index()
    logger.info("RAG engine ready. Index size: %d vectors", _index.ntotal if _index else 0)


def _rebuild_index() -> None:
    # faiss, but not the model: the vectors are already in SQLite.
    import faiss

    global _index, _chunk_id_map
    _index = faiss.IndexFlatIP(EMBEDDING_DIM)
    _chunk_id_map = []

    chunks = document_store.load_all_chunks_for_index()
    if not chunks:
        return

    embeddings = [c["embedding"] for c in chunks if c.get("embedding") is not None]
    ids = [c["id"] for c in chunks if c.get("embedding") is not None]

    if not embeddings:
        return

    matrix = np.stack(embeddings).astype(np.float32)
    faiss.normalize_L2(matrix)
    _index.add(matrix)
    _chunk_id_map = ids
    logger.info("FAISS index rebuilt with %d vectors.", len(ids))


# ---------------------------------------------------------------------------
# Text parsing
# ---------------------------------------------------------------------------

def parse_pdf(file_bytes: bytes) -> list[tuple[int, str]]:
    import fitz  # PyMuPDF
    pages: list[tuple[int, str]] = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append((i + 1, text))
    doc.close()
    return pages


def parse_txt(file_bytes: bytes) -> list[tuple[int, str]]:
    text = file_bytes.decode("utf-8", errors="replace").strip()
    return [(1, text)]


_SEPARATOR_LINE_RE = re.compile(r"^[━_\-=*~#—–]{3,}$")


_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n")


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Split on blank lines, returning (start, end) offsets into `text`.

    Offsets rather than strings, because a chunk is later emitted as the single
    slice text[first_start:last_end]. Slicing is what makes a chunk a verbatim
    contiguous substring of its source by construction, for every separator
    form the source happens to use -- a doubled blank line, a blank line holding
    spaces, an indented or trailing-space paragraph. Rebuilding chunks by
    joining stripped paragraphs with a literal "\n\n" only reproduced the source
    when the source already used exactly that separator.

    Each span is trimmed of surrounding whitespace, so it starts and ends on a
    non-space character, and empty spans are dropped.
    """
    bounds: list[tuple[int, int]] = []
    pos = 0
    for match in _BLANK_LINE_RE.finditer(text):
        bounds.append((pos, match.start()))
        pos = match.end()
    bounds.append((pos, len(text)))

    spans: list[tuple[int, int]] = []
    for start, end in bounds:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))
    return spans


def _split_paragraphs(text: str) -> list[str]:
    """The paragraphs of `text`, each an exact substring of it."""
    return [text[start:end] for start, end in _paragraph_spans(text)]


def _is_heading_block(paragraph: str) -> bool:
    """True for a decorative section-heading paragraph: one or more separator
    lines (a run of box-drawing/dash characters) wrapping exactly one title
    line, e.g. "━━━\nSECTION 3: HEALTH AND WELLNESS BENEFITS\n━━━"."""
    lines = [l.strip() for l in paragraph.splitlines() if l.strip()]
    if not lines:
        return False
    non_sep = [l for l in lines if not _SEPARATOR_LINE_RE.match(l)]
    has_sep = any(_SEPARATOR_LINE_RE.match(l) for l in lines)
    return has_sep and len(non_sep) == 1


def _chunk_pages(pages: list[tuple[int, str]]) -> list[dict]:
    """Pack paragraphs into ~CHUNK_SIZE chunks, never splitting a paragraph
    mid-sentence and never leaving a section heading stranded without the body
    text that follows it (a heading always starts a fresh chunk).

    A chunk is emitted as a single slice of its source page, spanning from the
    start of its first paragraph to the end of its last. So every chunk is a
    verbatim, contiguous substring of the source by construction -- citation
    grounding is structurally guaranteed rather than dependent on the source
    separating its paragraphs with exactly "\n\n".
    """
    chunks: list[dict] = []
    for page_num, page_text in pages:
        spans = _paragraph_spans(page_text)

        buffer: list[tuple[int, int]] = []
        buffer_len = 0

        def flush():
            nonlocal buffer, buffer_len
            if buffer:
                text = page_text[buffer[0][0]:buffer[-1][1]]
                if len(text) > 40:
                    chunks.append({"text": text, "page_num": page_num})
            buffer = []
            buffer_len = 0

        for span in spans:
            para_len = span[1] - span[0]
            if _is_heading_block(page_text[span[0]:span[1]]):
                flush()  # heading always opens a new chunk together with its body
                buffer.append(span)
                buffer_len = para_len
                continue

            # +2 for the blank line that will separate this paragraph from the
            # previous one; the packing budget is unchanged from the original.
            if buffer and buffer_len + para_len + 2 > CHUNK_SIZE:
                flush()

            buffer.append(span)
            buffer_len += para_len + 2

        flush()
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed(texts: list[str]) -> np.ndarray:
    model = get_model()
    vecs = list(model.embed(texts))
    return np.array(vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_document(file_bytes: bytes, filename: str, file_size: int) -> dict:
    """Parse → chunk → embed → persist → rebuild index."""
    ext = Path(filename).suffix.lower()
    file_type = "pdf" if ext == ".pdf" else "txt"

    pages = parse_pdf(file_bytes) if file_type == "pdf" else parse_txt(file_bytes)
    if not pages:
        raise ValueError("No text could be extracted from this file.")

    chunks = _chunk_pages(pages)
    if not chunks:
        raise ValueError("Document is too short to process.")

    texts = [c["text"] for c in chunks]
    embeddings = embed(texts)

    doc_id = document_store.create_document(filename, file_type, file_size)
    document_store.save_chunks(doc_id, chunks, embeddings)

    _rebuild_index()
    logger.info("Processed '%s': %d chunks, doc_id=%s", filename, len(chunks), doc_id)

    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunk_count": len(chunks),
        "status": "ready",
    }


def search(question: str, doc_id: Optional[str] = None, top_k: int = TOP_K) -> list[dict]:
    """Return top-k relevant chunks for a question."""
    if _index is None or _index.ntotal == 0:
        return []

    import faiss

    q_vec = embed([question])
    faiss.normalize_L2(q_vec)

    # Fetch extra candidates when filtering by doc_id
    k = min(_index.ntotal, top_k * 6 if doc_id else top_k)
    scores, indices = _index.search(q_vec, k)

    results: list[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_chunk_id_map):
            continue
        chunk = document_store.get_chunk_by_id(_chunk_id_map[idx])
        if chunk is None:
            continue
        if doc_id and chunk["doc_id"] != doc_id:
            continue

        results.append({
            "chunk_id": chunk["id"],
            "chunk_text": chunk["chunk_text"],
            "page_num": chunk["page_num"],
            "doc_id": chunk["doc_id"],
            "relevance_score": round(float(score), 4),
        })
        if len(results) >= top_k:
            break

    return results


def remove_document(doc_id: str) -> bool:
    deleted = document_store.delete_document(doc_id)
    if deleted:
        _rebuild_index()
    return deleted


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------
# WHY THIS EXISTS. Until 2026-09-04 the database sat on the container's
# writable layer, so every visitor upload was swept away by the next deploy.
# That was not a policy, it was an accident of storage, and the sweep ran
# every six to nineteen days -- the observed gap between docmind deploys. In
# that window one visitor's file was listed to, and answerable by, the next.
#
# Putting the database on a volume fixed a memory cost and made that window
# unbounded. So the window is now a decision instead of a side effect.
#
# FOUR HOURS. A recruiter trying the demo needs minutes. Four hours covers
# someone who opens it, is interrupted, and comes back after lunch, and is
# short enough that two different visitors are unlikely to overlap in it. Two
# felt tight for the interrupted case; a working day is long enough that the
# window stops meaning anything.
#
# All four numbers below take an environment variable so they can be tuned on
# the running service without a deploy, and so a short TTL can be set briefly
# to watch the sweep actually fire instead of waiting four hours to believe
# it. The defaults are the policy; the variables are for operating it.
def _num(env: str, default: float) -> float:
    raw = os.getenv(env)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", env, raw, default)
        return default
    if value <= 0:
        logger.warning("%s=%r is not positive; using %s", env, raw, default)
        return default
    return value


UPLOAD_TTL_HOURS = _num("DOCMIND_TTL_HOURS", 4)

#: Ceilings, all on visitor uploads only -- the seed is never counted or
#: evicted. The volume is billed on what is used rather than the 4.88GB
#: provisioned, so 200MB caps the volume line at roughly three cents a month
#: while still allowing a real document to be tried.
MAX_VISITOR_DOCUMENTS = int(_num("DOCMIND_MAX_DOCUMENTS", 20))
MAX_VISITOR_BYTES = int(_num("DOCMIND_MAX_MB", 200) * 1024 * 1024)


def enforce_retention() -> dict:
    """Expire old visitor uploads and keep the store under its ceilings.

    Called on the request path rather than by a scheduler: a demo with no
    visitors needs no sweeping, and adding a cron service to Railway would
    cost more per month than the storage it polices.

    Everything deletes through remove_document, which raises rather than
    touching the seed.
    """
    from datetime import datetime, timedelta

    removed_expired: list[str] = []
    removed_capacity: list[str] = []
    cutoff = datetime.utcnow() - timedelta(hours=UPLOAD_TTL_HOURS)

    for doc in document_store.visitor_documents_oldest_first():
        try:
            uploaded = datetime.fromisoformat(doc["upload_time"])
        except (ValueError, TypeError):
            continue                      # unparseable: leave it to the caps
        if uploaded < cutoff:
            if _safe_remove(doc["doc_id"]):
                removed_expired.append(doc["doc_id"])

    count, total = document_store.visitor_totals()
    for doc in document_store.visitor_documents_oldest_first():
        if count <= MAX_VISITOR_DOCUMENTS and total <= MAX_VISITOR_BYTES:
            break
        if _safe_remove(doc["doc_id"]):
            removed_capacity.append(doc["doc_id"])
            count -= 1
            total -= doc.get("file_size") or 0

    if removed_expired or removed_capacity:
        logger.info("Retention: %d expired, %d evicted for capacity",
                    len(removed_expired), len(removed_capacity))
    return {"expired": removed_expired, "evicted": removed_capacity}


def _safe_remove(doc_id: str) -> bool:
    """remove_document, but a protected seed is a no-op rather than a 500."""
    try:
        return remove_document(doc_id)
    except document_store.SeedDocumentProtected:
        logger.warning("Retention tried to remove the seed document %s. "
                       "Refused. This should not happen: the queries it walks "
                       "already exclude seeds.", doc_id)
        return False
