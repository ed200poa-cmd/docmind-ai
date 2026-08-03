import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from fastembed import TextEmbedding

import document_store

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
        logger.info("Loading embedding model '%s'…", EMBEDDING_MODEL)
        _model = TextEmbedding(model_name=EMBEDDING_MODEL)
        logger.info("Embedding model ready.")
    return _model


def init_rag() -> None:
    """Rebuild FAISS index from persisted SQLite chunks on startup."""
    _rebuild_index()
    logger.info("RAG engine ready. Index size: %d vectors", _index.ntotal if _index else 0)


def _rebuild_index() -> None:
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
