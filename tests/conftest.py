"""Shared fixtures.

Two rules hold for every test in this suite:

1. No test calls the Anthropic API. Nothing here imports `claude_qa`.
2. No test touches the real `docmind.db`. Every database fixture points
   `document_store.DB_PATH` at a per-test temporary file.

Search tests use a deterministic bag-of-words embedder rather than the real
fastembed model. That keeps the suite hermetic (no model weights to download)
and makes retrieval assertions depend on the retrieval code under test rather
than on the semantics of MiniLM. The real model is exercised by an opt-in test
in `test_search.py`.
"""

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import document_store  # noqa: E402
import rag_engine  # noqa: E402

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def deterministic_embed(texts: list[str]) -> np.ndarray:
    """A stand-in for `rag_engine.embed` with no model behind it.

    Hashes each token into a fixed dimension and L2-normalises the result, so
    cosine similarity reduces to token overlap. Uses blake2b rather than
    `hash()` so vectors are identical across runs, machines, and Python
    versions -- the tests assert on ranking, which requires that stability.
    """
    out = np.zeros((len(texts), rag_engine.EMBEDDING_DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            out[row, int.from_bytes(digest, "big") % rag_engine.EMBEDDING_DIM] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (out / norms).astype(np.float32)


@pytest.fixture(autouse=True)
def reset_rag_globals():
    """`rag_engine` keeps the index in module globals; isolate them per test."""
    rag_engine._index = None
    rag_engine._chunk_id_map = []
    yield
    rag_engine._index = None
    rag_engine._chunk_id_map = []


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "docmind_test.db"
    monkeypatch.setattr(document_store, "DB_PATH", db_path)
    # Guard against a refactor that stops the patch from taking effect.
    assert document_store.DB_PATH.parent == tmp_path
    document_store.init_db()
    return db_path


@pytest.fixture
def stub_embeddings(monkeypatch):
    monkeypatch.setattr(rag_engine, "embed", deterministic_embed)
    return deterministic_embed


@pytest.fixture
def make_document(temp_db, stub_embeddings):
    """Persist a document and its chunks, returning the new doc_id."""

    def _make(filename: str, chunk_texts: list[str], page_nums: list[int] | None = None) -> str:
        pages = page_nums or [1] * len(chunk_texts)
        chunks = [
            {"text": text, "page_num": page}
            for text, page in zip(chunk_texts, pages)
        ]
        doc_id = document_store.create_document(
            filename, "txt", sum(len(t) for t in chunk_texts)
        )
        document_store.save_chunks(doc_id, chunks, deterministic_embed(chunk_texts))
        return doc_id

    return _make


@pytest.fixture(scope="session")
def demo_policy_text() -> str:
    """The document the published evaluation results were measured against."""
    return (PROJECT_ROOT / "demo_docs" / "company_policy.txt").read_text(
        encoding="utf-8"
    ).strip()
