import os
import sqlite3
import uuid
import numpy as np
from datetime import datetime
from pathlib import Path

# WHERE THE DATABASE LIVES, and why it is not just "docmind.db" any more.
#
# This file holds the documents, the chunks and their embeddings as BLOBs, and
# rag_engine rebuilds the FAISS index from those embeddings at startup. So the
# database is not a cache: it is the only copy of work that cost an embedding
# model to produce.
#
# On a relative path it sat in the container's writable layer and vanished on
# every deploy. The visible cost was not the lost uploads -- a demo's uploads
# are disposable -- it was that main.py's lifespan re-ingested the demo
# document on every single boot, because filename_exists() was always False,
# and that ingestion is the one thing on the startup path that needs the
# embedding model. Loading it there held ~200MB resident for the life of the
# process to answer about four requests a week.
#
# DOCMIND_DB_PATH points at a Railway volume in production. Unset, the
# behaviour is exactly what it was, so local runs and the test suite are
# unaffected.
DB_PATH = Path(os.getenv("DOCMIND_DB_PATH") or "docmind.db")


def init_db() -> None:
    # The volume mount exists but its subdirectories do not, and sqlite3 will
    # not create a parent for its file.
    if DB_PATH.parent != Path("."):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id      TEXT PRIMARY KEY,
                filename    TEXT NOT NULL,
                file_type   TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                file_size   INTEGER DEFAULT 0,
                upload_time TEXT NOT NULL
            )
        """)
        # is_seed marks the pre-loaded demo document. A flag, not a filename
        # comparison: a visitor can upload their own company_policy.txt and a
        # name check would then either protect their file or expose ours,
        # depending on which one it matched first.
        #
        # ALTER rather than adding it to the CREATE above, because the volume
        # already holds a table without it. Guarded, so it runs once.
        cols = {r[1] for r in conn.execute("PRAGMA table_info(documents)")}
        if "is_seed" not in cols:
            conn.execute(
                "ALTER TABLE documents ADD COLUMN is_seed INTEGER NOT NULL DEFAULT 0")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id      TEXT    NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text  TEXT    NOT NULL,
                page_num    INTEGER DEFAULT 1,
                embedding   BLOB    NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            )
        """)
        conn.commit()


def create_document(filename: str, file_type: str, file_size: int) -> str:
    doc_id = str(uuid.uuid4())[:8]
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO documents (doc_id, filename, file_type, file_size, upload_time) VALUES (?, ?, ?, ?, ?)",
            (doc_id, filename, file_type, file_size, datetime.utcnow().isoformat()),
        )
        conn.commit()
    return doc_id


def save_chunks(doc_id: str, chunks: list[dict], embeddings: np.ndarray) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            "INSERT INTO chunks (doc_id, chunk_index, chunk_text, page_num, embedding) VALUES (?, ?, ?, ?, ?)",
            [
                (doc_id, i, chunk["text"], chunk.get("page_num", 1), emb.tobytes())
                for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
            ],
        )
        conn.execute(
            "UPDATE documents SET chunk_count = ? WHERE doc_id = ?",
            (len(chunks), doc_id),
        )
        conn.commit()


def get_all_documents() -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT doc_id, filename, file_type, chunk_count, file_size, "
            "upload_time, is_seed FROM documents ORDER BY upload_time DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_document(doc_id: str) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None


class SeedDocumentProtected(Exception):
    """Raised when something tries to delete the pre-loaded demo document."""


def mark_seed(doc_id: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE documents SET is_seed = 1 WHERE doc_id = ?", (doc_id,))
        conn.commit()


def get_seed() -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM documents WHERE is_seed = 1 LIMIT 1").fetchone()
        return dict(row) if row else None


def is_seed(doc_id: str) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT is_seed FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return bool(row and row[0])


def delete_document(doc_id: str) -> bool:
    """Delete one document. REFUSES the seed.

    The refusal lives here rather than only in the route, so that everything
    that deletes -- the endpoint, the TTL sweep, the size eviction -- inherits
    it from one place. A sweep that deleted the seed would empty the demo, and
    the demo is what the link on the CV points at.
    """
    if is_seed(doc_id):
        raise SeedDocumentProtected(doc_id)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        result = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        conn.commit()
        return result.rowcount > 0


def visitor_documents_oldest_first() -> list[dict]:
    """Everything a visitor uploaded, oldest first. Never the seed."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT doc_id, filename, file_size, upload_time FROM documents "
            "WHERE is_seed = 0 ORDER BY upload_time ASC").fetchall()
        return [dict(r) for r in rows]


def visitor_totals() -> tuple[int, int]:
    """(count, total bytes) of visitor uploads. The seed is not counted."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(file_size), 0) "
            "FROM documents WHERE is_seed = 0").fetchone()
        return int(row[0]), int(row[1])


def get_chunk_by_id(chunk_id: int) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT id, doc_id, chunk_text, page_num FROM chunks WHERE id = ?",
            (chunk_id,),
        ).fetchone()
        return dict(row) if row else None


def load_all_chunks_for_index() -> list[dict]:
    """Return all chunks with deserialized embeddings, ordered by id."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, doc_id, chunk_text, page_num, embedding FROM chunks ORDER BY id"
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            if d["embedding"]:
                d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32).copy()
            result.append(d)
        return result


def filename_exists(filename: str) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT 1 FROM documents WHERE filename = ? LIMIT 1", (filename,)
        ).fetchone()
        return row is not None


def get_document_by_filename(filename: str) -> dict | None:
    """Used once, at startup, to adopt a seed row written before is_seed
    existed. Oldest first, so a visitor's later upload of the same name
    cannot be adopted ahead of the real one."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM documents WHERE filename = ? "
            "ORDER BY upload_time ASC LIMIT 1", (filename,)).fetchone()
        return dict(row) if row else None
