import sqlite3
import uuid
import numpy as np
from datetime import datetime
from pathlib import Path

DB_PATH = Path("docmind.db")


def init_db() -> None:
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
            "SELECT doc_id, filename, file_type, chunk_count, file_size, upload_time "
            "FROM documents ORDER BY upload_time DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_document(doc_id: str) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None


def delete_document(doc_id: str) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        result = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        conn.commit()
        return result.rowcount > 0


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
