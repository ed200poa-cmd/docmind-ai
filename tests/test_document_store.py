"""Tests for `document_store`: schema init, chunk round-trip, cascade delete.

Every test runs against a temporary SQLite file supplied by the `temp_db`
fixture. The real `docmind.db` is never opened.
"""

import sqlite3

import numpy as np
import pytest

import document_store


def _tables(db_path) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    return {r[0] for r in rows}


def _chunk_rows(db_path, doc_id: str) -> list[tuple]:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT id, chunk_index, chunk_text, page_num FROM chunks "
            "WHERE doc_id = ? ORDER BY chunk_index",
            (doc_id,),
        ).fetchall()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_init_db_creates_both_tables(temp_db):
    assert {"documents", "chunks"} <= _tables(temp_db)


def test_init_db_is_idempotent(temp_db):
    document_store.init_db()
    document_store.init_db()
    assert {"documents", "chunks"} <= _tables(temp_db)


def test_chunks_declare_a_cascading_foreign_key(temp_db):
    with sqlite3.connect(temp_db) as conn:
        cursor = conn.execute("PRAGMA foreign_key_list(chunks)")
        columns = [c[0] for c in cursor.description]
        fks = [dict(zip(columns, row)) for row in cursor.fetchall()]

    assert fks, "chunks has no foreign key to documents"
    assert any(
        fk["table"] == "documents"
        and fk["from"] == "doc_id"
        and fk["on_delete"] == "CASCADE"
        for fk in fks
    )


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_create_document_returns_a_readable_record(temp_db):
    doc_id = document_store.create_document("policy.txt", "txt", 1234)

    record = document_store.get_document(doc_id)
    assert record is not None
    assert record["doc_id"] == doc_id
    assert record["filename"] == "policy.txt"
    assert record["file_type"] == "txt"
    assert record["file_size"] == 1234
    assert record["chunk_count"] == 0
    assert record["upload_time"]


def test_chunk_round_trip_preserves_text_page_and_order(temp_db):
    doc_id = document_store.create_document("handbook.pdf", "pdf", 99)
    chunks = [
        {"text": "Leave is accrued monthly.", "page_num": 1},
        {"text": "Expenses require a receipt.", "page_num": 2},
        {"text": "Remote work is approved per team.", "page_num": 7},
    ]
    embeddings = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)

    document_store.save_chunks(doc_id, chunks, embeddings)

    rows = _chunk_rows(temp_db, doc_id)
    assert [r[1] for r in rows] == [0, 1, 2]
    assert [r[2] for r in rows] == [c["text"] for c in chunks]
    assert [r[3] for r in rows] == [1, 2, 7]


def test_save_chunks_updates_the_document_chunk_count(temp_db):
    doc_id = document_store.create_document("handbook.pdf", "pdf", 99)
    document_store.save_chunks(
        doc_id,
        [{"text": f"chunk {i}", "page_num": 1} for i in range(5)],
        np.zeros((5, 4), dtype=np.float32),
    )
    assert document_store.get_document(doc_id)["chunk_count"] == 5


def test_page_num_defaults_to_one_when_absent(temp_db):
    doc_id = document_store.create_document("notes.txt", "txt", 10)
    document_store.save_chunks(
        doc_id, [{"text": "no page key"}], np.zeros((1, 4), dtype=np.float32)
    )
    assert _chunk_rows(temp_db, doc_id)[0][3] == 1


def test_embeddings_survive_the_blob_round_trip_bit_for_bit(temp_db):
    doc_id = document_store.create_document("vectors.txt", "txt", 10)
    original = np.array(
        [[0.1, -0.25, 3.5, 0.0], [1e-7, 42.0, -1.0, 0.333]], dtype=np.float32
    )
    document_store.save_chunks(
        doc_id,
        [{"text": "first chunk text", "page_num": 1},
         {"text": "second chunk text", "page_num": 1}],
        original,
    )

    loaded = document_store.load_all_chunks_for_index()
    assert len(loaded) == 2
    restored = np.stack([c["embedding"] for c in loaded])
    assert restored.dtype == np.float32
    np.testing.assert_array_equal(restored, original)


def test_load_all_chunks_for_index_is_ordered_by_id(temp_db):
    first = document_store.create_document("a.txt", "txt", 1)
    document_store.save_chunks(
        first, [{"text": "alpha", "page_num": 1}], np.zeros((1, 4), dtype=np.float32)
    )
    second = document_store.create_document("b.txt", "txt", 1)
    document_store.save_chunks(
        second, [{"text": "beta", "page_num": 1}], np.zeros((1, 4), dtype=np.float32)
    )

    loaded = document_store.load_all_chunks_for_index()
    assert [c["id"] for c in loaded] == sorted(c["id"] for c in loaded)
    assert [c["doc_id"] for c in loaded] == [first, second]


def test_get_chunk_by_id_returns_the_matching_chunk(temp_db):
    doc_id = document_store.create_document("a.txt", "txt", 1)
    document_store.save_chunks(
        doc_id,
        [{"text": "the only chunk", "page_num": 3}],
        np.zeros((1, 4), dtype=np.float32),
    )
    chunk_id = _chunk_rows(temp_db, doc_id)[0][0]

    chunk = document_store.get_chunk_by_id(chunk_id)
    assert chunk == {
        "id": chunk_id,
        "doc_id": doc_id,
        "chunk_text": "the only chunk",
        "page_num": 3,
    }


def test_get_all_documents_is_newest_first(temp_db):
    import time

    older = document_store.create_document("older.txt", "txt", 1)
    time.sleep(0.01)
    newer = document_store.create_document("newer.txt", "txt", 1)

    assert [d["doc_id"] for d in document_store.get_all_documents()] == [newer, older]


def test_lookups_for_missing_records_return_none(temp_db):
    assert document_store.get_document("does-not-exist") is None
    assert document_store.get_chunk_by_id(999_999) is None
    assert document_store.get_all_documents() == []


def test_filename_exists(temp_db):
    assert not document_store.filename_exists("policy.txt")
    document_store.create_document("policy.txt", "txt", 1)
    assert document_store.filename_exists("policy.txt")
    assert not document_store.filename_exists("Policy.TXT")


# ---------------------------------------------------------------------------
# Cascade delete
# ---------------------------------------------------------------------------

def test_delete_document_removes_its_chunks(temp_db):
    doc_id = document_store.create_document("gone.txt", "txt", 1)
    document_store.save_chunks(
        doc_id,
        [{"text": f"chunk {i}", "page_num": 1} for i in range(4)],
        np.zeros((4, 4), dtype=np.float32),
    )
    assert len(_chunk_rows(temp_db, doc_id)) == 4

    assert document_store.delete_document(doc_id) is True

    assert document_store.get_document(doc_id) is None
    assert _chunk_rows(temp_db, doc_id) == []


def test_delete_leaves_no_orphaned_chunks_anywhere(temp_db):
    kept = document_store.create_document("kept.txt", "txt", 1)
    removed = document_store.create_document("removed.txt", "txt", 1)
    for doc_id in (kept, removed):
        document_store.save_chunks(
            doc_id,
            [{"text": f"{doc_id} chunk {i}", "page_num": 1} for i in range(3)],
            np.zeros((3, 4), dtype=np.float32),
        )

    document_store.delete_document(removed)

    with sqlite3.connect(temp_db) as conn:
        orphans = conn.execute(
            "SELECT COUNT(*) FROM chunks "
            "WHERE doc_id NOT IN (SELECT doc_id FROM documents)"
        ).fetchone()[0]
    assert orphans == 0
    # The surviving document is untouched.
    assert len(_chunk_rows(temp_db, kept)) == 3
    assert document_store.get_document(kept) is not None


def test_deleting_a_missing_document_reports_false(temp_db):
    assert document_store.delete_document("does-not-exist") is False


@pytest.mark.parametrize("call", ["load_all_chunks_for_index", "get_all_documents"])
def test_readers_work_on_an_empty_database(temp_db, call):
    assert getattr(document_store, call)() == []
