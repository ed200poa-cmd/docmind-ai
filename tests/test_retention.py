"""The seed document survives everything that deletes. 2026-09-04.

The database moved onto a volume so the embedding model would stop loading at
boot. That fixed a memory cost and created a retention problem: uploads used
to be swept away by the next deploy -- every six to nineteen days, going by
docmind's actual deploy history -- and now they persist until something
removes them.

`rag_engine.enforce_retention` is that something. It runs on the request path,
and the failure that matters is not that it removes too little. It is that it
removes the seed, because then a recruiter opening the link finds an empty
demo and a chat box with nothing to answer from.

So these tests are mostly about what must NOT be deleted.
"""

from datetime import datetime, timedelta

import pytest

import document_store
import rag_engine


@pytest.fixture
def seeded(make_document):
    """A seed document plus one visitor upload, both indexed."""
    seed_id = make_document("company_policy.txt",
                            ["Vacation accrues at ten days a year."])
    document_store.mark_seed(seed_id)
    visitor_id = make_document("visitor_upload.txt",
                               ["Engineers receive a laptop and a monitor."])
    return seed_id, visitor_id


def _age(doc_id: str, hours: float) -> None:
    """Backdate a document's upload_time."""
    import sqlite3
    when = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    with sqlite3.connect(document_store.DB_PATH) as conn:
        conn.execute("UPDATE documents SET upload_time = ? WHERE doc_id = ?",
                     (when, doc_id))
        conn.commit()


class TestTheSeedIsProtected:
    def test_delete_document_refuses_it(self, seeded):
        seed_id, _ = seeded
        with pytest.raises(document_store.SeedDocumentProtected):
            document_store.delete_document(seed_id)
        assert document_store.get_document(seed_id) is not None

    def test_remove_document_refuses_it(self, seeded):
        seed_id, _ = seeded
        with pytest.raises(document_store.SeedDocumentProtected):
            rag_engine.remove_document(seed_id)

    def test_the_ttl_sweep_leaves_it_however_old_it_is(self, seeded):
        """The one that would empty the demo. The seed is older than any
        visitor upload by construction -- it is ingested at first boot and
        never touched again -- so an age-based sweep that forgot to exclude it
        would take the seed FIRST."""
        seed_id, visitor_id = seeded
        _age(seed_id, hours=24 * 30)
        _age(visitor_id, hours=24 * 30)

        rag_engine.enforce_retention()

        assert document_store.get_document(seed_id) is not None, \
            "the sweep deleted the seed document; the demo is now empty"
        assert document_store.get_document(visitor_id) is None

    def test_the_capacity_eviction_leaves_it(self, seeded, monkeypatch):
        seed_id, visitor_id = seeded
        monkeypatch.setattr(rag_engine, "MAX_VISITOR_DOCUMENTS", 0)
        rag_engine.enforce_retention()
        assert document_store.get_document(seed_id) is not None
        assert document_store.get_document(visitor_id) is None

    def test_it_is_not_identified_by_filename(self, make_document):
        """A visitor uploading their own company_policy.txt must not become
        protected, and must not make the real seed deletable."""
        seed_id = make_document("company_policy.txt", ["Ours, the real seed."])
        document_store.mark_seed(seed_id)
        impostor_id = make_document("company_policy.txt",
                                    ["Theirs, uploaded by a visitor."])

        assert document_store.is_seed(seed_id) is True
        assert document_store.is_seed(impostor_id) is False
        assert document_store.delete_document(impostor_id) is True
        assert document_store.get_document(seed_id) is not None


class TestVisitorUploadsExpire:
    def test_an_upload_past_the_ttl_is_removed(self, seeded):
        _, visitor_id = seeded
        _age(visitor_id, hours=rag_engine.UPLOAD_TTL_HOURS + 1)
        result = rag_engine.enforce_retention()
        assert visitor_id in result["expired"]
        assert document_store.get_document(visitor_id) is None

    def test_a_fresh_upload_is_left_alone(self, seeded):
        _, visitor_id = seeded
        rag_engine.enforce_retention()
        assert document_store.get_document(visitor_id) is not None

    def test_one_just_inside_the_ttl_is_left_alone(self, seeded):
        _, visitor_id = seeded
        _age(visitor_id, hours=rag_engine.UPLOAD_TTL_HOURS - 0.5)
        rag_engine.enforce_retention()
        assert document_store.get_document(visitor_id) is not None

    def test_the_index_no_longer_answers_from_an_expired_upload(
            self, seeded, stub_embeddings):
        """Deleting the row is not enough: the FAISS index is built from the
        chunks and would keep serving the text until it is rebuilt."""
        _, visitor_id = seeded
        rag_engine.init_rag()
        before = rag_engine.search("laptop and monitor")
        assert any(c["doc_id"] == visitor_id for c in before)

        _age(visitor_id, hours=rag_engine.UPLOAD_TTL_HOURS + 1)
        rag_engine.enforce_retention()

        after = rag_engine.search("laptop and monitor")
        assert not any(c["doc_id"] == visitor_id for c in after)


class TestCeilings:
    def test_the_oldest_visitor_upload_goes_first(self, seeded, monkeypatch):
        seed_id, first = seeded
        second = document_store.create_document("second.txt", "txt", 10)
        _age(first, hours=2)

        monkeypatch.setattr(rag_engine, "MAX_VISITOR_DOCUMENTS", 1)
        rag_engine.enforce_retention()

        assert document_store.get_document(first) is None, "oldest should go"
        assert document_store.get_document(second) is not None
        assert document_store.get_document(seed_id) is not None

    def test_totals_exclude_the_seed(self, seeded):
        """A ceiling that counted the seed would evict a visitor's upload to
        make room for a document that is never going away."""
        count, _ = document_store.visitor_totals()
        assert count == 1
