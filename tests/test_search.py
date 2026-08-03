"""Tests for `rag_engine._rebuild_index`, `search`, and `remove_document`.

FAISS runs locally and the embeddings here come from the deterministic stub in
`conftest.py`, so nothing in this file touches the network or the Anthropic API.
Using a stub rather than MiniLM is deliberate: these tests are about the
retrieval plumbing -- index rebuild, `top_k`, `doc_id` filtering, id mapping --
and a fixed embedder makes the ranking assertions reproducible.
"""

import os

import numpy as np
import pytest

import document_store
import rag_engine

LEAVE = "Employees accrue vacation leave at twenty hours per calendar month."
EXPENSE = "Expense reimbursement requires an itemized receipt within thirty days."
PARENTAL = "Parental leave provides sixteen weeks of paid time away from work."
SECURITY = "Laptops must use full disk encryption and a screen lock timeout."
TRAVEL = "Booking airfare needs manager approval before the itinerary is issued."

POLICY_CHUNKS = [LEAVE, EXPENSE, PARENTAL, SECURITY, TRAVEL]


@pytest.fixture
def indexed_policy(make_document):
    """One document, five chunks, index built."""
    doc_id = make_document("policy.txt", POLICY_CHUNKS)
    rag_engine._rebuild_index()
    return doc_id


@pytest.fixture
def two_indexed_documents(make_document):
    first = make_document("policy.txt", POLICY_CHUNKS)
    second = make_document("engineering.txt", [
        "Pull requests need one approving review before merge.",
        "Production deploys are frozen during the release window.",
    ])
    rag_engine._rebuild_index()
    return first, second


# ---------------------------------------------------------------------------
# Index rebuild
# ---------------------------------------------------------------------------

def test_index_size_equals_chunk_count(indexed_policy):
    assert rag_engine._index.ntotal == len(POLICY_CHUNKS)
    assert len(rag_engine._chunk_id_map) == len(POLICY_CHUNKS)


def test_rebuild_maps_faiss_positions_to_real_chunk_ids(indexed_policy):
    for chunk_id in rag_engine._chunk_id_map:
        assert document_store.get_chunk_by_id(chunk_id) is not None


def test_rebuild_is_idempotent(indexed_policy):
    rag_engine._rebuild_index()
    rag_engine._rebuild_index()
    assert rag_engine._index.ntotal == len(POLICY_CHUNKS)


def test_rebuild_covers_every_document(two_indexed_documents):
    assert rag_engine._index.ntotal == len(POLICY_CHUNKS) + 2


def test_init_rag_builds_the_index_from_persisted_chunks(make_document):
    make_document("policy.txt", POLICY_CHUNKS)
    assert rag_engine._index is None

    rag_engine.init_rag()

    assert rag_engine._index.ntotal == len(POLICY_CHUNKS)


# ---------------------------------------------------------------------------
# Empty index
# ---------------------------------------------------------------------------

def test_rebuild_on_an_empty_database_raises_nothing(temp_db, stub_embeddings):
    rag_engine._rebuild_index()
    assert rag_engine._index.ntotal == 0
    assert rag_engine._chunk_id_map == []


def test_search_on_an_empty_index_returns_no_results(temp_db, stub_embeddings):
    rag_engine._rebuild_index()
    assert rag_engine.search("anything at all") == []


def test_search_before_any_index_is_built_returns_no_results(temp_db, stub_embeddings):
    assert rag_engine._index is None
    assert rag_engine.search("anything at all") == []


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def test_a_known_query_returns_the_expected_chunk_within_top_k(indexed_policy):
    results = rag_engine.search("how many hours of vacation do employees accrue")
    assert results, "expected at least one result"
    assert LEAVE in [r["chunk_text"] for r in results]
    assert results[0]["chunk_text"] == LEAVE


@pytest.mark.parametrize(
    "question, expected",
    [
        ("itemized receipt for reimbursement", EXPENSE),
        ("weeks of paid parental time away", PARENTAL),
        ("full disk encryption on laptops", SECURITY),
        ("manager approval to book airfare", TRAVEL),
    ],
)
def test_distinctive_queries_rank_their_own_chunk_first(indexed_policy, question, expected):
    results = rag_engine.search(question)
    assert results[0]["chunk_text"] == expected


@pytest.mark.parametrize("top_k", [1, 2, 3, 5])
def test_top_k_is_respected(indexed_policy, top_k):
    assert len(rag_engine.search("leave policy", top_k=top_k)) == top_k


def test_top_k_larger_than_the_index_returns_everything_once(indexed_policy):
    results = rag_engine.search("policy", top_k=50)
    assert len(results) == len(POLICY_CHUNKS)
    assert len({r["chunk_id"] for r in results}) == len(POLICY_CHUNKS)


def test_default_top_k_matches_the_configured_constant(indexed_policy):
    assert len(rag_engine.search("policy")) == rag_engine.TOP_K == 5


def test_results_are_ordered_by_descending_relevance(indexed_policy):
    scores = [r["relevance_score"] for r in rag_engine.search("leave and expenses")]
    assert scores == sorted(scores, reverse=True)


def test_result_records_carry_the_fields_the_api_returns(indexed_policy):
    result = rag_engine.search("vacation leave")[0]
    assert set(result) == {
        "chunk_id", "chunk_text", "page_num", "doc_id", "relevance_score"
    }
    assert isinstance(result["chunk_id"], int)
    assert isinstance(result["relevance_score"], float)
    assert result["doc_id"] == indexed_policy


# ---------------------------------------------------------------------------
# doc_id filtering
# ---------------------------------------------------------------------------

def test_doc_id_filter_restricts_results_to_that_document(two_indexed_documents):
    _, engineering = two_indexed_documents
    results = rag_engine.search("approval before release", doc_id=engineering)
    assert results
    assert {r["doc_id"] for r in results} == {engineering}


def test_an_unfiltered_search_can_span_documents(two_indexed_documents):
    policy, engineering = two_indexed_documents
    results = rag_engine.search("approval review leave receipt deploy", top_k=7)
    assert {r["doc_id"] for r in results} == {policy, engineering}


def test_filtering_by_an_unknown_doc_id_returns_no_results(two_indexed_documents):
    assert rag_engine.search("approval", doc_id="no-such-doc") == []


def test_filter_still_honours_top_k(two_indexed_documents):
    policy, _ = two_indexed_documents
    assert len(rag_engine.search("policy", doc_id=policy, top_k=2)) == 2


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------

def test_remove_document_shrinks_the_index(two_indexed_documents):
    policy, engineering = two_indexed_documents

    assert rag_engine.remove_document(engineering) is True

    assert rag_engine._index.ntotal == len(POLICY_CHUNKS)
    assert {r["doc_id"] for r in rag_engine.search("leave", top_k=5)} == {policy}


def test_removing_every_document_leaves_an_empty_searchable_index(indexed_policy):
    rag_engine.remove_document(indexed_policy)
    assert rag_engine._index.ntotal == 0
    assert rag_engine.search("leave") == []


def test_removing_a_missing_document_reports_false(indexed_policy):
    assert rag_engine.remove_document("does-not-exist") is False
    assert rag_engine._index.ntotal == len(POLICY_CHUNKS)


# ---------------------------------------------------------------------------
# Real embedding model (opt-in)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("DOCMIND_TEST_REAL_MODEL") != "1",
    reason=(
        "set DOCMIND_TEST_REAL_MODEL=1 to exercise the real fastembed model; "
        "it downloads weights on first run, so CI stays on the stub embedder"
    ),
)
def test_real_fastembed_returns_normalised_float32_vectors():
    vectors = rag_engine.embed(["annual leave policy", "expense receipts"])
    assert vectors.shape == (2, rag_engine.EMBEDDING_DIM)
    assert vectors.dtype == np.float32
    assert np.isfinite(vectors).all()
