"""The MCP surface is a promise about what a model can reach, so pin it.

`mcp_server` is the process an MCP client spawns. Its docstring says two things
that matter more than anything else in this repository: deletion is not exposed,
and answer synthesis is not exposed. Both are claims about what a language model
holding this connection is able to do. A docstring cannot enforce either. These
tests do.

The count of tools is pinned too, because the risk is not that a tool changes,
it is that a fourth one appears without anyone deciding that it should.

Like the rest of the suite these tests make no API call and never open the real
`docmind.db`: every one of them points `document_store.DB_PATH` at a temporary
file, and tracing is switched off so a test run cannot append to the server's
own trace log.
"""
import asyncio
import hashlib
from pathlib import Path

import pytest

import document_store
import mcp_server
import rag_engine

EXPECTED_TOOLS = {"search_documents", "list_documents", "get_document_info"}
DOCUMENTS_RESOURCE = "docmind://documents"


@pytest.fixture(autouse=True)
def isolate_server(monkeypatch):
    """Reset the server's one piece of global state and silence its trace file.

    `_ensure_ready` builds the index once and then short circuits, so without
    this the second test in the file would run against the first test's index.
    """
    monkeypatch.setattr(mcp_server, "_ready", False)
    monkeypatch.setattr(mcp_server, "TRACE_PATH", Path("off"))
    yield
    mcp_server._ready = False


@pytest.fixture
def indexed(temp_db, stub_embeddings, make_document):
    """A temporary database with one small document in it."""
    doc_id = make_document("handbook.txt", [
        "Standard working hours are Monday to Friday, 9am to 5pm.",
        "Employees accrue 15 days of paid vacation per year.",
    ])
    return doc_id


def tool_names() -> set:
    return {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}


def resource_uris() -> set:
    return {str(r.uri) for r in asyncio.run(mcp_server.mcp.list_resources())}


def db_digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def db_contents(path) -> tuple:
    docs = tuple(sorted((d["doc_id"], d["filename"], d["chunk_count"])
                        for d in document_store.get_all_documents()))
    chunks = tuple(sorted((c["id"], c["chunk_text"])
                          for c in document_store.load_all_chunks_for_index()))
    return docs, chunks


def test_exactly_three_tools_are_exposed():
    """Not "at least three". A fourth tool is a widened surface."""
    assert tool_names() == EXPECTED_TOOLS


def test_deletion_is_not_exposed():
    names = tool_names()
    assert "delete_document" not in names
    # Anything that removes, not only the name the docstring happens to use.
    assert not [n for n in names if "delete" in n or "remove" in n or "drop" in n]
    # And the absence is a decision rather than an oversight: the capability
    # exists in the layer underneath and was not wired up.
    assert callable(rag_engine.remove_document)


def test_answer_synthesis_is_not_exposed():
    names = tool_names()
    assert "answer_question" not in names
    assert not [n for n in names if "answer" in n or "generate" in n]
    # The server must not even import the module that would call the API. If it
    # did, a future edit could reach synthesis without adding a tool.
    assert not hasattr(mcp_server, "claude_qa")
    source = Path(mcp_server.__file__).read_text(encoding="utf-8")
    assert "import claude_qa" not in source


def test_the_documents_resource_is_registered():
    assert DOCUMENTS_RESOURCE in resource_uris()


def test_every_exposed_tool_describes_itself():
    """A tool with no description is a tool a model has to guess about."""
    for tool in asyncio.run(mcp_server.mcp.list_tools()):
        assert tool.description and tool.description.strip()
        assert tool.inputSchema["type"] == "object"


def test_search_returns_excerpts_from_the_indexed_document(indexed):
    results = mcp_server.search_documents("working hours", top_k=2)
    assert results
    assert all(r["filename"] == "handbook.txt" for r in results)
    assert all(r["doc_id"] == indexed for r in results)
    # Plain text has no pagination, so a client must not be handed a page to cite.
    assert all(r["page"] is None for r in results)


def test_list_documents_and_get_document_info_agree(indexed):
    listed = mcp_server.list_documents()
    assert [d["doc_id"] for d in listed] == [indexed]
    info = mcp_server.get_document_info(indexed)
    assert info["filename"] == listed[0]["filename"]
    assert info["chunk_count"] == listed[0]["chunk_count"] == 2


def test_the_read_only_tools_leave_the_database_byte_identical(indexed, temp_db):
    """The read-only claim, checked against the file rather than the wording."""
    before_digest, before_contents = db_digest(temp_db), db_contents(temp_db)

    mcp_server.search_documents("vacation", top_k=5)
    mcp_server.search_documents("hours", doc_id=indexed, top_k=1)
    mcp_server.list_documents()
    mcp_server.get_document_info(indexed)
    mcp_server.documents_resource()

    assert db_contents(temp_db) == before_contents
    assert db_digest(temp_db) == before_digest


def test_a_failing_call_also_leaves_the_database_alone(indexed, temp_db):
    """Error paths are where a half-finished write would hide."""
    before_digest, before_contents = db_digest(temp_db), db_contents(temp_db)
    for call in (lambda: mcp_server.search_documents(""),
                 lambda: mcp_server.search_documents("x", top_k=99),
                 lambda: mcp_server.search_documents("x", doc_id="nope"),
                 lambda: mcp_server.get_document_info("nope")):
        with pytest.raises(ValueError):
            call()
    assert db_contents(temp_db) == before_contents
    assert db_digest(temp_db) == before_digest


def test_the_resource_reports_what_is_indexed(indexed):
    text = mcp_server.documents_resource()
    assert "handbook.txt" in text
    assert indexed in text
    assert "1 indexed document(s):" in text
