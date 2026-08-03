"""Tests for `rag_engine._chunk_pages`.

The headline test here is
`test_every_chunk_is_a_verbatim_contiguous_substring_of_its_source_page`.
DocMind's citation-grounding claim rests on chunks being exact slices of the
source rather than reflowed text: if that holds, a citation can be verified by
substring search instead of trusted. This file is the executable form of that
argument, so the invariant is asserted directly rather than inferred from
downstream metrics.
"""

import pytest

import rag_engine
from rag_engine import CHUNK_SIZE, _chunk_pages, _is_heading_block, _split_paragraphs

RULE = "━" * 20


def _para(letter: str, length: int) -> str:
    """A paragraph of an exact length, distinguishable from its neighbours."""
    word = f"{letter * 4} "
    return (word * ((length // len(word)) + 1))[:length].strip().ljust(length, letter)


HEADING_1 = f"{RULE}\nSECTION 1: LEAVE POLICY\n{RULE}"
HEADING_2 = f"{RULE}\nSECTION 2: EXPENSE POLICY\n{RULE}"
P_A, P_B, P_C = _para("a", 200), _para("b", 200), _para("c", 200)
P_D, P_E = _para("d", 200), _para("e", 200)

# Two headings, three body paragraphs. Each heading opens a fresh chunk.
SECTIONED_PAGE = "\n\n".join([HEADING_1, P_A, P_B, HEADING_2, P_C])

# No headings: chunking is driven purely by CHUNK_SIZE packing.
FLOWING_PAGE = "\n\n".join([P_A, P_B, P_C, P_D, P_E])

# A single paragraph larger than CHUNK_SIZE, which must not be split.
OVERSIZED_PAGE = _para("x", CHUNK_SIZE * 2)

MULTIPAGE = [(1, SECTIONED_PAGE), (2, FLOWING_PAGE), (3, OVERSIZED_PAGE)]


# ---------------------------------------------------------------------------
# The grounding invariant
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "label, pages",
    [
        ("sectioned", [(1, SECTIONED_PAGE)]),
        ("flowing", [(1, FLOWING_PAGE)]),
        ("oversized", [(1, OVERSIZED_PAGE)]),
        ("multipage", MULTIPAGE),
    ],
)
def test_every_chunk_is_a_verbatim_contiguous_substring_of_its_source_page(label, pages):
    """Citation grounding is structural, not incidental.

    For every chunk there must exist an offset `i` into the *page it was taken
    from* such that `page_text[i:i + len(chunk)] == chunk`. That is stronger
    than "the words appear somewhere": it says the chunk is one unbroken slice,
    which is exactly what makes a citation verifiable by substring search.
    """
    page_text_by_num = dict(pages)
    chunks = _chunk_pages(pages)
    assert chunks, "fixture produced no chunks; the invariant would be vacuous"

    for chunk in chunks:
        source = page_text_by_num[chunk["page_num"]]
        offset = source.find(chunk["text"])
        assert offset != -1, (
            f"[{label}] chunk on page {chunk['page_num']} is not present verbatim "
            f"in its source page: {chunk['text'][:120]!r}"
        )
        assert source[offset:offset + len(chunk["text"])] == chunk["text"], (
            f"[{label}] chunk is not a contiguous slice of its source page"
        )


def test_demo_corpus_chunks_are_all_verbatim_substrings(demo_policy_text):
    """The same invariant against the corpus the published eval numbers used."""
    chunks = _chunk_pages([(1, demo_policy_text)])
    assert len(chunks) == 17, "demo corpus chunking changed; re-check RESULTS.md"

    not_verbatim = [c["text"] for c in chunks if c["text"] not in demo_policy_text]
    assert not_verbatim == [], (
        f"{len(not_verbatim)} of {len(chunks)} demo-corpus chunks are not verbatim"
    )


@pytest.mark.parametrize(
    "label, page_text",
    [
        ("doubled blank line", _para("a", 60) + "\n\n\n" + _para("b", 60)),
        ("trailing spaces", _para("a", 60) + "   \n\n" + _para("b", 60)),
        ("blank line with spaces", _para("a", 60) + "\n   \n" + _para("b", 60)),
        ("indented paragraph", _para("a", 60) + "\n\n    " + _para("b", 60)),
        ("tabs in blank line", _para("a", 60) + "\n\t\n" + _para("b", 60)),
        ("trailing tab", _para("a", 60) + "\t\n\n" + _para("b", 60)),
        ("leading blank lines", "\n\n  " + _para("a", 60) + "\n\n\n" + _para("b", 60)),
        ("trailing blank lines", _para("a", 60) + "\n\n\n" + _para("b", 60) + "  \n\n"),
        ("mixed separators", "\n".join([
            _para("a", 60), "", _para("b", 60), "   ", _para("c", 60), "", "",
            _para("d", 60),
        ])),
    ],
)
def test_invariant_holds_for_every_separator_form(label, page_text):
    """The invariant must not depend on how the source separates paragraphs.

    Regression test for the defect this replaced: the chunker used to strip each
    paragraph and rejoin with a literal "\\n\\n", which reproduced the source
    only when the source already used exactly that separator. Chunks are now
    emitted as slices, so every case below is a substring by construction.
    """
    chunks = _chunk_pages([(1, page_text)])
    assert chunks, f"[{label}] fixture produced no chunks"
    for chunk in chunks:
        offset = page_text.find(chunk["text"])
        assert offset != -1, f"[{label}] chunk not verbatim: {chunk['text'][:80]!r}"
        assert page_text[offset:offset + len(chunk["text"])] == chunk["text"]


def test_chunk_text_never_starts_or_ends_with_whitespace():
    pages = [
        (1, "\n\n  " + _para("a", 60) + "\n   \n" + _para("b", 60) + "\t\n\n"),
        (2, SECTIONED_PAGE),
    ]
    for chunk in _chunk_pages(pages):
        assert chunk["text"] == chunk["text"].strip()


def test_separator_whitespace_is_preserved_inside_a_chunk():
    """A chunk is a slice, so it carries the source's own separators.

    This is the mechanism behind the fix: the chunker no longer normalises
    "\\n\\n\\n" down to "\\n\\n", which is precisely what used to break the
    substring guarantee.
    """
    page_text = _para("a", 60) + "\n\n\n" + _para("b", 60)
    chunk = _chunk_pages([(1, page_text)])[0]
    assert "\n\n\n" in chunk["text"]
    assert chunk["text"] == page_text


# ---------------------------------------------------------------------------
# Packing behaviour
# ---------------------------------------------------------------------------

def test_chunk_count_for_known_input():
    # HEADING_1 + P_A + P_B pack into one chunk; HEADING_2 forces a new one.
    assert len(_chunk_pages([(1, SECTIONED_PAGE)])) == 2
    # Five 200-char paragraphs pack two-per-chunk under a 500-char budget.
    assert len(_chunk_pages([(1, FLOWING_PAGE)])) == 3
    assert len(_chunk_pages(MULTIPAGE)) == 2 + 3 + 1


def test_a_heading_always_opens_a_new_chunk_and_keeps_its_body():
    chunks = _chunk_pages([(1, SECTIONED_PAGE)])

    assert chunks[0]["text"].startswith(HEADING_1)
    assert chunks[1]["text"].startswith(HEADING_2)
    # A heading is never stranded alone: body text follows it in the same chunk.
    for chunk in chunks:
        assert len(chunk["text"]) > len(HEADING_1) + 10


def test_paragraphs_are_never_split_mid_paragraph():
    """Every chunk decomposes back into whole source paragraphs."""
    source_paragraphs = set(_split_paragraphs(FLOWING_PAGE))
    for chunk in _chunk_pages([(1, FLOWING_PAGE)]):
        for part in chunk["text"].split("\n\n"):
            assert part in source_paragraphs


def test_chunks_from_one_page_are_disjoint_and_in_source_order():
    """This chunker uses no overlap; adjacent chunks must not share text."""
    chunks = _chunk_pages([(1, FLOWING_PAGE)])
    cursor = 0
    for chunk in chunks:
        start = FLOWING_PAGE.find(chunk["text"], cursor)
        assert start >= cursor, "chunks are out of source order or overlap"
        cursor = start + len(chunk["text"])


def test_chunk_exceeds_max_size_only_when_a_single_paragraph_is_oversized():
    """CHUNK_SIZE is a packing budget, not a hard cap.

    The chunker refuses to split a paragraph, so an oversized paragraph is
    emitted whole. Anything larger than CHUNK_SIZE must be exactly that case.
    """
    for chunk in _chunk_pages(MULTIPAGE):
        if len(chunk["text"]) > CHUNK_SIZE:
            assert "\n\n" not in chunk["text"], (
                "a multi-paragraph chunk exceeded CHUNK_SIZE; packing is wrong"
            )


def test_no_empty_or_whitespace_only_chunks():
    pages = MULTIPAGE + [(4, "   \n\n\t\n   "), (5, "short")]
    for chunk in _chunk_pages(pages):
        assert chunk["text"].strip() == chunk["text"]
        assert chunk["text"].strip() != ""
        assert len(chunk["text"]) > 40


def test_page_numbers_are_preserved_per_chunk():
    chunks = _chunk_pages(MULTIPAGE)
    assert {c["page_num"] for c in chunks} == {1, 2, 3}
    assert [c["page_num"] for c in chunks] == sorted(c["page_num"] for c in chunks)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_document_produces_no_chunks():
    assert _chunk_pages([]) == []
    assert _chunk_pages([(1, "")]) == []
    assert _chunk_pages([(1, "   \n\n   \n")]) == []


def test_single_page_shorter_than_the_minimum_is_dropped():
    assert _chunk_pages([(1, "Too short to be useful.")]) == []


def test_single_page_single_paragraph_produces_one_chunk():
    chunks = _chunk_pages([(1, P_A)])
    assert len(chunks) == 1
    assert chunks[0] == {"text": P_A, "page_num": 1}


def test_heading_detection():
    assert _is_heading_block(HEADING_1)
    assert _is_heading_block(f"{RULE}\nJUST A TITLE")
    assert not _is_heading_block(P_A)
    assert not _is_heading_block("")
    # Two title lines between rules is a body block, not a heading.
    assert not _is_heading_block(f"{RULE}\nTITLE\nSUBTITLE\n{RULE}")


def test_chunk_size_constant_is_what_the_tests_assume():
    assert rag_engine.CHUNK_SIZE == 500
