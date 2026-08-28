"""The boundary list has to stay tied to the tracked runs.

docs/what-can-and-cannot-be-claimed.md is the page someone reads just before
being asked about a number, which makes it the worst place for a figure to go
stale. Every quantity it states is re-derived here from evals/results/ and
required to be present verbatim.

Like the rest of this suite these tests make no API call. They read the five
tracked artifacts and the demo document, nothing else.
"""
import hashlib
import json
import re
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "what-can-and-cannot-be-claimed.md"
RESULTS = ROOT / "evals" / "results"

BASELINE = "eval_20260720T124830Z"
FINAL = "eval_20260803T022554Z"
PAIR = ("eval_20260730T122037Z", "eval_20260730T122154Z")
SETTLED = ("eval_20260730T014202Z", "eval_20260730T122037Z",
           "eval_20260730T122154Z", "eval_20260803T022554Z")


@pytest.fixture(scope="module")
def text():
    return DOC.read_text(encoding="utf-8")


def load(name):
    path = RESULTS / f"{name}.json"
    assert path.exists(), f"{name} is not tracked in evals/results/"
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_run_named_in_the_document_is_tracked(text):
    for name in set(re.findall(r"eval_20\d{6}T\d{6}Z", text)):
        assert (RESULTS / f"{name}.json").exists(), \
            f"the document names {name}, which is not tracked"


def test_the_endpoint_is_the_median_of_the_four_settled_runs(text):
    medians = [load(n)["metrics"]["latency"]["median_sec"] for n in SETTLED]
    exact = statistics.median(medians)
    assert f"{exact:.4f}" == "1.0765", f"the four medians now give {exact}"
    # Pin the sentence that defines the endpoint, not merely the digits. The
    # rounded value appears in several places, so a bare substring check passes
    # even when the definition itself is wrong.
    assert f"**median of those four, {exact:.2f}s**" in text, \
        f"the document does not define the endpoint as {exact:.2f}s"
    assert f"({exact:.4f}s before rounding)" in text, \
        "the document does not carry the unrounded endpoint"
    for m in medians:
        assert f"{m}s" in text, f"run median {m}s is not listed in the document"


def test_the_range_of_the_settled_runs_is_stated(text):
    medians = [load(n)["metrics"]["latency"]["median_sec"] for n in SETTLED]
    lo, hi = min(medians), max(medians)
    assert f"{hi - lo:.3f}s" in text, f"the range {hi - lo:.3f}s is not stated"
    assert f"{lo}s to {hi}s" in text, f"the range bounds {lo}s to {hi}s are not stated"


def test_the_baseline_and_the_total_drop_are_stated(text):
    baseline = load(BASELINE)["metrics"]["latency"]["median_sec"]
    assert f"{baseline}s" in text
    endpoint = statistics.median(
        [load(n)["metrics"]["latency"]["median_sec"] for n in SETTLED])
    assert f"{baseline - endpoint:.2f}s" in text, \
        f"the total drop {baseline - endpoint:.2f}s is not stated"


def test_the_baseline_median_re_derives_from_its_own_per_case_values():
    """The document says this re-derives exactly. It has to keep doing so."""
    run = load(BASELINE)
    lat = [r["latency_sec"] for r in run["results"]]
    assert len(lat) == 30
    assert round(statistics.median(lat), 3) == run["metrics"]["latency"]["median_sec"]


def test_the_quality_figures_match_the_final_run(text):
    m = load(FINAL)["metrics"]
    rec = m["retrieval_recall"]
    for k in (1, 3, 5):
        assert f"{rec['overall'][f'recall_at_{k}']:.1f}%" in text
    assert f"{rec['by_category']['multi_chunk']['recall_at_1']:.1f}%" in text
    c = m["answer_correctness"]
    for value in (c["correct_pct"], c["partially_correct_pct"], c["incorrect_pct"]):
        assert f"{value:.1f}%" in text
    assert f"n={c['n_judged']}" in text
    assert f"{c['by_category']['multi_chunk']['correct_pct']:.1f}%" in text
    g = m["citation_grounding"]
    assert f"{g['total_citations_checked']} of {g['total_citations_checked']}" in text
    r = m["refusal_accuracy"]
    assert f"{r['n_unanswerable']} of {r['n_unanswerable']}" in text


def test_the_byte_level_variation_between_the_pair_is_stated(text):
    a, b = (load(n) for n in PAIR)
    A = {r["id"]: r["generated_answer"] for r in a["results"]}
    B = {r["id"]: r["generated_answer"] for r in b["results"]}
    identical = sum(1 for i in A if A[i] == B[i])
    assert f"{identical} of {len(A)}" in text
    assert f"{len(A) - identical} of the\n30" in text or f"{len(A) - identical} of 30" in text \
        or f"{len(A) - identical} differ" in text


def test_the_verdicts_of_the_pair_still_agree(text):
    a, b = (load(n) for n in PAIR)
    va = [r["judge_verdict"] for r in a["results"]]
    vb = [r["judge_verdict"] for r in b["results"]]
    assert va == vb, "the document claims verdict-level agreement on all 30 cases"


def test_the_chunk_text_hash_across_the_chunker_fix_is_current(text):
    """The document pins ca4ba8543fcd786b as the evidence that the timed path
    did identical work either side of 9f1b845."""
    before, after = load(PAIR[1]), load(FINAL)
    digests = []
    for run in (before, after):
        pool = sorted({ch["chunk_text"] for r in run["results"]
                       for ch in r["retrieved_chunks"]})
        digests.append(hashlib.sha256("".join(pool).encode()).hexdigest())
        assert f"{len(pool)} distinct chunks" in text or f"same {len(pool)} distinct chunks" in text
    assert digests[0] == digests[1], "the chunker fix no longer leaves chunk text identical"
    assert digests[0][:16] in text, f"the document pins a stale digest; now {digests[0][:16]}"


def test_grounding_cannot_fail_because_chunks_are_substrings(text):
    doc = (ROOT / "demo_docs" / "company_policy.txt").read_text(encoding="utf-8")
    for name in (BASELINE,) + SETTLED:
        run = load(name)
        assert all(ch["chunk_text"] in doc for r in run["results"]
                   for ch in r["retrieved_chunks"]), f"{name} cites text outside the demo document"
    assert "cannot fail" in text


def test_the_case_set_is_identical_between_baseline_and_final(text):
    a, b = load(BASELINE), load(FINAL)
    ida = [r["id"] for r in a["results"]]
    idb = [r["id"] for r in b["results"]]
    assert ida == idb
    assert {r["id"]: r["question"] for r in a["results"]} == \
           {r["id"]: r["question"] for r in b["results"]}
    assert "same 30\ncase ids" in text or "same 30 case ids" in text


def test_the_document_does_not_use_em_dashes(text):
    assert "—" not in text


def test_the_closing_line_is_present(text):
    assert text.rstrip().endswith(
        "Any claim not in this document is not supported by this repository.")
