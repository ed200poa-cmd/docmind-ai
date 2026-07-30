"""API-free retrieval probe for DocMind AI.

Measures the *retrieval* half of the eval — recall@k and a verbatim-substring
check on retrieved chunks — using only the local fastembed embeddings and the
live FAISS index. It makes **zero** Claude API calls, so it can be run for free
and as often as needed while iterating on the retrieval/chunking path
(`rag_engine.py`).

What it does NOT measure: answer correctness, answer-side citation grounding
(the cited-source substring test in `run_eval.py`), and refusal accuracy — all
three require an answer call, i.e. the API key. Use `run_eval.py` for those.

The recall numbers here reproduce `run_eval.py`'s retrieval block exactly,
because both call the same `rag_engine.search(...)` against the same index and
score the same `expected_source_snippet` per case.

Usage:
    python evals/retrieval_probe.py                 # all answerable cases
    python evals/retrieval_probe.py --category multi_chunk
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
APP_ROOT = EVALS_DIR.parent
DATASET_PATH = EVALS_DIR / "dataset.json"
RESULTS_DIR = EVALS_DIR / "results"
DOC_PATH = APP_ROOT / "demo_docs" / "company_policy.txt"
K_VALUES = (1, 3, 5)

sys.path.insert(0, str(APP_ROOT))


def load_dataset(category=None):
    # Read-only: never rewrites dataset.json, only selects a subset for a run.
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if category:
        available = sorted({c["category"] for c in data})
        if category not in available:
            raise SystemExit(
                f"ERROR: --category {category!r} not found. Available: {', '.join(available)}."
            )
        data = [c for c in data if c["category"] == category]
    return data


def recall_block(cases_hits):
    """cases_hits: list of {str(k): bool} dicts -> {k: pct}."""
    n = len(cases_hits) or 1
    return {
        str(k): round(sum(1 for h in cases_hits if h[str(k)]) / n * 100, 1)
        for k in K_VALUES
    }


def main():
    parser = argparse.ArgumentParser(description="API-free retrieval probe (recall@k + verbatim check).")
    parser.add_argument("--category", type=str, default=None,
                        help="Restrict to one category, e.g. multi_chunk.")
    args = parser.parse_args()

    import os
    os.chdir(APP_ROOT)  # rag_engine/document_store use app-root-relative paths (docmind.db)
    import document_store
    import rag_engine

    document_store.init_db()
    rag_engine.init_rag()
    if rag_engine._index is None or rag_engine._index.ntotal == 0:
        raise SystemExit(
            "ERROR: no documents indexed (docmind.db empty). Launch the app once so the demo "
            "document auto-loads, then re-run."
        )

    doc_text = DOC_PATH.read_text(encoding="utf-8")
    dataset = load_dataset(args.category)
    answerable = [c for c in dataset if c["category"] != "unanswerable"]

    per_case = []
    verbatim_flags = []
    for case in answerable:
        retrieved = rag_engine.search(question=case["question"], doc_id=None, top_k=max(K_VALUES))
        texts = [c["chunk_text"] for c in retrieved]
        snippet = case["expected_source_snippet"]
        hits = {str(k): any(snippet in t for t in texts[:k]) for k in K_VALUES}
        per_case.append({"id": case["id"], "category": case["category"], "recall_at_k": hits})
        # Structural check: is every retrieved chunk a verbatim substring of the source doc?
        for t in texts:
            verbatim_flags.append(t in doc_text)

    def by_cat(cat):
        return [c["recall_at_k"] for c in per_case if c["category"] == cat]

    overall = recall_block([c["recall_at_k"] for c in per_case])
    cats = sorted({c["category"] for c in per_case})
    verbatim_pct = round(sum(verbatim_flags) / (len(verbatim_flags) or 1) * 100, 1)

    print(f"Retrieval probe (API-free) — {len(answerable)} answerable cases, "
          f"index={rag_engine._index.ntotal} chunks, model={rag_engine.EMBEDDING_MODEL}\n")
    header = "|            | k=1 | k=3 | k=5 |"
    print(header)
    print("|---|---|---|---|")
    print(f"| overall    | {overall['1']}% | {overall['3']}% | {overall['5']}% |")
    for cat in cats:
        b = recall_block(by_cat(cat))
        print(f"| {cat:10s} | {b['1']}% | {b['3']}% | {b['5']}% |")
    print(f"\nRetrieved-chunk verbatim-substring rate: {verbatim_pct}% "
          f"({sum(verbatim_flags)}/{len(verbatim_flags)} chunks are exact substrings of the source)")
    print("API calls made: 0")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"retrieval_probe_{ts}.json"
    out_path.write_text(json.dumps({
        "timestamp": ts,
        "kind": "retrieval_probe_api_free",
        "category_filter": args.category,
        "index_chunks": rag_engine._index.ntotal,
        "embedding_model": rag_engine.EMBEDDING_MODEL,
        "n_answerable": len(answerable),
        "recall": {"overall": overall, **{c: recall_block(by_cat(c)) for c in cats}},
        "retrieved_verbatim_substring_pct": verbatim_pct,
        "per_case": per_case,
        "api_calls": 0,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path.relative_to(APP_ROOT)}")


if __name__ == "__main__":
    main()
