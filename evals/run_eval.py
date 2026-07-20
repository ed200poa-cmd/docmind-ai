#!/usr/bin/env python3
"""Evaluation harness for DocMind AI's RAG pipeline.

Runs evals/dataset.json through the application's real retrieval
(rag_engine.search) and answering (claude_qa.answer_question) code paths
-- imported, not reimplemented -- and reports:

  - retrieval recall@k (k = 1, 3, 5), overall and by category
  - answer correctness, via an LLM-as-judge call (temperature 0)
  - citation grounding: are cited source chunks verbatim in the source doc
  - refusal accuracy on unanswerable questions, and a list of hallucinations
  - latency (median, p95)

Usage:
    python evals/run_eval.py                  # full run, with judge
    python evals/run_eval.py --limit 5         # first 5 cases only
    python evals/run_eval.py --no-judge        # retrieval metrics only, no LLM calls for grading

See evals/README.md for how to interpret the output.
"""
import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
APP_ROOT = EVALS_DIR.parent
DATASET_PATH = EVALS_DIR / "dataset.json"
RESULTS_DIR = EVALS_DIR / "results"
DOC_PATH = APP_ROOT / "demo_docs" / "company_policy.txt"
DEMO_DOC_FILENAME = "company_policy.txt"

sys.path.insert(0, str(APP_ROOT))  # so `import rag_engine`, `import claude_qa` resolve like the app does

REFUSAL_PHRASE = "This information is not found in the uploaded documents."
K_VALUES = (1, 3, 5)
# NOTE: the newer model families available in this account (claude-sonnet-5,
# claude-opus-4-8) reject an explicit `temperature` param ("deprecated for this
# model") -- there is no way to force temperature=0 on them. claude-haiku-4-5,
# the same model claude_qa.py uses to answer, is the only model in this account
# that accepts temperature=0, so it is used for judging too. This means the
# judge is not an independent model from the one being graded -- a known
# limitation of this harness; see evals/README.md.
JUDGE_MODEL = "claude-haiku-4-5-20251001"
ANSWER_TEMPERATURE = 0
JUDGE_TEMPERATURE = 0

JUDGE_SYSTEM_PROMPT = """You are a strict grading judge for a RAG (retrieval-augmented generation) question-answering system.

You will be given a question, a short reference answer (the ground truth), and the system's generated answer.
Grade the generated answer against the reference answer only -- do not use outside knowledge, and do not reward
style or verbosity.

Return ONLY a single line of JSON, no markdown fences, no commentary:
{"verdict": "correct" | "partially_correct" | "incorrect", "reason": "<one short sentence>"}

Grading rules:
- "correct": the generated answer states the same key fact(s) as the reference answer, with no contradictions
  and no missing key fact.
- "partially_correct": the generated answer is on-topic, and gets at least one key fact right, but is incomplete,
  imprecise, or only partially matches the reference.
- "incorrect": the generated answer contradicts the reference, omits the key fact entirely, states a wrong number
  or wrong condition, or is a refusal / "not found" response when the reference shows a real answer exists."""


def load_dataset(limit: int | None):
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if limit:
        data = data[:limit]
    return data


def judge_answer(client, question: str, expected_answer: str, generated_answer: str) -> tuple[str, str]:
    user_msg = (
        f"Question: {question}\n"
        f"Reference answer: {expected_answer}\n"
        f"Generated answer: {generated_answer}\n\n"
        "Grade the generated answer against the reference answer."
    )
    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=200,
        temperature=JUDGE_TEMPERATURE,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if "\n" in raw:
            raw = raw.split("\n", 1)[1]
    try:
        parsed = json.loads(raw)
        verdict = parsed.get("verdict", "incorrect")
        reason = parsed.get("reason", "")
    except (json.JSONDecodeError, AttributeError):
        verdict = "incorrect"
        reason = f"judge output not parseable as JSON: {raw[:200]!r}"
    if verdict not in ("correct", "partially_correct", "incorrect"):
        verdict = "incorrect"
    return verdict, reason


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(pct / 100 * (len(s) - 1))))
    return s[idx]


def run_eval(limit: int | None, no_judge: bool) -> dict:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY is not set. Export it (e.g. via the app's .env) before running the eval.",
            file=sys.stderr,
        )
        sys.exit(1)

    os.chdir(APP_ROOT)  # rag_engine / document_store use paths relative to the app root (e.g. docmind.db)

    import document_store
    import rag_engine
    import claude_qa
    import anthropic

    document_store.init_db()
    rag_engine.init_rag()
    if rag_engine._index is None or rag_engine._index.ntotal == 0:
        print(
            "ERROR: no documents are indexed (docmind.db is empty). Run the app once so the demo "
            "document auto-loads (or upload one), then re-run the eval.",
            file=sys.stderr,
        )
        sys.exit(1)

    doc_text = DOC_PATH.read_text(encoding="utf-8")
    dataset = load_dataset(limit)

    judge_client = None if no_judge else anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    api_calls = {"answer_calls": 0, "judge_calls": 0}
    results = []

    print(f"Running {len(dataset)} eval cases against top_k={max(K_VALUES)} retrieval "
          f"(embedding model: {rag_engine.EMBEDDING_MODEL}, answer model: {claude_qa.MODEL})…\n")

    for i, case in enumerate(dataset, 1):
        is_answerable = case["category"] != "unanswerable"

        t0 = time.perf_counter()
        retrieved = rag_engine.search(question=case["question"], doc_id=None, top_k=max(K_VALUES))
        answer_result = claude_qa.answer_question(
            question=case["question"],
            chunks=retrieved,
            doc_name=DEMO_DOC_FILENAME,
            temperature=ANSWER_TEMPERATURE,
        )
        latency = time.perf_counter() - t0
        api_calls["answer_calls"] += 1

        generated_answer = answer_result["answer"]
        retrieved_texts = [c["chunk_text"] for c in retrieved]

        recall_hits = {}
        if is_answerable:
            snippet = case["expected_source_snippet"]
            for k in K_VALUES:
                recall_hits[str(k)] = any(snippet in t for t in retrieved_texts[:k])
        else:
            recall_hits = {str(k): None for k in K_VALUES}

        cited_sources = answer_result.get("sources", [])
        if is_answerable:
            grounded_flags = [c["chunk_text"] in doc_text for c in cited_sources]
        else:
            grounded_flags = []

        judge_verdict = None
        judge_reason = None
        if is_answerable and not no_judge:
            judge_verdict, judge_reason = judge_answer(
                judge_client, case["question"], case["expected_answer"], generated_answer
            )
            api_calls["judge_calls"] += 1

        refused = REFUSAL_PHRASE in generated_answer
        hallucinated = (not is_answerable) and (not refused)

        result = {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "expected_answer": case["expected_answer"],
            "expected_source_snippet": case["expected_source_snippet"],
            "generated_answer": generated_answer,
            "retrieved_chunk_ids": [c.get("chunk_id") for c in retrieved],
            "retrieved_chunks": [
                {
                    "chunk_id": c.get("chunk_id"),
                    "page_num": c.get("page_num"),
                    "relevance_score": c.get("relevance_score"),
                    "chunk_text": c.get("chunk_text"),
                }
                for c in retrieved
            ],
            "recall_at_k": recall_hits,
            "citation_grounded_flags": grounded_flags,
            "citation_grounded_pct": (
                round(sum(grounded_flags) / len(grounded_flags) * 100, 1) if grounded_flags else None
            ),
            "judge_verdict": judge_verdict,
            "judge_reason": judge_reason,
            "refused": refused,
            "hallucinated_on_unanswerable": hallucinated,
            "latency_sec": round(latency, 3),
        }
        results.append(result)

        verdict_str = judge_verdict or ("refused" if refused else ("hallucinated" if hallucinated else "-"))
        print(f"  [{i:2d}/{len(dataset)}] {case['id']:16s} {case['category']:14s} "
              f"{latency:5.2f}s  {verdict_str}")

    metrics = compute_metrics(results, no_judge)
    return {
        "results": results,
        "metrics": metrics,
        "api_calls": api_calls,
    }


def compute_metrics(results: list[dict], no_judge: bool) -> dict:
    def by_category(cat):
        return [r for r in results if r["category"] == cat]

    answerable = [r for r in results if r["category"] != "unanswerable"]
    unanswerable = [r for r in results if r["category"] == "unanswerable"]

    def recall_block(cases):
        block = {}
        for k in K_VALUES:
            hits = [r["recall_at_k"][str(k)] for r in cases if r["recall_at_k"][str(k)] is not None]
            block[f"recall_at_{k}"] = round(sum(hits) / len(hits) * 100, 1) if hits else None
        return block

    retrieval = {
        "overall": recall_block(answerable),
        "by_category": {
            "factual": recall_block(by_category("factual")),
            "multi_chunk": recall_block(by_category("multi_chunk")),
        },
    }

    correctness = None
    if not no_judge:
        judged = [r for r in answerable if r["judge_verdict"] is not None]
        n = len(judged) or 1
        correctness = {
            "n_judged": len(judged),
            "correct_pct": round(sum(1 for r in judged if r["judge_verdict"] == "correct") / n * 100, 1),
            "partially_correct_pct": round(
                sum(1 for r in judged if r["judge_verdict"] == "partially_correct") / n * 100, 1
            ),
            "incorrect_pct": round(sum(1 for r in judged if r["judge_verdict"] == "incorrect") / n * 100, 1),
            "by_category": {
                cat: {
                    "n": len(by_category(cat)),
                    "correct_pct": round(
                        sum(1 for r in by_category(cat) if r["judge_verdict"] == "correct")
                        / (len(by_category(cat)) or 1) * 100, 1
                    ),
                }
                for cat in ("factual", "multi_chunk")
            },
        }

    all_flags = [f for r in answerable for f in r["citation_grounded_flags"]]
    citation_grounding = {
        "total_citations_checked": len(all_flags),
        "verifiable_pct": round(sum(all_flags) / len(all_flags) * 100, 1) if all_flags else None,
    }

    refused_count = sum(1 for r in unanswerable if r["refused"])
    hallucinated_cases = [r["id"] for r in unanswerable if r["hallucinated_on_unanswerable"]]
    refusal = {
        "n_unanswerable": len(unanswerable),
        "refusal_accuracy_pct": round(refused_count / len(unanswerable) * 100, 1) if unanswerable else None,
        "hallucinated_case_ids": hallucinated_cases,
    }

    latencies = [r["latency_sec"] for r in results]
    latency = {
        "median_sec": round(statistics.median(latencies), 3) if latencies else None,
        "p95_sec": round(percentile(latencies, 95), 3) if latencies else None,
        "min_sec": round(min(latencies), 3) if latencies else None,
        "max_sec": round(max(latencies), 3) if latencies else None,
    }

    return {
        "retrieval_recall": retrieval,
        "answer_correctness": correctness,
        "citation_grounding": citation_grounding,
        "refusal_accuracy": refusal,
        "latency": latency,
    }


def print_summary(metrics: dict, api_calls: dict, no_judge: bool) -> None:
    r = metrics["retrieval_recall"]
    print("\n" + "=" * 72)
    print("RETRIEVAL RECALL @ K  (expected source snippet found in top-k chunks)")
    print("=" * 72)
    print(f"{'':14s} {'k=1':>8s} {'k=3':>8s} {'k=5':>8s}")
    for label, block in [("overall", r["overall"]), ("factual", r["by_category"]["factual"]),
                          ("multi_chunk", r["by_category"]["multi_chunk"])]:
        print(f"{label:14s} {fmt_pct(block['recall_at_1']):>8s} {fmt_pct(block['recall_at_3']):>8s} "
              f"{fmt_pct(block['recall_at_5']):>8s}")

    if not no_judge and metrics["answer_correctness"]:
        c = metrics["answer_correctness"]
        print("\n" + "=" * 72)
        print(f"ANSWER CORRECTNESS  (LLM judge, n={c['n_judged']})")
        print("=" * 72)
        print(f"  correct:            {fmt_pct(c['correct_pct'])}")
        print(f"  partially_correct:  {fmt_pct(c['partially_correct_pct'])}")
        print(f"  incorrect:          {fmt_pct(c['incorrect_pct'])}")
        for cat, block in c["by_category"].items():
            print(f"    {cat:12s} correct: {fmt_pct(block['correct_pct'])}  (n={block['n']})")

    cg = metrics["citation_grounding"]
    print("\n" + "=" * 72)
    print("CITATION GROUNDING  (cited source chunk text verbatim in source doc)")
    print("=" * 72)
    print(f"  verifiable: {fmt_pct(cg['verifiable_pct'])}  ({cg['total_citations_checked']} citations checked)")

    ref = metrics["refusal_accuracy"]
    print("\n" + "=" * 72)
    print(f"REFUSAL ACCURACY  (unanswerable cases, n={ref['n_unanswerable']})")
    print("=" * 72)
    print(f"  correctly declined: {fmt_pct(ref['refusal_accuracy_pct'])}")
    if ref["hallucinated_case_ids"]:
        print(f"  HALLUCINATED on: {', '.join(ref['hallucinated_case_ids'])}  <-- most important failures")
    else:
        print("  hallucinated on: none")

    lat = metrics["latency"]
    print("\n" + "=" * 72)
    print("LATENCY (seconds, full search+answer pipeline per query)")
    print("=" * 72)
    print(f"  median: {lat['median_sec']}   p95: {lat['p95_sec']}   min: {lat['min_sec']}   max: {lat['max_sec']}")

    print("\n" + "=" * 72)
    total = api_calls["answer_calls"] + api_calls["judge_calls"]
    print(f"API CALLS: {total} total  (answer: {api_calls['answer_calls']}, judge: {api_calls['judge_calls']})")
    print("=" * 72)


def fmt_pct(v) -> str:
    return "n/a" if v is None else f"{v:.1f}%"


def write_markdown(path: Path, run_data: dict, timestamp: str) -> None:
    m = run_data["metrics"]
    lines = [f"# Eval run — {timestamp}", ""]

    r = m["retrieval_recall"]
    lines += ["## Retrieval recall @ k", "", "| | k=1 | k=3 | k=5 |", "|---|---|---|---|"]
    for label, block in [("overall", r["overall"]), ("factual", r["by_category"]["factual"]),
                          ("multi_chunk", r["by_category"]["multi_chunk"])]:
        lines.append(f"| {label} | {fmt_pct(block['recall_at_1'])} | {fmt_pct(block['recall_at_3'])} | "
                      f"{fmt_pct(block['recall_at_5'])} |")
    lines.append("")

    if m["answer_correctness"]:
        c = m["answer_correctness"]
        lines += ["## Answer correctness (LLM judge)", "",
                  f"- n judged: {c['n_judged']}",
                  f"- correct: {fmt_pct(c['correct_pct'])}",
                  f"- partially_correct: {fmt_pct(c['partially_correct_pct'])}",
                  f"- incorrect: {fmt_pct(c['incorrect_pct'])}", ""]
        for cat, block in c["by_category"].items():
            lines.append(f"  - {cat}: {fmt_pct(block['correct_pct'])} correct (n={block['n']})")
        lines.append("")

    cg = m["citation_grounding"]
    lines += ["## Citation grounding", "",
              f"- verifiable: {fmt_pct(cg['verifiable_pct'])} "
              f"({cg['total_citations_checked']} citations checked)", ""]

    ref = m["refusal_accuracy"]
    lines += ["## Refusal accuracy", "",
              f"- n unanswerable cases: {ref['n_unanswerable']}",
              f"- correctly declined: {fmt_pct(ref['refusal_accuracy_pct'])}",
              f"- hallucinated on: {', '.join(ref['hallucinated_case_ids']) if ref['hallucinated_case_ids'] else 'none'}",
              ""]

    lat = m["latency"]
    lines += ["## Latency (seconds)", "",
              f"- median: {lat['median_sec']}", f"- p95: {lat['p95_sec']}",
              f"- min: {lat['min_sec']}", f"- max: {lat['max_sec']}", ""]

    lines += ["## API calls", "",
              f"- answer calls: {run_data['api_calls']['answer_calls']}",
              f"- judge calls: {run_data['api_calls']['judge_calls']}", ""]

    lines += ["## Failing cases (incorrect or hallucinated)", ""]
    for res in run_data["results"]:
        if res["judge_verdict"] == "incorrect" or res["hallucinated_on_unanswerable"]:
            lines.append(f"- **{res['id']}** ({res['category']}): {res.get('judge_reason') or 'hallucinated on unanswerable question'}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run the DocMind AI RAG evaluation harness.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N cases in the dataset.")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-judge correctness grading (retrieval + refusal metrics only).")
    args = parser.parse_args()

    run_data = run_eval(limit=args.limit, no_judge=args.no_judge)
    print_summary(run_data["metrics"], run_data["api_calls"], args.no_judge)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = RESULTS_DIR / f"eval_{timestamp}.json"
    md_path = RESULTS_DIR / f"eval_{timestamp}.md"

    out = {
        "timestamp": timestamp,
        "dataset_path": str(DATASET_PATH.relative_to(APP_ROOT)),
        "num_cases": len(run_data["results"]),
        "config": {
            "top_k_values_checked": list(K_VALUES),
            "answer_model": None,
            "judge_model": None if args.no_judge else JUDGE_MODEL,
            "answer_temperature": ANSWER_TEMPERATURE,
            "judge_temperature": None if args.no_judge else JUDGE_TEMPERATURE,
            "no_judge": args.no_judge,
            "limit": args.limit,
        },
        "api_calls": run_data["api_calls"],
        "metrics": run_data["metrics"],
        "results": run_data["results"],
    }
    # fill in answer_model without importing at module scope (claude_qa imported lazily inside run_eval)
    sys.path.insert(0, str(APP_ROOT))
    import claude_qa as _claude_qa
    out["config"]["answer_model"] = _claude_qa.MODEL

    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, run_data, timestamp)

    print(f"\nWrote:\n  {json_path.relative_to(APP_ROOT)}\n  {md_path.relative_to(APP_ROOT)}")


if __name__ == "__main__":
    main()
