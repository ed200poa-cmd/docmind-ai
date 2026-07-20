# DocMind AI — RAG Evaluation Harness

Measures the quality of the retrieval + answering pipeline against a fixed
30-question dataset over the pre-loaded demo document (`demo_docs/company_policy.txt`),
so changes to chunking, retrieval, or prompting can be compared against a baseline
instead of eyeballed.

It imports and calls the app's real code paths (`rag_engine.search`,
`claude_qa.answer_question`) — it does not reimplement retrieval or answering logic.

## Running it

From the `docmind_demo/` directory (or anywhere — the script `chdir`s to the app root
itself so relative paths like `docmind.db` resolve correctly):

```bash
export ANTHROPIC_API_KEY=sk-...          # required, or the script exits with an error
python evals/run_eval.py                 # full run: 30 cases, retrieval + judge + refusal + latency
python evals/run_eval.py --limit 5       # quick smoke test on the first 5 cases
python evals/run_eval.py --no-judge      # retrieval + refusal + latency only, zero judge API calls
```

The demo document must already be indexed in `docmind.db` (it auto-loads the first
time the app starts — run the app once, or just let `main.py`'s lifespan hook do it,
before running the eval).

Each run prints a summary table to the console and writes two timestamped files to
`evals/results/` (gitignored — these can contain model output and are not committed):

- `eval_<UTC timestamp>.json` — every metric plus full per-case raw data (question,
  retrieved chunk ids/text, generated answer, judge verdict + reason, latency). This is
  the file to diff between two runs.
- `eval_<UTC timestamp>.md` — human-readable summary of the same run.

## What each metric means

**Retrieval recall@k** — for each answerable case, was the case's
`expected_source_snippet` (a verbatim phrase from the source document) present in any
of the top-k retrieved chunks? Reported at k=1, 3, and 5, overall and split by
`factual` vs `multi_chunk`. The app always retrieves `TOP_K=5` chunks per query
(`rag_engine.TOP_K`), so recall@5 reflects what the app actually does today; recall@1
and recall@3 exist to show whether `TOP_K` is set higher than it needs to be (if
recall@1 ≈ recall@5, the app could retrieve fewer chunks for the same quality) or too
low (if recall@5 is still poor, no amount of prompting will fix it — the right chunk
was never retrieved).

**Answer correctness** — an LLM judge (temperature 0) compares the generated answer to
`expected_answer` and returns `correct`, `partially_correct`, or `incorrect`, plus a
one-line reason. This is graded on answerable cases only (`factual` + `multi_chunk`).
It is a judgment call, not ground truth — always read `judge_reason` on failing cases
before trusting the label, and spot-check a sample against the raw `generated_answer`
if the numbers look surprising.

Known limitation: the judge uses the **same model** as the one being graded
(`claude-haiku-4-5-20251001`, `evals/run_eval.py:JUDGE_MODEL`), not an independent
stronger model. This is not a design preference — the newer model families available
in this account (`claude-sonnet-5`, `claude-opus-4-8`) reject an explicit `temperature`
param outright ("`temperature` is deprecated for this model"), and the task requires
temperature=0 for determinism, so haiku-4-5 was the only model in this account that
could satisfy both constraints. A same-model judge is more prone to self-preference
bias than an independent one. If a temperature-controllable stronger model becomes
available, change `JUDGE_MODEL` in `run_eval.py` — nothing else needs to change.

**Citation grounding** — for each answerable case, checks whether every chunk the app
returned as a `source` for its answer is a verbatim substring of the real source
document. Because chunks are produced by literally slicing the document text
(`rag_engine._chunk_pages`), this is expected to sit at ~100% by construction — it is
a structural sanity check on the retrieval/storage pipeline (catches bugs like
encoding corruption, cross-document leakage, or a stale/wrong index), not a check on
whether the model's *prose* faithfully represents the citation. A drop below 100% here
means something is actually broken in retrieval or storage, not that the LLM
hallucinated wording.

**Refusal accuracy** — for the 7 `unanswerable` cases (real HR questions the document
does not cover), the correct behavior is for the app to reply with its exact instructed
refusal string, `"This information is not found in the uploaded documents."`
(`claude_qa.SYSTEM_PROMPT`). Refusal accuracy is the percent of unanswerable cases
where that phrase appears in the answer. Any case where it does *not* appear is logged
in `hallucinated_case_ids` — these are the most important failures in the whole eval,
since they mean the app fabricated an HR policy answer that doesn't exist in the
document. Always read the actual `generated_answer` for these ids; a case can also land
here if the model refused in different wording than the exact instructed phrase, which
is a prompt-adherence issue worth knowing about even if it isn't a true hallucination.

**Latency** — wall-clock seconds for the full per-query pipeline (`rag_engine.search`
+ `claude_qa.answer_question`), median and p95 across all cases run. Judge calls are
not included (they're grading overhead, not user-facing latency).

## Interpreting a regression

Diff two JSON result files (by id, since ids are stable across runs):

- **Recall@5 drops** → retrieval regressed (embedding model, index, chunking, or top_k
  changed). Look at `retrieved_chunks` for the affected case ids to see what got
  retrieved instead of the expected chunk.
- **Recall stays flat but correctness drops** → the right context is being retrieved,
  but the prompt/model is failing to use it well. Compare `generated_answer` and
  `judge_reason` between the two runs for the same case id.
- **Citation grounding drops below 100%** → treat as a bug, not noise (see above).
- **Refusal accuracy drops / `hallucinated_case_ids` grows** → highest priority. The
  system is now inventing HR policy that doesn't exist — check the new
  `generated_answer` text and, if the prompt changed, whether the new prompt weakened
  the refusal instruction.
- **Latency p95 jumps** → check for infra issues (model swap, network) rather than
  code logic, unless chunk count or context size also changed.

Because the answer call runs at `temperature=0`, a regression from a genuine app code
change (chunking, retrieval, or the system prompt) should reproduce deterministically
between two runs on the same case. If a case flips between runs with an unchanged app,
suspect API-side non-determinism (temperature 0 reduces but does not guarantee
byte-identical output) or FAISS/embedding non-determinism (should be exact/deterministic
for `IndexFlatIP`, but re-embedding after a document re-upload can shift chunk ids).

## Adding new test cases

Append an object to the JSON array in `evals/dataset.json`:

```json
{
  "id": "factual_16",
  "question": "...",
  "expected_answer": "short reference answer, or null for unanswerable",
  "expected_source_snippet": "a verbatim phrase copy-pasted from demo_docs/company_policy.txt, or null for unanswerable",
  "category": "factual | multi_chunk | unanswerable"
}
```

Rules that keep the dataset trustworthy:

- `expected_source_snippet` must be an exact, verbatim substring of
  `demo_docs/company_policy.txt` (copy-paste it, don't retype it — retyping risks
  smart-quote/en-dash mismatches that silently break recall checks). It's what recall@k
  is checked against, so if it isn't a real substring the case will always show as a
  retrieval miss regardless of what the app actually retrieves.
- `factual` = the answer is stated in one paragraph/bullet block (i.e., you did not
  need to read two different `SECTION` headers to answer it).
- `multi_chunk` = you genuinely needed facts from two or more different `SECTION`
  headers to answer it in full.
- `unanswerable` = a plausible HR question that the document simply does not address
  (check by reading the whole document — don't guess). Set both `expected_answer` and
  `expected_source_snippet` to `null`.
- Keep ids stable once added — other runs' JSON files are diffed by id, and renumbering
  breaks that history.

If you add a new source document instead of/alongside `company_policy.txt`, note that
`run_eval.py` currently assumes a single indexed document and does not filter by
`doc_id`; you'd need to either pass `doc_id` through per-case or keep the eval
single-document.
