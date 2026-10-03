# What can and cannot be claimed

A boundary list, not an argument. `evals/RESULTS.md` carries the per-change
analysis. This page is for the five minutes before someone asks about a number.

Every figure below names the run it came from. All seven runs are tracked in
`evals/results/`, with their SHA-256 in `evals/results/README.md`.

| run | date | what it is |
|---|---|---|
| `eval_20260720T124830Z` | 2026-07-20 | pre-optimisation baseline |
| `eval_20260730T014202Z` | 2026-07-30 | full run |
| `eval_20260730T122037Z` | 2026-07-30 | confirmation run 1 |
| `eval_20260730T122154Z` | 2026-07-30 | confirmation run 2 |
| `eval_20260803T022554Z` | 2026-08-03 | after the chunker fix; source of the published quality figures |
| `eval_20261003T142527Z` | 2026-10-03 | re-measurement run 1, no code change to retrieval or answering |
| `eval_20261003T142910Z` | 2026-10-03 | re-measurement run 2 |

---

## 1. What this repository measured

**Retrieval recall on 30 frozen cases.** Run `eval_20260803T022554Z`, n=23
answerable: overall recall@1 87.0%, recall@3 95.7%, recall@5 100.0%. By
category, factual (n=15) 100.0% at every k, multi_chunk (n=8) 62.5% / 87.5% /
100.0%. The baseline `eval_20260720T124830Z` measured 73.9% / 91.3% / 95.7%.

**Answer correctness, LLM judge at temperature 0.** Run
`eval_20260803T022554Z`, n=23: 91.3% correct, 8.7% partially_correct, 0.0%
incorrect. factual (n=15) 100.0%, multi_chunk (n=8) 75.0%. The two
`partially_correct` cases are `multi_chunk_03` and `multi_chunk_07`.

**That answer correctness is a range, not a point.** Across the six tracked runs
after the baseline, meaning `eval_20260730T014202Z`, `eval_20260730T122037Z`,
`eval_20260730T122154Z`, `eval_20260803T022554Z`, `eval_20261003T142527Z` and
`eval_20261003T142910Z`, `correct` reads 91.3% or 95.7% and nothing else, and
`incorrect` is 0.0% in all six. Four of those six read 91.3%, which is the
figure published from `eval_20260803T022554Z`; the two 2026-10-03 runs read
95.7%. Those two re-ran the same 30 cases with no change to retrieval,
chunking, the index, the prompts or the judge. The baseline
`eval_20260720T124830Z` sits outside this range and is stated separately: 87.0%
correct, 8.7% `partially_correct`, 4.3% `incorrect`.

**What moved inside that range.** One case of 23. `multi_chunk_03` reads
`partially_correct` in `eval_20260803T022554Z` and `correct` in both
2026-10-03 runs, while `multi_chunk_07` reads `partially_correct` in all three.
Retrieval was identical in 30 of 30 cases across those three runs, so the index
is not the cause. The generated answer for `multi_chunk_03` changed, and it
changed in the one respect the August judge named, the missing `$500` annual
limit. Both 2026-10-03 files were checked before being tracked on the standard
in `evals/results/README.md`: the full credential and personal-data pattern set
with no match, `gitleaks 8.30.1` with this repository's `.gitleaks.toml`
reporting `no leaks found` over the full byte length of each, and provenance
confirmed over 150 `chunk_text` and 30 `question` values per file.
`docs/reports/hillclimb-2026-10-03.md` carries the full comparison. The two
2026-10-03 figures are not re-derived by the tests in
`tests/test_claims_doc.py`, whose run constants name the first five runs only.

**Verdict-level reproducibility.** Runs `eval_20260730T122037Z` and
`eval_20260730T122154Z`, executed back to back with no code change between them,
returned the same verdict on all 30 cases and identical aggregates. n=30 cases,
2 runs.

**That reproducibility is at the verdict level only.** Between those same two
runs, 26 of 30 generated answers are byte-identical and 4 differ in wording, at
temperature 0. Total answer length moved by 1.7% between them.

**Citation presence.** 115 citations checked in every one of the five runs, all
present verbatim in the source document. Section 3 says what that does and does
not mean.

**Refusal on unanswerable questions.** Run `eval_20260803T022554Z`: 7 of 7
declined, no hallucinations. n=7.

**Latency, end to end.** `run_eval.py` starts its clock at line 174, before
`rag_engine.search`, and stops it at line 182, after
`claude_qa.answer_question` returns. It therefore covers query embedding, FAISS
search and one Anthropic answer call, and excludes the judge call. n=30 cases
per run. The baseline `eval_20260720T124830Z` measured a median of 1.72s. The
four later runs measured 1.018s, 1.145s, 1.128s and 1.025s, so the end state is
reported as the **median of those four, 1.08s** (1.0765s before rounding), with
the range 1.018s to 1.145s.

**That those four runs are comparable.** `rag_engine.py` was untouched between
`18ecd33` (2026-07-20) and `9f1b845` (2026-08-02), and `claude_qa.py` has been
untouched since `fb04ca7` (2026-07-20). Across `9f1b845`, the chunker fix, the
retrieved chunk text is identical case for case: the same 17 distinct chunks,
concatenated SHA-256 `ca4ba8543fcd786b` on both sides. The timed span was doing
the same work in all four.

**That the case set did not change.** Baseline and final run carry the same 30
case ids in the same order, the same `question` for every id, and the same
`expected_source_snippet` for every id. Compared as sets, not read off a
sentence.

**API cost per run.** 30 answer calls and 23 judge calls, 53 total, unchanged
across every run.

**An offline suite of 110 tests**, 109 passing and 1 skipped, with no API call and
no network. The suite has two kinds of test in it and they are worth keeping
apart. **96 of them cover** chunking, storage, retention, search and the MCP tool surface:
they exercise the code a caller reaches, and they are the count meant by
"offline tests" wherever this project reports one. The other 14 re-derive every
figure in this document
from the five tracked runs and fail if the prose and the artifacts disagree;
they test documentation, not the pipeline. Run the pipeline suite alone with
`pytest -m "not docs"`. The split is enforced rather than described: a `docs`
marker separates them, and one of those 14 tests reads the collected counts back
out of pytest and fails if the numbers above stop being true. It did fail when
the MCP surface tests were added, which is how the numbers above came to be
updated in the same commit.

---

## 2. What this repository did not measure

**Whether the answer model fabricates citations.** It was never in a position
to. `claude_qa.answer_question` builds its `sources` list by echoing back the
chunks it was handed, before any of the model's text is read, so the model does
not select, filter or name its citations.

**p95 latency, reproducibly.** It moved 2.58s, 3.42s, 2.52s, 2.34s across the
Session-1 runs and then 4.89s and 2.23s across the 2026-07-30 pair, while the
medians of that pair sat 0.017s apart. n=30 is a small sample for a tail
statistic, and the tail is where a single slow call lands.

**Anything about a corpus larger than this one.** Every run here indexes one
document, `demo_docs/company_policy.txt`, as 17 chunks. Retrieval selects 5 of
17, so roughly a third of the corpus reaches the prompt on every query.

**Production conditions.** These are local runs against the demo document. The
deployed application was never measured.

**Whether both facts of a multi-fact question were retrieved.** The 8
multi_chunk cases each ask for two facts but carry a single
`expected_source_snippet` string, so multi_chunk recall measures whether that
one named snippet appeared in the top k. The judge sees both facts when grading
the answer, so the gap is specific to the recall numbers.

**How often any single-run behaviour would recur.** Two runs of the confirmation
pair and four runs of the settled pipeline. That is enough to see a range and
not enough to state a rate.

---

## 3. Per figure: what holds, and what gets taken apart

| the figure | this can be said | this gets taken apart |
|---|---|---|
| **median latency 1.72s to 1.08s** | Baseline and endpoint both n=30, same 30 cases verified id by id, temperature 0 on the answer call inside the timed span. The endpoint is the median of four runs whose timed path did identical work, range **1.018s to 1.145s**. The total drop of **0.64s** is more than five times that range, so the direction survives the noise. | Quoting a single run as the endpoint. The four runs of the unchanged pipeline span 0.127s, so any one of them is a draw. Also attributing an improvement to the last step: the 0.12s that step claimed is inside the 0.127s range, and 1.018s was already reached before it. That claim is withdrawn in `evals/RESULTS.md`. |
| **p95 latency 1.83s** | Nothing. This figure is not reproducible run to run and is labelled that way. | Any use at all. Across six runs it measured 2.58s, 3.42s, 2.52s, 2.34s, 4.89s, 2.23s and 1.83s, while the medians of a same-code pair sat 0.017s apart. A single slow API call moves it, and n=30 is a small sample for a tail statistic. |
| **recall@1 / @3 / @5 = 87.0% / 95.7% / 100.0%** | On this 30-case set against this one 17-chunk document, the answering chunk is in the top 5 for every answerable case. Run `eval_20260803T022554Z`, n=23. | "Recall is 100%." It is recall@5 on a corpus of 17 chunks where the top 5 is about a third of everything indexed. multi_chunk recall@1 is 62.5%, and multi_chunk cases are scored on one of the two snippets they require. On a 2566-chunk corpus the same cases give recall@5 87.0% (measured in `copilot-mcp-bridge`, not here). |
| **answer correctness 91.3%** | 21 of 23 answerable cases graded correct by an LLM judge at temperature 0, with 0.0% incorrect. Run `eval_20260803T022554Z`. The same verdicts appeared in both 2026-07-30 confirmation runs. | "91.3% is the system's accuracy." n=23, one document, questions written against that document. The two non-correct cases are both `multi_chunk`, where the category rate is 75.0% (n=8). On a 2566-chunk corpus the same cases give 73.9% (measured in `copilot-mcp-bridge`, not here). |
| **citation grounding 100.0%, 115 of 115** | Every cited chunk is present verbatim in the source document. No quotation was fabricated, in any of the five runs. | "The citations were relevant", or that this says anything about the answer model. The `sources` field is the retrieval result echoed back, and `_chunk_pages` emits every chunk as a contiguous slice of its page, so the check tests a substring for membership in the string it was cut from. Verified here: every retrieved chunk in all five runs is a substring of `demo_docs/company_policy.txt`. On a single-document corpus this check cannot fail. |
| **refusal accuracy 100.0%, 7 of 7** | All seven unanswerable questions were declined, with zero hallucinations, in run `eval_20260803T022554Z`. n=7. | "The system does not hallucinate." n=7, and every one of those questions was asked against a corpus containing nothing that resembles an answer. Add four near-domain documents and the same seven cases give 6 of 7 (measured in `copilot-mcp-bridge`, not here). |
| **110 offline tests** | 109 passing and 1 skipped, no API call and no network, run in CI. **96 cover chunking**, storage, retention, search and the MCP tool surface, and that 96 is what "offline tests" counts here; the other 14 check that this document's figures still match the tracked artifacts. | Quoting one total without saying which kind. A count that mixes pipeline tests with tests of this page's own prose describes two different things. And presenting either as evidence for a retrieval or answer figure: every quality number here comes from the five tracked runs, not from the suite. |

---

## 4. If someone asks

### "How was this latency measured?"

It is wall clock around two calls: `run_eval.py` starts a `perf_counter` before
`rag_engine.search` and stops it after `claude_qa.answer_question` returns, so
it covers query embedding, the FAISS lookup and one Anthropic answer call, and
excludes the judge call. Each run gives 30 per-case values and the reported
figure is their median. Because the answer call is inside the window, the number
moves with how long an answer the model writes, and answer length dropped about
24% in total characters between the baseline and the later runs.

### "Is the baseline reproducible?"

The 1.72s baseline itself was measured once, on 2026-07-20, and has not been
re-run, so I would not claim it reproduces. What I can show is the artifact it
came from, now tracked, and that its median re-derives exactly from the 30
per-case values inside it. What has been repeated is the end state: four runs of
the settled pipeline, which is why the endpoint is given as a median and a range
rather than a single number.

### "You say temperature 0. Does that make the number stable?"

It makes the verdicts stable and not the text. Two runs made back to back with
no code change returned the same judge verdict on all 30 cases, and yet 4 of the
30 generated answers differed in wording, with total answer length moving 1.7%.
Since the answer call sits inside the timed window, that variation is inside the
latency number too, which is exactly why the endpoint is reported as a range.

### "Why 1.08 and not your best run?"

Because the four runs behind it measure a pipeline that was doing identical
work: the two files in the timed path were unchanged, and the one commit between
them left the retrieved chunk text byte for byte the same, which I checked by
hashing. Their medians are 1.018s, 1.145s, 1.128s and 1.025s, so picking 1.018s
would be reporting the low draw of a distribution as a result. The median of the
four is the honest summary, and the range is published beside it so nobody has
to take my word for the spread.

---

## 5. This repository and copilot-mcp-bridge

Two repositories measure the same 30 cases with the same scoring code, and their
numbers must not be quoted as if they came from one place.

**DocMind** measures this application: one document, `company_policy.txt`, 17
chunks, FAISS only, and a latency figure that includes the answer call. Every
figure in sections 1 and 3 above is a DocMind figure.

**copilot-mcp-bridge** measures three vector stores against each other. It
imports `evals/run_eval.py` from here rather than copying it, so the metric
definitions are identical, and it loads a 2566-chunk corpus in which this demo
document is 17 chunks and the rest is public-domain US labour law and the
Federalist Papers.

Where they must not be mixed:

- **Latency.** DocMind's 1.08s is end to end and includes an Anthropic call. The
  bridge's 12.30ms, 21.97ms and 48.54ms are retrieval alone, no model call.
  These are different quantities that happen to share a unit family.
- **Quality figures.** 91.3% correct, 7 of 7 refusals and recall@5 100.0% are
  DocMind numbers on a 17-chunk corpus. The bridge's 73.9% correct, 6 of 7
  refusals and recall@5 87.0% are the same cases and the same scoring code on a
  2566-chunk corpus. Same definition, different difficulty. Neither supersedes
  the other, and quoting one while describing the other's conditions is the
  error to avoid.
- **Answer variation.** DocMind measured 4 of 30 answers differing between two
  runs. The bridge measured 12 of 30 between two of its runs. Both are real and
  each belongs to its own corpus and prompt.
- **Citation grounding.** Both report 100.0% of 115. In DocMind that cannot
  fail, because there is one document. In the bridge the same check still passes
  while 69 of the 115 citations are US labour law, which is the observation that
  showed what the check actually tests.

---

Any claim not in this document is not supported by this repository.
