# RAG Improvement Results

Three changes, applied one at a time, each measured against the same unmodified
30-case dataset (`evals/dataset.json`) and the same metric definitions
(`evals/run_eval.py`) used to produce the original baseline. No test case and no
metric formula changed at any point in this process — every number below is
directly comparable.

Result files referenced below, in order: `eval_20260720T124830Z` (baseline) →
`eval_20260720T132125Z` (after chunking) → `eval_20260720T132538Z` (after refusal
prompt) → `eval_20260720T132826Z` (after conciseness prompt, current).

## Before / after, every metric

| Metric | Baseline | After Step 2 (chunking) | After Step 3 (refusal) | After Step 4 (conciseness) | Δ total |
|---|---|---|---|---|---|
| recall@1 (overall) | 73.9% | 87.0% | 87.0% | 87.0% | **+13.1** |
| recall@3 (overall) | 91.3% | 95.7% | 95.7% | 95.7% | **+4.4** |
| recall@5 (overall) | 95.7% | 100.0% | 100.0% | 100.0% | **+4.3** |
| recall@1 (multi_chunk) | 37.5% | 62.5% | 62.5% | 62.5% | **+25.0** |
| recall@3 (multi_chunk) | 75.0% | 87.5% | 87.5% | 87.5% | **+12.5** |
| recall@5 (multi_chunk) | 87.5% | 100.0% | 100.0% | 100.0% | **+12.5** |
| answer correctness — correct | 87.0% | 91.3% | 95.7% | 95.7% | **+8.7** |
| answer correctness — partially_correct | 8.7% | 8.7% | 4.3% | 4.3% | -4.4 |
| answer correctness — incorrect | 4.3% | 0.0% | 0.0% | 0.0% | **-4.3** |
| citation grounding (protected) | 100.0% | 100.0% | 100.0% | 100.0% | 0 (held) |
| refusal accuracy (protected) | 100.0% | 100.0% | 100.0% | 100.0% | 0 (held) |
| median latency | 1.72s | 1.60s | 1.60s | 1.42s | **-0.30s** |
| p95 latency | 2.58s | 3.42s | 2.52s | 2.34s | -0.24s |
| API calls per full run | 53 | 53 | 53 | 53 | 0 |

Both protected metrics (citation grounding, refusal accuracy) stayed at 100% through
every step — nothing was reverted, because nothing broke them.

## Change 1: paragraph + heading-aware chunking

**File:** `rag_engine.py` (`_chunk_pages`). **Commit:** `18ecd33`.

**Diagnosis first** (see the conversation / commit message for the full evidence):
for `multi_chunk_05`, the target chunk ranked 6th of 13 by cosine similarity, one
place outside `top_k=5`, at only a 0.02 score gap from the cutoff. Reading the raw
chunk text showed why: the old fixed 500-char/50-overlap slicer cut straight through
section-heading blocks, so `"SECTION 3: HEALTH AND WELLNESS BENEFITS"` ended up
appended to the *previous* chunk (about vacation carryover — unrelated) while the
chunk holding the actual 30-day eligibility fact had no heading at all. The single
strongest semantic signal for "this is about health benefits" was attached to the
wrong content. This was diagnosed as a chunking defect, not an embedding-model or
top_k problem (the embedding still ranked the orphaned chunk above 7 of 13 others).

**Fix:** pack paragraphs (split on blank lines) into ~500-char chunks without ever
splitting a paragraph mid-sentence, and always start a new chunk when a section
heading is encountered, so the heading and the paragraph(s) immediately following it
land in the same chunk. Chunks are still built by rejoining paragraphs with the exact
same `"\n\n"` that originally separated them, so every chunk remains a verbatim,
contiguous substring of the source document — this is why citation grounding did not
just "happen to stay" at 100%, it's structurally guaranteed by how the chunker works.

**Cases fixed:** `multi_chunk_05` (recall@5 False → True, judge incorrect → correct).
`factual_03`, `multi_chunk_03`, `multi_chunk_07` moved from a recall@1 miss to a
recall@1 hit (previously only recovered at k=3/5). Chunk count went 13 → 17 (sections
with 2+ paragraphs now split more granularly instead of via arbitrary 500-char cuts).

**Why decomposition/top_k weren't also tried:** the task's stated preference was
chunking and decomposition over raising top_k. Chunking alone took recall@5 — the
value the app actually operates at — to 100%. There was no remaining recall problem
to justify the extra API calls, latency, and complexity of a query-decomposition step,
so it was not built. If a future, larger corpus reintroduces recall@5 misses,
decomposition is the next lever to reach for before top_k.

## Change 2: stop all-or-nothing refusal

**File:** `claude_qa.py` (`SYSTEM_PROMPT`). **Commit:** `17710ff`.

**Problem:** `multi_chunk_05` and `multi_chunk_07` both showed the model quoting a
correct, retrieved fact and then still opening with the blanket refusal sentence for
the *entire* question, because the old prompt only had one refusal rule with no
concept of a partially-answerable question — turning a retrieval success into an
answer failure.

**Fix:** rewrote the rule set to handle multi-part questions part by part: answer
each part the excerpts support (citing the excerpt; applying a stated policy to the
specific situation asked about even if the wording differs, without adding unstated
facts), use the exact refusal sentence only for the specific part that isn't covered,
and keep the original whole-question refusal when nothing retrieved is relevant at
all — which is exactly the behavior the 7 unanswerable cases still need.

**Cases fixed:** `multi_chunk_05` was already fixed by the chunking change by this
point; this change's independent contribution is visible in `multi_chunk_04`'s answer
becoming cleaner and, most directly, in `multi_chunk_07`'s answer no longer
self-contradicting — it now says "not eligible" for the merit-increase part (correct,
cited) and narrows the refusal to specifically the laptop-replacement part instead of
refusing the whole question.

**Verification that refusal accuracy didn't regress:** all 7 `unanswerable_*` cases
were re-checked individually post-change — every one still returns the exact refusal
sentence and nothing else. No hallucinations were introduced.

## Change 3: stop volunteering unrequested information

**File:** `claude_qa.py` (`SYSTEM_PROMPT`). **Commit:** `fb04ca7`.

**Problem:** `factual_01` ("What are ACME's standard working hours?") correctly
answered 9AM–5PM, then appended an unrelated sentence about flexible work
arrangements that nobody asked about, pulled from the same excerpt. The judge is
lenient about this (it marked the answer "correct" by the time this change was made,
since factual correctness had already reached 100% via the chunking fix), but the
underlying behavior — padding a correct answer with unrequested adjacent facts — is
exactly what the baseline eval flagged and is worth fixing regardless of whether it
moves the correctness score.

**Fix:** added an explicit rule to answer only what was asked and not surface other
facts/caveats from the excerpt just because they're nearby.

**Effect:** correctness percentages were already at their ceiling for `factual`
(100% since the chunking change), so this change shows up in answer length rather
than score: average generated-answer length dropped 31.0 → 28.6 words for `factual`
cases and 89.1 → 77.9 words for `multi_chunk` cases. `factual_01`'s answer no longer
includes the unrelated flexible-work-arrangements sentence. Median latency also
improved (1.60s → 1.42s), plausibly because shorter answers mean fewer output tokens.

## What is still failing

- **`multi_chunk_07` (partially_correct).** The model answers the merit-increase part
  correctly and cites it, and gives a narrowly-scoped refusal specifically for the
  laptop-replacement part: the document describes requesting *additional* equipment
  (standing desks, monitors) through the IT portal, but never explicitly says that
  process also covers *replacing broken* equipment. The model's caution here is a
  defensible reading of what's actually grounded in the text, not a repeat of the
  original bug (compare to before Change 2, where it refused the *entire* question
  including the part it could answer). Pushing the prompt to bridge this specific gap
  risks reintroducing exactly the over-answering failure mode Change 2 fixed, so it
  was left as-is. Note that this also means the eval dataset's own `expected_answer`
  for `multi_chunk_07` assumes a slightly more liberal reading than the source text
  literally supports — worth knowing if this case is revisited, though the dataset
  itself was not touched per the task's constraint.
- **recall@1 is not 100%** for `multi_chunk_01`, `multi_chunk_05`, and
  `multi_chunk_08` (62.5% overall on `multi_chunk`) — but the application always
  retrieves `top_k=5`, and recall@5 is 100%, so this has no effect on live answer
  quality today. It would only matter if `TOP_K` were ever lowered.
- **p95 latency (2.34s) is still noisy** run to run — it moved 2.58s → 3.42s → 2.52s
  → 2.34s across the four runs while median stayed flat or improved, consistent with
  ordinary per-call API variance (a slow outlier call landing on a different case each
  run) rather than anything caused by the changes. Worth re-checking on a larger run
  if p95 matters operationally, since n=30 is a small sample for a tail statistic.

## Cost / latency impact

- **API calls per full eval run:** unchanged at 53 (30 answer calls + 23 judge calls)
  through every step — none of the three changes added or removed a retrieval call, an
  answer call, or a judge call; only the chunking algorithm and the system prompt text
  changed.
- **Indexed chunk count:** 13 → 17 after the chunking change (more, smaller
  section-aligned chunks instead of arbitrary 500-char slices). This does not change
  `top_k` or how many chunks are sent to Claude per query (still 5), so it has no
  effect on per-query answer-call cost.
- **Latency net effect is a small improvement**, not a regression: median 1.72s →
  1.42s (-17%) end to end. p95 fluctuated but did not trend upward.
- **Total cost of running this optimization exercise itself:** 4 full eval runs ×
  53 API calls = 212 Claude API calls (embedding is local/CPU via fastembed, no API
  cost), plus one local-only diagnostic script (pure numpy/FAISS scoring, zero API
  calls) for the Step 1 diagnosis.

## Session 3 (2026-07-29): API-free subset tooling + retrieval re-verification

**Constraint this session:** no `ANTHROPIC_API_KEY` was available (no `.env`, no env
var), so the full harness (`run_eval.py`) — which makes an answer call per case plus a
judge call — could not be run. That means the answer-side metrics (correctness,
citation grounding on *cited* sources, refusal accuracy) were **not** re-measured this
session; they carry forward unchanged from the last full run with the key,
`eval_20260720T132826Z` (the "After Step 4" column above), because no answer-path or
retrieval-path code was changed. Retrieval recall@k is computed purely from local
fastembed + FAISS, so it *was* re-verified, API-free.

### What was added (tooling the task asked for, no dataset change)

1. **`--category` on `run_eval.py`.** Previously only `--limit N` existed, which takes
   the *first* N cases, so there was no way to run "multi_chunk only." Added
   `--category multi_chunk` (a read-only subset selector; `dataset.json` is never
   rewritten — verified with `git status`), so repeated experiments can target the one
   weak category instead of paying for all 30 cases. An unknown category name errors
   out listing the valid ones, rather than silently running everything.
2. **`evals/retrieval_probe.py` — an API-free retrieval measurement path.** Reuses the
   app's real `rag_engine.search(...)` to compute recall@1/@3/@5 and a
   verbatim-substring check on retrieved chunks, making **zero** Claude API calls. This
   is the "retrieval-only, no-LLM" path the task asked for so search-side experiments
   can iterate for free.

### Retrieval re-verified against the shipped Session-1 chunking fix

`python evals/retrieval_probe.py` reproduces the Session-1 retrieval numbers exactly,
independently confirming the paragraph+heading-aware chunker is what's live (index =
17 chunks):

| | k=1 | k=3 | k=5 |
|---|---|---|---|
| overall | 87.0% | 95.7% | 100.0% |
| factual | 100.0% | 100.0% | 100.0% |
| multi_chunk | 62.5% | 87.5% | 100.0% |

Retrieved-chunk verbatim-substring rate: **100.0% (115/115)** — every retrieved chunk
is an exact substring of `company_policy.txt`, the structural reason citation grounding
holds. `factual` recall@1 is 100.0% (was 93.3% at baseline) — the guardrail that it
must not drop is satisfied. Result files:
`results/retrieval_probe_*.json` (full + `--category multi_chunk`).

### Retrieval NOT changed this session — deliberate, with evidence

No new retrieval change was shipped, and this is the correct call, not an omission:

- **`multi_chunk recall@1` (62.5%) is a fixed-dataset measurement artifact, not a
  retrieval defect** — re-confirmed independently this session by inspecting which
  chunk ranks #1 for each miss. `multi_chunk_05`: rank #1 is the *parental-leave* chunk
  (a genuinely required half of the two-part answer); the single `expected_source_snippet`
  the metric scores is the *health* chunk, at rank 5 — both are in the top-5, so the
  question is fully answerable, but recall@1 checks only the one marked snippet.
  `multi_chunk_08`: rank #1 is the *harassment-policy* chunk (a required half); the
  marked snippet is at rank 2. No retrieval method can push these to recall@1 = 100%
  without editing the frozen `dataset.json`, which is forbidden.
- **recall@3 and recall@5 are already at ceiling** (95.7% / 100.0% overall; the app
  operates at `top_k=5`), so there is no live retrieval gap for a reranker / hybrid
  search / query rewrite to close.
- **Any change that reshuffles the top-5 set risks a guarded metric I cannot verify.**
  Refusal accuracy (hard 100%) depends on what the 7 unanswerable cases retrieve, and
  re-checking it needs an answer call, i.e. the API key that isn't available. Under the
  task's rule "reject any change that lowers grounding or refusal, however large the
  gain," an unverifiable-refusal change cannot be adopted. So the safe choice is to
  ship no retrieval change and record why.

### Frontend (mobile) fix this session

`static/index.html`: the responsive breakpoint was `@media (max-width: 768px)`, which
renders **exactly 768px** (a common tablet width) as *mobile*, contradicting the spec's
"≥768px = desktop." Changed to `max-width: 767.98px` so <768px stacks and ≥768px keeps
the two-column desktop layout. Verified with Playwright screenshots at 390/767/768/1440
(saved to `/tmp/layout-check/`): 390 & 767 stack chat-first with the upload/documents
panel below and zero horizontal overflow; 768 & 1440 render the two-column desktop
layout; the 1440 screenshot is byte-identical before and after the change, so desktop
is provably untouched.

### API calls spent this session: 0 (all measurement local).
