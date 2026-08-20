# DocMind AI

DocMind AI is a retrieval-augmented document Q&A service: upload a PDF or text file, ask
questions in natural language, and get an answer with the exact source excerpts it was drawn
from. Every answer is constrained to the retrieved passages — when the documents do not
contain the answer, the system says so instead of guessing.

**Live:** https://docmind-ai-production-ae3a.up.railway.app

## Stack

| Layer | Choice |
|---|---|
| API / server | FastAPI + Uvicorn |
| Embeddings | fastembed, `sentence-transformers/all-MiniLM-L6-v2` running locally via ONNX — no embedding API calls, no per-query embedding cost |
| Vector search | FAISS (`faiss-cpu`), in-process index rebuilt from persisted chunks on startup |
| Persistence | SQLite — documents, chunks, and embeddings |
| PDF parsing | PyMuPDF |
| Answer generation | Claude API (`claude-haiku-4-5-20251001`) |
| Deployment | Railway |

Retrieval is entirely local: only answer generation and eval grading call an external API.

## How it works

1. An uploaded document is split into ~500-character chunks by a paragraph- and
   heading-aware chunker. Paragraphs are never split mid-sentence, and a section heading
   always starts a new chunk so the heading stays attached to the text it introduces.
2. Each chunk is embedded locally and indexed in FAISS.
3. A question is embedded the same way; the top `k=5` chunks by cosine similarity are
   retrieved.
4. Only those chunks are sent to Claude. The system prompt forbids answering from general
   knowledge or training data, requires each answered part to cite the excerpt supporting it,
   and requires a fixed refusal sentence for any part the excerpts do not cover.

Chunks are assembled by rejoining paragraphs with the original separators, so every chunk
remains a verbatim contiguous substring of the source document. This is what makes citation
grounding structurally verifiable rather than incidental.

## Evaluation

Behaviour is measured by a harness, not by spot-checking. `evals/run_eval.py` runs a frozen
30-case dataset (`evals/dataset.json`) through the application's real retrieval and answering
code paths — imported, not reimplemented — and reports five metric families:

- **Retrieval recall@k** (k = 1, 3, 5) — is the expected source snippet in the top k chunks
- **Answer correctness** — LLM-as-judge, graded `correct` / `partially_correct` / `incorrect`
  against a reference answer, at temperature 0
- **Citation grounding** — is every cited chunk a verbatim substring of the source document
- **Refusal accuracy** — do the unanswerable cases get the refusal sentence and nothing else
- **Latency** — median and p95 over the full search-and-answer pipeline

The dataset is 30 cases: 15 `factual`, 8 `multi_chunk`, 7 `unanswerable`. A full run costs 53
API calls (30 answer, 23 judge).

### Current measured results

| Metric | Value |
|---|---|
| recall@1 / @3 / @5 (overall) | 87.0% / 95.7% / 100.0% |
| recall@1 (factual) | 100.0% |
| recall@1 / @3 / @5 (multi_chunk) | 62.5% / 87.5% / 100.0% |
| answer correctness (n=23) | 91.3% correct, 8.7% partially correct, 0.0% incorrect |
| answer correctness — factual (n=15) | 100.0% correct |
| answer correctness — multi_chunk (n=8) | 75.0% correct |
| citation grounding | 100.0% (115 of 115 citations verified verbatim) |
| refusal accuracy | 100.0% (7 of 7), zero hallucinations |
| median latency | 1.03s |
| p95 latency | 1.83s |

Citation grounding and refusal accuracy are treated as protected metrics: a change that
lowers either is reverted regardless of what it improves elsewhere. Both have held at 100%
across every run.

The figures above come from the most recent full run
(`evals/results/eval_20260803T022554Z.json`, 2026-08-03, after the chunker fix). Every
correctness, grounding, refusal, and recall figure is identical to the two consecutive
confirmation runs that preceded it (`eval_20260730T122037Z.json` and
`eval_20260730T122154Z.json`, 2026-07-30), which agreed with each other on all 30 case
verdicts. Latency is the one column that moved: median went from 1.15s / 1.13s to 1.03s. It
is wall-clock rather than a verdict, and p95 has never been reproducible run to run (the
2026-07-30 pair measured 4.89s and 2.23s, where run 1 reflects a single slow outlier call).
The judge runs at temperature 0, which makes verdicts reproducible; it does not make the
generated text byte-identical, and the free-text answer and judge rationale still vary
slightly between runs.

For reference, the pre-optimisation baseline (`eval_20260720T124830Z.json`) measured 73.9%
recall@1 and 4.3% incorrect answers. Full history and per-change analysis:
[`evals/RESULTS.md`](evals/RESULTS.md).

### multi_chunk recall@1 is a metric artifact, not a retrieval defect

`multi_chunk` recall@1 of 62.5% looks like the weakest number here. It is a property of how
the metric is defined against a frozen dataset, not a retrieval failure.

`dataset.json` marks exactly one `expected_source_snippet` per case, but a `multi_chunk`
question by construction requires two or more chunks to answer. recall@1 is scored only
against that single marked snippet. So when the *other* equally-required chunk legitimately
ranks first, recall@1 records a miss even though retrieval succeeded and the question is
fully answerable from the retrieved set.

Concretely, for `multi_chunk_05` the rank-1 chunk is the parental-leave passage — a genuinely
required half of the two-part answer — while the marked snippet is the health passage at rank
5. Both are in the top 5. For `multi_chunk_08` the rank-1 chunk is the harassment-policy
passage and the marked snippet is at rank 2.

**recall@5 is the operative number, because the application retrieves `top_k=5`, and it is
100.0% overall.** No retrieval method could raise `multi_chunk` recall@1 to 100% without
editing the frozen dataset, which would invalidate comparison against the baseline.

## Running the app

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then set ANTHROPIC_API_KEY
uvicorn main:app --reload   # http://127.0.0.1:8000
```

The embedding model downloads once on first run and is cached locally. A demo policy
document auto-loads on startup, so the app is queryable immediately.

Endpoints: `POST /upload`, `POST /ask`, `GET /documents`, `DELETE /documents/{doc_id}`,
`GET /health`.

## MCP server (optional)

`mcp_server.py` publishes the same retrieval layer over the Model Context Protocol, so an
MCP client — Claude Code, for instance — can search the indexed documents directly instead
of going through the HTTP API. It is a separate entry point: the FastAPI app is untouched
and none of its endpoints change.

Three tools are exposed, plus a `docmind://documents` resource that summarises the
collection:

| Tool | What it does |
|---|---|
| `search_documents(question, doc_id?, top_k?)` | Semantic search over the FAISS index. Returns excerpts with filename, page, and relevance score so the client can cite what it used. |
| `list_documents()` | Every indexed document with its id and chunk count. |
| `get_document_info(doc_id)` | Metadata for a single document. |

Two capabilities are deliberately withheld, and both omissions are the point rather than
gaps to fill in later.

**No delete tool.** `rag_engine.remove_document()` stays out of the tool surface. Whatever
is on the other end of an MCP connection is a model deciding for itself which tools to
invoke, and dropping a document from the index cannot be undone. A destructive action with
no recovery path needs a person to approve it, so it stays in the HTTP API where a human is
driving.

**No answering tool.** `claude_qa.answer_question()` is not wrapped either. In the HTTP app
the server has to call a model, because the caller is a browser. Over MCP the caller
already is a model, so the server's useful contribution ends once it has handed back the
right passages and their sources — anything further would be a second model paraphrasing
the first one's context, at twice the cost.

Setup, client registration, and the transport constraints that shaped the implementation
are in [`README_MCP.md`](README_MCP.md).

## Running the evaluation

```bash
.venv/bin/python evals/run_eval.py                      # full 30-case run, with judge
.venv/bin/python evals/run_eval.py --limit 5            # first 5 cases
.venv/bin/python evals/run_eval.py --category multi_chunk   # one category only
.venv/bin/python evals/run_eval.py --no-judge           # retrieval + refusal only
```

Each run writes a timestamped JSON and Markdown summary to `evals/results/`. `--no-judge`
skips grading calls; `--category` scopes a run to one category so retrieval experiments do
not pay for all 30 cases.

For retrieval work with no API cost at all:

```bash
.venv/bin/python evals/retrieval_probe.py               # recall@k, zero API calls
```

`evals/dataset.json` is read-only by convention. The harness only selects subsets of it; it
never rewrites the file, so every number above stays comparable to the baseline.

See [`evals/README.md`](evals/README.md) for metric definitions and how to interpret output.

## Testing

The eval harness measures model and retrieval behaviour. The test suite is a separate
concern: it covers the application code that the eval numbers depend on.

```bash
pip install -r requirements-dev.txt
.venv/bin/python -m pytest                    # whole suite, ~0.6s
.venv/bin/python -m pytest tests/test_chunking.py -v
```

The suite runs fully offline. No test calls the Anthropic API, no test opens the real
`docmind.db` (every database fixture uses a temporary file), and the search tests use a
deterministic bag-of-words embedder instead of downloading model weights — so retrieval
assertions test the retrieval code rather than the semantics of MiniLM. CI on push and pull
request runs it with no secrets configured, which is what keeps that property honest.

Current coverage:

- `tests/test_chunking.py` — the grounding invariant asserted directly: for every chunk
  there is an offset `i` into its source page where `page[i:i+len(chunk)] == chunk`. Also
  paragraph packing, heading boundaries, disjointness, size limits, and edge cases.
- `tests/test_document_store.py` — schema init, chunk and embedding round-trip, cascade
  delete with an explicit no-orphans check.
- `tests/test_search.py` — index rebuild from persisted chunks, index size, `top_k`,
  `doc_id` filtering, ranking, removal, and empty-index behaviour.

### A defect the suite found, and the fix

Writing the invariant test surfaced a real bug. `_split_paragraphs` used to strip each
paragraph and rejoin them with a literal `"\n\n"`, which reproduces the source only when
the source already separates paragraphs with exactly that. Any other form — a blank line
containing spaces or tabs, a doubled blank line, an indented or trailing-space paragraph —
produced a chunk that was **not** a verbatim substring of its source, silently breaking the
guarantee citation grounding rests on. The demo corpus never triggers it, which is why the
published 115/115 figure was never wrong; an arbitrary uploaded PDF could.

The fix tracks paragraph *offsets* into the page and emits each chunk as a single slice
`page[first_start:last_end]`, so a chunk is a contiguous substring by construction rather
than by reconstruction. Chunk boundaries and the packing budget are unchanged: on
`demo_docs/company_policy.txt` the fixed chunker produces byte-identical output to the old
one (17 chunks, same SHA-256), and a full 30-case eval re-run after the fix reproduced every
metric exactly, both protected metrics included — citation grounding 100.0% (115/115) and
refusal accuracy 100.0% (7/7). See `evals/RESULTS.md`, Session 4.

`test_invariant_holds_for_every_separator_form` covers nine separator forms and is the
regression test for this.

Still to come:

- API contract tests for each endpoint via `fastapi.testclient`, with the Claude call stubbed
- A scheduled CI job running the API-free `retrieval_probe.py`, which needs the embedding
  model and so does not belong in the hermetic per-push run
