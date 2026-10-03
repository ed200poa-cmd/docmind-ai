# Tracked eval runs

Most files in this directory are ignored. Seven are not.

These seven are the raw material behind every latency, recall, correctness,
grounding and refusal figure published in `../RESULTS.md` and in the repository
README. They were untracked until 2026-08-28, which meant the prose could be
read but the numbers behind it could not be re-derived by anyone else. Checking
the latency claim in August required opening these files, and that check found
an error, so the files are now part of the repository rather than part of one
laptop.

## What is here

| file | run | why it is kept |
|---|---|---|
| `eval_20260720T124830Z.json` | 2026-07-20 baseline | the 1.72s median and the pre-optimisation recall and correctness figures |
| `eval_20260730T014202Z.json` | 2026-07-30 | median 1.018s, the run that shows 1.03s was not a new floor |
| `eval_20260730T122037Z.json` | 2026-07-30 confirmation, run 1 | median 1.145s, and half of the verdict-level reproducibility check |
| `eval_20260730T122154Z.json` | 2026-07-30 confirmation, run 2 | median 1.128s, the other half |
| `eval_20260803T022554Z.json` | 2026-08-03, after the chunker fix | median 1.025s, and the source of the published recall, correctness, grounding and refusal figures |
| `eval_20261003T142527Z.json` | 2026-10-03 re-measurement, run 1 | 95.7% correct against the published 91.3%, and half of the evidence that answer correctness is a range |
| `eval_20261003T142910Z.json` | 2026-10-03 re-measurement, run 2 | the other half; same 95.7%, same single `partially_correct` case |

The four runs after the baseline measure a timed path that does identical work.
Their medians span 1.018s to 1.145s, which is the range the retraction in
`../RESULTS.md` rests on.

The two 2026-10-03 runs were produced by an instrumented harness and are the
only files here that carry `token_usage` and `cost` blocks; the other five
record `api_calls` counts and no token or dollar figures, so the two sets are
not the same shape. Those two runs also re-measured the baseline without any
code change to retrieval or answering: retrieval was identical case for case
across all three of `eval_20260803T022554Z`, `eval_20261003T142527Z` and
`eval_20261003T142910Z`, while `answer correct` read 91.3%, 95.7% and 95.7%.
`docs/reports/hillclimb-2026-10-03.md` carries that comparison.

## SHA-256

```
9f82bb8a1003212788b8204cc1c5191bb9b26a5e7cb8f218e2983def14407351  eval_20260720T124830Z.json
91333f486986f0b43824ab0e76856a601aa1cd5829c66b9e94c89210b33d1de1  eval_20260730T014202Z.json
4734e0110d276e9f0cbe4e789df2f9bcba166d2473204660f24ae66bcf7330e0  eval_20260730T122037Z.json
c0072b4611ded8ece79f48a2fd1f2e158f4e2f826158b08f38523ff829693aff  eval_20260730T122154Z.json
aa37e6ef5bc5d6bc1c70019c01a3baa7da723f8910f35fee343706553a95c36b  eval_20260803T022554Z.json
84c09b6570951d46eef4669ee8be5eca8c429774305f21481be14a431d3fd4ef  eval_20261003T142527Z.json
3d51024fb8e111edb01710b010b96979f64c5e1b2058af1729e701a01585f29c  eval_20261003T142910Z.json
```

Regenerate with `shasum -a 256 eval_*.json` from this directory.

## What was checked before adding them

Eval artifacts hold full model output, so they were inspected rather than
assumed safe. Each file was scanned for Anthropic, OpenAI and AWS key formats,
bearer tokens, authorization headers, generic secret assignments, email
addresses, absolute home paths, the local username, IP addresses, and URLs
carrying credentials. No pattern matched in any of the seven. `gitleaks` was run
over all seven as a second opinion and reported no leaks — over the first five
in 2026-08 (commit `c0e3d26`), and over the two 2026-10-03 runs with
`gitleaks 8.30.1` and this repository's `.gitleaks.toml`, which scanned
136,048 and 135,875 bytes, the full size of each file, and reported
`no leaks found` for both. One false positive came up on the 2026-10-03 pair
and is worth naming so the next scan is not re-run in alarm: a case-insensitive
search for the local username matches the word "computer" twelve times per
file, every occurrence inside the tracked demo document's sentence about a
laptop computer. The username itself, matched case-sensitively, appears zero
times.

Provenance of the text inside them was checked too, because these files embed
the retrieved passages verbatim. Every `chunk_text` in all seven files is a
substring of `demo_docs/company_policy.txt`, this repository's own fictional
demo handbook, which is already tracked here. Every `question` comes from
`evals/dataset.json`, also tracked. `dataset_path` is recorded as a relative
path. There is no third-party document and no customer material in these runs.
Counted for the 2026-10-03 pair: 150 `chunk_text` values per file (17 distinct),
30 `question`, 30 `expected_answer` and 30 `expected_source_snippet` values per
file, with zero outside their tracked source.

## What they do not contain

No API key, because `run_eval.py` reads it from the environment and records only
the model names and temperatures in `config`. No request or response headers.
No timing beyond the per-case wall clock. The judge's free-text `reason` strings
are included, and they are the judge's own words about the demo document.

## Adding more

The ignore rule is `evals/results/*` with one negation per tracked file, so a
new run stays untracked unless it is named explicitly. Do the same inspection
first, then add the negation and its SHA-256 here.
