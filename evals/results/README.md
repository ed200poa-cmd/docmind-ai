# Tracked eval runs

Most files in this directory are ignored. Five are not.

These five are the raw material behind every latency, recall, correctness,
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

The four runs after the baseline measure a timed path that does identical work.
Their medians span 1.018s to 1.145s, which is the range the retraction in
`../RESULTS.md` rests on.

## SHA-256

```
9f82bb8a1003212788b8204cc1c5191bb9b26a5e7cb8f218e2983def14407351  eval_20260720T124830Z.json
91333f486986f0b43824ab0e76856a601aa1cd5829c66b9e94c89210b33d1de1  eval_20260730T014202Z.json
4734e0110d276e9f0cbe4e789df2f9bcba166d2473204660f24ae66bcf7330e0  eval_20260730T122037Z.json
c0072b4611ded8ece79f48a2fd1f2e158f4e2f826158b08f38523ff829693aff  eval_20260730T122154Z.json
aa37e6ef5bc5d6bc1c70019c01a3baa7da723f8910f35fee343706553a95c36b  eval_20260803T022554Z.json
```

Regenerate with `shasum -a 256 eval_*.json` from this directory.

## What was checked before adding them

Eval artifacts hold full model output, so they were inspected rather than
assumed safe. Each file was scanned for Anthropic, OpenAI and AWS key formats,
bearer tokens, authorization headers, generic secret assignments, email
addresses, absolute home paths, the local username, IP addresses, and URLs
carrying credentials. No pattern matched in any of the five. `gitleaks` was run
over all five as a second opinion and reported no leaks.

Provenance of the text inside them was checked too, because these files embed
the retrieved passages verbatim. Every `chunk_text` in all five files is a
substring of `demo_docs/company_policy.txt`, this repository's own fictional
demo handbook, which is already tracked here. Every `question` comes from
`evals/dataset.json`, also tracked. `dataset_path` is recorded as a relative
path. There is no third-party document and no customer material in these runs.

## What they do not contain

No API key, because `run_eval.py` reads it from the environment and records only
the model names and temperatures in `config`. No request or response headers.
No timing beyond the per-case wall clock. The judge's free-text `reason` strings
are included, and they are the judge's own words about the demo document.

## Adding more

The ignore rule is `evals/results/*` with one negation per tracked file, so a
new run stays untracked unless it is named explicitly. Do the same inspection
first, then add the negation and its SHA-256 here.
