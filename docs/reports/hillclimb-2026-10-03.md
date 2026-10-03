# Hillclimb attempt — 2026-10-03

Branch: `hillclimb-2026-10-03`
HEAD at time of writing: `a655cae16c58e22239b0e0a9a811905758c05deb`
("Merge pull request #1 from ed200poa-cmd/add-mit-license", Sat Sep 26 14:27:33 2026 -0400)
Not merged. Not committed.

**Outcome: hillclimb was not run.** Task 1 ended at step 1-2 (baseline
re-measurement). The reason is not the cost ceiling — see
[Conclusion](#conclusion).

---

## 1-0 Start conditions

| # | Condition | Result |
|---|---|---|
| a | `claude --version` | Pass (two different binaries, see below) |
| b | `/claude-api build-eval`, `/claude-api hillclimb` exist | Pass |
| c | DocMind branch, last commit | Pass |
| d | Golden dataset, 30 cases | Pass |
| e | API key configured | Pass (second attempt) |

### a. Versions — two different binaries

The agent session's shell and the operator's Mac do not run the same
`claude`. Both figures are recorded because the distinction matters for
Task 2.

Session sandbox:

```
$ which claude
/usr/local/bin/claude
$ claude --version
2.1.286 (Claude Code)
$ ls -l --time-style=long-iso /usr/local/bin/claude
-rwxr-xr-x 1 nobody nogroup 241667256 2026-10-01 17:00 /usr/local/bin/claude
```

Operator's Mac (operator's terminal output):

```
Current version: 2.1.267
Updating to 2.1.288...
Successfully updated from 2.1.267 to version 2.1.288
2.1.288 (Claude Code)
```

The sandbox binary is baked into the image (2026-10-01) and is unaffected
by `claude update` on the Mac.

### b. `build-eval` and `hillclimb`

Not listed in `claude --help` (they are in-session commands, not CLI
subcommands). Present in the binary's subcommand list:

```
$ strings /usr/local/bin/claude | grep -oE '.{0,90}build-eval.{0,90}'
u=["cost-optimize","migrate","managed-agents-onboard","prompt-audit",
"upgrade","build-eval","hillclimb","preserved-thinking-migration"]
```

Bundled documentation resources: `shared/evals/build-eval.md`,
`shared/evals/cost-hillclimb.md`, `shared/evals/eval-hillclimb.md`.
Documented flow layout: `.claude/hillclimb/<name>`.

### c. Branch and last commit

```
$ git branch --show-current
hillclimb-2026-10-03
$ git log -1
commit a655cae16c58e22239b0e0a9a811905758c05deb
Merge: 2d2b1e7 34f373d
Author: Destarlink LLC <254977870+ed200poa-cmd@users.noreply.github.com>
Date:   Sat Sep 26 14:27:33 2026 -0400

    Merge pull request #1 from ed200poa-cmd/add-mit-license

    Add MIT license
```

The branch was created by the operator in the terminal, not by the agent
(`.git` writes are blocked from the sandbox mount — see
[Environment limits](#environment-limits)).

### d. Golden dataset

Path `evals/dataset.json` (10,292 bytes, 2026-07-20 08:38).

```
top-level type: list
len(d) = 30
first item keys: ['id','question','expected_answer',
                  'expected_source_snippet','category']
Counter({'factual': 15, 'multi_chunk': 8, 'unanswerable': 7})
```

Not one case was added or modified. Harness: `evals/run_eval.py`,
`evals/retrieval_probe.py`.

### e. API key

First check failed and was reported rather than worked around:

```
$ ls -l --time-style=full-iso .env
-rw------- 1 ... 127 2026-07-29 21:37:48.000000000 -0400 .env
```

The backup `.env.before-hillclimb` had been created at 09:25 that day but
`.env` itself was untouched since July, so the dedicated key was not yet
in place. The run was not started on the old key. After the operator
replaced it:

```
125  2026-10-03 09:39:40.259279288  .env
```

The key value was never printed in any output. `.env.before-hillclimb`
was never opened, and no hash comparison was made against it (that would
require reading it).

`.env` is ignored and untracked:

```
$ git check-ignore -v .env
.gitignore:2:.env	.env
$ git ls-files | grep -i 'env'
.env.example
```

How the harness reads it:

```
evals/run_eval.py:43  from dotenv import load_dotenv
evals/run_eval.py:44  load_dotenv(APP_ROOT / ".env")
evals/run_eval.py:136 if not os.getenv("ANTHROPIC_API_KEY"):   → exit(1)
```

---

## Instrumentation added

The harness recorded API call *counts* but no tokens and no dollars, so
the per-question cost that step 1-2 requires could not be produced. This
was reported first and added only on the operator's instruction.

### Files changed (2)

`claude_qa.py`

- `_usage_dict()` reads `response.usage`: `input_tokens`,
  `output_tokens`, `cache_creation_input_tokens`,
  `cache_read_input_tokens`.
- `"usage"` key added to the returned dict (`None` on paths that make no
  call).
- Prompt, model, `max_tokens` and `temperature` untouched.

`evals/run_eval.py`

- `PRICING_USD_PER_MTOK` constant, with source URL and verification date
  in the comment.
- `_empty_usage`, `_add_usage`, `_usage_from_response`, `cost_usd`,
  `build_cost_report`, `print_cost`.
- `judge_answer` also returns usage. The judge prompt and the grading
  rules are unchanged.
- Answer and judge accumulated separately; cache fields accumulated
  separately.
- `token_usage` and `cost` written to both stdout and the results JSON;
  per-case `usage` recorded too.

Not touched: `evals/dataset.json`, the grading logic, the judge model,
any prompt.

### Price source

| Claude Haiku 4.5 | USD / MTok |
|---|---|
| input | $1 |
| output | $5 |
| cache write 5m | $1.25 |
| cache write 1h | $2 |
| cache read | $0.10 |

Source row, verbatim:

```
haiku-4-5/overview)The fastest model with near-frontier intelligence
| $1 / MTok | $5 / MTok | $1.25 / MTok | $2 / MTok | $0.10 / MTok |
```

URL: <https://platform.claude.com/docs/en/about-claude/pricing>
(requested as `docs.claude.com/en/docs/about-claude/pricing`, which
redirects there). Verified 2026-10-03. Both the answer model and the
judge model are `claude-haiku-4-5-20251001`, so one rate card applies.

Known limit recorded in the code: the API returns a single
`cache_creation_input_tokens` figure and does not say whether a write used
the 5-minute or the 1-hour TTL, which are priced differently. The harness
sends no `cache_control`, so these fields are expected to be 0; if they
are ever non-zero, writes are costed at the 5-minute rate and the summary
says so.

### Guard verification

Arithmetic checks against known inputs, no API calls:

```
[1] 1M in + 1M out -> 6.0 (expected 6.0) status= priced
[2] 1M cache_w + 1M cache_r -> 1.35 (expected 1.35)
[3] unpriced model -> {'usd': None, 'status': 'unverified-price', ...}
[4] accumulator -> {'input_tokens': 15, 'output_tokens': 5, ...}
[6] usd_run_total= 0.06  per_question= 0.002
    expected: 39000/1e6*1 + 4200/1e6*5 = 0.06
```

Then violations were planted to confirm the guard fires — an empty
measurement must not read as a cheap run:

```
[A] all 53 calls report no usage ->
    status= no-usage-reported  usd_run_total= None
    RUN TOTAL: unknown (no call reported a usage object) -- not $0.00
    calls that returned no usage object: answer=30, judge=23
[B] 5 calls missing ->
    status= incomplete-usage
    RUN TOTAL: $0.039000 ... [floor, some calls reported no usage]
[C] none missing -> status= priced
```

---

## 110-test gate

CI command, verbatim (`.github/workflows/tests.yml` lines 31-35):

```
      # No secrets are configured for this job on purpose. The suite must run
      # offline: no Anthropic API calls, and no embedding-model download (search
      # tests use the deterministic stub embedder in tests/conftest.py).
      - name: Run tests
        run: python -m pytest
```

No marker filter, so CI's number is the full collection:

```
$ python -m pytest --collect-only -o addopts='--strict-markers'
rootdir: /…/docmind
configfile: pytest.ini
collected 110 items
========================= 110 tests collected in 0.64s =========================
```

110 and 96 are both real but mean different things:

| Selection | Count | Is it CI's number? |
|---|---|---|
| `python -m pytest` (no filter) | 110 | Yes |
| `-m "not docs"` (pipeline suite) | 96 | No |
| `-m "docs"` (document checks) | 14 | No |

```
$ python -m pytest --collect-only -m "not docs"
collected 110 items / 14 deselected / 96 selected
$ python -m pytest --collect-only -m "docs"
collected 110 items / 96 deselected / 14 selected
```

After the instrumentation change:

```
$ python -m pytest
........................................................................ [ 65%]
.....................................s                                   [100%]
109 passed, 1 skipped, 1 warning in 3.81s
[exit=0]
```

109 + 1 = 110. The one skip is pre-existing and unrelated:

```
$ python -m pytest -rs
SKIPPED [1] tests/test_search.py:208: set DOCMIND_TEST_REAL_MODEL=1 to
exercise the real fastembed model; it downloads weights on first run,
so CI stays on the stub embedder
```

The repository's own self-check on published counts ran and passed — it
did not skip:

```
$ python -m pytest -v | grep published
tests/test_claims_doc.py::test_the_published_test_counts_match_the_collected_suite PASSED [ 38%]
```

---

## 1-2 Baseline re-measurement

Two runs on the Mac, `.venv/bin/python` (Python 3.13.12). The agent
session made no API calls.

| | file | mtime |
|---|---|---|
| AUG | `evals/results/eval_20260803T022554Z.json` | 2026-08-02 22:25 |
| R1 | `evals/results/eval_20261003T142527Z.json` | 2026-10-03 10:25 |
| R2 | `evals/results/eval_20261003T142910Z.json` | 2026-10-03 10:29 |

All three: `num_cases=30`, `api_calls={answer 30, judge 23}`,
`answer_model = judge_model = claude-haiku-4-5-20251001`, `limit=None`,
`category=None`.

"Published" below is the figure set given in the task instruction.

| Metric | Published | AUG | R1 | R2 | Verdict |
|---|---|---|---|---|---|
| recall@1 overall | 87.0 | 87.0 | 87.0 | 87.0 | same |
| recall@3 overall | (none) | 95.7 | 95.7 | 95.7 | no baseline |
| recall@5 overall | 100 | 100.0 | 100.0 | 100.0 | same |
| recall@1 factual | (none) | 100.0 | 100.0 | 100.0 | no baseline |
| recall@3 factual | (none) | 100.0 | 100.0 | 100.0 | no baseline |
| recall@5 factual | (none) | 100.0 | 100.0 | 100.0 | no baseline |
| recall@1 multi_chunk | (none) | 62.5 | 62.5 | 62.5 | no baseline |
| recall@3 multi_chunk | (none) | 87.5 | 87.5 | 87.5 | no baseline |
| recall@5 multi_chunk | (none) | 100.0 | 100.0 | 100.0 | no baseline |
| n_judged | (none) | 23 | 23 | 23 | no baseline |
| **answer correct %** | **91.3** | **91.3** | **95.7** | **95.7** | **differs** |
| partially_correct % | (none) | 8.7 | 4.3 | 4.3 | no baseline |
| incorrect % | 0 | 0.0 | 0.0 | 0.0 | same |
| factual correct % | (none) | 100.0 | 100.0 | 100.0 | no baseline |
| multi_chunk correct % | (none) | 75.0 | 87.5 | 87.5 | no baseline |
| citation grounding % | 100 | 100.0 | 100.0 | 100.0 | same |
| citations checked | 115 | 115 | 115 | 115 | same |
| refusal accuracy % | 100 | 100.0 | 100.0 | 100.0 | same |
| n_unanswerable | 7 | 7 | 7 | 7 | same |
| hallucinated ids | (none) | [] | [] | [] | no baseline |
| median latency s | 1.08 | 1.025 | 0.932 | 0.988 | same (within ±0.15) |
| p95 latency s | (none) | 1.831 | 1.692 | 1.913 | no baseline |
| min latency s | (none) | 0.493 | 0.566 | 0.517 | no baseline |
| max latency s | (none) | 2.209 | 6.917 | 2.088 | no baseline |

Latency rule: published 1.08 ±0.15 gives 0.93–1.23. AUG 1.025, R1 0.932,
R2 0.988 all fall inside, so latency counts as unchanged. R1's max of
6.917s is a single outlier; median and p95 did not move, so it does not
affect any reported metric. Its cause was not investigated.

**Stop condition met.** `answer correct` 91.3 vs 95.7 differs from the
published figure, so per the instruction hillclimb was not started and
the two numbers are reported side by side.

### The index was not rebuilt

`run_eval.py` does open `docmind.db` for writing at startup, which is why
its mtime changed, but it writes no data.

```
    document_store.init_db()
    rag_engine.init_rag()
```

`init_db()` runs `PRAGMA foreign_keys = ON`, two
`CREATE TABLE IF NOT EXISTS`, an `is_seed` column check, and `commit()`.
Both tables and the column already exist, so nothing is created and
nothing is deleted — but the committed transaction updates the file's
mtime.

`init_rag()` → `_rebuild_index()` rebuilds the **in-memory** FAISS index
from vectors already in SQLite. The comment states it directly:

```
    # faiss, but not the model: the vectors are already in SQLite.
```

No embeddings are recomputed. The model is used for exactly one thing per
case, the query vector (`rag_engine.py:271  q_vec = embed([question])`).

Seed loading lives only in `main.py:40 lifespan()` (FastAPI startup).
`run_eval.py` does not import `main`. Write-capable calls in
`run_eval.py`, counted:

```
init_db 1, init_rag 1, process_document 0, create_document 0,
save_chunks 0, remove_document 0, enforce_retention 0, _safe_remove 0,
delete 0, load_seed 0
```

Confirmed in the data, read-only (`file:docmind.db?mode=ro`):

```
documents: [('08b72bd4', 'company_policy.txt', 17, 7036,
             '2026-08-03T02:24:42.088890', 0)]
chunk id range: 18 ~ 34,  count 17
sqlite_sequence: [('chunks', 34)]
embedding blob sha256:
  4ec7ffc86d673c0d7f84e3643b22ea3099e95f9cac047f5e4e8304104730ce6e
```

`upload_time` is still 2026-08-03 and `AUTOINCREMENT` still sits at 34; a
re-load would have moved both. And retrieval is identical across all
three runs:

```
cases compared: 30
AUG==R1 identical ranking: 30 / 30
R1==R2 identical ranking: 30 / 30
differing cases: none
chunk ids seen (AUG): [18..34]   (R1): [18..34]
```

So retrieval and the index are ruled out as causes of the correctness
change, and the recall comparison is valid.

---

## multi_chunk_03

The only case whose verdict differs across the three runs.

```
cases compared: 30
multi_chunk_03 [multi_chunk]  AUG=partially_correct  R1=correct  R2=correct
```

Partially-correct cases per run:

```
R1:  ['multi_chunk_07']   1
R2:  ['multi_chunk_07']   1
AUG: ['multi_chunk_03', 'multi_chunk_07']   2
```

Both of today's runs land on the same single case, `multi_chunk_07`. The
grading did not wobble between the two runs; `multi_chunk_07` simply sits
on the boundary.

### Retrieval was identical

```
AUG: [33, 34, 25, 18, 23]
R1:  [33, 34, 25, 18, 23]
R2:  [33, 34, 25, 18, 23]
```

### The answer text changed

```
AUG==R1 answer: False
R1==R2 answer: True
```

AUG, the relevant sentence:

> employees can use the wellness stipend to get reimbursement for gym
> memberships by submitting "receipts through the expense portal."

No `$500`.

R1 and R2, the relevant sentence:

> the employee would need to submit receipts through the expense portal
> to be reimbursed using their $500 annual wellness stipend.

`$500` present.

### Judge reasons, verbatim

AUG 2026-08-03, `verdict=partially_correct`:

```
The generated answer correctly identifies the IT portal for standing
desks and the wellness stipend for gym membership reimbursement, but
specifies the 'expense portal' for reimbursement rather than just the
wellness stipend process, and doesn't mention the $500 annual limit
that is in the reference answer.
```

R1 2026-10-03, `verdict=correct`:

```
The generated answer correctly identifies both processes: standing desk
through IT portal (with manager approval and budget availability) and
gym membership reimbursement through the $500 annual wellness stipend,
matching all key facts in the reference answer.
```

R2 2026-10-03, `verdict=correct`: byte-for-byte identical to R1's reason.

The August judge's stated ground for `partially_correct` was the missing
`$500`, and today's answer contains it. For this one case the question
"better answer or looser grading?" is answerable: the answer changed, and
it changed in exactly the respect the earlier judge named.

---

## Answers are not stable at temperature 0

`ANSWER_TEMPERATURE = 0` and `JUDGE_TEMPERATURE = 0`, yet the generated
text differs run to run:

```
R1 vs AUG, differing answers: 7 cases
  factual_01, factual_12, multi_chunk_02, multi_chunk_03,
  multi_chunk_04, multi_chunk_05, multi_chunk_06
R2 vs R1, differing answers: 6 cases
  factual_01, multi_chunk_02, multi_chunk_04, multi_chunk_05,
  multi_chunk_06, multi_chunk_08
```

Retrieval, by contrast, was identical in 30/30. So whether a given run
mentions `$500` is itself variable. Two runs agreeing does not establish
that it always will.

## The judge is not independent

From `evals/run_eval.py` lines 48-57:

```
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
```

The same model grades its own answers. For `multi_chunk_03` the question
was separable because the answer text itself changed in the named
respect. In general it is not: a shift in correctness cannot be
attributed to the answer rather than the grader without evidence of this
kind.

---

## Cost

| | input | output | cache_w | cache_r | USD |
|---|---|---|---|---|---|
| R1 answer | 29,260 | 1,875 | 0 | 0 | $0.038635 |
| R1 judge | 8,988 | 1,234 | 0 | 0 | $0.015158 |
| **R1 total** | **38,248** | **3,109** | **0** | **0** | **$0.053793** |
| R2 answer | 29,260 | 1,867 | 0 | 0 | $0.038595 |
| R2 judge | 8,980 | 1,209 | 0 | 0 | $0.015025 |
| **R2 total** | **38,240** | **3,076** | **0** | **0** | **$0.053620** |

`calls_without_usage = {answer 0, judge 0}` in both runs: all 53 calls
were measured.

| | USD | per question |
|---|---|---|
| R1 | $0.053793 | $0.001793 |
| R2 | $0.053620 | $0.001787 |
| mean | $0.053707 | $0.001790 |

Projection, mean × runs:

```
17 runs (hillclimb max 15 + final 2) = $0.053707 × 17 = $0.913011
already spent, 2 runs               = $0.107413
2 + 17                              = $1.020424
```

Against the $4.73 ceiling: 17 runs would be 19.30% of it, and 21.57%
including the two already spent. The ceiling was never the constraint.
Per-step cost would vary in a real hillclimb, since changing prompt,
model or effort changes tokens per call — a more expensive model would
make the figure above a floor, and successful cost-reduction steps would
make it an overestimate.

Totals: **$0.107413** spent (two Mac runs). **$0.00** from the agent
session — it made no API calls at any point.

---

## Conclusion

**Answer accuracy sits in a 91.3–95.7 range, not at a single value.** The
spread is one case out of 23 judged. Everything protected held in all
three runs: citation grounding 100% (115 of 115), refusal accuracy 100%
(7 of 7), incorrect 0%. Retrieval was byte-identical in 30/30 cases.

**Hillclimb was not run, and the reason is that there is nothing to
climb — not the cost ceiling.** Specifically:

1. The headroom is 1–2 cases. `multi_chunk_03` moved from
   `partially_correct` to `correct` and `multi_chunk_07` remains
   `partially_correct` in every run. `incorrect` is already 0.
2. Those cases move on their own. The generated answer changed in 7 of 30
   cases between AUG and R1 and in 6 of 30 between R1 and R2, at
   temperature 0. An optimiser cannot distinguish its own effect from
   this noise at n=1 run per step.
3. Cost per question is already $0.001790. The stated hillclimb objective
   was to reduce per-question cost; there is little left to reduce, and
   the full 17-run budget is 21.57% of the ceiling, so stopping is not a
   budget decision.
4. The judge is the model under test. Optimising a score produced by the
   same model that generates the answers risks moving the grader rather
   than the system.

The 91.3 → 95.7 difference is **not an improvement.** It is a baseline
re-measurement, with no hillclimb step between the two figures. 95.7 had
already appeared twice on 2026-07-20 (`eval_20260720T132538Z.json`,
`eval_20260720T132826Z.json`), so it is not a new high. Of the ten
`eval_*.json` files, five contain 91.3 and four contain 95.7.

---

## Next suggestions

Recorded, not acted on.

1. **Record the embedding model name in the index.** `docmind.db` stores
   no model name, and 384 dimensions is shared by
   `sentence-transformers/all-MiniLM-L6-v2` and
   `BAAI/bge-small-en-v1.5` (which is `fastembed`'s default when
   `model_name` is omitted; this repository always passes it explicitly,
   and the string appears nowhere in the repository's own files or git
   history). So the database cannot prove which model produced its
   vectors. The evidence today is indirect: the constant has not changed
   since the initial commit (2026-06-12), and a probe recorded 33 seconds
   before the index was built names MiniLM. A `meta` row would make this
   direct.
2. **Use an independent judge model.** The current constraint is that
   `claude-haiku-4-5` is the only model in this account accepting
   `temperature=0`. A judge that cannot be pinned to 0 but is a different
   model may trade one weakness for a lesser one; worth measuring before
   any future hillclimb, since a self-graded objective is not a safe
   thing to optimise against.
3. **Freeze the start-condition checks in one script** so each run leaves
   the same raw output in the same order.
4. **Decide what the published figure should say.** A single number
   (91.3) describes a system whose measured range is 91.3–95.7 across
   identical configuration. Either a range or an n-run median would be
   more honest than either endpoint.

---

## Files to commit — operator's decision

```
$ git status --short
 M claude_qa.py
 M evals/run_eval.py
```

Nothing was committed and nothing was staged. `.git` writes are blocked
from the session mount in any case.

Proposed: the two modified files, both instrumentation-only.

### Whether to include today's results files

```
$ git ls-files evals/results
evals/results/.gitkeep
evals/results/README.md
evals/results/eval_20260720T124830Z.json
evals/results/eval_20260730T014202Z.json
evals/results/eval_20260730T122037Z.json
evals/results/eval_20260730T122154Z.json
evals/results/eval_20260803T022554Z.json
```

```
$ sed -n '24,35p' .gitignore
# Eval results (may contain generated answers / API responses; keep the dir, not the runs)
evals/results/*
!evals/results/.gitkeep
# Five runs are the evidence behind the published numbers, so they are tracked.
# Each was checked for credentials and personal data before being added; the
# corpus in them is this repository's own demo document. See evals/results/README.md.
!evals/results/README.md
!evals/results/eval_20260720T124830Z.json
!evals/results/eval_20260730T014202Z.json
!evals/results/eval_20260730T122037Z.json
!evals/results/eval_20260730T122154Z.json
!evals/results/eval_20260803T022554Z.json
```

```
$ git check-ignore -v evals/results/eval_20261003T142527Z.json
.gitignore:25:evals/results/*	evals/results/eval_20261003T142527Z.json
$ git check-ignore -v evals/results/eval_20261003T142527Z.md
.gitignore:25:evals/results/*	evals/results/eval_20261003T142527Z.md
```

Tracking is **selective, by explicit allow-list**, not blanket. The rule
is `evals/results/*` ignored, with five named exceptions described as
"the evidence behind the published numbers". The August file is tracked
because it is one of those five; today's files are ignored because no
exception names them.

Both readings are defensible and the decision is the operator's:

- **Include them.** They are evidence for the same published numbers —
  in fact they are the evidence that the published 91.3 is one point in a
  range. That is the stated purpose of the allow-list. This requires
  adding two `!evals/results/eval_20261003T…` lines to `.gitignore` and
  a `git add -f`, plus the same credentials-and-personal-data check the
  comment says each of the five received.
- **Leave them out.** The allow-list was written for the five runs behind
  the figures as published. Today's runs are a re-measurement that
  concludes those figures should change; tracking them before that
  decision bundles two questions into one commit.

A note either way: these two files were produced by the instrumented
harness and so carry `token_usage` and `cost` blocks that the five
tracked files do not. If they are added, the `evals/results/README.md`
description of the tracked set should say so.

---

## Environment limits

Recorded because they shaped how the work was done.

- `.git` writes fail from the session mount
  (`unable to unlink '.git/index.lock': Operation not permitted`), so the
  branch was created by the operator and nothing was committed.
- The session sandbox cannot delete files inside the repository. Two
  empty write-probe files were created there by the agent and had to be
  removed by the operator —
  `.writetest-sandbox` and `evals/results/.writetest`. The probe should
  have been done without creating files.
- Step 1-2 was run on the Mac rather than in the session: the sandbox has
  Python 3.10.12 against the project's 3.13, and no cached embedding
  weights, so a measurement there would not have been comparable to the
  published figures.
