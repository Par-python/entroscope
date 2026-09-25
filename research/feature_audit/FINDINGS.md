# Feature-audit prototype: findings and gate status

Date: 2026-09-25. Status: **technical gate PASSED. User validation gate:
AWAITING PARTICIPANTS. Package promotion: NOT STARTED** (it requires the user
gate plus explicit owner approval).

This file records what was run and what came out, including failures and
caveats. Passing the technical gate is a mechanism check on synthetic data. It
is not evidence of usefulness, predictive value or novelty.

## Environment

- Apple M4, 16 GiB RAM, macOS 26.6.2 (arm64).
- Python 3.13.5, numpy 2.4.6, pandas 3.0.3, scipy 1.17.1, scikit-learn 1.9.1
  (the repository `.venv`). Nothing was installed for this work.
- Python 3.9 **runtime compatibility is unverified**: no 3.9 interpreter was
  available locally. The source avoids post-3.9 syntax and Ruff targets py39.

## Technical validation (frozen protocol)

Protocol: [PROTOCOL.md](PROTOCOL.md). It was frozen, with the config and
source hashes in [artifacts/protocol_config.json](artifacts/protocol_config.json),
before any held-out seed ran.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m research.feature_audit.run_validation --case all --seeds 7000:7100 \
  --resume --workers 8 --out research/feature_audit/artifacts/technical.json
.venv/bin/python -m research.feature_audit.run_validation --summarize \
  research/feature_audit/artifacts/technical.json
```

- Records: 200/200 (`complete: true`). Wall-clock time: 39 min (10:46–11:25)
  with 8 workers. The summed per-record time was 16,369 s, inflated by
  parallel load; the median record took 74 s.
- Config hash `a0dafce0…5c`; source hash `123567ac…e172`. Both match the
  freeze record.

### Null case (independent labels, 5 normal + 5 balanced four-category predictors, n=400)

| Measure | Result | Gate |
|---|---|---|
| Datasets with ANY Holm rejection | **6 / 100** | ≤ 10 |
| Exact 95% Clopper–Pearson interval | **[0.022, 0.126]** | lower bound ≤ 0.05 |
| Gate | **PASS** | |

False-positive datasets, all retained. Each had exactly one rejected value test:
7013 (c0, raw p 0.0045, Holm 0.045), 7045 (c1, 0.0045 → 0.045), 7063 (k3,
0.0005 → 0.005), 7072 (c1, 0.003 → 0.03), 7079 (k0, 0.0015 → 0.015),
7080 (k3, 0.002 → 0.02). No missingness tests exist in this case.

The observed family-wise rate of 0.06 is consistent with the nominal 0.05.
The interval is wide (upper bound 0.126). This check does not establish a
universal guarantee.

### Signal case (y = 1[|x| > 1], zero linear correlation by construction)

| Measure | Result | Gate |
|---|---|---|
| `x` value test above the null | **100 / 100** | ≥ 90 |
| `id` value inference excluded (`high_cardinality`) | **100 / 100** | all |
| `(x, x_copy)` duplicate reported | **100 / 100** | all |
| Gate | **PASS** | |

Context, not gated:
- Max |Pearson r(x, y)| was 1.9e-17, so a linear screen would miss x entirely.
- The raw MI of x ranged from 0.986 to 0.993 bits against a 1-bit target
  entropy. The largest Holm p for x was 0.0045.
- `x_copy` was also above the null in every dataset, as expected, since it is
  an exact copy.
- **Additional rejections of pure-noise columns: 5 / 100 datasets**, one
  categorical noise column each: 7008 cat5, 7024 cat4, 7057 cat3,
  7079 cat0, 7084 cat5. Holm controls the family-wise error across the whole
  family, including the true signals, so this is expected at about this rate.
  It is disclosed rather than gated.

### Post-run source change (verification, not replication)

After the held-out run, Ruff reported two E721 findings (`dtype == object`),
and formatting drift, in the frozen sources. They were fixed with
`pd.api.types.is_object_dtype` and `ruff format`; no logic changed. This
changed the source hash to `bf827bd4…e970`. All 40 development records were
rerun under the new hash
([artifacts/development-postlint.json](artifacts/development-postlint.json)).
Their test tables, statuses, duplicates and rejections are **identical** to
the original development run. The held-out artifact keeps its original hash
and was not rerun.

### Second post-run change: Markdown rendering fix (final review)

The final code review found two rendering defects. The target name was
entity-escaped inside a code span, so it displayed as `churn&#95;flag`. And a
name starting with a block marker (`#`, `1.`, `-`, `+`), containing `~~`, or
forming a bare URL could change the report's Markdown structure. The fix
escapes those characters as entities, character by character (which also
repairs apostrophes), and drops the code span. Tests render the reports with
markdown-it to check both.

Only `prototype/report.py` Markdown output changed. `to_dict()` and every
number are untouched. The source hash is now `c169d45b…7489`. Seed 6000 of
both cases was rerun and its records are identical. The walkthroughs were
regenerated under this hash. `resource.json` was measured under `bf827bd4…`;
the audit computation is unchanged since then.

Source-hash history: `123567ac…` (frozen protocol; held-out run) →
`bf827bd4…` (lint and format; 40 development records identical;
resource check) → `c169d45b…` (rendering fix; seed 6000 identical) → `f425c1ce…` (minor fixes below; seed 6000 identical;
walkthroughs). Each walkthrough JSON records the hash that produced it.

### Third post-run change: minor fixes (review follow-up)

- A pandas Categorical column is now validated on the categories that
  actually occur. Unused categories of another scalar type no longer cause a
  rejection; mixed types that occur are still rejected.
- `--resume` now refuses to continue if the Python or library versions differ
  from those stored with the run.
- Walkthrough JSON files record the prototype source hash. Their comparison
  text uses plain column names; only the Markdown is escaped.

None of these touch the validation computations. Seed 6000 of both cases
was rerun and its records are identical.

### Development run

Seeds 6000–6019 ([artifacts/development.json](artifacts/development.json)):
null 1/20 datasets rejected; signal 20/20; id excluded 20/20; duplicate
reported 20/20. No correctness defects were found. The first development run
used default library threading and oversubscribed the CPU. Seed 6000 was
rerun with single-threaded BLAS/OpenMP and reproduced its records exactly;
the held-out run then used single-threaded libraries.

## Resource target (dedicated process, one worker)

```bash
.venv/bin/python -m research.feature_audit.run_validation --resource-check \
  --out research/feature_audit/artifacts/resource.json
```

n=1,000, 5 continuous + 5 categorical predictors, defaults (B=1999, 30 subsamples):
**36.0 s** (target < 300 s) and **34.2 MB incremental peak RSS** (target < 1 GiB).
**PASS.** Memory comes from `getrusage` `ru_maxrss`, which macOS reports in
bytes. The figure is the high-water mark after the audit minus the mark before it.

## Walkthroughs (default resampling)

```bash
.venv/bin/python -m research.feature_audit.demo --dataset {synthetic,wine,breast_cancer} \
  --output-dir research/feature_audit/artifacts --overwrite
```

| Dataset | Training rows (70% stratified, seed 42) | Baseline | Audit | Outcome |
|---|---|---|---|---|
| synthetic | 600 (whole frame) | 1.3 s | 20.1 s | All planted mechanisms were surfaced: the duplicate, `customer_id` → high_cardinality, `row_number` → possible_identifier, the constant column, the rare region, `support_calls` → ambiguous_discrete, and informative missingness in `last_login_days`. |
| wine | 124 (holdout 54, unused) | 0.2 s | 51.2 s | 13/13 value tests above the null. **Every stability summary is empty** (`insufficient_subsample_support`): 80% subsamples of 124 rows have 98 rows, below the 100-row support floor. |
| breast_cancer | 398 (holdout 171, unused) | 0.2 s | 116.2 s | 22/30 above the null. 7 features have positive raw MI but are not distinguished from the null after Holm. |

Source notes come from the installed dataset descriptions and are recorded in
each JSON file. **Neither description states license terms**, and none were
checked online (no network access was used). The artifacts contain aggregates
and column names only, never rows. The breast-cancer walkthrough is labelled
as not medical guidance.

What the baseline already gives: pandas missing and distinct counts and
exact duplicates cover the review-flag tasks for simple cases, and raw MI
gives the same point estimates. What the audit adds on these datasets: null
context and Holm adjustment (the 7 breast-cancer features whose positive raw
MI does not survive), explicit exclusions where raw MI looks informative but
reflects identifiers or sparse categories (synthetic `customer_id` raw MI =
1.0 bit), separate missingness tests, and type-sensitive duplicates. Whether
any of that saves a practitioner time is **untested** and is the purpose of
the pilot.

## Tests and verification

Task 9 results are in the handoff message and the ledger. The research suite
passes, as does the existing package suite plus `research/change_detection`.

## Known limitations and deferred items

- Datasets with 100–124 training rows can never produce stability summaries.
  This is a property of the fixed 100-row floor combined with 80% subsamples,
  and it is disclosed in the report as `insufficient_subsample_support`.
- Names are escaped as HTML entities in Markdown (for example `&#95;` for
  `_`). This is safe and reversible, but the raw text is harder to read.
- Everything in the design's non-goals list remains out of scope:
  regression targets, dependent rows, interactions and conditional MI,
  automatic selection, hosted use.

## Gate status and next owner action

1. Technical gate: **passed** (above).
2. User gate: **awaiting real participants.** None were contacted, and no
   feedback exists. The owner recruits five
   consenting practitioners, and the gate needs at least 3/5 verified useful
   findings or time savings with no unresolved seriously misleading claims.
3. Package promotion (plan Task 8): **not started**. It requires gate 2 plus
   explicit owner approval.
