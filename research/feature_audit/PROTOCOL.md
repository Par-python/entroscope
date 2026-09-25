# Feature-audit technical validation protocol

Status: **FROZEN 2026-09-25** before held-out seeds 7000–7099 were run (see section 5).

This protocol tests known mechanisms on synthetic data. It is not a claim of
downstream accuracy, utility, or novel predictive power. The thresholds below
are predeclared engineering choices taken from the design spec, not theorems.

## 1. Design (copied verbatim from spec section 8)

> Technical validation tests known mechanisms, not a claim of downstream accuracy.
> Unit fixtures include a balanced categorical copy (1 bit), independent Cartesian
> categorical pairs (0 bits), a unique ID with empirical MI but blocked inference,
> an exact duplicate, all-missing/constant fields, rare categories, a symmetric
> nonlinear continuous relationship, and balanced XOR (individually zero MI).
> Always explain the interaction blind spot; no automatic feature selection.
>
> Freeze an execution protocol before scoring, with 100 held-out synthetic seeds
> 7000–7099, n=400, ten predictors, B=1999, 30 subsamples, configuration above.
> Development seeds 6000–6019 are disjoint. Use a separate stream per fixture case.
> Independent-null case: balanced binary y shuffled independently of five normal
> and five balanced four-category predictors; no missingness. Record whether ANY
> Holm rejection occurs per dataset. Gate: at most 10 of 100 datasets reject and
> the two-sided 95% Clopper–Pearson lower bound is <=0.05. A conservative result
> below 0.05 must not fail simply because its interval lies below 0.05. This engineering check
> does not establish a universal guarantee; report the observed interval.
>
> Signal case: n=400 with x consisting of 200 evenly spaced positive values in
> (0,2] and their negatives, shuffled; y=1[abs(x)>1]. Include x, x's exact copy,
> one continuous noise feature, six categorical noise features, and one unique
> categorical ID. Gate: x is above null in >=90/100 datasets; ID value inference
> is always excluded; duplicate pair always reported. Retain correlation baseline
> (near zero by construction), raw sklearn MI, timings and every failure. Passing
> is a mechanism check, not evidence of utility or novel predictive power.
>
> Also run two real-data walkthroughs: sklearn's bundled wine and breast-cancer
> datasets, accessed locally with `load_wine(as_frame=True)` and
> `load_breast_cancer(as_frame=True)`. Read installed dataset descriptions and
> record original source/license notes before exporting anything. No raw rows in
> artifacts. Use only a stratified 70% training split (seed=42); holdout unused.
> The latter is a statistical walkthrough, never medical guidance. Report runtime
> and limitations, not discoveries asserted as causal truths.
>
> Resource target: under five minutes and under 1 GB incremental peak memory for
> n=1,000, 10 predictors (five continuous/five categorical), default resampling,
> on the recorded local machine with one worker. Measure with a dedicated process;
> document OS-specific memory units. If it misses, profile and report the failure;
> do not silently reduce B or change method. Keep the full 100-seed study outside
> default unit tests and checkpoint resumably by case/seed/config/source hash.

## 2. Operational definitions

Implemented in `research/feature_audit/run_validation.py`:

- **Generators.** `make_case(case, seed)` draws from
  `np.random.default_rng(SeedSequence([seed, case_id]))`, with case ids
  null=1, nonlinear=2, resource=3. Every case/seed pair therefore has its own stream.
  - `null`: n=400. `c0..c4` are standard normal. `k0..k4` are shuffled tilings of
    `[0,1,2,3]` (100 of each code), declared categorical. `y` is a shuffled
    tiling of `[0,1]`, drawn independently. There is no missingness.
  - `nonlinear`: `x` is a fixed shuffle of
    `np.r_[np.arange(1,201)/100, -np.arange(1,201)/100]`, and `y = 1[|x| > 1]`
    (200 ones, 200 zeros). The other columns are `x_copy` (an exact copy), one
    standard-normal `noise` column, six shuffled balanced four-code categorical
    columns `cat0..cat5`, and `id` (unique strings `id-<k>`, declared
    categorical). That makes ten predictors. No fitting and no seed selection.
  - `resource`: n=1,000, five normal continuous columns, five balanced
    four-code categorical columns, and a balanced independent binary `y`.
- **Audit configuration (official).** `n_permutations=1999`, `n_subsamples=30`,
  `random_state=0`. Otherwise the prototype defaults apply: 80% stratified
  subsamples without replacement, k=3, alpha=0.05, and Holm adjustment across
  all eligible value and missingness tests in each report.
- **Seeds.** Development: 6000–6019. Held-out acceptance: 7000–7099
  (`--seeds 7000:7100`, STOP excluded). The two sets are disjoint.
- **Null gate.** For each held-out null dataset, record whether ANY test is
  `above_permutation_null`. Let k be the number of datasets with a rejection,
  n=100. The gate passes iff all 100 records exist, k ≤ 10, and the exact
  two-sided 95% Clopper–Pearson lower bound is ≤ 0.05. The bounds are
  `0 if k==0 else beta.ppf(.025,k,n-k+1)` and
  `1 if k==n else beta.ppf(.975,k+1,n-k)`. An interval lying below 0.05 does not fail.
- **Signal gate.** Over the 100 held-out nonlinear datasets: the `x` value test
  is `above_permutation_null` in at least 90; `id` has `value_status` other than
  `eligible` in all 100; the pair `(x, x_copy)` is reported in all 100.
  Recorded for context but not gated: the Pearson r of x with y (near zero by
  construction), raw MI for every test, timings, and every miss or false positive.
- **Technical gate** = null gate AND signal gate.
- **Resource target** (reported, not part of the statistical gate). One default
  audit of the `resource` case, run in a dedicated process
  (`--resource-check`) with one worker, must take under 300 s and add under
  1 GiB to peak memory. Peak memory is measured with `getrusage` `ru_maxrss`,
  which reports bytes on macOS and KiB on Linux; the runner converts it to
  bytes and subtracts the pre-audit high-water mark. If the target is missed,
  profile the run and report the miss. Do not reduce B or change the method.

## 3. Execution rules

1. Run development seeds first. They may reveal correctness defects. Fix any
   defect, record it, and freeze the code before running held-out seeds.
2. Run held-out seeds once with
   `python -m research.feature_audit.run_validation --case all --seeds 7000:7100 --resume --out research/feature_audit/artifacts/technical.json`.
   Parallel workers (`--workers`) are allowed. They do not change any
   computation, only wall-clock time. Per-record timings under parallel load
   are not the resource measurement.
3. Resume is allowed only with identical config and source hashes; the runner
   enforces this. A source change during a run invalidates it.
4. Retain every record regardless of outcome. Summaries come from the stored
   records via `--summarize`, and the record count must be 200.
5. If a statistical criterion fails, progression stops and the failure is
   reported. A correctness fix may rerun the same seeds, but that rerun is
   disclosed as verification, not independent replication. Original artifacts
   are preserved under a new versioned filename. Thresholds, seeds and the
   method do not change after held-out outcomes have been seen.

## 4. Limitations

- Synthetic mechanisms only. Passing says the null calibration, exclusions and
  duplicate detection behave as designed on these generators. It says nothing
  about real-data utility, and nothing about dependence structures that violate
  the assumption of independent rows.
- 100 datasets bound the family-wise rate only coarsely; the observed interval
  is reported, not a universal guarantee.
- Holm's guarantee is conditional on exchangeability under the null.

## 5. Freeze record

Frozen on 2026-09-25, **before any held-out seed was run**.

- Config hash (sha256 of the sorted official config JSON): `a0dafce07b03d8cf47bbe190191a65562759e80069bd3eca561f900ba929ff5c`
- Combined source hash (prototype/*.py + run_validation.py): `123567ac9dda8b511a7ef9fbe026193464980ef0d353f46b6ccc0ce92ef6e172`
- Frozen config and gates: `artifacts/protocol_config.json`

| File | sha256 (first 16 hex) |
|---|---|
| `prototype/__init__.py` | `2d789d8f20f6ae77` |
| `prototype/api.py` | `3b7d73699767be4e` |
| `prototype/association.py` | `70c30f2341f65a6b` |
| `prototype/describe.py` | `c839db9e8f0df123` |
| `prototype/duplicates.py` | `8f964797ea7ea7cd` |
| `prototype/report.py` | `6d2625f1c7dacb9d` |
| `prototype/stability.py` | `50ca121d33e39a81` |
| `prototype/validation.py` | `0312e5c4d85776d9` |
| `run_validation.py` | `e4dc48db90cc564e` |

Development outcomes (seeds 6000–6019, official config, `artifacts/development.json`):
1/20 null datasets had any Holm rejection; `x` was above the
null in 20/20 nonlinear datasets; `id` was excluded
(`high_cardinality`) in 20/20; `(x, x_copy)` was reported in 20/20; max |Pearson r| of x
with y was 9.6e-18. No correctness defect was found, and no
code was changed after the development run.

Execution note: the development run used 8 parallel workers with default library
threading, which oversubscribed the CPU (about 105 s per record). Rerunning seed 6000 of both cases
with `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=VECLIB_MAXIMUM_THREADS=1`
reproduced the stored test records exactly, so the held-out run sets those variables.
This changes only wall-clock time.
