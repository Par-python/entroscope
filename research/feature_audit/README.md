# Feature association audit — research prototype

**Status: research prototype, not part of the supported `entroscope` API.**
Nothing here is installed by `pip install entroscope`. Evidence status is
recorded in [FINDINGS.md](FINDINGS.md). User validation is still pending.

The audit answers three questions about a small or medium tabular
**classification training** dataset:

1. Which individual feature–target associations are above a shuffled-label
   null, after Holm multiple-testing adjustment?
2. How much does each mutual-information (MI) estimate move across stratified
   80% subsamples?
3. Which columns need manual review before their scores mean anything
   (all missing, constant, too little support, high-cardinality or rare
   categories, too few distinct values, possible row identifiers, exact duplicates)?

It does not select features, fit models, detect leakage, or make causal claims.
The MI estimators come from scikit-learn. What the prototype adds is the
interpretation around them: null context, multiplicity adjustment, stability,
and explicit exclusions.

## Usage

Run from the repository root. The prototype uses the namespace path
`research.feature_audit.prototype`.

```python
import pandas as pd
from research.feature_audit.prototype import audit

train = pd.read_csv("my_training_split.csv")      # training data only
report = audit(
    train,
    target="churned",                             # class labels, 2–20 classes
    feature_types={                               # one entry per predictor, no guessing
        "age": "continuous",
        "plan": "categorical",
        "region_code": "categorical",             # integer codes are categories, not numbers
    },
    assume_iid=True,                              # you confirm rows are independent
)
print(report.to_markdown())
payload = report.to_dict()                        # JSON-compatible builtins
```

### Declaring types

- `categorical` for labels and codes, including integer codes such as ZIP or
  product IDs. No ordering or distance is assumed.
- `continuous` only for real numeric measurements. A string column declared
  continuous is rejected, not coerced. Continuous columns with fewer than 10
  distinct values are excluded as `ambiguous_discrete`: reconsider the type.

### Input limits

- Input is a pandas DataFrame with 100–5,000 rows and 1–50 predictors. Larger
  inputs are rejected, not silently sampled.
- Column names must be unique strings, and the row index must be unique.
- The target must have no missing values, between 2 and 20 classes, and at
  least 10 rows per class. Numeric labels must be whole numbers.
- Predictors may contain missing values. Infinities, datetimes, complex
  numbers, nested values and object columns that mix scalar types are rejected.
- Rows must be independent. Repeated subjects, grouped records and time series
  are out of scope, and `assume_iid=True` is your acknowledgment of that. The
  audit does not test it.

### Cost

Each eligible hypothesis runs 1 observed and 1,999 permuted MI estimates, plus
30 subsample estimates. Continuous features dominate the cost (about 2 ms per
kNN estimate at n=400 on an Apple M4). A 400-row, 10-predictor frame (5
continuous, 5 categorical) took about 27 s with defaults on that machine. The
measured n=1,000 resource check is in [FINDINGS.md](FINDINGS.md).
`n_permutations` and `n_subsamples` can be lowered for exploration. Lowering
`n_permutations` coarsens the attainable p-values, and the report warns when it does.

### Privacy

The audit is local-only. It makes no network calls, sends no telemetry and
uses no LLM. The report keeps only aggregate numbers: no rows, index values,
category labels or class names. Feature and target names do appear in it, so
review them before sharing. Nothing is written to disk unless you write it yourself.

### Dependencies

This needs scikit-learn, the existing optional extra:
`pip install 'entroscope[sklearn]'` (or `pip install scikit-learn`). No other
dependency is added.

## Reading the output

- `value_status` is `eligible` or the first applicable exclusion, in this order:
  `all_missing`, `constant`, `insufficient_support`, `high_cardinality`,
  `rare_categories`, `ambiguous_discrete`, `possible_identifier`.
  `warning_codes` keeps every applicable code. These are conservative policies,
  not leakage verdicts.
- Tests are `value` tests on rows where the feature is observed, or
  `missingness` tests: does "is missing" relate to the target, on all rows?
  `above_permutation_null` means Holm-adjusted p ≤ 0.05.
  `not_distinguished_from_null` does not mean useless.
- `subsample_mi_p10/p90` are sensitivity summaries, not confidence intervals.
- Each feature is tested alone. Features that matter only jointly (XOR-like)
  can show zero individual MI.

## Layout and commands

```text
prototype/        validation, describe, association, stability, duplicates, report, api
tests/            unit tests (not collected by default; pass the path explicitly)
PROTOCOL.md       frozen technical-validation protocol
run_validation.py resumable held-out synthetic study
demo.py           synthetic and bundled-sklearn walkthroughs
FINDINGS.md       evidence, failures and gate status
artifacts/        aggregate outputs only
```

```bash
.venv/bin/python -m pytest research/feature_audit/tests -q
.venv/bin/python -m research.feature_audit.demo --dataset synthetic --output-dir /tmp/audit-demo
```
