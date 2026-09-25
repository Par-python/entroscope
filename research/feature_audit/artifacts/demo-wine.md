# Feature-audit walkthrough: wine

_A statistical walkthrough of a public teaching dataset. Associations are descriptive only: no causal or predictive claim is made._

## Data

- Source: UCI ML Wine recognition data, as bundled with scikit-learn (load_wine). The installed description names R.A. Fisher as creator and Forina, M. et al. (PARVUS, Institute of Pharmaceutical and Food Analysis and Technologies, Genoa) as original owners, and cites Lichman, M. (2013), UCI Machine Learning Repository. The installed description states no license terms.
- Split: {"stratified": true, "train_fraction": 0.7, "random_state": 42, "train_rows": 124, "holdout_rows": 54, "holdout_used": false}
- Types: all predictors declared continuous; exclusions are reported as they fall.

## Run

- Baseline (pandas + raw sklearn MI): 0.17 s; audit: 51.2 s.

## Baseline

Pandas already shows missing counts, distinct counts and exact duplicates; the audit repeats those deliberately. Raw MI is scikit-learn's estimate with the same masks, types and estimator seed as the audit, but without null context.

| feature | missing | distinct | Pearson r (binary target) | raw MI (bits) |
|---|---|---|---|---|
| alcohol | 0 | 98 | — | 0.7219 |
| malic&#95;acid | 0 | 95 | — | 0.4696 |
| ash | 0 | 67 | — | 0.2141 |
| alcalinity&#95;of&#95;ash | 0 | 51 | — | 0.332 |
| magnesium | 0 | 48 | — | 0.3106 |
| total&#95;phenols | 0 | 76 | — | 0.5951 |
| flavanoids | 0 | 97 | — | 0.9528 |
| nonflavanoid&#95;phenols | 0 | 34 | — | 0.2207 |
| proanthocyanins | 0 | 80 | — | 0.3937 |
| color&#95;intensity | 0 | 103 | — | 0.8635 |
| hue | 0 | 65 | — | 0.619 |
| od280/od315&#95;of&#95;diluted&#95;wines | 0 | 92 | — | 0.734 |
| proline | 0 | 90 | — | 0.7418 |

Pandas exact duplicates (untyped): none

## What the audit adds here

- Raw MI is positive but the audit withholds value inference (0): none. The exclusion codes say why. For identifiers and sparse categories, positive MI can reflect memorization; other codes flag a declared-type or support problem.
- Positive raw MI, not distinguished from the shuffled-label null after Holm (0): none.
- Value associations above the permutation null (13): alcohol, malic&#95;acid, ash, alcalinity&#95;of&#95;ash, magnesium, total&#95;phenols, flavanoids, nonflavanoid&#95;phenols, proanthocyanins, color&#95;intensity, hue, od280/od315&#95;of&#95;diluted&#95;wines, proline. This is association in this training sample, not causation or model value.
- Missingness associated with the target (0): none.
- Holm family size m=13. Features are tested one at a time, so interactions are not examined.

## Audit report

### Feature association audit (research prototype)

#### Assumptions

- Target target is treated as a classification label with 3 classes (empirical entropy 1.565 bits).
- Rows were declared independent by the user (assume_iid=True); this was not tested.
- Feature types are the user's explicit declarations; nothing was inferred.
- 124 rows; B=1999 label permutations (minimum attainable raw p = 0.0005); 30 stratified 80% subsamples without replacement; k=3; seed=0; Holm family size m=13; alpha=0.05.
- MI values are raw empirical estimates in bits, shown next to their shuffled-label null median. Rows keep input order; this is not a ranking.

#### Features

| feature | type | observed/total | missing | unique | dominant | entropy (bits) | value status | warning codes | subsample MI median [p10, p90] |
|---|---|---|---|---|---|---|---|---|---|
| alcohol | continuous | 124/124 | 0 | 98 | 0.0484 | — | eligible | insufficient_subsample_support | — |
| malic&#95;acid | continuous | 124/124 | 0 | 95 | 0.0565 | — | eligible | insufficient_subsample_support | — |
| ash | continuous | 124/124 | 0 | 67 | 0.0484 | — | eligible | insufficient_subsample_support | — |
| alcalinity&#95;of&#95;ash | continuous | 124/124 | 0 | 51 | 0.0968 | — | eligible | insufficient_subsample_support | — |
| magnesium | continuous | 124/124 | 0 | 48 | 0.0806 | — | eligible | insufficient_subsample_support | — |
| total&#95;phenols | continuous | 124/124 | 0 | 76 | 0.0565 | — | eligible | insufficient_subsample_support | — |
| flavanoids | continuous | 124/124 | 0 | 97 | 0.0323 | — | eligible | insufficient_subsample_support | — |
| nonflavanoid&#95;phenols | continuous | 124/124 | 0 | 34 | 0.0726 | — | eligible | insufficient_subsample_support | — |
| proanthocyanins | continuous | 124/124 | 0 | 80 | 0.0484 | — | eligible | insufficient_subsample_support | — |
| color&#95;intensity | continuous | 124/124 | 0 | 103 | 0.0242 | — | eligible | insufficient_subsample_support | — |
| hue | continuous | 124/124 | 0 | 65 | 0.0403 | — | eligible | insufficient_subsample_support | — |
| od280/od315&#95;of&#95;diluted&#95;wines | continuous | 124/124 | 0 | 92 | 0.0323 | — | eligible | insufficient_subsample_support | — |
| proline | continuous | 124/124 | 0 | 90 | 0.0323 | — | eligible | insufficient_subsample_support | — |

#### Association tests

| feature | hypothesis | n | MI (bits) | null median | MI − null median | p | Holm p | assessment |
|---|---|---|---|---|---|---|---|---|
| alcohol | value | 124 | 0.7219 | 0 | 0.7219 | 0.0005 | 0.0065 | above_permutation_null |
| malic&#95;acid | value | 124 | 0.4696 | 0 | 0.4696 | 0.0005 | 0.0065 | above_permutation_null |
| ash | value | 124 | 0.2141 | 0 | 0.2141 | 0.002 | 0.0065 | above_permutation_null |
| alcalinity&#95;of&#95;ash | value | 124 | 0.332 | 0 | 0.332 | 0.0005 | 0.0065 | above_permutation_null |
| magnesium | value | 124 | 0.3106 | 0 | 0.3106 | 0.001 | 0.0065 | above_permutation_null |
| total&#95;phenols | value | 124 | 0.5951 | 0 | 0.5951 | 0.0005 | 0.0065 | above_permutation_null |
| flavanoids | value | 124 | 0.9528 | 0 | 0.9528 | 0.0005 | 0.0065 | above_permutation_null |
| nonflavanoid&#95;phenols | value | 124 | 0.2207 | 0 | 0.2207 | 0.0025 | 0.0065 | above_permutation_null |
| proanthocyanins | value | 124 | 0.3937 | 0 | 0.3937 | 0.0005 | 0.0065 | above_permutation_null |
| color&#95;intensity | value | 124 | 0.8635 | 0 | 0.8635 | 0.0005 | 0.0065 | above_permutation_null |
| hue | value | 124 | 0.619 | 0 | 0.619 | 0.0005 | 0.0065 | above_permutation_null |
| od280/od315&#95;of&#95;diluted&#95;wines | value | 124 | 0.734 | 0 | 0.734 | 0.0005 | 0.0065 | above_permutation_null |
| proline | value | 124 | 0.7418 | 0 | 0.7418 | 0.0005 | 0.0065 | above_permutation_null |

#### Exact duplicate columns

None found among same-type, same-dtype pairs.

#### Warnings

- Some eligible features had fewer than 10 subsamples meeting the support rules (insufficient_subsample_support); their stability summary is left empty and their value status is unchanged.
- Feature and target names appear in this report and may themselves be sensitive; review before sharing.

#### Limitations

- Run this on training data only; results describe this sample, not a population or a deployment setting.
- Rows are assumed independent (acknowledged via assume_iid=True, not tested). Repeated subjects, groups or time order invalidate the permutation null.
- Value tests use rows where the feature is observed; missingness tests use all rows. These are different populations, so scores are not a universal ranking across features.
- 'not_distinguished_from_null' is not evidence that a feature is useless: power can be low and marginal tests miss interactions.
- Interaction blind spot: each feature is tested alone against the target. Features that matter only jointly (for example XOR) can show zero individual MI.
- No causal claim, leakage verdict, accuracy guarantee or column-removal advice is made. Exclusion codes are conservative policies, and duplicate pairs are facts about values, not recommendations.
- Holm adjustment controls the family-wise error rate across this report's tests under exchangeability; adjusted p-values are not FDR q-values and change when the family of tests changes.
- subsample_mi_p10/p90 summarize sensitivity to stratified 80% subsamples without replacement; they are not confidence intervals.
- Continuous MI uses scikit-learn's k-nearest-neighbour estimator (k=3), which assumes genuinely continuous values; categorical MI is a plug-in estimate that is biased upward when categories are many relative to rows.
- possible_identifier only catches unique values that are strictly monotone in row order; it can flag a legitimate measurement, and a shuffled numeric ID cannot be recognized reliably.
