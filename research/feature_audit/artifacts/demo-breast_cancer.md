# Feature-audit walkthrough: breast_cancer

_A statistical walkthrough of a public teaching dataset. This is NOT medical guidance, and associations here say nothing clinical, causal or predictive._

## Data

- Source: UCI ML Breast Cancer Wisconsin (Diagnostic) data, as bundled with scikit-learn (load_breast_cancer). The installed description names Wolberg, Street and Mangasarian as creators and Nick Street as donor (November 1995). The installed description states no license terms.
- Split: {"stratified": true, "train_fraction": 0.7, "random_state": 42, "train_rows": 398, "holdout_rows": 171, "holdout_used": false}
- Types: all predictors declared continuous; exclusions are reported as they fall.

## Run

- Baseline (pandas + raw sklearn MI): 0.22 s; audit: 116.2 s.

## Baseline

Pandas already shows missing counts, distinct counts and exact duplicates; the audit repeats those deliberately. Raw MI is scikit-learn's estimate with the same masks, types and estimator seed as the audit, but without null context.

| feature | missing | distinct | Pearson r (binary target) | raw MI (bits) |
|---|---|---|---|---|
| mean radius | 0 | 337 | 0.7404 | 0.5282 |
| mean texture | 0 | 350 | 0.3947 | 0.1685 |
| mean perimeter | 0 | 373 | 0.7533 | 0.5883 |
| mean area | 0 | 385 | 0.7295 | 0.5162 |
| mean smoothness | 0 | 344 | 0.3834 | 0.1056 |
| mean compactness | 0 | 382 | 0.6103 | 0.3673 |
| mean concavity | 0 | 380 | 0.6928 | 0.5199 |
| mean concave points | 0 | 381 | 0.7848 | 0.6377 |
| mean symmetry | 0 | 331 | 0.342 | 0.07378 |
| mean fractal dimension | 0 | 357 | -0.003663 | 0 |
| radius error | 0 | 383 | 0.6 | 0.3203 |
| texture error | 0 | 370 | -0.05658 | 0.0822 |
| perimeter error | 0 | 376 | 0.5857 | 0.3243 |
| area error | 0 | 385 | 0.6429 | 0.4798 |
| smoothness error | 0 | 386 | -0.1154 | 0.0647 |
| compactness error | 0 | 386 | 0.2854 | 0.06729 |
| concavity error | 0 | 371 | 0.2372 | 0.2302 |
| concave points error | 0 | 368 | 0.4031 | 0.1902 |
| symmetry error | 0 | 359 | -0.01125 | 0.03362 |
| fractal dimension error | 0 | 389 | 0.0445 | 0.02273 |
| worst radius | 0 | 342 | 0.7805 | 0.6601 |
| worst texture | 0 | 368 | 0.4462 | 0.2148 |
| worst perimeter | 0 | 369 | 0.7868 | 0.6983 |
| worst area | 0 | 386 | 0.7424 | 0.6545 |
| worst smoothness | 0 | 311 | 0.4369 | 0.1244 |
| worst compactness | 0 | 380 | 0.6125 | 0.3632 |
| worst concavity | 0 | 382 | 0.6771 | 0.4826 |
| worst concave points | 0 | 356 | 0.8056 | 0.6577 |
| worst symmetry | 0 | 366 | 0.4415 | 0.1271 |
| worst fractal dimension | 0 | 387 | 0.3404 | 0.06747 |

Pandas exact duplicates (untyped): none

## What the audit adds here

- Raw MI is positive but the audit withholds value inference (0): none. The exclusion codes say why. For identifiers and sparse categories, positive MI can reflect memorization; other codes flag a declared-type or support problem.
- Positive raw MI, not distinguished from the shuffled-label null after Holm (7): mean symmetry, texture error, smoothness error, compactness error, symmetry error, fractal dimension error, worst fractal dimension.
- Value associations above the permutation null (22): mean radius, mean texture, mean perimeter, mean area, mean smoothness, mean compactness, mean concavity, mean concave points, radius error, perimeter error, area error, concavity error, concave points error, worst radius, worst texture, worst perimeter, worst area, worst smoothness, worst compactness, worst concavity, worst concave points, worst symmetry. This is association in this training sample, not causation or model value.
- Missingness associated with the target (0): none.
- Holm family size m=30. Features are tested one at a time, so interactions are not examined.

## Audit report

### Feature association audit (research prototype)

#### Assumptions

- Target target is treated as a classification label with 2 classes (empirical entropy 0.9521 bits).
- Rows were declared independent by the user (assume_iid=True); this was not tested.
- Feature types are the user's explicit declarations; nothing was inferred.
- 398 rows; B=1999 label permutations (minimum attainable raw p = 0.0005); 30 stratified 80% subsamples without replacement; k=3; seed=0; Holm family size m=30; alpha=0.05.
- MI values are raw empirical estimates in bits, shown next to their shuffled-label null median. Rows keep input order; this is not a ranking.

#### Features

| feature | type | observed/total | missing | unique | dominant | entropy (bits) | value status | warning codes | subsample MI median [p10, p90] |
|---|---|---|---|---|---|---|---|---|---|
| mean radius | continuous | 398/398 | 0 | 337 | 0.00754 | — | eligible | — | 0.5336 [0.5029, 0.5688] (30 valid) |
| mean texture | continuous | 398/398 | 0 | 350 | 0.00754 | — | eligible | — | 0.176 [0.1336, 0.2006] (30 valid) |
| mean perimeter | continuous | 398/398 | 0 | 373 | 0.00754 | — | eligible | — | 0.5832 [0.5544, 0.6147] (30 valid) |
| mean area | continuous | 398/398 | 0 | 385 | 0.00503 | — | eligible | — | 0.5216 [0.4987, 0.5539] (30 valid) |
| mean smoothness | continuous | 398/398 | 0 | 344 | 0.0101 | — | eligible | — | 0.1195 [0.0894, 0.1524] (30 valid) |
| mean compactness | continuous | 398/398 | 0 | 382 | 0.00503 | — | eligible | — | 0.3722 [0.3459, 0.4052] (30 valid) |
| mean concavity | continuous | 398/398 | 0 | 380 | 0.0251 | — | eligible | — | 0.5223 [0.4883, 0.5447] (30 valid) |
| mean concave points | continuous | 398/398 | 0 | 381 | 0.0251 | — | eligible | — | 0.6255 [0.6118, 0.6652] (30 valid) |
| mean symmetry | continuous | 398/398 | 0 | 331 | 0.0101 | — | eligible | — | 0.0728 [0.02975, 0.09845] (30 valid) |
| mean fractal dimension | continuous | 398/398 | 0 | 357 | 0.00754 | — | eligible | — | 0 [0, 0.01775] (30 valid) |
| radius error | continuous | 398/398 | 0 | 383 | 0.00754 | — | eligible | — | 0.3339 [0.3066, 0.3609] (30 valid) |
| texture error | continuous | 398/398 | 0 | 370 | 0.00754 | — | eligible | — | 0.06438 [0.02814, 0.09567] (30 valid) |
| perimeter error | continuous | 398/398 | 0 | 376 | 0.0101 | — | eligible | — | 0.3282 [0.2963, 0.3548] (30 valid) |
| area error | continuous | 398/398 | 0 | 385 | 0.00754 | — | eligible | — | 0.4682 [0.4413, 0.4966] (30 valid) |
| smoothness error | continuous | 398/398 | 0 | 386 | 0.00503 | — | eligible | — | 0.06956 [0.02817, 0.1089] (30 valid) |
| compactness error | continuous | 398/398 | 0 | 386 | 0.00754 | — | eligible | — | 0.07895 [0.05925, 0.1195] (30 valid) |
| concavity error | continuous | 398/398 | 0 | 371 | 0.0251 | — | eligible | — | 0.2474 [0.2222, 0.2765] (30 valid) |
| concave points error | continuous | 398/398 | 0 | 368 | 0.0251 | — | eligible | — | 0.1918 [0.1622, 0.2142] (30 valid) |
| symmetry error | continuous | 398/398 | 0 | 359 | 0.00754 | — | eligible | — | 0.02931 [0, 0.05796] (30 valid) |
| fractal dimension error | continuous | 398/398 | 0 | 389 | 0.00503 | — | eligible | — | 0.02497 [0, 0.06061] (30 valid) |
| worst radius | continuous | 398/398 | 0 | 342 | 0.0101 | — | eligible | — | 0.657 [0.6309, 0.6792] (30 valid) |
| worst texture | continuous | 398/398 | 0 | 368 | 0.00754 | — | eligible | — | 0.2271 [0.1951, 0.256] (30 valid) |
| worst perimeter | continuous | 398/398 | 0 | 369 | 0.00754 | — | eligible | — | 0.684 [0.6656, 0.7177] (30 valid) |
| worst area | continuous | 398/398 | 0 | 386 | 0.00503 | — | eligible | — | 0.6552 [0.6278, 0.6803] (30 valid) |
| worst smoothness | continuous | 398/398 | 0 | 311 | 0.0101 | — | eligible | — | 0.1193 [0.0942, 0.1558] (30 valid) |
| worst compactness | continuous | 398/398 | 0 | 380 | 0.00503 | — | eligible | — | 0.3567 [0.33, 0.3817] (30 valid) |
| worst concavity | continuous | 398/398 | 0 | 382 | 0.0251 | — | eligible | — | 0.4997 [0.468, 0.524] (30 valid) |
| worst concave points | continuous | 398/398 | 0 | 356 | 0.0251 | — | eligible | — | 0.6528 [0.6395, 0.6751] (30 valid) |
| worst symmetry | continuous | 398/398 | 0 | 366 | 0.00754 | — | eligible | — | 0.1291 [0.1035, 0.1756] (30 valid) |
| worst fractal dimension | continuous | 398/398 | 0 | 387 | 0.00503 | — | eligible | — | 0.0742 [0.04443, 0.1192] (30 valid) |

#### Association tests

| feature | hypothesis | n | MI (bits) | null median | MI − null median | p | Holm p | assessment |
|---|---|---|---|---|---|---|---|---|
| mean radius | value | 398 | 0.5282 | 0 | 0.5282 | 0.0005 | 0.015 | above_permutation_null |
| mean texture | value | 398 | 0.1685 | 0 | 0.1685 | 0.0005 | 0.015 | above_permutation_null |
| mean perimeter | value | 398 | 0.5883 | 0 | 0.5883 | 0.0005 | 0.015 | above_permutation_null |
| mean area | value | 398 | 0.5162 | 0 | 0.5162 | 0.0005 | 0.015 | above_permutation_null |
| mean smoothness | value | 398 | 0.1056 | 0 | 0.1056 | 0.003 | 0.027 | above_permutation_null |
| mean compactness | value | 398 | 0.3673 | 0 | 0.3673 | 0.0005 | 0.015 | above_permutation_null |
| mean concavity | value | 398 | 0.5199 | 0 | 0.5199 | 0.0005 | 0.015 | above_permutation_null |
| mean concave points | value | 398 | 0.6377 | 0 | 0.6377 | 0.0005 | 0.015 | above_permutation_null |
| mean symmetry | value | 398 | 0.07378 | 0.0002427 | 0.07354 | 0.0195 | 0.1365 | not_distinguished_from_null |
| mean fractal dimension | value | 398 | 0 | 0 | 0 | 1 | 1 | not_distinguished_from_null |
| radius error | value | 398 | 0.3203 | 0.0005053 | 0.3198 | 0.0005 | 0.015 | above_permutation_null |
| texture error | value | 398 | 0.0822 | 0 | 0.0822 | 0.0095 | 0.076 | not_distinguished_from_null |
| perimeter error | value | 398 | 0.3243 | 0 | 0.3243 | 0.0005 | 0.015 | above_permutation_null |
| area error | value | 398 | 0.4798 | 0 | 0.4798 | 0.0005 | 0.015 | above_permutation_null |
| smoothness error | value | 398 | 0.0647 | 0 | 0.0647 | 0.035 | 0.1675 | not_distinguished_from_null |
| compactness error | value | 398 | 0.06729 | 0 | 0.06729 | 0.026 | 0.156 | not_distinguished_from_null |
| concavity error | value | 398 | 0.2302 | 0.0001192 | 0.2301 | 0.0005 | 0.015 | above_permutation_null |
| concave points error | value | 398 | 0.1902 | 0 | 0.1902 | 0.0005 | 0.015 | above_permutation_null |
| symmetry error | value | 398 | 0.03362 | 0.0006183 | 0.033 | 0.177 | 0.531 | not_distinguished_from_null |
| fractal dimension error | value | 398 | 0.02273 | 0 | 0.02273 | 0.241 | 0.531 | not_distinguished_from_null |
| worst radius | value | 398 | 0.6601 | 0 | 0.6601 | 0.0005 | 0.015 | above_permutation_null |
| worst texture | value | 398 | 0.2148 | 0 | 0.2148 | 0.0005 | 0.015 | above_permutation_null |
| worst perimeter | value | 398 | 0.6983 | 0 | 0.6983 | 0.0005 | 0.015 | above_permutation_null |
| worst area | value | 398 | 0.6545 | 0 | 0.6545 | 0.0005 | 0.015 | above_permutation_null |
| worst smoothness | value | 398 | 0.1244 | 0 | 0.1244 | 0.0005 | 0.015 | above_permutation_null |
| worst compactness | value | 398 | 0.3632 | 0 | 0.3632 | 0.0005 | 0.015 | above_permutation_null |
| worst concavity | value | 398 | 0.4826 | 0 | 0.4826 | 0.0005 | 0.015 | above_permutation_null |
| worst concave points | value | 398 | 0.6577 | 0 | 0.6577 | 0.0005 | 0.015 | above_permutation_null |
| worst symmetry | value | 398 | 0.1271 | 0 | 0.1271 | 0.001 | 0.015 | above_permutation_null |
| worst fractal dimension | value | 398 | 0.06747 | 0 | 0.06747 | 0.0335 | 0.1675 | not_distinguished_from_null |

#### Exact duplicate columns

None found among same-type, same-dtype pairs.

#### Warnings

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
