# Changelog

## 0.2.0 — 2026-06-03

- **Transfer entropy** (`entroscope.transfer`) — the first bivariate measure:
  directional information flow `TE(X→Y)` (Schreiber 2000) with two estimators,
  Kraskov KSG k-NN (default, no binning) and a binned histogram cross-check.
  API: `compute`, `rolling`, `delta`, `plot`. A sample-size guard warns when the
  data is too thin for the embedding dimension. Correctness is validated against
  three independent authorities: the bivariate-Gaussian closed form, the
  Kraskov-2004 analytic mutual information, and a hand-checked isolated embedding.
- **Divergence** (`entroscope.divergence`) — distribution-vs-distribution
  measures: `kl` (Kullback-Leibler, directional) and `js` (Jensen-Shannon,
  symmetric, bounded `[0, 1]` bits), plus `plot`. Inputs are two raw samples,
  binned over a shared range; KL uses epsilon smoothing to stay finite on empty
  bins. Useful for data-drift detection (training vs production distributions).
- **Fixed:** importing a measure no longer forces matplotlib's `Agg` backend
  (issue #2). The library no longer mutates global matplotlib state on import, so
  interactive plotting (e.g. `df.plot()` in a notebook) keeps working. Headless,
  Docker, and CI users who want a guaranteed non-interactive backend should set
  `MPLBACKEND=Agg` in their environment.
- **Correlation-stability example** (`examples/correlation_stability.py`) —
  entropy of a rolling correlation series as a regime-stability gauge, shipped
  alongside a rolling-std baseline so the comparison stays honest.
- New worked examples: `examples/information_flow.py` (directional transfer
  entropy) and `examples/drift.py` (KL/JS distribution drift).

## 0.1.1 — 2026-06-01

- Added a `LICENSE` file (MIT) and a `CONTRIBUTING.md` guide.
- Rewrote the README: clearer intro, working PyPI/CI/Python/license badges, and
  links to the example write-ups.
- Added PyPI trove classifiers (supported Python versions, license, topics) so
  the Python-versions badge and PyPI sidebar populate correctly.
- Runnable example scripts in `examples/`: four medical/biomedical scenarios
  (HRV/sample, EEG/permutation+spectral, respiration/approximate,
  glucose/differential) and four business scenarios (sales/shannon,
  web-traffic/spectral, prices/permutation, QC/multiscale), plus matching
  `docs/examples/medical.md` and `docs/examples/business.md` pages.
- Smoke tests (`tests/test_examples.py`) run the example scripts in CI so they
  can't silently break; CI lint now covers `examples/` too.
- CI/CD via GitHub Actions: lint (ruff check + format), test matrix across
  Python 3.9 to 3.14 with a 90% coverage gate, and a build + `twine check` job.
- Tag-triggered PyPI publish workflow using trusted publishing (inert until a
  PyPI publisher is configured).
- Applied `ruff format` across the codebase; added `ruff` to the `dev` extra.

## 0.1.0 — 2026-06-01

Initial release.

- Seven entropy measures: shannon, permutation, sample, approximate, spectral,
  differential, multiscale.
- Consistent API across measures: `compute`, `rolling`, `delta`, `plot`, plus
  `normalized` (shannon/permutation/spectral) and `geographic` (shannon).
- Works on `pd.Series` (index preserved) and `np.ndarray`.
- Visualization helpers: `plot.compare`, `plot.dashboard`, `plot.drop_events`.
