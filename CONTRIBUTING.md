# Contributing to entroscope

Thanks for your interest in helping. entroscope aims to be the definitive entropy
toolkit for time series, so contributions that add measures, sharpen the math,
improve docs, or fix bugs are all welcome.

## Getting set up

```bash
git clone https://github.com/Par-python/entroscope.git
cd entroscope

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # installs the package + pytest, pytest-cov, ruff
```

## Running the checks

The same three checks run in CI; please run them locally before opening a PR.

```bash
# tests, with the 90% coverage gate CI enforces
pytest --cov=entroscope --cov-fail-under=90

# lint and formatting (ruff)
ruff check entroscope tests examples
ruff format --check entroscope tests examples
```

`ruff format entroscope tests examples` (without `--check`) applies the
formatting for you.

## How the library is organized

A small shared core does the heavy lifting; each measure is a thin module on top.

- `entroscope/_core.py`: input coercion (Series/ndarray), the `rolling` and
  `delta` drivers, and the plotting scaffold. The "Series in, Series out
  (index preserved); ndarray in, ndarray out" contract lives here, once.
- `entroscope/utils/`: `windows.py` (sliding windows), `normalize.py`
  (scale to [0, 1]), `plot.py` (the cross-measure `compare`/`dashboard`/
  `drop_events` helpers).
- `entroscope/<measure>.py`: each measure (shannon, permutation, spectral,
  sample, approximate, differential, multiscale) defines a private
  `_kernel(values, **params) -> float` and thin public functions that delegate
  to the core drivers.
- `tests/`: one file per measure, plus `test_consistency.py` (verifies every
  measure honors the shared API) and `test_examples.py` (runs the example
  scripts end-to-end).

## Adding a new entropy measure

Follow the existing modules as a template (shannon is the simplest):

1. Create `entroscope/<name>.py` with a `_kernel(values, **params) -> float`
   that computes the single-value entropy, then `compute`, `rolling`, `delta`,
   and `plot` that delegate to `entroscope._core`. Add `normalized` only if the
   measure has a well-defined theoretical maximum.
2. Export it from `entroscope/__init__.py`.
3. Add `tests/test_<name>.py` with at least: a known-answer case, a
   higher-vs-lower-entropy ordering check, type/shape checks for `rolling`, and
   a validation-error case.
4. If it fits the cross-measure API, add it to the registry in
   `entroscope/utils/plot.py` and to `tests/test_consistency.py`.
5. Keep coverage at or above 90%.

## Pull requests

- Branch off `master`, keep PRs focused on one change.
- Write a clear description of what changed and why; reference an issue if there
  is one.
- Make sure tests, coverage, and ruff all pass.
- Add a line to `CHANGELOG.md` under an `## Unreleased` heading.
- Match the style of the surrounding code (ruff handles formatting).

## Reporting bugs and requesting features

Open an issue at https://github.com/Par-python/entroscope/issues. For bugs, a
minimal reproducible example (the input series, the call, what you expected, what
you got) makes it much faster to fix.

## License

By contributing, you agree that your contributions are licensed under the MIT
License, the same as the rest of the project.
