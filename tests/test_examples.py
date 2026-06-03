"""Smoke tests: the shipped example scripts must run end-to-end.

These guard against the public API drifting out from under the documented
examples. We import each example module and run its `main()`, asserting it
completes without error.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _load(module_name):
    path = EXAMPLES_DIR / f"{module_name}.py"
    # Ensure the examples dir is importable so `import _synthetic` resolves.
    sys.path.insert(0, str(EXAMPLES_DIR))
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["medical", "business", "correlation_stability"])
def test_example_runs(name, capsys):
    module = _load(name)
    module.main()  # must not raise
    out = capsys.readouterr().out
    assert "entropy" in out.lower()


def test_synthetic_generators_return_series():
    import pandas as pd

    data = _load("_synthetic")
    for gen in [
        data.heart_rate,
        data.eeg,
        data.respiration,
        data.glucose,
        data.daily_sales,
        data.web_traffic,
        data.stock_prices,
        data.qc_sensor,
    ]:
        series = gen()
        assert isinstance(series, pd.Series)
        assert len(series) > 0


def test_correlated_assets_has_regime_breakdown():
    """The generator must actually deliver the stable->broken regime it claims.

    The correlation-stability example is only meaningful if the first half is
    genuinely more correlated than the second. Assert that behavioral contract,
    not just the type — if the regime logic is broken later, this fails loudly.
    """
    import pandas as pd

    data = _load("_synthetic")
    df = data.correlated_assets()

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]
    assert len(df) > 0

    half = len(df) // 2
    stable_corr = df["a"].iloc[:half].corr(df["b"].iloc[:half])
    broken_corr = df["a"].iloc[half:].corr(df["b"].iloc[half:])
    assert stable_corr > broken_corr
