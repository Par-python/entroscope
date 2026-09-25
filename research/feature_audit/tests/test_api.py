"""End-to-end prototype entry point (spec section 5)."""

from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from research.feature_audit.prototype import audit


def test_local_report_is_json_safe_and_does_not_mutate():
    df = pd.DataFrame(
        {
            "plan": ["private-label-a", "private-label-b"] * 100,
            "copy": ["private-label-a", "private-label-b"] * 100,
            "y": [0, 1] * 100,
        }
    )
    original = df.copy(deep=True)
    result = audit(
        df,
        target="y",
        feature_types={"plan": "categorical", "copy": "categorical"},
        assume_iid=True,
        n_permutations=19,
        n_subsamples=10,
        random_state=0,
    )
    payload = json.dumps(result.to_dict(), allow_nan=False)
    assert "private-label-a" not in payload
    assert "private-label-b" not in payload
    assert result.duplicates == [("plan", "copy")]
    pd.testing.assert_frame_equal(df, original)


def test_signature_defaults_match_spec():
    sig = inspect.signature(audit)
    assert sig.parameters["assume_iid"].default is False
    assert sig.parameters["n_permutations"].default == 1999
    assert sig.parameters["n_subsamples"].default == 30
    assert sig.parameters["random_state"].default == 0
    for name in ("target", "feature_types"):
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_iid_acknowledgment_required_by_default():
    df = pd.DataFrame({"x": np.linspace(0, 1, 200), "y": [0, 1] * 100})
    with pytest.raises(ValueError, match="independent"):
        audit(df, target="y", feature_types={"x": "continuous"})


def _mixed():
    rng = np.random.default_rng(5)
    y = np.tile([0, 1, 2], 100)[:300]
    x = rng.normal(size=300) + y
    miss = pd.Series(rng.normal(size=300))
    miss[(y == 2) & (rng.random(300) < 0.6)] = np.nan
    df = pd.DataFrame(
        {
            "signal": x,
            "noise_cat": rng.integers(0, 3, 300).astype(str),
            "with_missing": miss,
            "row_id": np.arange(300.0),
            "label": np.array(["cls-alpha", "cls-beta", "cls-gamma"])[y],
        }
    )
    types = {
        "signal": "continuous",
        "noise_cat": "categorical",
        "with_missing": "continuous",
        "row_id": "continuous",
    }
    return df, types


def test_end_to_end_metadata_and_accounting():
    df, types = _mixed()
    report = audit(
        df,
        target="label",
        feature_types=types,
        assume_iid=True,
        n_permutations=19,
        n_subsamples=10,
        random_state=1,
    )
    assert list(report.features["feature"]) == list(types)
    status = dict(zip(report.features["feature"], report.features["value_status"]))
    assert status["row_id"] == "possible_identifier"
    meta = report.metadata
    assert meta["schema_version"] == "0.1"
    assert meta["target_name"] == "label"
    assert meta["n_rows"] == 300
    assert meta["target_class_count"] == 3
    assert meta["target_entropy_bits"] == pytest.approx(np.log2(3))
    assert meta["n_tests"] == len(report.tests)
    assert meta["assume_iid"] is True
    cfg = meta["configuration"]
    assert cfg["n_permutations"] == 19 and cfg["n_subsamples"] == 10
    assert cfg["random_state"] == 1
    assert cfg["minimum_attainable_p"] == pytest.approx(1 / 20)
    assert {"python", "numpy", "pandas", "scipy", "sklearn"} <= set(meta["versions"])
    assert len(meta["limitations"]) >= 6
    kinds = set(zip(report.tests["feature"], report.tests["kind"]))
    assert ("with_missing", "missingness") in kinds
    assert ("row_id", "value") not in kinds
    assert any("Reduced permutation count" in w for w in report.warnings)
    assert any("missingness" in w.lower() for w in report.warnings)
    assert "cls-" not in json.dumps(report.to_dict())


def test_deterministic_reruns():
    df, types = _mixed()
    kwargs = dict(
        target="label",
        feature_types=types,
        assume_iid=True,
        n_permutations=19,
        n_subsamples=10,
        random_state=3,
    )
    assert audit(df, **kwargs).to_dict() == audit(df, **kwargs).to_dict()


def test_insufficient_subsample_support_warning_does_not_change_status():
    x = np.r_[np.random.default_rng(0).normal(size=124), [np.nan] * 76]
    df = pd.DataFrame({"x": x, "y": np.tile([0, 1], 100)})
    report = audit(
        df,
        target="y",
        feature_types={"x": "continuous"},
        assume_iid=True,
        n_permutations=19,
        n_subsamples=10,
        random_state=0,
    )
    row = report.features.iloc[0]
    assert row["value_status"] == "eligible"
    assert "insufficient_subsample_support" in row["warning_codes"]
    assert row["subsample_valid_count"] == 0
