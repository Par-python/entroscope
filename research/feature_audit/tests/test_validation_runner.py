"""Tests for the frozen technical-validation runner (spec section 8).

These use a two-seed, low-B TEST configuration that is explicitly distinct
from the official acceptance configuration; no 100-seed run happens here.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from research.feature_audit import run_validation as rv
from research.feature_audit.run_validation import make_case


def test_nonlinear_fixture_has_zero_linear_correlation():
    df, types = make_case("nonlinear", 6000)
    assert len(df) == 400
    assert abs(np.corrcoef(df["x"], df["y"])[0, 1]) < 1e-12
    pd.testing.assert_series_equal(df["x"], df["x_copy"], check_names=False)
    assert types["id"] == "categorical"


def test_nonlinear_fixture_mechanism():
    df, types = make_case("nonlinear", 6001)
    assert len(types) == 10 and "y" not in types
    np.testing.assert_allclose(np.sort(np.abs(df["x"]))[::2], np.arange(1, 201) / 100)
    assert (df["y"] == (df["x"].abs() > 1)).all()
    assert df["y"].sum() == 200
    assert df["id"].is_unique
    assert sum(t == "categorical" for t in types.values()) == 7


def test_null_fixture_shape_balance_and_no_missingness():
    df, types = make_case("null", 7000)
    assert df.shape == (400, 11)
    assert df["y"].value_counts().tolist() == [200, 200]
    assert not df.isna().any().any()
    assert sorted(types.values()) == ["categorical"] * 5 + ["continuous"] * 5
    for name, kind in types.items():
        if kind == "categorical":
            assert df[name].value_counts().tolist() == [100] * 4


def test_generator_is_deterministic_and_case_streams_differ():
    a, _ = make_case("null", 6005)
    b, _ = make_case("null", 6005)
    c, _ = make_case("null", 6006)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)
    n, _ = make_case("nonlinear", 6005)
    assert not np.allclose(a["c0"].to_numpy(), n["noise"].to_numpy())


def test_seed_partitions_do_not_overlap():
    assert set(rv.DEVELOPMENT_SEEDS).isdisjoint(rv.HELDOUT_SEEDS)
    assert list(rv.HELDOUT_SEEDS) == list(range(7000, 7100))
    assert list(rv.DEVELOPMENT_SEEDS) == list(range(6000, 6020))


def test_official_config_is_frozen_to_spec():
    cfg = rv.OFFICIAL_CONFIG
    assert cfg["n_permutations"] == 1999 and cfg["n_subsamples"] == 30
    assert cfg["random_state"] == 0 and cfg["n_rows"] == 400
    assert rv.TEST_CONFIG["label"] == "test" and rv.TEST_CONFIG != cfg


def test_clopper_pearson_endpoints():
    assert rv.clopper_pearson(0, 100) == (0.0, pytest.approx(stats.beta.ppf(0.975, 1, 100)))
    assert rv.clopper_pearson(100, 100) == (pytest.approx(stats.beta.ppf(0.025, 100, 1)), 1.0)
    lo, hi = rv.clopper_pearson(10, 100)
    assert lo == pytest.approx(stats.beta.ppf(0.025, 10, 91))
    assert hi == pytest.approx(stats.beta.ppf(0.975, 11, 90))


def _null_records(k, n=100):
    return [
        {"case": "null", "seed": 7000 + i, "any_rejection": i < k, "elapsed_s": 1.0}
        for i in range(n)
    ]


def test_null_gate_logic():
    assert rv.null_gate(_null_records(10))["passed"] is True
    assert rv.null_gate(_null_records(11))["passed"] is False
    zero = rv.null_gate(_null_records(0))
    assert zero["passed"] is True  # conservative results must not fail
    assert zero["ci_lower"] == 0.0
    incomplete = rv.null_gate(_null_records(0, n=99))
    assert incomplete["passed"] is False and incomplete["complete"] is False


def _run(tmp_path, *extra, name="out.json"):
    out = tmp_path / name
    rv.main(["--case", "null", "--seeds", "6000:6002", "--test-config", "--out", str(out), *extra])
    return out


def _strict_load(path):
    def reject(token):
        raise ValueError(token)

    return json.loads(path.read_text(), parse_constant=reject)


def test_runner_writes_strict_json_without_raw_data(tmp_path):
    payload = _strict_load(_run(tmp_path))
    assert payload["config"]["label"] == "test"
    assert payload["complete"] is True
    assert payload["expected_records"] == 2
    assert [r["seed"] for r in payload["records"]] == [6000, 6001]
    for rec in payload["records"]:
        assert rec["partition"] == "development"
        assert rec["source_hash"] == payload["source_hash"]
        assert isinstance(rec["any_rejection"], bool)
    text = json.dumps(payload)
    df, _ = make_case("null", 6000)
    assert f"{df['c0'].iloc[0]!r}" not in text


def test_resume_skips_done_and_rejects_changed_config(tmp_path, monkeypatch):
    out = _run(tmp_path)
    calls = []
    real = rv.run_case
    monkeypatch.setattr(rv, "run_case", lambda *a, **k: calls.append(a) or real(*a, **k))
    rv.main(
        ["--case", "null", "--seeds", "6000:6003", "--test-config", "--resume", "--out", str(out)]
    )
    assert [a[1] for a in calls] == [6002]
    with pytest.raises(SystemExit, match="config"):
        rv.main(["--case", "null", "--seeds", "6000:6002", "--resume", "--out", str(out)])


def test_resume_rejects_changed_source(tmp_path, monkeypatch):
    out = _run(tmp_path)
    monkeypatch.setattr(rv, "source_hash", lambda: "different")
    with pytest.raises(SystemExit, match="source"):
        rv.main(
            [
                "--case",
                "null",
                "--seeds",
                "6000:6002",
                "--test-config",
                "--resume",
                "--out",
                str(out),
            ]
        )


def test_existing_output_requires_resume(tmp_path):
    out = _run(tmp_path)
    with pytest.raises(SystemExit, match="resume"):
        _run(tmp_path)
    assert out.exists()


def test_interrupted_run_is_marked_incomplete(tmp_path, monkeypatch):
    real = rv.run_case

    def flaky(case, seed, config):
        if seed == 6001:
            raise KeyboardInterrupt
        return real(case, seed, config)

    monkeypatch.setattr(rv, "run_case", flaky)
    out = tmp_path / "partial.json"
    with pytest.raises(KeyboardInterrupt):
        rv.main(["--case", "null", "--seeds", "6000:6002", "--test-config", "--out", str(out)])
    payload = _strict_load(out)
    assert payload["complete"] is False
    assert len(payload["records"]) == 1


def test_source_change_during_run_is_detected(tmp_path, monkeypatch):
    hashes = iter(["before", "after", "after"])
    monkeypatch.setattr(rv, "source_hash", lambda: next(hashes))
    with pytest.raises(SystemExit, match="changed during"):
        _run(tmp_path)
    assert _strict_load(tmp_path / "out.json")["complete"] is False


def test_nonlinear_record_fields(tmp_path):
    out = tmp_path / "nl.json"
    rv.main(["--case", "nonlinear", "--seeds", "6000:6001", "--test-config", "--out", str(out)])
    rec = _strict_load(out)["records"][0]
    assert rec["id_value_status"] == "high_cardinality"
    assert rec["duplicate_pair_reported"] is True
    assert abs(rec["x_pearson_r"]) < 1e-12
    assert rec["x_mi_bits"] > 0.5
    assert {"x_above_null", "x_p_holm", "tests", "elapsed_s"} <= set(rec)


def test_resume_rejects_changed_library_versions(tmp_path, monkeypatch):
    out = _run(tmp_path)
    real = rv._environment()
    monkeypatch.setattr(rv, "_environment", lambda: {**real, "sklearn": "0.0.0"})
    with pytest.raises(SystemExit, match="librar"):
        rv.main(
            [
                "--case",
                "null",
                "--seeds",
                "6000:6002",
                "--test-config",
                "--resume",
                "--out",
                str(out),
            ]
        )
