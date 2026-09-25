"""Smoke tests for the walkthrough CLI (reduced resampling, clearly recorded)."""

from __future__ import annotations

import json

import pytest

from research.feature_audit import demo


def _strict(path):
    def reject(token):
        raise ValueError(token)

    return json.loads(path.read_text(), parse_constant=reject)


def _run(out, *extra, dataset="synthetic"):
    demo.main(
        [
            "--dataset",
            dataset,
            "--output-dir",
            str(out),
            "--permutations",
            "19",
            "--subsamples",
            "10",
            *extra,
        ]
    )


def test_synthetic_smoke_writes_only_requested_files(tmp_path):
    out = tmp_path / "demo"
    _run(out)
    assert sorted(p.name for p in tmp_path.rglob("*") if p.is_file()) == [
        "demo-synthetic.json",
        "demo-synthetic.md",
    ]
    payload = _strict(out / "demo-synthetic.json")
    cfg = payload["audit"]["metadata"]["configuration"]
    assert cfg["n_permutations"] == 19 and cfg["n_subsamples"] == 10
    assert payload["run"]["reduced_resampling"] is True
    assert payload["run"]["audit_seconds"] > 0 and payload["run"]["baseline_seconds"] >= 0
    md = (out / "demo-synthetic.md").read_text()
    assert "reduced resampling" in md.lower()
    assert "## Baseline" in md and "## Audit report" in md


def test_no_raw_rows_or_labels_in_outputs(tmp_path):
    _run(tmp_path)
    df, _ = demo.synthetic_frame()
    text = (tmp_path / "demo-synthetic.json").read_text()
    text += (tmp_path / "demo-synthetic.md").read_text()
    for value in df["customer_id"].iloc[:20]:
        assert value not in text
    for label in df["plan"].dropna().unique():
        assert str(label) not in text
    assert repr(float(df["tenure"].iloc[0])) not in text


def test_refuses_overwrite_without_flag(tmp_path):
    _run(tmp_path)
    before = (tmp_path / "demo-synthetic.json").read_text()
    with pytest.raises(SystemExit, match="overwrite"):
        _run(tmp_path)
    assert (tmp_path / "demo-synthetic.json").read_text() == before
    _run(tmp_path, "--overwrite")


def test_synthetic_demo_shows_expected_mechanisms(tmp_path):
    _run(tmp_path)
    payload = _strict(tmp_path / "demo-synthetic.json")
    status = {f["feature"]: f["value_status"] for f in payload["audit"]["features"]}
    assert status["customer_id"] == "high_cardinality"
    assert status["row_number"] == "possible_identifier"
    assert status["promo_flag"] == "constant"
    assert ["tenure", "tenure_copy"] in payload["audit"]["duplicates"]
    kinds = {(t["feature"], t["kind"]) for t in payload["audit"]["tests"]}
    assert ("last_login_days", "missingness") in kinds
    baseline = {b["feature"]: b for b in payload["baseline"]["features"]}
    assert baseline["tenure"]["raw_mi_bits"] is not None
    assert payload["baseline"]["exact_duplicates"] == [["tenure", "tenure_copy"]]


def test_wine_uses_training_split_only(tmp_path):
    _run(tmp_path, dataset="wine")
    payload = _strict(tmp_path / "demo-wine.json")
    split = payload["dataset"]["split"]
    assert split["train_rows"] == payload["audit"]["metadata"]["n_rows"] == 124
    assert split["holdout_rows"] == 54 and split["holdout_used"] is False
    assert split["random_state"] == 42 and split["stratified"] is True
    assert "Forina" in payload["dataset"]["source_notes"]
    assert all(f["declared_type"] == "continuous" for f in payload["audit"]["features"])


def test_json_has_plain_names_and_source_hash(tmp_path):
    from research.feature_audit.run_validation import source_hash

    _run(tmp_path)
    payload = _strict(tmp_path / "demo-synthetic.json")
    assert payload["run"]["source_hash"] == source_hash()
    text = json.dumps(payload["comparison"])
    assert "customer_id" in text and "&#" not in text
    md = (tmp_path / "demo-synthetic.md").read_text()
    assert "customer&#95;id" in md
