"""Report serialization, privacy and Markdown rendering (spec section 7)."""

from __future__ import annotations

import dataclasses
import html
import json

import numpy as np
import pandas as pd
import pytest

from research.feature_audit.prototype import AuditReport, audit
from research.feature_audit.prototype.describe import FEATURE_COLUMNS
from research.feature_audit.prototype.association import TEST_COLUMNS
from research.feature_audit.prototype.report import escape_markdown

NASTY = "bad<script>alert(1)</script>|x`y\nz\\*_[a](b)"


def _report():
    y = np.tile([0, 1], 100)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            NASTY: np.where(y == 1, "secret-yes", "secret-no"),
            "gone": [np.nan] * 200,
            "noise": rng.normal(size=200),
            "y": y,
        }
    )
    types = {NASTY: "categorical", "gone": "continuous", "noise": "continuous"}
    return audit(
        df,
        target="y",
        feature_types=types,
        assume_iid=True,
        n_permutations=19,
        n_subsamples=10,
        random_state=0,
    )


def test_escape_is_faithful_and_inert():
    escaped = escape_markdown(NASTY)
    assert html.unescape(escaped) == NASTY
    for bad in ("<", ">", "|", "`", "\n", "\\", "*", "_", "["):
        assert bad not in escaped


def test_markdown_has_no_injected_html_or_table_cells():
    md = _report().to_markdown()
    assert "<script>" not in md
    assert escape_markdown(NASTY) in md
    lines = md.splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("| feature | type"))
    table = []
    for line in lines[header:]:
        if not line.startswith("|"):
            break
        table.append(line)
    assert len(table) == 2 + 3  # header, separator, one row per feature
    assert len({line.count("|") for line in table}) == 1


def test_markdown_sections_and_all_features_present():
    md = _report().to_markdown()
    for heading in (
        "## Assumptions",
        "## Features",
        "## Association tests",
        "## Exact duplicate columns",
        "## Warnings",
        "## Limitations",
    ):
        assert heading in md
    assert "all_missing" in md
    assert "gone" in md and "noise" in md
    assert "interaction" in md.lower()
    assert "secret-yes" not in md


def test_to_dict_is_strict_json_with_nulls_and_no_labels():
    payload = _report().to_dict()
    text = json.dumps(payload, allow_nan=False)
    assert "secret-yes" not in text and "secret-no" not in text
    assert "NaN" not in text and "Infinity" not in text and '"nan"' not in text
    gone = next(f for f in payload["features"] if f["feature"] == "gone")
    assert gone["entropy_bits"] is None
    assert gone["dominant_fraction"] is None
    assert gone["subsample_mi_median"] is None
    assert isinstance(gone["warning_codes"], list)
    assert set(payload) == {"features", "tests", "duplicates", "warnings", "metadata"}
    assert list(payload["features"][0]) == FEATURE_COLUMNS
    assert list(payload["tests"][0]) == TEST_COLUMNS


def test_report_retains_only_aggregate_fields():
    report = _report()
    assert isinstance(report, AuditReport)
    assert [f.name for f in dataclasses.fields(report)] == [
        "features",
        "tests",
        "duplicates",
        "warnings",
        "metadata",
    ]
    assert list(report.features.columns) == FEATURE_COLUMNS
    assert list(report.tests.columns) == TEST_COLUMNS
    assert isinstance(report.features.index, pd.RangeIndex)
    meta_text = json.dumps(report.to_dict()["metadata"])
    assert "secret" not in meta_text


def test_serializers_return_copies():
    report = _report()
    payload = report.to_dict()
    payload["features"][0]["warning_codes"].append("tampered")
    payload["metadata"]["configuration"]["n_permutations"] = -1
    payload["duplicates"].append(["a", "b"])
    again = report.to_dict()
    assert "tampered" not in again["features"][0]["warning_codes"]
    assert again["metadata"]["configuration"]["n_permutations"] == 19
    assert again["duplicates"] == []


def _render(md):
    markdown_it = pytest.importorskip("markdown_it")
    return markdown_it.MarkdownIt("commonmark").enable("table").enable("strikethrough").render(md)


def test_snake_case_target_renders_faithfully():
    df = pd.DataFrame({"x": np.linspace(0, 1, 200), "churn_flag": np.tile([0, 1], 100)})
    md = audit(
        df,
        target="churn_flag",
        feature_types={"x": "continuous"},
        assume_iid=True,
        n_permutations=19,
        n_subsamples=10,
        random_state=0,
    ).to_markdown()
    rendered = _render(md)
    assert "&amp;#95;" not in rendered
    assert "churn_flag" in html.unescape(rendered)


def _dup_report_md(name):
    y = np.tile([0, 1], 100)
    values = np.where(y == 1, "u", "v")
    df = pd.DataFrame({name: values, "dup": values, "y": y})
    return audit(
        df,
        target="y",
        feature_types={name: "categorical", "dup": "categorical"},
        assume_iid=True,
        n_permutations=19,
        n_subsamples=10,
        random_state=0,
    ).to_markdown()


@pytest.mark.parametrize(
    "name", ["# heading", "1. item", "~~struck~~", "https://evil.example/x", "+ plus", "- dash"]
)
def test_block_markers_and_autolinks_in_names_are_inert(name):
    md = _dup_report_md(name)
    assert "https://evil" not in md
    rendered = _render(md)
    reference = _render(_dup_report_md("plainname"))
    for tag in ("<h1>", "<h2>", "<ol", "<ul>", "<li>", "<s>", "<del>", "<a ", "<code>"):
        assert rendered.count(tag) == reference.count(tag), tag
    assert name in html.unescape(rendered)
    assert html.unescape(escape_markdown(name)) == name


@pytest.mark.parametrize("name", ["o'brien", 'say "hi"', "a&b", "&#35;literal", "x;y"])
def test_escape_round_trips_quotes_and_ampersands(name):
    assert html.unescape(escape_markdown(name)) == name
