"""Aggregate audit result and its JSON/Markdown serializers (spec section 7).

The report holds aggregate tables only: no input frame, target array, row
index, category label or example record. Nothing here writes files.
"""

from __future__ import annotations

import copy
import html
import math
import numbers
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd

LIMITATIONS = [
    "Run this on training data only; results describe this sample, not a population "
    "or a deployment setting.",
    "Rows are assumed independent (acknowledged via assume_iid=True, not tested). "
    "Repeated subjects, groups or time order invalidate the permutation null.",
    "Value tests use rows where the feature is observed; missingness tests use all "
    "rows. These are different populations, so scores are not a universal ranking "
    "across features.",
    "'not_distinguished_from_null' is not evidence that a feature is useless: power "
    "can be low and marginal tests miss interactions.",
    "Interaction blind spot: each feature is tested alone against the target. "
    "Features that matter only jointly (for example XOR) can show zero individual MI.",
    "No causal claim, leakage verdict, accuracy guarantee or column-removal advice is "
    "made. Exclusion codes are conservative policies, and duplicate pairs are facts "
    "about values, not recommendations.",
    "Holm adjustment controls the family-wise error rate across this report's tests "
    "under exchangeability; adjusted p-values are not FDR q-values and change when "
    "the family of tests changes.",
    "subsample_mi_p10/p90 summarize sensitivity to stratified 80% subsamples "
    "without replacement; they are not confidence intervals.",
    "Continuous MI uses scikit-learn's k-nearest-neighbour estimator (k=3), which "
    "assumes genuinely continuous values; categorical MI is a plug-in estimate that "
    "is biased upward when categories are many relative to rows.",
    "possible_identifier only catches unique values that are strictly monotone in row "
    "order; it can flag a legitimate measurement, and a shuffled numeric ID cannot be "
    "recognized reliably.",
]

_EXTRA_ESCAPES = {
    "|": "&#124;",
    "`": "&#96;",
    "\\": "&#92;",
    "*": "&#42;",
    "_": "&#95;",
    "[": "&#91;",
    "]": "&#93;",
    "\n": "&#10;",
    "\r": "&#13;",
    # Block markers, strikethrough and autolink triggers (':' breaks bare URLs).
    "#": "&#35;",
    "~": "&#126;",
    ":": "&#58;",
    "+": "&#43;",
    "-": "&#45;",
    ".": "&#46;",
    "!": "&#33;",
}


def escape_markdown(text: Any) -> str:
    """Escape HTML and Markdown-significant characters as reversible entities."""
    return "".join(_EXTRA_ESCAPES.get(ch) or html.escape(ch, quote=True) for ch in str(text))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool,)) or type(value).__name__ == "bool_":
        return bool(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    raise TypeError(f"unexpected value in report: {type(value).__name__}")


def _fmt(value, digits: int = 4) -> str:
    if value is None or value is pd.NA:
        return "—"
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        if not math.isfinite(float(value)):
            return "—"
        return f"{float(value):.{digits}g}"
    return escape_markdown(value)


def _table(headers: List[str], rows: List[List[str]]) -> List[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


@dataclass
class AuditReport:
    """Aggregate result of :func:`audit`."""

    features: pd.DataFrame
    tests: pd.DataFrame
    duplicates: List[Tuple[str, str]]
    warnings: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-compatible builtins; unsupported numbers become ``None``."""
        return {
            "features": [_jsonable(r) for r in self.features.to_dict("records")],
            "tests": [_jsonable(r) for r in self.tests.to_dict("records")],
            "duplicates": [[a, b] for a, b in self.duplicates],
            "warnings": list(self.warnings),
            "metadata": _jsonable(copy.deepcopy(self.metadata)),
        }

    def to_markdown(self) -> str:
        """Deterministic plain-text report; names are escaped, no HTML is emitted."""
        meta = self.metadata
        cfg = meta["configuration"]
        out = [
            "# Feature association audit (research prototype)",
            "",
            "## Assumptions",
            "",
            f"- Target {escape_markdown(meta['target_name'])} is treated as a "
            f"classification label with {meta['target_class_count']} classes "
            f"(empirical entropy {_fmt(meta['target_entropy_bits'])} bits).",
            "- Rows were declared independent by the user (assume_iid=True); this was not tested.",
            "- Feature types are the user's explicit declarations; nothing was inferred.",
            f"- {meta['n_rows']} rows; B={cfg['n_permutations']} label permutations "
            f"(minimum attainable raw p = {_fmt(cfg['minimum_attainable_p'])}); "
            f"{cfg['n_subsamples']} stratified {cfg['subsample_fraction']:.0%} subsamples "
            f"without replacement; k={cfg['n_neighbors']}; seed={cfg['random_state']}; "
            f"Holm family size m={meta['n_tests']}; alpha={cfg['alpha']}.",
            "- MI values are raw empirical estimates in bits, shown next to their "
            "shuffled-label null median. Rows keep input order; this is not a ranking.",
            "",
            "## Features",
            "",
        ]
        feature_rows = []
        for r in self.features.to_dict("records"):
            codes = ", ".join(r["warning_codes"]) or "—"
            if r["subsample_valid_count"]:
                stab = (
                    f"{_fmt(r['subsample_mi_median'])} "
                    f"[{_fmt(r['subsample_mi_p10'])}, {_fmt(r['subsample_mi_p90'])}] "
                    f"({r['subsample_valid_count']} valid)"
                )
            else:
                stab = "—"
            feature_rows.append(
                [
                    escape_markdown(r["feature"]),
                    r["declared_type"],
                    f"{r['n_observed']}/{r['n_total']}",
                    _fmt(r["missing_fraction"], 3),
                    _fmt(r["n_unique_observed"]),
                    _fmt(r["dominant_fraction"], 3),
                    _fmt(r["entropy_bits"]),
                    r["value_status"],
                    codes,
                    stab,
                ]
            )
        out += _table(
            [
                "feature",
                "type",
                "observed/total",
                "missing",
                "unique",
                "dominant",
                "entropy (bits)",
                "value status",
                "warning codes",
                "subsample MI median [p10, p90]",
            ],
            feature_rows,
        )
        out += ["", "## Association tests", ""]
        if self.tests.empty:
            out.append("No eligible hypotheses; no p-values were computed.")
        else:
            test_rows = [
                [
                    escape_markdown(r["feature"]),
                    r["kind"],
                    _fmt(r["n"]),
                    _fmt(r["mi_bits"]),
                    _fmt(r["null_median_bits"]),
                    _fmt(r["excess_bits"]),
                    _fmt(r["p_value"]),
                    _fmt(r["p_holm"]),
                    r["assessment"],
                ]
                for r in self.tests.to_dict("records")
            ]
            out += _table(
                [
                    "feature",
                    "hypothesis",
                    "n",
                    "MI (bits)",
                    "null median",
                    "MI − null median",
                    "p",
                    "Holm p",
                    "assessment",
                ],
                test_rows,
            )
        out += ["", "## Exact duplicate columns", ""]
        if self.duplicates:
            out += [
                f"- {escape_markdown(a)} = {escape_markdown(b)} (identical values and "
                "missingness; not a removal recommendation)"
                for a, b in self.duplicates
            ]
        else:
            out.append("None found among same-type, same-dtype pairs.")
        out += ["", "## Warnings", ""]
        # Warnings and limitations are fixed prototype text and never contain names.
        out += [f"- {w}" for w in self.warnings] or ["None."]
        out += ["", "## Limitations", ""]
        out += [f"- {text}" for text in meta["limitations"]]
        return "\n".join(out) + "\n"
