"""Frozen technical validation of the feature-audit prototype (see PROTOCOL.md).

Usage (from the repository root)::

    python -m research.feature_audit.run_validation --case all --seeds 7000:7100 \
        --resume --out research/feature_audit/artifacts/technical.json
    python -m research.feature_audit.run_validation --summarize \
        research/feature_audit/artifacts/technical.json
    python -m research.feature_audit.run_validation --resource-check \
        --out research/feature_audit/artifacts/resource.json

Records hold aggregate outcomes only, never generated series. Output is
checkpointed after every (case, seed) and keyed by config and source hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from research.feature_audit.prototype import audit

HERE = Path(__file__).resolve().parent
CASE_IDS = {"null": 1, "nonlinear": 2, "resource": 3}
ACCEPTANCE_CASES = ("null", "nonlinear")
DEVELOPMENT_SEEDS = range(6000, 6020)
HELDOUT_SEEDS = range(7000, 7100)

OFFICIAL_CONFIG = {
    "label": "official",
    "n_rows": 400,
    "n_permutations": 1999,
    "n_subsamples": 30,
    "random_state": 0,
}
TEST_CONFIG = {
    "label": "test",
    "n_rows": 400,
    "n_permutations": 19,
    "n_subsamples": 10,
    "random_state": 0,
}
NULL_MAX_REJECTIONS = 10
NULL_CI_LOWER_MAX = 0.05
SIGNAL_MIN_RECALL = 90


def _rng(case: str, seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), CASE_IDS[case]]))


def make_case(case: str, seed: int) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Deterministic synthetic fixture; one independent stream per case and seed."""
    rng = _rng(case, seed)
    if case == "null":
        n = 400
        cols = {f"c{i}": rng.normal(size=n) for i in range(5)}
        cols.update({f"k{i}": rng.permutation(np.tile([0, 1, 2, 3], n // 4)) for i in range(5)})
        cols["y"] = rng.permutation(np.tile([0, 1], n // 2))
        types = {f"c{i}": "continuous" for i in range(5)}
        types.update({f"k{i}": "categorical" for i in range(5)})
    elif case == "nonlinear":
        base = np.r_[np.arange(1, 201) / 100, -np.arange(1, 201) / 100]
        x = rng.permutation(base)
        n = len(x)
        cols = {"x": x, "x_copy": x.copy(), "noise": rng.normal(size=n)}
        cols.update({f"cat{i}": rng.permutation(np.tile([0, 1, 2, 3], n // 4)) for i in range(6)})
        cols["id"] = [f"id-{i}" for i in rng.permutation(n)]
        cols["y"] = (np.abs(x) > 1).astype(int)
        types = {"x": "continuous", "x_copy": "continuous", "noise": "continuous"}
        types.update({f"cat{i}": "categorical" for i in range(6)})
        types["id"] = "categorical"
    elif case == "resource":
        n = 1000
        cols = {f"c{i}": rng.normal(size=n) for i in range(5)}
        cols.update({f"k{i}": rng.permutation(np.tile([0, 1, 2, 3], n // 4)) for i in range(5)})
        cols["y"] = rng.permutation(np.tile([0, 1], n // 2))
        types = {f"c{i}": "continuous" for i in range(5)}
        types.update({f"k{i}": "categorical" for i in range(5)})
    else:
        raise ValueError(f"unknown case {case!r}")
    return pd.DataFrame(cols), types


def source_hash() -> str:
    """SHA-256 over the prototype sources and this runner."""
    digest = hashlib.sha256()
    for path in sorted((HERE / "prototype").glob("*.py")) + [HERE / "run_validation.py"]:
        digest.update(path.relative_to(HERE).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def partition(seed: int) -> str:
    if seed in DEVELOPMENT_SEEDS:
        return "development"
    if seed in HELDOUT_SEEDS:
        return "heldout"
    return "other"


def run_case(case: str, seed: int, config: dict) -> dict:
    """Audit one generated dataset and return aggregate outcomes only."""
    df, types = make_case(case, seed)
    start = time.perf_counter()
    report = audit(
        df,
        target="y",
        feature_types=types,
        assume_iid=True,
        n_permutations=config["n_permutations"],
        n_subsamples=config["n_subsamples"],
        random_state=config["random_state"],
    )
    elapsed = time.perf_counter() - start
    tests = report.to_dict()["tests"]
    features = report.to_dict()["features"]
    rejections = [t for t in tests if t["assessment"] == "above_permutation_null"]
    record = {
        "case": case,
        "seed": int(seed),
        "partition": partition(seed),
        "elapsed_s": elapsed,
        "n_tests": len(tests),
        "n_rejections": len(rejections),
        "any_rejection": bool(rejections),
        "rejected": [[t["feature"], t["kind"]] for t in rejections],
        "value_status": {f["feature"]: f["value_status"] for f in features},
        "duplicates": report.to_dict()["duplicates"],
        "tests": [
            {
                k: t[k]
                for k in (
                    "feature",
                    "kind",
                    "n",
                    "mi_bits",
                    "null_median_bits",
                    "p_value",
                    "p_holm",
                    "assessment",
                )
            }
            for t in tests
        ],
    }
    if case == "nonlinear":
        x_test = next((t for t in tests if t["feature"] == "x" and t["kind"] == "value"), None)
        record.update(
            x_pearson_r=float(np.corrcoef(df["x"], df["y"])[0, 1]),
            x_mi_bits=None if x_test is None else x_test["mi_bits"],
            x_p_holm=None if x_test is None else x_test["p_holm"],
            x_above_null=bool(x_test and x_test["assessment"] == "above_permutation_null"),
            id_value_status=record["value_status"]["id"],
            duplicate_pair_reported=["x", "x_copy"] in record["duplicates"],
        )
    return record


def clopper_pearson(k: int, n: int) -> Tuple[float, float]:
    """Exact two-sided 95% binomial interval."""
    lower = 0.0 if k == 0 else float(stats.beta.ppf(0.025, k, n - k + 1))
    upper = 1.0 if k == n else float(stats.beta.ppf(0.975, k + 1, n - k))
    return lower, upper


def _heldout(records: List[dict], case: str) -> List[dict]:
    return [r for r in records if r["case"] == case and r["seed"] in HELDOUT_SEEDS]


def null_gate(records: List[dict]) -> dict:
    rows = [r for r in records if r["case"] == "null"]
    n = len(rows)
    k = sum(bool(r["any_rejection"]) for r in rows)
    lower, upper = clopper_pearson(k, n) if n else (None, None)
    complete = n == len(HELDOUT_SEEDS)
    passed = complete and k <= NULL_MAX_REJECTIONS and lower <= NULL_CI_LOWER_MAX
    return {
        "datasets": n,
        "complete": complete,
        "any_rejection_count": k,
        "rate": k / n if n else None,
        "ci_lower": lower,
        "ci_upper": upper,
        "passed": bool(passed),
    }


def signal_gate(records: List[dict]) -> dict:
    rows = [r for r in records if r["case"] == "nonlinear"]
    n = len(rows)
    recall = sum(r["x_above_null"] for r in rows)
    id_excluded = sum(r["id_value_status"] != "eligible" for r in rows)
    dup = sum(r["duplicate_pair_reported"] for r in rows)
    complete = n == len(HELDOUT_SEEDS)
    return {
        "datasets": n,
        "complete": complete,
        "x_above_null": recall,
        "id_excluded": id_excluded,
        "duplicate_reported": dup,
        "missed_seeds": [r["seed"] for r in rows if not r["x_above_null"]],
        "max_abs_pearson_r": max((abs(r["x_pearson_r"]) for r in rows), default=None),
        "passed": bool(complete and recall >= SIGNAL_MIN_RECALL and id_excluded == n and dup == n),
    }


def summarize(payload: dict) -> dict:
    records = payload["records"]
    null_rows = _heldout(records, "null")
    signal_rows = _heldout(records, "nonlinear")
    times = [r["elapsed_s"] for r in records]
    null_result = null_gate(null_rows)
    signal_result = signal_gate(signal_rows)
    return {
        "config": payload["config"],
        "source_hash": payload["source_hash"],
        "record_count": len(records),
        "null": null_result,
        "null_false_positive_seeds": [r["seed"] for r in null_rows if r["any_rejection"]],
        "signal": signal_result,
        "elapsed_s": {
            "total": float(np.sum(times)) if times else 0.0,
            "median": float(np.median(times)) if times else None,
            "max": float(np.max(times)) if times else None,
        },
        "technical_gate_passed": bool(null_result["passed"] and signal_result["passed"]),
    }


def _write(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, allow_nan=False, sort_keys=False) + "\n")
    os.replace(tmp, path)


def _environment() -> dict:
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }


def resource_check(out: Path) -> dict:
    """Time and peak-memory one default audit at n=1000 in this (dedicated) process."""
    import resource

    df, types = make_case("resource", 0)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    audit(df, target="y", feature_types=types, assume_iid=True)
    elapsed = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024  # macOS bytes, Linux KiB
    result = {
        "n_rows": len(df),
        "predictors": "5 continuous, 5 categorical",
        "configuration": "defaults (B=1999, 30 subsamples, one worker)",
        "elapsed_s": elapsed,
        "ru_maxrss_before_bytes": before * scale,
        "ru_maxrss_after_bytes": after * scale,
        "incremental_peak_bytes": (after - before) * scale,
        "ru_maxrss_unit_note": "ru_maxrss is bytes on macOS and KiB on Linux; "
        "converted to bytes here. Incremental = high-water mark after minus before.",
        "time_target_s": 300,
        "memory_target_bytes": 1024**3,
        "environment": _environment(),
        "source_hash": source_hash(),
    }
    result["passed"] = bool(elapsed < 300 and result["incremental_peak_bytes"] < 1024**3)
    _write(out, result)
    return result


def _parse_seeds(text: str) -> range:
    start, stop = (int(v) for v in text.split(":"))
    if stop <= start:
        raise argparse.ArgumentTypeError("seed range must be START:STOP with STOP > START")
    return range(start, stop)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", choices=["null", "nonlinear", "all"], default="all")
    parser.add_argument("--seeds", type=_parse_seeds, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--test-config",
        action="store_true",
        help="low-B configuration for tests; never an acceptance run",
    )
    parser.add_argument("--summarize", type=Path)
    parser.add_argument("--resource-check", action="store_true")
    args = parser.parse_args(argv)

    if args.summarize:
        payload = json.loads(args.summarize.read_text())
        print(json.dumps(summarize(payload), indent=1))
        return
    if args.out is None:
        parser.error("--out is required")
    if args.resource_check:
        print(json.dumps(resource_check(args.out), indent=1))
        return
    if args.seeds is None:
        parser.error("--seeds is required")

    config = TEST_CONFIG if args.test_config else OFFICIAL_CONFIG
    cases = list(ACCEPTANCE_CASES) if args.case == "all" else [args.case]
    wanted = [(c, s) for c in cases for s in args.seeds]
    src = source_hash()
    cfg_hash = config_hash(config)

    if args.out.exists():
        if not args.resume:
            raise SystemExit(f"{args.out} exists; pass --resume to continue it")
        payload = json.loads(args.out.read_text())
        if payload["config_hash"] != cfg_hash:
            raise SystemExit("resume refused: config hash differs from the stored run")
        if payload["source_hash"] != src:
            raise SystemExit("resume refused: source hash differs from the stored run")
        libraries = ("python", "numpy", "pandas", "scipy", "sklearn")
        current = _environment()
        stored = payload.get("environment", {})
        if any(stored.get(k) != current[k] for k in libraries):
            raise SystemExit(
                "resume refused: Python or library versions differ from the stored run"
            )
    else:
        payload = {
            "protocol": "research/feature_audit/PROTOCOL.md",
            "config": config,
            "config_hash": cfg_hash,
            "source_hash": src,
            "environment": _environment(),
            "records": [],
        }
    done = {(r["case"], r["seed"]) for r in payload["records"]}
    requested = sorted(set(payload.get("requested", [])) | {f"{c}:{s}" for c, s in wanted})
    payload["requested"] = requested
    payload["expected_records"] = len(requested)
    payload["complete"] = False
    todo = [(c, s) for c, s in wanted if (c, s) not in done]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write(args.out, payload)

    def store(record):
        payload["records"].append(record)
        payload["records"].sort(key=lambda r: (r["case"], r["seed"]))
        _write(args.out, payload)

    for record in _execute(todo, config, args.workers):
        record["source_hash"] = src
        record["config_hash"] = cfg_hash
        store(record)
        print(
            f"{record['case']} {record['seed']}: rejections={record['n_rejections']} "
            f"{record['elapsed_s']:.1f}s",
            flush=True,
        )

    if source_hash() != src:
        _write(args.out, payload)
        raise SystemExit("source files changed during the run; results are not valid")
    done = {f"{r['case']}:{r['seed']}" for r in payload["records"]}
    payload["complete"] = set(requested) <= done
    _write(args.out, payload)


def _execute(todo, config, workers):
    if workers <= 1:
        for case, seed in todo:
            yield run_case(case, seed, config)
        return
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_case, case, seed, config) for case, seed in todo]
        for future in as_completed(futures):
            yield future.result()


if __name__ == "__main__":
    main()
