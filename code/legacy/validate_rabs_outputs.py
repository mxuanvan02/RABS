#!/usr/bin/env python3
"""Validate RABS simulation outputs and write a reproducibility manifest."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "source_rows": 48360,
    "loops_per_step": 3,
    "raw_rows": 1600,
    "summary_rows": 20,
    "holdout_rows": 20,
    "pairwise_rows": 72,
}
FILES = {
    "source": ROOT / "data/source/safety_probability_calibration_raw.csv",
    "script_main": ROOT / "code/run_rabs_adaptive_bandwidth.py",
    "script_pairwise": ROOT / "code/analyze_rabs_pairwise.py",
    "script_validate": ROOT / "code/validate_rabs_outputs.py",
    "raw": ROOT / "outputs/rabs/rabs_raw.csv",
    "summary": ROOT / "outputs/rabs/rabs_summary.csv",
    "holdout": ROOT / "outputs/rabs/rabs_holdout_summary.csv",
    "pairwise": ROOT / "outputs/rabs/rabs_pairwise_deltas.csv",
    "table_summary": ROOT / "outputs/tables/table_rabs_summary.tex",
    "table_holdout": ROOT / "outputs/tables/table_rabs_holdout_summary.tex",
    "table_oracle_gap": ROOT / "outputs/tables/table_rabs_oracle_gap.tex",
    "table_pairwise": ROOT / "outputs/tables/table_rabs_pairwise_deltas.tex",
}
MANIFEST = ROOT / "outputs/rabs/run_manifest.json"
POLICIES = {
    "fixed_b1",
    "fixed_b2",
    "fixed_b3",
    "aoi_adaptive",
    "risk_adaptive",
    "burst_adaptive",
    "rabs_h",
    "rabs_l",
    "rabs_pd",
    "oracle_b",
}
NETWORKS = {"burst", "severe_burst"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def assert_true(cond: bool, message: str, errors: list[str]) -> None:
    if not cond:
        errors.append(message)


def main() -> None:
    errors: list[str] = []
    for name, path in FILES.items():
        assert_true(path.exists(), f"Missing {name}: {path}", errors)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    source = read_csv(FILES["source"])
    raw = read_csv(FILES["raw"])
    summary = read_csv(FILES["summary"])
    holdout = read_csv(FILES["holdout"])
    pairwise = read_csv(FILES["pairwise"])

    assert_true(len(source) == EXPECTED["source_rows"], f"source rows {len(source)} != {EXPECTED['source_rows']}", errors)
    assert_true(len(raw) == EXPECTED["raw_rows"], f"raw rows {len(raw)} != {EXPECTED['raw_rows']}", errors)
    assert_true(len(summary) == EXPECTED["summary_rows"], f"summary rows {len(summary)} != {EXPECTED['summary_rows']}", errors)
    assert_true(len(holdout) == EXPECTED["holdout_rows"], f"holdout rows {len(holdout)} != {EXPECTED['holdout_rows']}", errors)
    assert_true(len(pairwise) == EXPECTED["pairwise_rows"], f"pairwise rows {len(pairwise)} != {EXPECTED['pairwise_rows']}", errors)

    step_counts: dict[tuple[str, str], int] = {}
    for r in source:
        key = (r["window"], r["t"])
        step_counts[key] = step_counts.get(key, 0) + 1
    bad_steps = [k for k, v in step_counts.items() if v != EXPECTED["loops_per_step"]]
    assert_true(not bad_steps, f"{len(bad_steps)} source time steps do not have 3 loops", errors)

    raw_units = {(r["network"], r["window_start"], r["seed"], r["policy"]) for r in raw}
    assert_true(len(raw_units) == len(raw), "raw output has duplicate scenario-policy rows", errors)
    assert_true({r["network"] for r in raw} == NETWORKS, "raw networks mismatch", errors)
    assert_true({r["policy"] for r in raw} == POLICIES, "raw policies mismatch", errors)

    numeric_fields = [
        "objective",
        "loss_mean",
        "avg_bandwidth",
        "missed_pct",
        "avg_aoi",
        "packet_success_pct",
    ]
    for r in raw:
        for field in numeric_fields:
            value = float(r[field])
            assert_true(value == value and value not in (float("inf"), float("-inf")), f"non-finite {field}", errors)
            assert_true(value >= 0, f"negative {field}", errors)

    by_summary = {(r["network"], r["policy"]): r for r in summary}
    for net in NETWORKS:
        pd = by_summary[(net, "rabs_pd")]
        b3 = by_summary[(net, "fixed_b3")]
        assert_true(float(pd["avg_bandwidth_mean"]) < float(b3["avg_bandwidth_mean"]), f"RABS-PD does not save bandwidth in {net}", errors)
        assert_true(float(pd["objective_mean"]) < float(b3["objective_mean"]), f"RABS-PD objective not below Fixed-B3 in {net}", errors)

    pair_lookup = {(r["network"], r["baseline"], r["metric"]): r for r in pairwise}
    for net in NETWORKS:
        row = pair_lookup[(net, "fixed_b3", "objective")]
        assert_true(float(row["delta_mean"]) < 0, f"RABS-PD objective delta vs Fixed-B3 not negative in {net}", errors)
        assert_true(int(row["wins"]) > int(row["losses"]), f"RABS-PD objective wins not dominant vs Fixed-B3 in {net}", errors)

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "expected": EXPECTED,
        "row_counts": {
            "source": len(source),
            "raw": len(raw),
            "summary": len(summary),
            "holdout": len(holdout),
            "pairwise": len(pairwise),
        },
        "hashes": {name: sha256(path) for name, path in FILES.items()},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("RABS validation passed.")
    print(MANIFEST)


if __name__ == "__main__":
    main()
