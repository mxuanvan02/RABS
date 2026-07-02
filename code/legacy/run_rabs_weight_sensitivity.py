#!/usr/bin/env python3
"""Weight-sensitivity sweep for RABS family.

Reuses the canonical simulator in `run_rabs_adaptive_bandwidth.py` and
sweeps the five risk-score utility weights (w1..w5) over the configurations
documented in the manuscript. For each configuration, every (network, seed,
window) cell is re-simulated and per-policy objectives are averaged across
both scenarios so the resulting CSV measures robustness to weight choice
under the same canonical run as the main results.

Outputs `outputs/rabs_weight_sensitivity.csv` consumed by
`code/make_sensitivity_table.py`.
"""
from __future__ import annotations
import csv
from pathlib import Path

import run_rabs_adaptive_bandwidth as base


VARIANTS = [
    ("base",          (0.45, 0.20, 0.18, 0.10, 0.07)),
    ("risk_heavy",    (0.55, 0.18, 0.12, 0.08, 0.07)),
    ("aoi_heavy",     (0.35, 0.18, 0.30, 0.10, 0.07)),
    ("channel_heavy", (0.40, 0.18, 0.16, 0.20, 0.06)),
    ("balanced",      (0.30, 0.25, 0.20, 0.15, 0.10)),
]
TARGET_POLICIES = ["rabs_h", "rabs_l", "rabs_pd"]


def patched_risk_score(w):
    w1, w2, w3, w4, w5 = w

    def risk(records, hat, age, pbad):
        ps = [r["p"] for r in records]
        err = [base.proxy_error(records, hat, i) for i in range(3)]
        maxp = max(ps)
        meanp = sum(ps) / 3
        meana = sum(age) / 3
        return (
            w1 * maxp
            + w2 * meanp
            + w3 * min(meana / 8.0, 1.0)
            + w4 * pbad
            + w5 * max(err)
        )

    return risk


def sweep():
    steps = base.read_steps()
    per = 1000
    starts = [i * per for i in range(0, 16) if (i + 1) * per <= len(steps)]
    rows = []
    original = base.risk_score
    try:
        for variant, weights in VARIANTS:
            base.risk_score = patched_risk_score(weights)
            agg = {p: {"obj": [], "loss": [], "B": [], "miss": [], "aoi": []}
                   for p in TARGET_POLICIES}
            for net in base.NETWORKS:
                for start in starts:
                    for seed in base.SEEDS:
                        for pol in TARGET_POLICIES:
                            r = base.run_fixed(steps, pol, net, seed,
                                               start, start + per)
                            agg[pol]["obj"].append(r["objective"])
                            agg[pol]["loss"].append(r["loss_mean"])
                            agg[pol]["B"].append(r["avg_bandwidth"])
                            agg[pol]["miss"].append(r["missed_pct"])
                            agg[pol]["aoi"].append(r["avg_aoi"])
            ws = "/".join(f"{x:g}" for x in weights)
            for pol in TARGET_POLICIES:
                a = agg[pol]
                rows.append({
                    "variant": variant,
                    "weights": ws,
                    "policy": pol,
                    "objective_avg": sum(a["obj"]) / len(a["obj"]),
                    "loss_avg": sum(a["loss"]) / len(a["loss"]),
                    "bandwidth_avg": sum(a["B"]) / len(a["B"]),
                    "missed_avg": sum(a["miss"]) / len(a["miss"]),
                    "aoi_avg": sum(a["aoi"]) / len(a["aoi"]),
                })
    finally:
        base.risk_score = original

    out = base.ROOT / "outputs" / "rabs_weight_sensitivity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(out)


if __name__ == "__main__":
    sweep()
