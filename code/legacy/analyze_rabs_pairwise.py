#!/usr/bin/env python3
"""Pairwise statistical evidence for the RABS bandwidth-control simulation.

The script reads the deterministic raw simulation output and compares RABS-PD
against key baselines using paired scenario units: network, window_start, and seed.
It intentionally uses only the Python standard library so the evidence can be
regenerated on a minimal submission machine.
"""
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "outputs/rabs/rabs_raw.csv"
OUT = ROOT / "outputs/rabs/rabs_pairwise_deltas.csv"
OUTT = ROOT / "outputs/tables/table_rabs_pairwise_deltas.tex"
TARGET = "rabs_pd"
BASELINES = ["fixed_b3", "rabs_h", "rabs_l", "oracle_b"]
METRICS = [
    "objective",
    "loss_mean",
    "avg_bandwidth",
    "bandwidth_saving_vs_b3_pct",
    "missed_pct",
    "avg_aoi",
]
LOWER_BETTER = {"objective", "loss_mean", "avg_bandwidth", "missed_pct", "avg_aoi"}


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def ci95(xs: list[float]) -> float:
    return 1.96 * statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def sign_test_p_two_sided(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign-test p-value, ignoring ties."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def load_raw() -> list[dict[str, str]]:
    if not RAW.exists():
        raise FileNotFoundError(f"Missing raw simulation output: {RAW}")
    with RAW.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    rows = load_raw()
    keyed: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for r in rows:
        key = (r["network"], r["window_start"], r["seed"], r["policy"])
        keyed[key] = r

    networks = sorted({r["network"] for r in rows})
    units = sorted({(r["network"], r["window_start"], r["seed"]) for r in rows})
    out_rows: list[dict[str, object]] = []

    for network in networks + ["all"]:
        network_units = [u for u in units if network == "all" or u[0] == network]
        for baseline in BASELINES:
            for metric in METRICS:
                diffs: list[float] = []
                target_vals: list[float] = []
                base_vals: list[float] = []
                for net, window, seed in network_units:
                    t = keyed[(net, window, seed, TARGET)]
                    b = keyed[(net, window, seed, baseline)]
                    tv = float(t[metric])
                    bv = float(b[metric])
                    target_vals.append(tv)
                    base_vals.append(bv)
                    diffs.append(tv - bv)

                if metric in LOWER_BETTER:
                    wins = sum(1 for d in diffs if d < 0)
                    losses = sum(1 for d in diffs if d > 0)
                else:
                    wins = sum(1 for d in diffs if d > 0)
                    losses = sum(1 for d in diffs if d < 0)
                ties = len(diffs) - wins - losses
                out_rows.append(
                    {
                        "network": network,
                        "target": TARGET,
                        "baseline": baseline,
                        "metric": metric,
                        "n_pairs": len(diffs),
                        "target_mean": mean(target_vals),
                        "baseline_mean": mean(base_vals),
                        "delta_mean": mean(diffs),
                        "delta_ci95": ci95(diffs),
                        "wins": wins,
                        "losses": losses,
                        "ties": ties,
                        "win_rate_pct": 100 * wins / len(diffs),
                        "sign_test_p": sign_test_p_two_sided(wins, losses),
                    }
                )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(out_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    # Compact table for the manuscript: objective, missed violations, bandwidth.
    selected_metrics = ["objective", "missed_pct", "avg_bandwidth", "avg_aoi"]
    selected_baselines = ["fixed_b3", "rabs_h", "rabs_l", "oracle_b"]
    lines = [
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Kịch bản & Đối chứng & Chỉ số & $\Delta$ TB & CI95 & Thắng & $p$ \\",
        r"\midrule",
    ]
    for r in out_rows:
        if r["network"] == "all":
            continue
        if r["baseline"] not in selected_baselines or r["metric"] not in selected_metrics:
            continue
        p = float(r["sign_test_p"])
        ptxt = "<0.001" if p < 0.001 else f"{p:.3f}"
        lines.append(
            f"{tex_escape(str(r['network']))} & {tex_escape(str(r['baseline']))} & "
            f"{tex_escape(str(r['metric']))} & {float(r['delta_mean']):.4f} & "
            f"$\\pm${float(r['delta_ci95']):.4f} & {int(r['wins'])}/{int(r['n_pairs'])} & {ptxt} "
            + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUTT.parent.mkdir(parents=True, exist_ok=True)
    OUTT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(OUT)
    print(OUTT)


if __name__ == "__main__":
    main()
