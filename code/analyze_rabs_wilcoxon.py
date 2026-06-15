#!/usr/bin/env python3
"""
analyze_rabs_wilcoxon.py — Q2-grade paired statistics for RABS.

Upgrades the previous sign-test analysis to:
  - Wilcoxon signed-rank test (paired, two-sided)
  - Matched-pairs rank-biserial effect size r = Z / sqrt(N)
  - 95% CI of the mean paired difference (t-distribution)
  - Holm-Bonferroni correction across baselines per (network, metric)

Pairing key: (network, seed, window_start). Compares rabs_pd against each
baseline on identical seeds/windows/channel, matching the manuscript design.

Outputs:
  outputs/rabs/rabs_wilcoxon.csv
  outputs/tables/table_rabs_wilcoxon.tex   (Vietnamese-labelled)
"""
from __future__ import annotations
import csv
import math
import os
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "outputs", "rabs", "rabs_raw.csv")
OUT_CSV = os.path.join(ROOT, "outputs", "rabs", "rabs_wilcoxon.csv")
OUT_TEX = os.path.join(ROOT, "outputs", "tables", "table_rabs_wilcoxon.tex")

TARGET = "rabs_pd"
BASELINES = ["fixed_b3", "rabs_h", "rabs_l", "oracle_b"]
# Lower-is-better metrics; negative delta_mean = target better than baseline.
METRICS = ["objective", "loss_mean", "missed_pct", "avg_bandwidth", "avg_aoi"]
NETWORKS = ["burst", "severe_burst"]


def load_rows():
    rows = list(csv.DictReader(open(RAW)))
    # index[policy][network][(seed, window)] = {metric: value}
    idx = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        key = (int(r["seed"]), int(r["window_start"]))
        d = {m: float(r[m]) for m in METRICS}
        idx[r["policy"]][r["network"]][key] = d
    return idx


def paired_vectors(idx, network, baseline, metric):
    tgt = idx[TARGET][network]
    base = idx[baseline][network]
    keys = sorted(set(tgt) & set(base))
    a = np.array([tgt[k][metric] for k in keys])
    b = np.array([base[k][metric] for k in keys])
    return a, b


def analyze():
    idx = load_rows()
    results = []
    for network in NETWORKS:
        for metric in METRICS:
            block = []
            for baseline in BASELINES:
                a, b = paired_vectors(idx, network, baseline, metric)
                n = len(a)
                if n == 0:
                    continue
                diff = a - b
                delta_mean = float(np.mean(diff))
                sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
                # 95% CI of mean diff (t-dist)
                if n > 1 and sd > 0:
                    tcrit = stats.t.ppf(0.975, df=n - 1)
                    ci = tcrit * sd / math.sqrt(n)
                else:
                    ci = 0.0
                # Wilcoxon signed-rank (skip if all diffs zero)
                if np.allclose(diff, 0):
                    W, p = float("nan"), 1.0
                    r_eff = 0.0
                else:
                    W, p = stats.wilcoxon(a, b, zero_method="wilcox",
                                          alternative="two-sided",
                                          correction=False, mode="approx")
                    # Z from normal approx -> rank-biserial-ish effect r
                    z = stats.norm.isf(p / 2) * (1 if delta_mean < 0 else -1)
                    r_eff = float(z / math.sqrt(n))
                wins = int(np.sum(diff < 0))   # target lower (better)
                losses = int(np.sum(diff > 0))
                ties = int(np.sum(diff == 0))
                block.append({
                    "network": network, "metric": metric, "baseline": baseline,
                    "n": n, "target_mean": float(np.mean(a)),
                    "baseline_mean": float(np.mean(b)),
                    "delta_mean": delta_mean, "ci95": ci,
                    "wilcoxon_W": W, "p_raw": p, "effect_r": r_eff,
                    "wins": wins, "losses": losses, "ties": ties,
                })
            # Holm-Bonferroni within (network, metric) across baselines
            block_sorted = sorted(block, key=lambda x: x["p_raw"])
            m = len(block_sorted)
            for rank, item in enumerate(block_sorted):
                item["p_holm"] = min(1.0, item["p_raw"] * (m - rank))
            results.extend(block)
    return results


def write_csv(results):
    cols = ["network", "metric", "baseline", "n", "target_mean", "baseline_mean",
            "delta_mean", "ci95", "wilcoxon_W", "p_raw", "p_holm", "effect_r",
            "wins", "losses", "ties"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"wrote {OUT_CSV} ({len(results)} rows)")


def fmt_p(p):
    if p < 1e-3:
        return "$<$0,001"
    return f"{p:.3f}".replace(".", ",")


def write_tex(results):
    # Focus table: objective + missed_pct for both networks, vs fixed_b3 & rabs_h
    metric_vi = {"objective": "Hàm mục tiêu", "loss_mean": "Tổn thất",
                 "missed_pct": "Bỏ sót (\\%)", "avg_bandwidth": "Băng thông",
                 "avg_aoi": "AoI"}
    net_vi = {"burst": "Chùm lỗi", "severe_burst": "Chùm lỗi mạnh"}
    base_vi = {"fixed_b3": "Fixed-B3", "rabs_h": "RABS-H",
               "rabs_l": "RABS-L", "oracle_b": "Oracle-B"}
    show_metrics = ["objective", "missed_pct"]
    show_base = ["fixed_b3", "rabs_h", "rabs_l"]
    lines = []
    lines.append("\\begin{tabular}{lllrrrr}")
    lines.append("\\hline")
    lines.append("Kịch bản & Chỉ số & Đối chứng & $\\Delta$TB & KTC 95\\% & $p$ (Holm) & $r$ \\\\")
    lines.append("\\hline")
    by = {(r["network"], r["metric"], r["baseline"]): r for r in results}
    for network in NETWORKS:
        for metric in show_metrics:
            for baseline in show_base:
                r = by.get((network, metric, baseline))
                if not r:
                    continue
                dm = f"{r['delta_mean']:.4f}".replace(".", ",")
                ci = f"{r['ci95']:.4f}".replace(".", ",")
                p = fmt_p(r["p_holm"])
                reff = f"{r['effect_r']:.2f}".replace(".", ",")
                lines.append(
                    f"{net_vi[network]} & {metric_vi[metric]} & {base_vi[baseline]} "
                    f"& {dm} & {ci} & {p} & {reff} \\\\")
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT_TEX}")


if __name__ == "__main__":
    res = analyze()
    write_csv(res)
    write_tex(res)
    # Print a quick summary for rabs_pd vs fixed_b3 & rabs_h on objective
    print("\n=== Quick summary: objective ===")
    for r in res:
        if r["metric"] == "objective" and r["baseline"] in ("fixed_b3", "rabs_h"):
            print(f"{r['network']:13s} vs {r['baseline']:9s}: "
                  f"Δ={r['delta_mean']:+.4f}, p_holm={r['p_holm']:.2e}, "
                  f"r={r['effect_r']:+.2f}, wins={r['wins']}/{r['n']}")
