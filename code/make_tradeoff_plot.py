#!/usr/bin/env python3
"""Regenerate the bandwidth-vs-safety-objective trade-off figure for the RABS
manuscript directly from the fresh experiment summary.

Source of truth : outputs/rabs/rabs_summary.csv (severe_burst scenario)
Output          : <manuscript>/outputs/figures/tradeoff_plot.pdf (+ .png for inspection)

All plotted values are read from the CSV; no hand-entered numbers. Policy set and
labels are kept consistent with Table `tab:sota` in sections/04_method_and_eval.tex.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import os
ROOT = Path(__file__).resolve().parents[1]
# Source CSV can be overridden (e.g. the ERA5 summary) via RABS_SUMMARY_CSV.
SUMMARY = Path(os.environ.get("RABS_SUMMARY_CSV", ROOT / "outputs" / "rabs" / "rabs_summary.csv"))
# Output to a local, repo-relative figures/ dir by default. Set RABS_FIG_DIR
# to redirect the output elsewhere (e.g. a manuscript figures folder).
MAN_FIG = Path(os.environ.get("RABS_FIG_DIR", ROOT / "figures"))
MAN_FIG.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "legend.fontsize": 8.3,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Policies discussed in the manuscript tables/narrative (severe-burst scenario).
# annotate flag: label directly only for well-separated points. The tight
# B=2 cluster (Fixed-B2, Max-AoI, VoI) is left to the legend+markers to avoid
# overlapping text; their markers/colours already distinguish them.
SPEC = [
    # key            label             color       marker  size  dx      dy      ha       annotate
    ("fixed_b1",     "Fixed-B1",       "#9e9e9e",  "v",    70,    0.06,  0.000, "left",  True),
    ("fixed_b2",     "Fixed-B2",       "#6d6d6d",  "v",    70,    0.10,  0.0075, "left", False),
    ("fixed_b3",     "Fixed-B3",       "#3b3b3b",  "v",    70,   -0.06,  0.0015, "right", True),
    ("max_aoi",      "Max-AoI (B=2)",  "#f58518",  "X",    95,   -0.10, -0.0035, "right", False),
    ("voi_b2",       "VoI (B=2)",      "#8c564b",  "P",    80,    0.10, -0.0045, "left", False),
    ("rabs_h",       "RABS-H",         "#1f77b4",  "o",    65,    0.05,  0.0020, "left",  True),
    ("rabs_l",       "RABS-L",         "#2ca02c",  "s",    65,   -0.05,  0.000, "right", True),
    ("rabs_pd",      "RABS-PD (proposed)", "#d62728", "D", 110,   0.05,  0.0015, "left", True),
    ("oracle_b",     "Greedy clairvoyant", "#bcbd22", "*", 200,   0.05,  0.000, "left",  True),
]

df = pd.read_csv(SUMMARY)
sev = df[df.network == "severe_burst"].set_index("policy")

fig, ax = plt.subplots(figsize=(7.2, 4.3))
for key, label, color, marker, size, dx, dy, ha, annotate in SPEC:
    if key not in sev.index:
        print(f"[WARN] missing policy in CSV: {key}")
        continue
    r = sev.loc[key]
    x, y = float(r.avg_bandwidth_mean), float(r.objective_mean)
    ax.scatter(x, y, s=size, color=color, edgecolor="black",
               linewidth=0.8, label=label, zorder=3, marker=marker)
    # Label well-separated points directly; the tight B=2 cluster
    # (Fixed-B2/Max-AoI/VoI) relies on the legend to avoid overlapping text.
    if annotate:
        ax.annotate(label, (x, y), xytext=(x + dx, y + dy), ha=ha,
                    va="center", fontsize=7.0, color=color, zorder=4,
                    fontweight="bold" if key == "rabs_pd" else "normal")

# Shade the desirable low-bandwidth region (qualitative guide only).
ax.axvspan(0.9, 1.6, color="#2ca02c", alpha=0.06, zorder=0)
ax.text(0.95, sev.objective_mean.max(),
        "low-bandwidth\nregion", color="#2e7d32", fontsize=7.8,
        va="top", ha="left", alpha=0.85)

ax.set_xlabel(r"Average bandwidth per slot $\bar B_t$")
ax.set_ylabel("Composite safety objective (lower is better)")
ax.set_title("Bandwidth vs. safety-objective trade-off (severe burst loss)")
ax.set_xlim(0.85, 3.15)
ax.margins(y=0.12)
# Legend placed outside the plot area so no data point is occluded.
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
          framealpha=0.95, ncol=1, fontsize=8.4, title="Policy")
ax.yaxis.set_label_coords(-0.115, 0.5)
fig.subplots_adjust(left=0.16, right=0.74, top=0.92, bottom=0.12)

for ext in ("pdf", "png"):
    fig.savefig(MAN_FIG / f"tradeoff_plot.{ext}")
plt.close(fig)
print("Wrote:", MAN_FIG / "tradeoff_plot.pdf")
print("\nPlotted severe-burst points (from fresh CSV):")
for key, label, *_ in SPEC:
    if key in sev.index:
        r = sev.loc[key]
        print(f"  {label:22s} Bw={r.avg_bandwidth_mean:5.2f}  Obj={r.objective_mean:.4f}")
