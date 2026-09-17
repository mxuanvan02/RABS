#!/usr/bin/env python3
"""Regenerate the bandwidth-vs-safety-objective trade-off figure (v2).

Fixes the Reviewer-1 overlap issue in three ways:
  1. labels of the low-bandwidth cluster (RABS-H, RABS-L, RABS-PD, Greedy
     clairvoyant -- RABS-H and Greedy coincide at (1.20, 0.1028)) move to
     free positions with thin leader lines;
  2. the shaded-region caption moves to the top-centre of the band, clear of
     the Fixed-B1 marker, which is labelled below its marker;
  3. the tight B=2 column (Fixed-B2, Max-AoI, VoI) stays legend-only, as in v1.

Source of truth : outputs/rabs/rabs_era5_summary.csv (severe_burst scenario)
All plotted values are read from the CSV; no hand-entered numbers.
Label placements are layout constants only (same role as dx/dy in v1).
"""
from pathlib import Path
import csv
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = Path(os.environ.get("RABS_SUMMARY_CSV", ROOT / "outputs" / "rabs" / "rabs_era5_summary.csv"))
MAN_FIG = Path(os.environ.get("RABS_FIG_DIR", ROOT / "outputs" / "figures"))
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

with open(SUMMARY, newline="", encoding="utf-8") as f:
    sev = {row["policy"]: row for row in csv.DictReader(f)
           if row["network"] == "severe_burst"}
if not sev:
    raise SystemExit(f"No severe_burst rows found in {SUMMARY}")

def pt(key):
    r = sev[key]
    return float(r["avg_bandwidth_mean"]), float(r["objective_mean"])

# key, label, color, marker, size
BASE = [
    ("fixed_b1",  "Fixed-B1",           "#9e9e9e", "v",  70),
    ("fixed_b2",  "Fixed-B2",           "#6d6d6d", "v",  70),
    ("fixed_b3",  "Fixed-B3",           "#3b3b3b", "v",  70),
    ("max_aoi",   "Max-AoI (B=2)",      "#f58518", "X",  95),
    ("voi_b2",    "VoI (B=2)",          "#8c564b", "P",  80),
    ("rabs_h",    "RABS-H",             "#1f77b4", "o",  65),
    ("rabs_l",    "RABS-L",             "#2ca02c", "s",  65),
    ("rabs_pd",   "RABS-PD (proposed)", "#d62728", "D", 110),
    ("oracle_b",  "Greedy clairvoyant", "#bcbd22", "*", 200),
]

fig, ax = plt.subplots(figsize=(7.2, 4.3))
for key, label, color, marker, size in BASE:
    if key not in sev:
        print(f"[WARN] missing policy in CSV: {key}")
        continue
    x, y = pt(key)
    ax.scatter(x, y, s=size, color=color, edgecolor="black",
               linewidth=0.8, label=label, zorder=3, marker=marker)

# --- direct labels ---------------------------------------------------------
# legend-only points (v1 convention): fixed_b2, max_aoi, voi_b2 (tight B=2 column)
x1, y1 = pt("fixed_b1")
ax.annotate("Fixed-B1", xy=(x1, y1), xytext=(x1, y1 - 0.0028),
            ha="center", va="top", fontsize=7.0, color="#9e9e9e", zorder=4)

x3, y3 = pt("fixed_b3")
ax.annotate("Fixed-B3", xy=(x3, y3), xytext=(x3 - 0.06, y3 + 0.0015),
            ha="right", va="center", fontsize=7.0, color="#3b3b3b", zorder=4)

# low-bandwidth cluster with leader lines.
xh, yh = pt("rabs_h")      # (1.20, 0.1028) -- coincides with oracle
xl, yl = pt("rabs_l")      # (1.42, 0.1011)
xp, yp = pt("rabs_pd")     # (1.37, 0.1000)
xo, yo = pt("oracle_b")    # (1.20, 0.1028)
LEADERS = [
    ("RABS-H",             xh, yh, xh - 0.085, yh + 0.0060, "right",  "#1f77b4", "normal"),
    ("Greedy clairvoyant", xo, yo, xo + 0.465,  yo + 0.0194, "left",   "#bcbd22", "normal"),
    ("RABS-L",             xl, yl, xl + 0.115,  yl - 0.0049, "left",   "#2ca02c", "normal"),
    ("RABS-PD (proposed)", xp, yp, xp + 0.215,  yp + 0.0042, "left",   "#d62728", "bold"),
]
for label, mx, my, tx, ty, ha, color, weight in LEADERS:
    ax.annotate(label, xy=(mx, my), xytext=(tx, ty), ha=ha, va="center",
                fontsize=7.0, color=color, zorder=4, fontweight=weight,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.6,
                                alpha=0.75, shrinkA=1.5, shrinkB=2.5))

# --- shaded low-bandwidth region: caption at top-centre of the band --------
ax.axvspan(0.9, 1.6, color="#2ca02c", alpha=0.06, zorder=0)
ax.text(1.25, 0.1572, "low-bandwidth region", color="#2e7d32", fontsize=7.8,
        va="top", ha="center", alpha=0.85)

ax.set_xlabel(r"Average bandwidth per slot $\bar B_t$")
ax.set_ylabel("Composite safety objective (lower is better)")
ax.set_title("Bandwidth vs. safety-objective trade-off (severe burst loss)")
ax.set_xlim(0.85, 3.15)
ax.set_ylim(0.094, 0.160)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
          framealpha=0.95, ncol=1, fontsize=8.4, title="Policy")
ax.yaxis.set_label_coords(-0.115, 0.5)
fig.subplots_adjust(left=0.16, right=0.74, top=0.92, bottom=0.12)

for ext in ("pdf", "png"):
    fig.savefig(MAN_FIG / f"tradeoff_plot.{ext}")
plt.close(fig)
print("Wrote:", MAN_FIG / "tradeoff_plot.pdf")
print("\nPlotted severe-burst points (from fresh CSV):")
for key, label, *_ in BASE:
    if key in sev:
        x, y = pt(key)
        print(f"  {label:22s} Bw={x:5.2f}  Obj={y:.4f}")
