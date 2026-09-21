#!/usr/bin/env python3
"""Bandwidth-vs-safety-objective trade-off figure.

Layout notes:
  1. All nine policies are labelled directly in the plot, so no legend is
     needed; dropping it widens the axes from 0.58 to 0.82 of the figure,
     which matters because the paper places this figure at 0.52\\linewidth.
  2. Label colours for the three grey Fixed-B* markers are darkened for
     contrast at 7 pt; marker colours are unchanged.
  3. Draw order is largest-marker-first so that coincident points stay
     visible: RABS-H/greedy clairvoyant are 0.4 pt apart, Max-AoI/Fixed-B2
     1.5 pt, RABS-PD/RABS-L 9.1 pt.

Source of truth: outputs/rabs/rabs_era5_summary.csv (severe_burst).
All plotted values are read from the CSV; no hand-entered numbers.
Label offsets are layout constants only.
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


# key, label, marker colour, label colour, marker, size, dx, dy, ha, va, leader, bold
SPEC = [
    ("fixed_b1", "Fixed-B1",           "#9e9e9e", "#6f6f6f", "v",  70,
     0.000, -0.0028, "center", "top",    False, False),
    ("fixed_b2", "Fixed-B2",           "#6d6d6d", "#4f4f4f", "v",  70,
     0.060, -0.0088, "left",   "center", True,  False),
    ("fixed_b3", "Fixed-B3",           "#3b3b3b", "#2b2b2b", "v",  70,
     -0.060, 0.0015, "right",  "center", False, False),
    ("max_aoi",  "Max-AoI (B=2)",      "#f58518", "#c96a08", "X",  95,
     -0.080, 0.0000, "right",  "center", True,  False),
    ("voi_b2",   "VoI (B=2)",          "#8c564b", "#8c564b", "P",  80,
     0.050,  0.0030, "left",   "center", True,  False),
    ("rabs_h",   "RABS-H",             "#1f77b4", "#1a6399", "o",  65,
     -0.085, 0.0060, "right",  "center", True,  False),
    ("rabs_l",   "RABS-L",             "#2ca02c", "#238524", "s",  65,
     0.115, -0.0049, "left",   "center", True,  False),
    ("rabs_pd",  "RABS-PD (proposed)", "#d62728", "#b71c1c", "D", 110,
     0.215,  0.0042, "left",   "center", True,  True),
    ("oracle_b", "Greedy clairvoyant", "#bcbd22", "#8d8e18", "*", 200,
     0.465,  0.0194, "left",   "center", True,  False),
]

fig, ax = plt.subplots(figsize=(7.2, 4.3))

# Largest marker first so smaller coincident glyphs stay visible on top.
for key, label, mcol, lcol, marker, size, dx, dy, ha, va, leader, bold in \
        sorted(SPEC, key=lambda s: -s[5]):
    if key not in sev:
        print(f"[WARN] missing policy in CSV: {key}")
        continue
    x, y = pt(key)
    on_top = size < max(s[5] for s in SPEC)
    ax.scatter(x, y, s=size, color=mcol,
               edgecolor="white" if on_top else "black",
               linewidth=1.0 if on_top else 0.8,
               zorder=3 + (10 - size / 25.0), marker=marker)

for key, label, mcol, lcol, marker, size, dx, dy, ha, va, leader, bold in SPEC:
    if key not in sev:
        continue
    x, y = pt(key)
    if leader:
        ax.annotate(label, xy=(x, y), xytext=(x + dx, y + dy), ha=ha, va=va,
                    fontsize=7.0, color=lcol, zorder=4,
                    fontweight="bold" if bold else "normal",
                    arrowprops=dict(arrowstyle="-", color=lcol, lw=0.6,
                                    alpha=0.8, shrinkA=1.5, shrinkB=2.5))
    else:
        ax.annotate(label, xy=(x, y), xytext=(x + dx, y + dy), ha=ha, va=va,
                    fontsize=7.0, color=lcol, zorder=4,
                    fontweight="bold" if bold else "normal")

# Shaded low-bandwidth region, caption centred at the top of the band.
ax.axvspan(0.9, 1.6, color="#2ca02c", alpha=0.06, zorder=0)
ax.text(1.25, 0.1572, "low-bandwidth region", color="#2e7d32", fontsize=7.8,
        va="top", ha="center", alpha=0.85)

ax.set_xlabel(r"Average bandwidth per slot $\bar B_t$")
ax.set_ylabel("Composite safety objective (lower is better)")
ax.set_title("Bandwidth vs. safety-objective trade-off (severe burst loss)")
ax.set_xlim(0.85, 3.15)
ax.set_ylim(0.090, 0.160)
ax.yaxis.set_label_coords(-0.085, 0.5)
fig.subplots_adjust(left=0.13, right=0.98, top=0.92, bottom=0.12)

for ext in ("pdf", "png"):
    fig.savefig(MAN_FIG / f"tradeoff_plot.{ext}")
plt.close(fig)
print("Wrote:", MAN_FIG / "tradeoff_plot.pdf")
print("\nPlotted severe-burst points (from fresh CSV):")
for key, label, *_ in SPEC:
    if key in sev:
        x, y = pt(key)
        print(f"  {label:22s} Bw={x:5.3f}  Obj={y:.4f}  labelled=yes")
