#!/usr/bin/env python3
"""make_sensitivity_table.py — Q2 weight-sensitivity table for RABS.

Reads outputs/rabs_weight_sensitivity.csv and emits a Vietnamese LaTeX table
showing how objective/bandwidth/missed vary across 5 risk-weight configurations,
with the coefficient of variation (CV) per policy to quantify robustness.
"""
import csv
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "outputs", "rabs_weight_sensitivity.csv")
OUT = os.path.join(ROOT, "outputs", "tables", "table_rabs_weight_sensitivity.tex")

VARIANT_VI = {
    "base": "Cơ sở",
    "risk_heavy": "Ưu tiên rủi ro",
    "aoi_heavy": "Ưu tiên AoI",
    "channel_heavy": "Ưu tiên kênh",
    "balanced": "Cân bằng",
}
POL_VI = {"rabs_h": "RABS-H", "rabs_l": "RABS-L", "rabs_pd": "RABS-PD"}
ORDER = ["base", "risk_heavy", "aoi_heavy", "channel_heavy", "balanced"]


def num(x):
    return f"{x:.4f}".replace(".", ",")


def main():
    rows = list(csv.DictReader(open(SRC)))
    lines = ["\\begin{tabular}{lcccc}", "\\hline",
             "Cấu hình trọng số & Trọng số ($w_1$--$w_5$) & RABS-H & RABS-L & RABS-PD \\\\",
             "\\hline"]
    for v in ORDER:
        vr = [r for r in rows if r["variant"] == v]
        w = vr[0]["weights"].replace("/", "/")
        objs = {r["policy"]: float(r["objective_avg"]) for r in vr}
        lines.append(
            f"{VARIANT_VI[v]} & {w} & {num(objs['rabs_h'])} & "
            f"{num(objs['rabs_l'])} & {num(objs['rabs_pd'])} \\\\")
    lines.append("\\hline")
    # CV row
    cvs = {}
    for pol in ("rabs_h", "rabs_l", "rabs_pd"):
        o = [float(r["objective_avg"]) for r in rows if r["policy"] == pol]
        cvs[pol] = st.pstdev(o) / st.mean(o) * 100
    lines.append(
        f"Hệ số biến thiên CV (\\%) & -- & {cvs['rabs_h']:.1f} & "
        f"{cvs['rabs_l']:.1f} & {cvs['rabs_pd']:.1f} \\\\".replace(".", ","))
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    print("CV:", {k: round(v, 2) for k, v in cvs.items()})


if __name__ == "__main__":
    main()
