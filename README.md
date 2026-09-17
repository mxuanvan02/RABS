# RABS: Risk-Adaptive Bandwidth Scaling for Safety-Critical Smart-Agriculture IoT

This repository contains the public reproduction package for the paper:

> **Self-Tuning Risk-Adaptive Bandwidth Scaling for Safety-Critical Smart-Agriculture IoT Networks**

The repository is intentionally limited to the final paper-facing artifacts: source code, public input data/fetch scripts, and commands needed to reproduce the reported tables and figures.

## Reproduce the reported results

```bash
git clone https://github.com/mxuanvan02/RABS.git
cd RABS
bash reproduce.sh
```

`reproduce.sh` performs the full pipeline:

1. creates a local Python virtual environment;
2. installs `numpy`, `scipy`, and `matplotlib`;
3. fetches or reuses the public ERA5 hourly 2-m temperature data;
4. runs the main comparison, urgency ablation, and scalability experiments;
5. regenerates the LaTeX tables and trade-off figure under `outputs/`.

No API key is required. The simulation code uses fixed seeds so that repeated runs are deterministic, subject to the same Python/library environment and the same ERA5 source data.

## Requirements

- Python 3.9 or newer
- Internet access for the first ERA5 fetch, unless the CSV files are already present under `data/era5_vn/`
- Python packages: `numpy`, `scipy`, `matplotlib`

Manual setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib
```

## Dataset

The experiments use real ERA5 hourly 2-m temperature data for 20 Mekong-delta locations in 2024. Data are fetched from the public Open-Meteo archive by:

```bash
python3 data/fetch_era5_vn.py
```

The scalability study evaluates deployments using up to 20 distinct stations. The main comparison uses the three primary stations selected in the paper. The data are climate reanalysis traces used as a reproducible proxy for smart-agriculture heat-stress monitoring; they are not claimed to be in-field wireless sensor measurements.

## Manual reproduction steps

Run these commands from the repository root.

```bash
# 1. Fetch public ERA5 data if not already cached
python3 data/fetch_era5_vn.py

# 2. Main comparison: paper Tables 1 and 2
python3 code/run_rabs_era5_main.py

# 3. Urgency-channel ablation: paper Table 4
python3 code/run_rabs_era5_ablation.py

# 4. Scalability across Mekong-delta stations: paper Table 3
python3 code/run_rabs_era5_scaling.py

# 5. Regenerate LaTeX tables
python3 code/make_era5_tables.py

# 6. Regenerate the trade-off figure (v2: leader-line labels, no overlaps)
RABS_SUMMARY_CSV=outputs/rabs/rabs_era5_summary.csv \
  python3 code/make_tradeoff_plot_v2.py
```

## Revision-round experiments (peer-review response)

The revision round for the STAIS reviews adds three analyses, all deterministic
and seeded like the main pipeline:

```bash
# R1. Risk-channel decomposition (Reviewer 2, Q4): ablates the ranking risk
#     weight, the budget relaxation -c_R*R*b, and the missed-risk surrogate,
#     individually and jointly, on severe_burst AND a harsher extreme_burst
#     regime (12%/90% Gilbert-Elliott, ~48% overall loss).
#     -> outputs/rabs/rabs_q4_full_ablation.csv (+ _stats.csv)
python3 code/abl_q4_full.py

# R2. Revised Table 4 (consistent deployed raw-p basis, 8 rows; cross-checks
#     every overlapping value against the submitted ablation before writing):
#     -> outputs/rabs/rabs_q4_final_matrix.csv
#     -> outputs/tables/ablation_urgency.tex
python3 code/make_ablation_table_final.py

# R3. Surrogate-vs-realized calibration for the O(1/T) bound discussion
#     (Reviewer 2, Q5): per-window E[Mhat], E[m], correlation.
#     -> outputs/rabs/rabs_q5_surrogate_calibration.csv
python3 code/abl_q5_calibration.py

# R4. Full baseline comparison under the extreme_burst regime (Reviewer 2,
#     Q2: the missed-event ordering reverses in RABS-PD's favour there).
#     -> outputs/rabs/rabs_era5_extreme_{raw,summary}.csv
python3 code/run_rabs_era5_main_extreme.py
```

Expected generated outputs:

```text
outputs/rabs/rabs_era5_summary.csv
outputs/rabs/rabs_era5_ablation_summary.csv
outputs/rabs/rabs_era5_scaling_summary.csv
outputs/tables/sota_comparison.tex
outputs/tables/wilcoxon.tex
outputs/tables/scalability.tex
outputs/tables/ablation_urgency.tex
outputs/figures/tradeoff_plot.pdf
```

## Repository layout

```text
code/
  run_rabs_era5_main.py       Main comparison experiment
  run_rabs_era5_ablation.py   Urgency-channel ablation
  run_rabs_era5_scaling.py    Scalability experiment
  make_era5_tables.py         LaTeX table generator
  make_tradeoff_plot.py       Figure generator (v1, submitted layout)
  make_tradeoff_plot_v2.py    Figure generator (v2, leader-line labels)
  run_rabs_era5_main_extreme.py  Extreme-burst main comparison (revision Q2)
  abl_q4_full.py              Risk-channel decomposition, 2 regimes (revision Q4)
  make_ablation_table_final.py   Revised Table 4 generator with cross-check
  abl_q5_calibration.py       Surrogate-vs-realized calibration (revision Q5)

data/
  fetch_era5_vn.py            Public ERA5 data fetcher
  era5_vn/*.csv               Cached ERA5 station traces

reproduce.sh                  One-command reproduction script
```

## Reproducibility notes

- Generated outputs are not committed; regenerate them with `bash reproduce.sh`.
- The code package contains only the final paper-facing pipeline.
- The comparison includes practical baselines and a non-deployable post-observation reference used only to contextualize achievable performance.
- If Open-Meteo updates service behavior or availability, use the cached `data/era5_vn/*.csv` files in this repository for exact reproduction of the submitted results.

## License

See `LICENSE`.
