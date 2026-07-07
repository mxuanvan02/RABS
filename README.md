# RABS: Risk-Adaptive Bandwidth Scaling for Safety-Critical Smart-Agriculture IoT

This repository contains the public reproduction package for the paper:

> **Self-Tuning Risk-Adaptive Bandwidth Scaling for Safety-Critical Smart-Agriculture IoT Networks**

The repository is intentionally limited to the final paper-facing artifacts: source code, public input data/fetch scripts, and commands needed to reproduce the reported tables and figures. Development notes, exploratory variants, drafts, and internal working material are not part of this release.

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

# 6. Regenerate the trade-off figure
RABS_SUMMARY_CSV=outputs/rabs/rabs_era5_summary.csv \
  python3 code/make_tradeoff_plot.py
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
  make_tradeoff_plot.py       Figure generator

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
