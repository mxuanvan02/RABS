# RABS code directory

Run all commands from the repository root.

## 1. Fetch/reuse ERA5 data

```bash
python3 data/fetch_era5_vn.py
```

Writes cached station traces under:

```text
data/era5_vn/
```

## 2. Main comparison

```bash
python3 code/run_rabs_era5_main.py
```

Writes:

```text
outputs/rabs/rabs_era5_raw.csv
outputs/rabs/rabs_era5_summary.csv
```

## 3. Urgency-channel ablation

```bash
python3 code/run_rabs_era5_ablation.py
```

Writes:

```text
outputs/rabs/rabs_era5_ablation_raw.csv
outputs/rabs/rabs_era5_ablation_summary.csv
```

## 4. Scalability experiment

```bash
python3 code/run_rabs_era5_scaling.py
```

Writes:

```text
outputs/rabs/rabs_era5_scaling_raw.csv
outputs/rabs/rabs_era5_scaling_summary.csv
```

## 5. Tables and figure

```bash
python3 code/make_era5_tables.py
RABS_SUMMARY_CSV=outputs/rabs/rabs_era5_summary.csv \
  python3 code/make_tradeoff_plot.py
```

Writes LaTeX tables under `outputs/tables/` and the trade-off figure under `outputs/figures/`.

## Notes

- This directory contains the final paper-facing reproduction pipeline only.
- Use `bash reproduce.sh` from the repository root for the full end-to-end run.
