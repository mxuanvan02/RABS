# Code directory — RABS ERA5 pipeline

Run from the project root. Simulators are stdlib-only; the table generator
needs `numpy`+`scipy` for paired Wilcoxon statistics.

## 0. Fetch data (reproducible)

```bash
python3 data/fetch_era5_vn.py
```

Writes real ERA5 hourly 2-m temperature (2024) for the three Mekong-delta
stations:

```text
data/era5_vn/can_tho_2024.csv
data/era5_vn/soc_trang_2024.csv
data/era5_vn/ca_mau_2024.csv
```

## 1. Main comparison

```bash
python3 code/run_rabs_era5_main.py
```

Baselines (Fixed-B1/B2/B3, Max-AoI, VoI-B2, RABS-H/L, RABS-PD, greedy
clairvoyant) over Gilbert–Elliott burst / severe-burst channels. Urgency ranks
sensors by violation probability. Safety band `[22, 34] °C`. Writes:

```text
outputs/rabs/rabs_era5_raw.csv
outputs/rabs/rabs_era5_summary.csv
```

## 2. Urgency-term ablation

```bash
python3 code/run_rabs_era5_ablation.py   # -> outputs/rabs/rabs_era5_ablation_summary.csv
```

Zeroes each urgency term and compares the deployed raw-probability channel
against a decision-uncertainty variant `g(p)=4p(1-p)`. On this heavy-tailed
regime raw probability wins, because heat episodes push `p_vio -> 1` and the
at-risk zones must not be de-ranked.

## 3. Scalability stress test

```bash
python3 code/run_rabs_era5_scaling.py    # -> outputs/rabs/rabs_era5_scaling_summary.csv
```

Synthetic multi-zone traces (N = 3, 8, 12, 20) anchored on the three real
stations (independent circular time offsets + small thermal biases). A
scheduler stress test, not a claim of extra measured data.

## 4. Manuscript tables + figure

```bash
python3 code/make_era5_tables.py         # sota / wilcoxon / scalability / ablation .tex
RABS_SUMMARY_CSV=outputs/rabs/rabs_era5_summary.csv \
  python3 code/make_tradeoff_plot.py     # trade-off figure
```

## legacy/

Earlier greenhouse + VoU exploratory pipeline. Not part of the paper's results.
