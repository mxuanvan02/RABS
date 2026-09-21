# RABS code directory

Experiment scripts for the paper. Run everything from the repository root.

The fastest path is a single command from the root — `bash reproduce.sh` — which
sets up the environment, fetches (or reuses cached) ERA5 data, runs every
experiment, and regenerates all tables and figures. The individual entry points
below are listed for reference and to run a single stage on its own.

## Pipeline stages

```text
data/fetch_era5_vn.py             fetch/reuse public ERA5 station traces
code/run_rabs_era5_main.py        main baseline comparison
code/run_rabs_era5_ablation.py    urgency-channel ablation
code/run_rabs_era5_scaling.py     scalability across stations
code/run_rabs_era5_main_extreme.py  extreme-burst loss sensitivity
code/risk_channel_decomposition.py  per-channel risk ablation (two regimes)
code/risk_signal_diagnostic.py    p vs delta spread/saturation diagnostic
code/surrogate_calibration.py     surrogate-vs-realized calibration
code/make_era5_tables.py          LaTeX table generator
code/make_ablation_table.py       urgency/risk decomposition table
code/make_tradeoff_plot.py        bandwidth-vs-objective trade-off figure
```

Each script writes CSVs under `outputs/rabs/` and LaTeX/figure artifacts under
`outputs/tables/` and `outputs/figures/`. All runs are seeded and deterministic;
each script's docstring documents its inputs, outputs, and parameters.
