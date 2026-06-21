# RABS: Risk-Adaptive Bandwidth Scaling for Smart Greenhouse IoT Networks

This repository contains the public reproducibility code for the RABS manuscript:

> Self-Tuning Risk-Aware Bandwidth Scheduling for Safety-Critical Greenhouse IoT Networks

RABS-PD is a primal-dual-inspired online scheduler that adapts the per-slot
transmission budget through feedback-updated safety, AoI, and bandwidth
penalties, instead of fixing the budget ahead of time.

The release ships **code and source data only**. Result tables, figures, and the
manuscript itself are not tracked: readers regenerate every number by running
the scripts below and check them against the paper.

## Repository layout

```text
code/                    RABS simulator, baselines, analysis, and plotting scripts
data/source/             Source calibration data needed to run the experiments
```

Running the scripts creates `outputs/` and `figures/` locally; both are
intentionally untracked.

## Requirements

The core simulation (`run_rabs_adaptive_bandwidth.py`) is **stdlib-only**.
Plotting and table scripts additionally use `pandas` and `matplotlib`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas matplotlib scipy
```

## Reproduce the main results

Run from the repository root, in order:

```bash
# 1. Main RABS experiment (stationary burst + severe-burst channels)
python3 code/run_rabs_adaptive_bandwidth.py

# 2. Weight-sensitivity sweep (risk-score weights w1..w5)
python3 code/run_rabs_weight_sensitivity.py

# 3. Non-stationary (drifting) channel experiment
python3 code/run_rabs_nonstationary.py

# 4. Paired statistics (Wilcoxon signed-rank vs. baselines)
python3 code/analyze_rabs_wilcoxon.py
python3 code/analyze_rabs_pairwise.py

# 5. Dual-feasibility check for the bandwidth penalty
python3 code/verify_dual_convergence.py

# 6. Sensitivity table + trade-off figure
python3 code/make_sensitivity_table.py
python3 code/make_tradeoff_plot.py
```

Each script reads `data/source/` and writes its outputs into `outputs/` (CSV
summaries) and `figures/` (plots) in your local checkout. Compare the
regenerated numbers against the manuscript tables to verify reproducibility.

See `code/README.md` for the per-script input/output details.

## Notes

- Throwaway exploratory experiments and internal backups are excluded from this
  public release.
- Please cite the associated manuscript if you use this code.
