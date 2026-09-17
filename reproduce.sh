#!/usr/bin/env bash
# =============================================================================
# RABS — one-command reproduction of every number in the paper.
#
#   Self-Tuning Risk-Adaptive Bandwidth Scaling for
#   Safety-Critical Smart-Agriculture IoT Networks
#
# What this does, end to end:
#   1. creates an isolated Python virtual environment (.venv)
#   2. installs the third-party deps (numpy + scipy for paired stats,
#      matplotlib for the trade-off figure)
#   3. fetches the real ERA5 hourly 2-m temperature (2024) for 20 Mekong-delta
#      stations from the public Open-Meteo archive  (skipped if already cached)
#   4. runs the main comparison, urgency ablation, and scalability evaluation
#   5. regenerates every LaTeX table and the trade-off figure
#
# All results land under outputs/. The manuscript tables are emitted by
# code/make_era5_tables.py, so the paper cannot drift from the code.
#
# Usage:
#   bash reproduce.sh              # full pipeline
#   REFETCH=1 bash reproduce.sh    # force re-download of the ERA5 data
#
# Requirements: python3 (>=3.9) and internet access for the one-time fetch.
# The simulators themselves are stdlib-only and fully deterministic.
# =============================================================================
set -euo pipefail

# Always run from the repository root (the directory this script lives in).
cd "$(dirname "$0")"
ROOT="$(pwd)"
echo "==> RABS reproduction — project root: $ROOT"

# ----------------------------------------------------------------------------
# 1. Isolated virtual environment
# ----------------------------------------------------------------------------
if [ ! -d .venv ]; then
  echo "==> [1/5] creating virtual environment (.venv)"
  python3 -m venv .venv
else
  echo "==> [1/5] reusing existing virtual environment (.venv)"
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ----------------------------------------------------------------------------
# 2. Dependencies (numpy + scipy for stats, matplotlib for the figure;
#    the simulators themselves are stdlib-only)
# ----------------------------------------------------------------------------
echo "==> [2/5] installing dependencies (numpy, scipy, matplotlib)"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet numpy scipy matplotlib

# ----------------------------------------------------------------------------
# 3. Fetch real ERA5 data (cached: only downloads if CSVs are missing)
# ----------------------------------------------------------------------------
N_CSV=$(find data/era5_vn -maxdepth 1 -name '*_2024.csv' 2>/dev/null | wc -l | tr -d ' ')
if [ "${REFETCH:-0}" = "1" ] || [ "$N_CSV" -lt 20 ]; then
  echo "==> [3/5] fetching real ERA5 data for 20 Mekong-delta stations (Open-Meteo)"
  python data/fetch_era5_vn.py
else
  echo "==> [3/5] ERA5 data already cached ($N_CSV stations) — skipping download"
  echo "          (run 'REFETCH=1 bash reproduce.sh' to force re-download)"
fi

# ----------------------------------------------------------------------------
# 4. Run the experiments (deterministic; fixed seeds)
# ----------------------------------------------------------------------------
echo "==> [4/5] running experiments"
echo "    - main comparison  (Table 1 sota, Table 2 wilcoxon)"
python code/run_rabs_era5_main.py
echo "    - urgency ablation (baseline for the revised Table 4 cross-check)"
python code/run_rabs_era5_ablation.py
echo "    - scalability, 20 real stations (Table 3)"
python code/run_rabs_era5_scaling.py
echo "    - revision round: extreme-burst main comparison (Reviewer 2, Q2)"
python code/run_rabs_era5_main_extreme.py
echo "    - revision round: risk-channel decomposition, 2 regimes (Reviewer 2, Q4)"
python code/abl_q4_full.py
echo "    - revision round: surrogate-vs-realized calibration (Reviewer 2, Q5)"
python code/abl_q5_calibration.py

# ----------------------------------------------------------------------------
# 5. Regenerate manuscript tables + figure from the fresh CSVs
# ----------------------------------------------------------------------------
echo "==> [5/5] regenerating LaTeX tables + trade-off figure"
python code/make_era5_tables.py
# The revised Table 4 must be written AFTER make_era5_tables.py, which emits the
# legacy 5-row ablation table. make_ablation_table_final.py rebuilds it on the
# consistent deployed raw-p basis (8 rows) and aborts if any overlapping value
# drifts from the submitted ablation summary.
python code/make_ablation_table_final.py
RABS_SUMMARY_CSV=outputs/rabs/rabs_era5_summary.csv \
  python code/make_tradeoff_plot_v2.py

echo
echo "============================================================"
echo " DONE. Reproduced artifacts:"
echo "   outputs/rabs/rabs_era5_summary.csv            (main)"
echo "   outputs/rabs/rabs_era5_ablation_summary.csv   (ablation baseline)"
echo "   outputs/rabs/rabs_era5_scaling_summary.csv    (scalability)"
echo "   outputs/rabs/rabs_era5_extreme_summary.csv    (extreme burst, Q2)"
echo "   outputs/rabs/rabs_q4_full_ablation.csv        (risk decomposition, Q4)"
echo "   outputs/rabs/rabs_q4_final_matrix.csv         (revised Table 4 source)"
echo "   outputs/rabs/rabs_q5_surrogate_calibration.csv (bound calibration, Q5)"
echo "   outputs/tables/{sota_comparison,wilcoxon,scalability,ablation_urgency}.tex"
echo "   outputs/figures/tradeoff_plot.pdf"
echo "============================================================"
