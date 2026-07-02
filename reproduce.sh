#!/usr/bin/env bash
# =============================================================================
# RABS — one-command reproduction of every number in the paper.
#
#   Self-Tuning Risk-Adaptive Bandwidth Scaling for
#   Safety-Critical Smart-Agriculture IoT Networks
#
# What this does, end to end:
#   1. creates an isolated Python virtual environment (.venv)
#   2. installs the only third-party deps (numpy + scipy, for paired stats)
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
# 2. Dependencies (numpy + scipy only; simulators are stdlib-only)
# ----------------------------------------------------------------------------
echo "==> [2/5] installing dependencies (numpy, scipy)"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet numpy scipy

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
echo "    - urgency ablation (Table 4)"
python code/run_rabs_era5_ablation.py
echo "    - scalability, 20 real stations (Table 3)"
python code/run_rabs_era5_scaling.py

# ----------------------------------------------------------------------------
# 5. Regenerate manuscript tables + figure from the fresh CSVs
# ----------------------------------------------------------------------------
echo "==> [5/5] regenerating LaTeX tables + trade-off figure"
python code/make_era5_tables.py
RABS_SUMMARY_CSV=outputs/rabs/rabs_era5_summary.csv \
  python code/make_tradeoff_plot.py

echo
echo "============================================================"
echo " DONE. Reproduced artifacts:"
echo "   outputs/rabs/rabs_era5_summary.csv           (main)"
echo "   outputs/rabs/rabs_era5_ablation_summary.csv  (ablation)"
echo "   outputs/rabs/rabs_era5_scaling_summary.csv   (scalability)"
echo "   outputs/tables/{sota_comparison,wilcoxon,scalability,ablation_urgency}.tex"
echo "   outputs/figures/tradeoff_plot.pdf"
echo "============================================================"
