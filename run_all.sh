#!/usr/bin/env bash
# Full pipeline, steps S01-S13, on the real data. Run from anywhere:   bash run_all.sh
# Each step writes its tables / figures / models to <step>/Results/ and appends its console output to <step>/Results/run_log.txt.
# Shared intermediate tables (features, edges, DE tables, CV groups) go to Data/processed/.
# Every script runs on its own, so you can also run the lines below one by one. About 6-7 h in total on a 12-core laptop.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1

run() {   # run <step folder> <script> [args...]; console output is appended to <step>/Results/run_log.txt
  local step=$1 script=$2; shift 2
  local log="$step/Results/run_log.txt"
  mkdir -p "$step/Results"
  echo "### $(date '+%F %T')  $script $*" >> "$log"
  echo "$(date '+%T')  $step/$script $*"
  # stdin from /dev/null: on Windows, joblib/loky worker pools (PyDESeq2) fail with WinError 6 when stdin is not a valid handle
  if ! "$PY" "$step/$script" "$@" < /dev/null >> "$log" 2>&1; then
    echo "FAILED: $step/$script - last lines of $log:"; tail -n 20 "$log"; return 1
  fi
}

run S01_Data_Download 01_download.py --extra huri string_exp || echo "WARNING: a download failed; continuing with files already in Data/raw"
# Manual inputs (see README): Data/labels/source/{gordon_2020_ST2.xlsx,baggen_2021_ST.xlsx}; CORUM, BioPlex, housekeeping list and
# STRING v10.5 / v11.5 / v12.0 files in Data/raw/string_versions (see README, section Data)
run S02_Label_Sets 00_build_labels.py
run S03_Differential_Expression 02_de.py
run S04_Network_Features 03_network.py --ppi string_v11
run S04_Network_Features 03_network.py --ppi huri
run S04_Network_Features 03c_shared_huri.py
run S05_Communities_And_CV_Groups 03b_groups.py --ppi string_v11
for L in gordon crispr; do run S06_Prediction_Models 04_models.py --ppi string_v11 --labels $L --B 2000; done
run S06_Prediction_Models 04_models.py --ppi string_v11 --labels crispr_min2 --B 2000 --no_rf
run S06_Prediction_Models 04_models.py --ppi huri --labels gordon --B 1000
run S06_Prediction_Models 04_models.py --ppi huri --labels crispr --B 1000 --no_rf
for n in string_v11 huri; do
  run S06_Prediction_Models 04_models.py --ppi $n --labels gordon --subset Data/processed/shared_huri_v11.csv --tag shared_${n}__gordon --no_rf --B 1000
done
run S07_Rewiring_Null 05_null.py --ppi string_v11 --labels gordon --R 200 --procs 8
run S08_STRING_Versions_And_Channels 06a_build_panel.py
run S08_STRING_Versions_And_Channels 06b_panel_analysis.py --labels gordon crispr
run S08_STRING_Versions_And_Channels 06c_panel_tests.py --labels gordon crispr --B 2000
run S09_Transfer 07_transfer.py
run S10_Interpretation 08_interpret.py --tag string_v11__gordon
run S11_HuRI_Sparsity_Control 09_sparsity_control.py
run S12_Extensions_v105_BioPlex_Sensitivity 11_extensions.py --B 500
run S13_Report_Figures 10_report_figures.py
echo "Done. Each step's outputs are in <step>/Results/; report figures F1-F9 and table T1 in S13_Report_Figures/Results/."
