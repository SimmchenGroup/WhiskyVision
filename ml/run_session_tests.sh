#!/bin/bash
# Imaging-session tests: within-session CV on raw features, session z-scoring,
# and full per-image normalisation (re-extract -> embed -> train).
#   bash run_session_tests.sh [results dir]
# Working data lives in $WHISKY_ML_WORK (default ~/whisky_ml; see ml_paths.py).
# Set PY to use another Python.
PY=${PY:-python}
H=${WHISKY_ML_WORK:-$HOME/whisky_ml}
R=${1:-$H/session_tests}
mkdir -p "$R"
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore
echo "STEP raw";       $PY -u train.py $H/ml_features $R/results_raw > $R/log_raw.txt 2>&1
echo "STEP sessionz";  $PY -u train.py $H/ml_features $R/results_sessionz --session-z > $R/log_sessionz.txt 2>&1
echo "STEP extract-norm"; $PY -u extract_features.py $H/ml_cache $H/ml_features_norm --norm --workers 14 > $R/log_extract_norm.txt 2>&1
echo "STEP embed-norm"; $PY -u embed_cnn.py $H/ml_cache $H/ml_features_norm > $R/log_embed_norm.txt 2>&1
echo "STEP train-norm"; $PY -u train.py $H/ml_features_norm $R/results_norm --normalised > $R/log_norm.txt 2>&1
echo "ALL DONE"
