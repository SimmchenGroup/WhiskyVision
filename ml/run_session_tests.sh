#!/bin/bash
# Imaging-session tests: within-session CV on raw features, session z-scoring,
# and full per-image normalisation (re-extract -> embed -> train).
PY=C:/Users/joshk/anaconda3/envs/PhD/python.exe
H=/c/Users/joshk/jarvis/jarvis/home/outputs/whisky
R=/c/Users/joshk/jarvis/jarvis/home/outputs/2026-11-13/ml
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore
echo "STEP raw";       $PY -u train.py $H/ml_features $R/results_raw > $R/log_raw.txt 2>&1
echo "STEP sessionz";  $PY -u train.py $H/ml_features $R/results_sessionz --session-z > $R/log_sessionz.txt 2>&1
echo "STEP extract-norm"; $PY -u extract_features.py $H/ml_cache $H/ml_features_norm --norm --workers 14 > $R/log_extract_norm.txt 2>&1
echo "STEP embed-norm"; $PY -u embed_cnn.py $H/ml_cache $H/ml_features_norm > $R/log_embed_norm.txt 2>&1
echo "STEP train-norm"; $PY -u train.py $H/ml_features_norm $R/results_norm --normalised > $R/log_norm.txt 2>&1
echo "ALL DONE"
