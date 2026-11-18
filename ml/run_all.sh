#!/bin/bash
# Full whisky-web ML pipeline (incremental feature extraction -> models -> plots).
set -e
PY=C:/Users/joshk/anaconda3/envs/PhD/python.exe
H=/c/Users/joshk/jarvis/jarvis/home/outputs/whisky
CACHE=$H/ml_cache
FEAT=$H/ml_features
OUT=${1:-/c/Users/joshk/jarvis/jarvis/home/outputs/2026-11-13/ml/results}
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore
echo "STEP extract"; $PY -u extract_features.py $CACHE $FEAT --workers 14
echo "STEP embed";   $PY -u embed_cnn.py $CACHE $FEAT | tail -1
echo "STEP train";   $PY -u train.py $FEAT $OUT
echo "STEP polar";   $PY -u polar_cnn.py $FEAT $OUT --epochs 30
echo "STEP polar+radial"; $PY -u polar_cnn.py $FEAT $OUT --epochs 30 --with-profile
echo "STEP plots";   $PY -u plots.py $OUT "../Whisky_Key.csv"
echo "ALL DONE"
