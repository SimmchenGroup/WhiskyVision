#!/bin/bash
# Full whisky-web ML pipeline (incremental feature extraction -> models -> plots).
set -e
#   bash run_all.sh [results dir]
# Working data lives in $WHISKY_ML_WORK (default ~/whisky_ml; see ml_paths.py),
# which must hold ml_cache/ from cache_images.py. Set PY to use another Python.
PY=${PY:-python}
H=${WHISKY_ML_WORK:-$HOME/whisky_ml}
CACHE=$H/ml_cache
FEAT=$H/ml_features
OUT=${1:-$H/results}
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore
echo "STEP extract"; $PY -u extract_features.py $CACHE $FEAT --workers 14
echo "STEP embed";   $PY -u embed_cnn.py $CACHE $FEAT | tail -1
echo "STEP train";   $PY -u train.py $FEAT $OUT
echo "STEP polar";   $PY -u polar_cnn.py $FEAT $OUT --epochs 30
echo "STEP polar+radial"; $PY -u polar_cnn.py $FEAT $OUT --epochs 30 --with-profile
echo "STEP plots";   $PY -u plots.py $OUT "../Whisky_Key.csv"
echo "ALL DONE"
