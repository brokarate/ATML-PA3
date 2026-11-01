#!/usr/bin/env bash
# Validate joins, score stored answers, and save fixed splits/orders. No GPU work.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-2}"
python -m pa3.cli task3-inspect
python -m pa3.cli task3-prepare --schema configs/rollout_schema.json \
  --mathvista-metadata external/MathVista/data/testmini.json \
  --mathvista-normalizer external/MathVista/evaluation/calculate_score.py
echo "Task 3 data prepared. Next steps: offline selection, adaptive analysis, and 50-question live generation."
