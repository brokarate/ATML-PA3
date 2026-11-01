#!/usr/bin/env bash
# Queue the prepared Task 2 run behind the currently running Task 1 finisher.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-2}"
device="${PA3_DEVICE:-mps}"
scorer=external/VQA/PythonEvaluationTools/vqaEvaluation/vqaEval.py
test -s data/task2/image_ids.json
test -s "$scorer"
test -f data/task2_downloads/train2014.extracted
test -f data/task2_downloads/val2014.extracted
if [[ -f checkpoints/task2/train.jsonl ]]; then
  echo "Task 2 already has a training log. Use its saved checkpoint to resume instead."
  exit 1
fi

# Run this from another Terminal tab. Wait for the actual shell process to exit,
# including any qualitative work it may perform after the benchmarks.
pids=$(pgrep -f '[b]ash scripts/task1_finish.sh') || {
  status=$?
  if [[ "$status" -ne 1 ]]; then
    echo "Cannot check Task 1's process; stopping before launching another GPU workload."
    exit 1
  fi
  pids=""
}
for pid in $pids; do
  echo "Waiting for Task 1 (process $pid). Leave this Terminal tab open."
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
done

# Task 1 may exit intentionally for the missing qualitative YAML. Quantitative
# work and diagnostics must nevertheless have finished before Task 2 starts.
for condition in baseline full_midpoint full l2sp lora4 lora16 oplora16; do
  test -s "results/task1/posttraining/evaluate-$condition.done" || {
    echo "Task 1 evaluation incomplete: $condition. Task 2 was not started."; exit 1;
  }
done
test -s results/task1/posttraining/cache-probes.done
for condition in full_midpoint full l2sp lora4 lora16 oplora16; do
  test -s "results/task1/posttraining/diagnostics-$condition.done" || {
    echo "Task 1 diagnostics incomplete: $condition. Task 2 was not started."; exit 1;
  }
done
test -s report/figures/task1_benchmarks.csv
test -s report/figures/task1_diagnostics.csv

mkdir -p results/task2
echo "Task 1 has released the GPU. Starting Task 2 at $(date)."
python -m pa3.cli task2-cache --device "$device" --images data/coco --batch-size 8 \
  2>&1 | tee results/task2/cache.log
python -m pa3.cli task2-train --device "$device" \
  2>&1 | tee results/task2/train.log
python -m pa3.cli task2-evaluate --device "$device" \
  --checkpoint checkpoints/task2/update_4500.pt --vqa-eval "$scorer" \
  2>&1 | tee results/task2/evaluate.log
python -m pa3.cli plots
echo "Task 2 computation completed at $(date). Review results and selected image/label examples."
