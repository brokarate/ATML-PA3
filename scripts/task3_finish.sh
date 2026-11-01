#!/usr/bin/env bash
# Run the remaining Task 3 computations after task3_prepare.sh has succeeded.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-2}"
export HF_HUB_OFFLINE=1
device="${PA3_DEVICE:-mps}"
progress=results/task3/progress
mkdir -p "$progress"
lock=results/task3/runner.lock
if ! mkdir "$lock" 2>/dev/null; then
  echo "A Task 3 runner is already active (or its lock remains after a forced shutdown)."
  exit 1
fi
trap 'rmdir "$lock"' EXIT

for file in data/task3/environment.json data/task3/sources.json \
  data/task3_downloads/math_questions.json ATML-PA3-Data/README.MD; do
  test -s "$file" || { echo "Preparation incomplete: missing $file. Let task3_prepare.sh finish first."; exit 1; }
done
for kind in math mathvista; do
  for suffix in candidates.jsonl labels.jsonl splits.json orders.json; do
    test -s "data/task3/${kind}_${suffix}"
  done
done

run_stage() {
  local name="$1"; shift
  local -a outputs=()
  while [[ "$1" != -- ]]; do outputs+=("$1"); shift; done
  shift
  local complete=true output
  for output in "${outputs[@]}"; do [[ -s "$output" ]] || complete=false; done
  if [[ -s "$progress/$name.done" && "$complete" == true ]]; then
    echo "Already completed: $name"
    return
  fi
  rm -f "$progress/$name.done"
  echo "Starting $name at $(date)"
  "$@" 2>&1 | tee "$progress/$name.log"
  for output in "${outputs[@]}"; do test -s "$output"; done
  date > "$progress/$name.done"
  echo "Completed $name at $(date)"
}

outputs=()
for kind in math mathvista; do
  for suffix in summary.csv per_question.jsonl question_means.csv accuracy_differences.json \
    signals.jsonl rollout_scores.jsonl trajectories.jsonl; do
    outputs+=("results/task3/offline/${kind}_${suffix}")
  done
done
run_stage offline "${outputs[@]}" -- python -m pa3.cli task3-offline

outputs=()
for kind in math mathvista; do
  for suffix in frozen_policies.json development_grid.json adaptive_summary.json \
    agreement_heldout.jsonl quality_heldout.jsonl; do
    outputs+=("results/task3/adaptive/${kind}_${suffix}")
  done
done
run_stage adaptive "${outputs[@]}" -- python -m pa3.cli task3-adaptive

run_stage live results/task3/live/calls.jsonl results/task3/live/per_question.jsonl \
  results/task3/live/environment.json -- python -m pa3.cli task3-live --device "$device" \
  --questions data/task3_downloads/math_questions.json \
  --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --rollout-readme ATML-PA3-Data/README.MD

python -m scripts.make_examples
python -m pa3.cli plots --images data/coco
python -m pa3.cli audit --output results/task3/completion-audit.json
python - <<'PY'
import json
from pathlib import Path
audit = json.loads(Path('results/task3/completion-audit.json').read_text())
failed = [r['requirement'] for r in audit['checks']
          if r['requirement'].startswith('Task 3 ') and not r['passed']]
if failed:
    raise SystemExit('Task 3 evidence checks failed: ' + ', '.join(failed))
PY
echo "Task 3 automated computation finished. Review the results/examples and write the report yourself."
