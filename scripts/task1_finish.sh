#!/usr/bin/env bash
# Finish Task 1 using existing training checkpoints. Never starts training.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-2}"
device="${PA3_DEVICE:-mps}"
batch_size="${PA3_EVAL_BATCH_SIZE:-1}"
prompts="${PA3_TASK1_PROMPTS:-first_plot_questions.yaml}"
progress=results/task1/posttraining
mkdir -p "$progress"

for method in full l2sp lora4 lora16 oplora16; do
  test -s "checkpoints/task1/$method/update_1250.pt" || {
    echo "Missing final checkpoint: $method"; exit 1;
  }
done
test -s checkpoints/task1/full/update_625.pt
test -s data/task1/probes.jsonl

# A completed stage is skipped only while its expected outputs still exist.
# An interrupted stage reruns from its beginning; training is never repeated.
run_stage() {
  local name="$1"; shift
  local -a outputs=()
  while [[ "$1" != -- ]]; do outputs+=("$1"); shift; done
  shift
  local complete=true
  local output
  for output in "${outputs[@]}"; do
    [[ -s "$output" ]] || complete=false
  done
  if [[ -f "$progress/$name.done" && "$complete" == true ]]; then
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

conditions=(baseline full_midpoint full l2sp lora4 lora16 oplora16)
for condition in "${conditions[@]}"; do
  checkpoint=()
  if [[ "$condition" == full_midpoint ]]; then
    checkpoint=(--checkpoint checkpoints/task1/full/update_625.pt)
  elif [[ "$condition" != baseline ]]; then
    checkpoint=(--checkpoint "checkpoints/task1/$condition/update_1250.pt")
  fi
  out="results/task1/$condition"
  run_stage "evaluate-$condition" "$out/gsm8k.json" "$out/gsm8k_samples.jsonl" \
    "$out/retention.json" "$out/merge_check.json" "$out/environment.json" -- \
    python -m pa3.cli task1-evaluate --device "$device" --batch-size "$batch_size" \
    ${checkpoint[@]+"${checkpoint[@]}"} --output "$out"
done

run_stage cache-probes data/task1/probe_cache.pt -- \
  python -m pa3.cli task1-cache-probes --device "$device"
for condition in full_midpoint full l2sp lora4 lora16 oplora16; do
  checkpoint="checkpoints/task1/$condition/update_1250.pt"
  [[ "$condition" != full_midpoint ]] || checkpoint=checkpoints/task1/full/update_625.pt
  output="results/task1/$condition/diagnostics.json"
  run_stage "diagnostics-$condition" "$output" -- \
    python -m pa3.cli task1-diagnostics --checkpoint "$checkpoint" --output "$output"
done
python -m pa3.cli plots

if [[ ! -s "$prompts" ]]; then
  echo "Benchmarks, diagnostics and figures finished."
  echo "Still required: the original first_plot_questions.yaml from the course."
  echo "Place it in the project folder and rerun this command to generate qualitative responses."
  exit 2
fi
for condition in "${conditions[@]}"; do
  checkpoint=()
  if [[ "$condition" == full_midpoint ]]; then
    checkpoint=(--checkpoint checkpoints/task1/full/update_625.pt)
  elif [[ "$condition" != baseline ]]; then
    checkpoint=(--checkpoint "checkpoints/task1/$condition/update_1250.pt")
  fi
  run_stage "qualitative-$condition" "results/task1/qualitative/$condition.jsonl" -- \
    python -m pa3.cli task1-qualitative --device "$device" ${checkpoint[@]+"${checkpoint[@]}"} \
    --condition "$condition" --prompts "$prompts"
done
echo "All Task 1 computation finished. Review outputs and complete the 42 required human annotations."
