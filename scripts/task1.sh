#!/usr/bin/env bash
set -euo pipefail
pa3 task1-prepare
pa3 task1-cache-probes
pa3 task1-evaluate --output results/task1/baseline
pa3 task1-qualitative --condition baseline
for method in full l2sp lora4 lora16 oplora16; do
  pa3 task1-train --method "$method"
  pa3 task1-evaluate --checkpoint "checkpoints/task1/$method/update_1250.pt" --output "results/task1/$method"
  pa3 task1-qualitative --checkpoint "checkpoints/task1/$method/update_1250.pt" --condition "$method"
  pa3 task1-diagnostics --checkpoint "checkpoints/task1/$method/update_1250.pt" --output "results/task1/$method/diagnostics.json"
done
pa3 task1-evaluate --checkpoint checkpoints/task1/full/update_625.pt --output results/task1/full_midpoint
pa3 task1-qualitative --checkpoint checkpoints/task1/full/update_625.pt --condition full_midpoint
pa3 task1-diagnostics --checkpoint checkpoints/task1/full/update_625.pt --output results/task1/full_midpoint/diagnostics.json
