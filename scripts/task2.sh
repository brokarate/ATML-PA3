#!/usr/bin/env bash
set -euo pipefail
# Paths are arguments, so no assumptions about where COCO was downloaded.
if [[ $# -ne 3 ]]; then
  echo 'Usage: bash scripts/task2.sh ANNOTATIONS_ROOT COCO_IMAGES_ROOT VQA_EVAL_PY'
  exit 2
fi
pa3 task2-prepare --annotations "$1" --vqa-eval "$3"
pa3 task2-cache --images "$2"
pa3 task2-train
pa3 task2-evaluate --checkpoint checkpoints/task2/update_4500.pt --vqa-eval "$3"
