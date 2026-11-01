#!/usr/bin/env bash
# Download original Task 2 data and build manifests; no GPU work or training.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-2}"
archives=data/task2_downloads
annotations=data/task2_annotations
images=data/coco
mkdir -p "$archives" "$annotations" "$images" manifests external

download() {
  local url="$1" archive="$archives/$2"
  if [[ ! -s "$archive" ]]; then
    echo "Downloading $2 (partial downloads can be resumed)"
    if [[ "${PA3_DOWNLOAD_WORKERS:-1}" != 1 ]]; then
      python -m scripts.parallel_download "$url" "$archive.part" \
        --workers "$PA3_DOWNLOAD_WORKERS"
    else
      curl --fail --location --retry 5 --connect-timeout 30 --continue-at - \
        --output "$archive.part" "$url"
    fi
    unzip -tq "$archive.part"
    mv "$archive.part" "$archive"
  fi
}

# Access the official public bucket through AWS's HTTPS endpoint: the custom
# images.cocodataset.org host currently has a certificate hostname mismatch.
download https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip annotations_trainval2014.zip
unzip -nq "$archives/annotations_trainval2014.zip" 'annotations/captions_*.json' 'annotations/instances_*.json' -d "$archives"
for split in train val; do
  for kind in captions instances; do
    cp -p "$archives/annotations/${kind}_${split}2014.json" "$annotations/"
  done
done
for split in Train Val; do
  for kind in Questions Annotations Complementary_Pairs; do
    file="v2_${kind}_${split}_mscoco.zip"
    download "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/$file" "$file"
    unzip -nq "$archives/$file" '*.json' -d "$annotations"
  done
done

if [[ ! -d external/VQA ]]; then
  git clone https://github.com/GT-Vision-Lab/VQA.git external/VQA
fi
git -C external/VQA rev-parse HEAD > manifests/vqa-source-revision.txt
python -m pa3.cli task2-prepare --annotations "$annotations" \
  --vqa-eval external/VQA/PythonEvaluationTools/vqaEvaluation/vqaEval.py

# These are the original 2014 images linked by the official VQA download page.
for split in train val; do
  download "https://s3.amazonaws.com/images.cocodataset.org/zips/${split}2014.zip" "${split}2014.zip"
  if [[ ! -f "$archives/${split}2014.extracted" ]]; then
    echo "Extracting ${split}2014 images"
    unzip -nq "$archives/${split}2014.zip" -d "$images"
    touch "$archives/${split}2014.extracted"
  fi
done
python - <<'PY'
import json
from pathlib import Path
root = Path('data/coco')
count = 0
for manifest in sorted(Path('data/task2').glob('*.jsonl')):
    split = 'train2014' if manifest.stem.endswith('_train') else 'val2014'
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        for example in row.get('examples', [row]):
            image_id = int(example['image_id'])
            image = root / split / f'COCO_{split}_{image_id:012d}.jpg'
            if not image.is_file():
                raise FileNotFoundError(image)
            count += 1
print(f'Checked image files for {count} selected example entries.')
PY
echo "Task 2 downloads and dataset selection finished."
echo "Next: inspect selected labels/images, then cache image features after Task 1 releases the GPU."
