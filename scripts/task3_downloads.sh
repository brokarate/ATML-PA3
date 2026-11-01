#!/usr/bin/env bash
# Download Task 3 public dependencies without loading any model onto the GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
mkdir -p external manifests data/task3_downloads
test -s ATML-PA3-Data/README.MD

if [[ ! -d external/MathVista ]]; then
  git clone --depth 1 https://github.com/lupantech/MathVista.git external/MathVista
fi
test -s external/MathVista/data/testmini.json
git -C external/MathVista rev-parse HEAD > manifests/mathvista-source-revision.txt

python - <<'PY'
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download

revision = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
readme = Path('ATML-PA3-Data/README.MD')
if revision not in readme.read_text():
    raise ValueError('Model revision no longer matches the supplied README')
source = Path('external/MathVista')
normalizers = [p for p in source.rglob('*.py')
               if 'def normalize_extracted_answer(' in p.read_text()]
if not normalizers:
    raise FileNotFoundError('Official MathVista answer normalizer not found')
print('MathVista metadata and official scoring source downloaded.', flush=True)
print('Downloading the README-pinned Qwen2.5-3B-Instruct weights/tokenizer; no GPU use.', flush=True)
location = snapshot_download(
    'Qwen/Qwen2.5-3B-Instruct',
    revision=revision,
    allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.tiktoken'],
    max_workers=4,
)
manifest = {
    'model': 'Qwen/Qwen2.5-3B-Instruct',
    'revision': revision,
    'local_snapshot': location,
    'rollout_readme_sha256': hashlib.sha256(readme.read_bytes()).hexdigest(),
    'mathvista_revision': Path('manifests/mathvista-source-revision.txt').read_text().strip(),
    'metadata': str(source / 'data/testmini.json'),
    'normalizer_sources': [str(p) for p in normalizers],
}
Path('data/task3_downloads/dependencies.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Task 3 downloads finished. Schema/scorer validation and experiments are separate steps.')
PY
