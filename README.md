# ATML PA3

Advanced Topics in Machine Learning, Fall 2026. Specification: `ATML-PA3.pdf`.

## Status

This repository contains experiment implementations, protocol tests, plotting,
and evidence-audit commands. Full training and benchmark runs have **not** been
performed. No experimental results are claimed. GPU integration and the exact
course-export schema must be checked when the required assets are available.

The original `first_plot_questions.yaml` has been obtained from the GitHub link
embedded on page 3 of the manual. Its original bytes are preserved; the loader
supports its 24 records, including the eight JSON-format system instructions.
Source URL and SHA256 are in `manifests/task1/qualitative-source.json`.
`ATML-PA3-Data/` (including its original `README.MD`) is still required for Task 3.
The PDF embeds its Google Drive link on page 11. Do not invent replacement data.

## Layout

The organization follows the earlier PA1/PA2 task-specific packages and shared
helpers. The manual specifies repository contents, not an exact directory tree.

| Path | Purpose |
| --- | --- |
| `common/` | Reproducibility, losses, checkpoints, LoRA/OPLoRA, VQA normalization |
| `task1_finetuning/` | Data filtering, five training runs, benchmark/qualitative evaluation, diagnostics |
| `task2_vlm/` | COCO/VQAv2 preparation, cached SigLIP features, visual model, curriculum, controls |
| `task3_inference/` | Rollout ingestion, answer scoring, offline selection, adaptive sampling, live inference |
| `configs/` | Fixed protocol and explicit course-export column mapping |
| `scripts/` | End-to-end task commands and environment checks |
| `manifests/` | Tracked selected example/image/pair IDs and saved rollout orders |
| `results/` | Tracked compact measurements, per-example outputs, environment records |
| `report/figures/` | Generated figures/tables; report prose is student-written |
| `manual/` | Requirement-to-code map |
| `tests/` | Numerical, leakage, masking, visual-interface and inference tests |
| `pa3/` | CLI dispatch, evidence audit, shared plotting |
| `data/`, `checkpoints/`, `external/` | Local raw assets, recovery state, upstream evaluators; excluded from Git |

## Install and check

Run from the repository root on Apple Silicon (MPS) or a CUDA machine using Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m scripts.check_environment
python -m pytest -q
pa3 --help
```

`requirements.txt` installs the project and the version constraints in
`pyproject.toml`. These are implementation pins, not staff-provided pins, and a
clean installation has not yet been validated here. Every experiment records its
observed package versions, GPU, CUDA version, model revision and Git commit.
Capture a full environment lock on the machine actually used:

```bash
python -m pip freeze > results/environment-lock.txt
```

Training/live-generation commands select CUDA when available, otherwise MPS.
Force Apple GPU execution with `--device mps` on a `pa3` command, or set
`export PA3_DEVICE=mps`. CPU-only processes fail explicitly for GPU experiments.
Offline Task 3 analysis and data preparation can run on CPU.

### Apple Silicon checks

```bash
python -m scripts.smoke --device mps
```

This checks native FP16 autocast, a finite FP32-master AdamW update, FP32 moment
states, forced-overflow step skipping/scale backoff, scaler/RNG checkpoint
roundtrips, checkpointed tiny-Qwen backpropagation, frozen FP16 weights with FP32
OPLoRA factors, projected merging, and generation. Synthetic smoke results are
not assignment experiment results. Physical MPS tests are skipped when the
executing process cannot access the Apple GPU; a CPU smoke pass does not qualify MPS.

Then test real Qwen3-0.6B Full FT at the prescribed microbatch and maximum length:

```bash
caffeinate -i python -m scripts.smoke --device mps --real-model --length 2048 --steps 2
```

The real-model test downloads weights if needed and uses synthetic tokens. Its
default requires two verified successful optimizer updates within at most 12
attempts (`--steps 2 --max-attempts 12`). Early overflow skips allow the native
dynamic scaler to calibrate; every attempt prints and saves loss, scale changes,
unscaled gradient norm and a parameter-update check. Exhausting the attempt
budget still fails and preserves the diagnostics. It
does not alter the selected training data or replace the full experiment. It
checks microbatch memory and optimizer operation, not the full 16-microbatch
accumulation schedule or the entire training/evaluation pipeline. Preserve its
JSON report before launching long runs. OOM or unsupported operations are
reported, never silently bypassed with smaller batches or different precision.

MPS uses eager attention and explicit **exact FP32 CPU SVD** for orthogonal bases.
Forward/backward, optimizer operations and generation remain on the selected GPU.
No quantization, BF16/FP32 training substitution, disabled scaler, reduced batch,
shortened sequence budget or fewer examples/updates are applied. Device changes
do not change the prescribed experiment parameters. Resume a run on its original
backend; cross-backend RNG/optimizer continuation is rejected.

PyTorch exposes current MPS tensor and driver allocations, rather than CUDA's
peak-allocator API. An in-process monitor samples both every 10 ms and logs their
high-water marks as **sampled lower bounds**, including the measurement method and
interval. These are not exact CUDA-equivalent peaks or a measurement of all macOS
unified-memory use. Record this limitation in resource comparisons. Timing uses
the matching backend's synchronization calls. MPS RNG state is saved/restored.
The code does not enable global automatic CPU-operation fallback or disable MPS
memory safety limits.

To keep a Mac awake for a long run, plug it in, keep the lid open, and prefix the
command with `caffeinate -i`. For example, after successful preparation/checks:

```bash
caffeinate -i pa3 task1-train --method full --device mps
```

This keeps the invoked process awake until it exits; it does not make an AI
session persistent or restart work after the app/session closes.

## Task 1

Public models and datasets are loaded through Hugging Face. The immutable Qwen
revision is resolved once during preparation and reused by all five methods,
baseline evaluation and diagnostics. The prepared selection and probe manifests
are saved in `manifests/task1/`.

```bash
pa3 task1-prepare
pa3 task1-cache-probes
pa3 task1-evaluate --output results/task1/baseline
pa3 task1-train --method full
pa3 task1-evaluate --checkpoint checkpoints/task1/full/update_625.pt --output results/task1/full_midpoint
pa3 task1-evaluate --checkpoint checkpoints/task1/full/update_1250.pt --output results/task1/full
```

The remaining method names are `l2sp`, `lora4`, `lora16`, and `oplora16`.
`bash scripts/task1.sh` runs the full required schedule, including all seven
quantitative/qualitative conditions and six weight/activation diagnostics.
It requires the qualitative YAML before reaching its generation steps.
After all five training runs have finished, use the following command to finish
Task 1 without repeating training:

```bash
caffeinate -i bash scripts/task1_finish.sh
```

This runs all seven benchmark conditions, caches the fixed activation probes,
computes six checkpoint diagnostics, creates figures and tables, and generates
the seven sets of qualitative responses when the original course YAML is present.
It defaults to MPS and evaluation batch size 1 to limit memory; this changes
evaluation throughput, not the prescribed training batches or scoring settings.
Stage logs and completion markers are saved in `results/task1/posttraining/`.
Rerunning the command skips completed stages with their output files present;
an interrupted benchmark condition starts again from its beginning. If the YAML
is missing, benchmarks/diagnostics/figures finish first, then the script exits
with a message requesting the original file. Rerun after placing it in the
project folder. Human annotations and report writing remain student work.

For resuming a saved training run:

```bash
pa3 task1-train --method full --resume checkpoints/task1/full/update_625.pt
```

Preparation retains the first 40,000 eligible source-order examples, excluding
GSM8K-test question overlaps, empty content and complete formatted sequences over
2,048 tokens. It never truncates an accepted response. Loss is normalized by
the total supervised response/end-of-turn tokens in each effective batch of 32.
Recovery checkpoints are at updates 312, 625, 937, 1250 and include optimizer,
scheduler, scaler, RNG and data position. Skipped scaler updates are recorded.
Task 1 computes the output head and FP32 cross-entropy in checkpointed chunks of
128 supervised tokens and releases microbatch outputs immediately after backward.
All tokens still pass through the transformer; masked tokens contribute no loss.
This avoids full-sequence vocabulary logits without changing microbatch size,
token weighting, sequence lengths or model precision. Update timings and an approximate
remaining-time estimate appear in the terminal and training JSONL log.
On MPS, each microbatch also collects dead autograd objects, synchronizes, and
releases unused allocator cache. Live tensor and total Metal-driver memory after
cleanup are printed per update. Cleanup adds runtime but changes no training
parameters; Metal-driver memory includes framework allocations as well as tensors.
MPS Task 1 training right-pads each microbatch to the smallest of 512, 1024 or
2048 positions that fits. Attention masks are zero and labels are -100 on added
padding. No real tokens are removed, no examples are regrouped or reordered,
and the response-token denominator stays unchanged. The padding policy is saved
in the run environment. A four-update full-FT MPS diagnostic including long
examples completed with finite gradients; full-run stability remains unverified.

Full FT uses FP32 trainable parameters with FP16 autocast. L2-SP sums over unique
parameters without averaging; its fixed reference is kept on CPU and transferred
parameter by parameter to limit GPU storage, with transfer/storage overhead
recorded. LoRA/OPLoRA use frozen FP16 originals and FP32 trainable adapters.
OPLoRA uses exact FP32 SVD bases, factorized projections and projected exports.
Its preprocessing can be expensive; it is not replaced by a randomized SVD.

Benchmarks use fixed installed harness definitions, complete prescribed splits,
zero-shot prompts, FP16 and stock scoring/stopping rules. The non-thinking chat
template applies only to GSM8K. Adapter checkpoints are merged into fresh
backbones and the merged/unmerged logits are checked. Full FT midpoint and final
and all four other finals receive diagnostics on layers 0/13/27. Diagnostic
activations use the stock benchmark prompts and the saved 32+32 probe IDs.

For qualitative generation the original course YAML is supported directly,
preserving `paraphrases[0]` as user text and `system` as a separate system message.
The loader also supports a normalized list of 24 records with keys
`question_id`, `format` (`plain`, `json`, `template`) and `prompt`. Preserve the
course file's exact 24 prompt strings and underlying plain-question order when
converting its schema. Run:

```bash
pa3 task1-qualitative --condition baseline
pa3 task1-qualitative --condition full --checkpoint checkpoints/task1/full/update_1250.pt
```

Each condition saves 24 responses. The first and last underlying plain questions
are flagged for manual annotation in all three formats and all seven conditions:
42 required annotations. Empty annotation fields are intentional student review
work, not machine-generated conclusions. Truncation flags and token IDs are saved.

## Task 2

To download the original annotations/images, obtain the official scorer and
prepare the dataset manifests without GPU computation, run:

```bash
caffeinate -i bash scripts/task2_prepare.sh
```

This can run in a second Terminal while Task 1 evaluation continues. Allow
roughly 20 GB of downloads and 40–45 GB of space for archives plus extracted
files. Partial downloads resume when the same command is rerun. It does not
start feature extraction or training. Inspect selected labels/images before
training; run image feature caching after the current GPU evaluation finishes.

Obtain COCO 2014 images, captions and instance annotations, plus VQAv2 2014
questions, human answers and complementary pairs. Keep the original filenames in
one annotations directory:

| File pattern | Splits |
| --- | --- |
| `captions_{split}2014.json` | train, val |
| `instances_{split}2014.json` | train, val |
| `v2_OpenEnded_mscoco_{split}2014_questions.json` | train, val |
| `v2_mscoco_{split}2014_annotations.json` | train, val |
| `v2_mscoco_{split}2014_complementary_pairs.json` | train, val |

Images must be in `COCO_IMAGES_ROOT/train2014/` and `val2014/` with standard COCO
filenames. Install the official VQA answer normalizer locally:

```bash
git clone https://github.com/GT-Vision-Lab/VQA.git external/VQA
git -C external/VQA rev-parse HEAD > manifests/vqa-source-revision.txt
bash scripts/task2.sh ANNOTATIONS_ROOT COCO_IMAGES_ROOT external/VQA/PythonEvaluationTools/vqaEvaluation/vqaEval.py
```

Resolve and record the upstream revision before experiments; reuse it throughout.
Only official normalization is called; no annotation or predicted answer is sent
to an external service. The normalizer source is fingerprinted in manifests.

Preparation enforces 6,000 training-caption images ×3 captions and 96 development
caption images; natural pair type counts 640/256/640 (train), 12/8/12 (dev), and
36/24/36 (test); grounding totals 24 development pairs and 64 final-test pairs.
Presence/counting evaluation is balanced (an implementation selection choice;
the manual constrains their combined count). Natural targets require 8/10
agreement for both images and different normalized consensus answers. All ten
human references are retained. Training pools are reusable and may be resampled.

All development/final image sets are globally disjoint across captions and both
VQA suites; training uses train2014 only. Mismatched-image mappings are fixed
before generation and exclude both images of the question's own pair. Inspect
the selected annotation-derived labels/images independently of model outputs,
since COCO annotations can omit objects or have ambiguous counts.

Feature caching saves final, already-normalized 196×768 SigLIP patch features in
FP16; it does not apply a second layer norm. Feature caching is a separate process
from Qwen training and records its own time/memory. The 768→1024→1024 GELU
connector scale is fixed by RMS matching on the first 128 selected training
captions, before training. The rule and scale are saved.

Stage 1 performs two passes through 18,000 captions. Later stages use seeded
random pair draws with the prescribed probabilities, without enforcing quotas.
Loss averages supervised-token losses within each example, then across the
effective batch of 16. Stages reset the optimizer/scheduler and use the stated
per-stage learning rates, warmup and cosine floor, retaining connector/adapters.
Resumable checkpoints include sampler RNG, stage/data position and progress.

```bash
pa3 task2-train --resume checkpoints/task2/update_2500.pt
pa3 task2-evaluate --checkpoint checkpoints/task2/update_4500.pt --vqa-eval external/VQA/PythonEvaluationTools/vqaEvaluation/vqaEval.py
```

Development runs before training, every 500 main-run updates and at all stage
endpoints. Final evaluation requires update 4500 and uses four image conditions,
batch size 8, greedy decoding and 16 new tokens. It reports original-reference
scores/NLL, pair accuracy, identical/empty answers, partner-answer following,
time per answer, type breakdowns and 4,000-resample paired bootstrap intervals.
No-image inputs omit visual tokens and rebuild positions/masks. Optional 49-token
compression is selected with `--compressed --output checkpoints/task2_49` for a
fresh complete training run using the same selected data.

## Task 3

The original export is now extracted under `ATML-PA3-Data/`. The downloaded
README and Parquet schemas have been checked and mapped in
`configs/rollout_schema.json`. Original MATH question strings were recovered
from `question_text` into `data/task3_downloads/math_questions.json`, so no
separate MATH dataset download is needed. Download provenance is recorded in
`data/task3_downloads/dependencies.json` and setup checks in
`results/validation/task3-setup-review.json`.

After downloads and setup, prepare the original candidate pools and fixed splits:

```bash
caffeinate -i bash scripts/task3_prepare.sh
```

This uses CPU computation for joining and answer scoring. The official MathVista
normalization function and its closest-choice helper are loaded unchanged from
the downloaded source, without importing its unrelated image/API modules.
The distance helper computes exact unit-cost Unicode Levenshtein distance,
preserving the official closest-choice and first-choice tie rules. The full
normalizer source and local implementation hashes are recorded for attribution
and reproducibility. No model judges or external answer-extraction calls run.

Once preparation finishes successfully, run all remaining Task 3 computations:

```bash
caffeinate -i bash scripts/task3_finish.sh
```

The runner executes offline selection, adaptive-policy development/evaluation,
50-question live generation, illustrative-record selection, figures/tables and
an output audit, stopping on errors. Logs/finished-stage markers are saved in
`results/task3/progress/`. Rerunning skips completed stages whose expected files
are present. Interrupted offline/adaptive stages restart that stage; live
generation resumes from complete question records, repeating only an unfinished
question. The original prompts, caps, seeds, sampling and actual-cost accounting
are unchanged. It never starts training. Leave the preparation command to finish
before invoking this runner, and do not run another GPU workload alongside it.

Place the original course export under `ATML-PA3-Data/`, keeping its README and
folder/file structure intact. Inspect the real schema before mapping its columns:

```bash
pa3 task3-inspect
```

Copy `configs/rollout_schema.example.json` to `configs/rollout_schema.json`, then
replace the explicitly marked example paths/columns using the supplied README
and the inspection output. This mapping cannot be finalized without the actual
export. Joins are one-to-one on dataset/sample/rollout IDs and validate every
retained token-list length and finite token measurement.

MathVista requires its released `testmini.json` metadata and official
`normalize_extracted_answer` function (from the upstream evaluation code). Clone
the source locally, preserve its license, record its revision, and install its
evaluation dependencies if required. No LLM answer extraction is used for the
stored rollouts: the course-supplied extractions are authoritative.

```bash
git clone https://github.com/lupantech/MathVista.git external/MathVista
git -C external/MathVista rev-parse HEAD > manifests/mathvista-source-revision.txt
pa3 task3-prepare --schema configs/rollout_schema.json --mathvista-metadata MATHVISTA_TESTMINI_JSON --mathvista-normalizer OFFICIAL_SCORE_PY
pa3 task3-offline
pa3 task3-adaptive
```

`OFFICIAL_SCORE_PY` must expose the upstream normalizer with arguments
`(extraction, choices, question_type, answer_type, precision)`. Record its exact
source revision and SHA256; the preparation command records the latter. MATH
equivalence uses pinned `math-verify`. MathVista preserves official answer-type,
choice and precision rules and uses option letters for voting. Missing
extractions remain in a single incorrect `NO_ANSWER` group. Gold labels are
stored separately from selector inputs. The scorer is fixed before held-out use.

The seeded 200/800 split follows lexicographic question-ID sorting. Ten saved
rollout permutations per question provide nested prefixes, shared across all
methods and controllers. RWC uses average tied ranks; score and answer-vote ties
use the earliest supporting rollout. VisConf recalibrates the token KL percentiles
on each revealed prefix only. FirstAns is charged for just its first candidate.
Question-level accuracy differences average over orders before 1,000 paired
bootstrap resamples. Quartile diagnostics use exactly eight candidates per tail,
breaking cutoff ties by original rollout order. Constant-signal correlations
are recorded as undefined, not converted into a numeric association.

Adaptive policies are selected using only development data, saved, then applied
to held-out data. Both controllers return self-consistency; quality only changes
when sampling stops. The fixed-32 fallback, agreement/no-answer checks, RWC margin,
one-percentage-point development tolerance and threshold tie rules are included.

The live-generation command requires the exact immutable model revision in the
original rollout README plus a JSON dictionary mapping course sample IDs to the
original problem strings (or `{ "problem": "..." }` records):

```bash
pa3 task3-live --questions ORIGINAL_MATH_PROBLEMS_JSON --revision ORIGINAL_README_REVISION --rollout-readme ATML-PA3-Data/README.MD
```

The saved first 50 held-out IDs determine the live subset. Long, three-way
parallel, and critique/revision strategies use the stated 3,072-token allowance.
The first parallel call is reused and charged to both parallel/refinement.
Prompts, independent seeds, generated IDs, extraction/truncation, input/output
cost, synchronized GPU time/memory and revision transitions are saved. The balanced-brace
extractor must be compared with the supplied README's exact rule once available;
it currently chooses the last `\\boxed{...}` and rejects unbalanced/empty boxes.

## Figures and final checks

```bash
pa3 plots --output report/figures
python -m scripts.make_examples
pa3 audit
```

Add `--images COCO_IMAGES_ROOT` to the plotting command to render the fixed
complementary-image example panels alongside their references/predictions.

Plotting reads saved measurements only. It creates learning curves, benchmark
and diagnostic/resource CSVs, VQA type tables, deterministic qualitative-example
records, selection accuracy/cost curves, signal tables, token trajectories,
stopping distributions and live-inference tables as measurements become available.
`scripts.make_examples` exports retrospective selection-disagreement, revision
and premature-stop cases using explicitly recorded deterministic selection rules.
It does not feed any correctness information back into selectors/controllers.
Missing experiments produce no fabricated plots. The evidence audit checks for
expected outputs and budgets; it is not a substitute for semantic review.

Keep compact JSON/JSONL/CSV logs, saved split IDs, environment information and
figures in Git. Raw datasets, downloaded models and recovery checkpoints remain
ignored. The repository must be public at submission. Submit the 8-page NeurIPS
report separately according to LMS instructions, with the repository link at the
end of the abstract. References and optional appendix are outside the page limit.

## Attribution and AI use

LLM coding assistance was used in this implementation. The student must inspect,
understand and take responsibility for each submitted line. No PDF-report prose,
interpretations or conclusions are generated here. Qualitative annotations and
the report's language/analysis must be the student's own.

Public dependencies and reference implementations:

- Qwen3 native non-thinking template: https://github.com/QwenLM/Qwen3
- Hugging Face Transformers model implementations, including SigLIP's already
  normalized patch outputs: https://github.com/huggingface/transformers
- Stock task definitions, answer filters and evaluation API:
  https://github.com/EleutherAI/lm-evaluation-harness
- Mathematical equivalence: https://github.com/huggingface/Math-Verify
- Official VQA answer normalization (called from a local checkout):
  https://github.com/GT-Vision-Lab/VQA
- Official MathVista answer normalization (called from a local checkout):
  https://github.com/lupantech/MathVista

The experiment logic and adapter code implement the formulas in the assignment;
there is no copied complete PA3 solution or invented course starter repository.
The required paper readings remain the student's responsibility.

## Restoring compressed result files

Large JSON/JSONL results and manifests are committed as `.gz` files. After cloning, restore them before using the analysis scripts:

```bash
python - <<'PY'
import gzip
import shutil
from pathlib import Path

for root in (Path("results"), Path("manifests")):
    for archive in root.rglob("*.gz"):
        target = archive.with_suffix("")
        if not target.exists():
            with gzip.open(archive, "rb") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
PY
```
