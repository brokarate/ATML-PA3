# Implementation map

The authoritative specification is the original `ATML-PA3.pdf` at repository root.
This is an implementation checklist, not report prose or experimental analysis.

| Manual requirement | Code / output |
| --- | --- |
| Seed, precision, independent initialization, reproducibility | `common/`, `configs/protocol.json`, run environment JSON |
| MPS/CUDA device routing, native AMP, RNG, synchronization and measurements | `common/device.py`, `scripts/smoke.py`; MPS sampled memory peaks are explicitly labeled lower bounds |
| Task 1 source-order filtering and leakage exclusion | `task1_finetuning/data.py` |
| Token-weighted loss, 1,250 updates, resumable quarter checkpoints | `task1_finetuning/train.py` |
| Full FT and unaveraged, unique-parameter L2-SP | `task1_finetuning/train.py` |
| Matched LoRA/OPLoRA initialization, exact bases and projected export | `common/adapters.py` |
| Seven conditions and four complete stock benchmarks | `task1_finetuning/evaluate.py`, `scripts/task1.sh` |
| Merge verification | `task1_finetuning/model.py` |
| Weight displacement, rho, eta and fixed activation probes | `task1_finetuning/diagnostics.py` |
| 168 qualitative responses, 42 manual annotation slots | `task1_finetuning/evaluate.py` |
| COCO/VQAv2 splits, 8/10 agreement, IDs, image-disjoint sets | `task2_vlm/data.py` |
| Presence/counting pairs, reusable pools, exact natural type counts | `task2_vlm/data.py` |
| Frozen SigLIP, final layer norm once, cached FP16 features | `task2_vlm/data.py` |
| Connector, chat insertion, masks, positions, training-only calibration | `task2_vlm/model.py` |
| Three-stage 4,500-update curriculum and seeded paired sampling | `task2_vlm/train.py` |
| Example-weighted loss and FP32 trainables/state | `task2_vlm/model.py`, `task2_vlm/train.py` |
| Development schedules, reference-answer NLL, image-swap penalty | `task2_vlm/train.py` |
| Four final controls, scoring, type/pair diagnostics and 4,000 pair bootstraps | `task2_vlm/evaluate.py` |
| Optional 49 visual tokens | `task2_vlm/model.py`, `--compressed` |
| Supplied rollout joins, schema and token alignment | `task3_inference/data.py` |
| Fixed lexicographic seeded split, saved orders and live subset | `task3_inference/data.py` |
| Equivalent answer groups formed without gold, missing answers retained | `task3_inference/scoring.py` |
| KL/logp, normalized vision, prefix VisConf, tied-rank RWC | `task3_inference/scoring.py` |
| Required selectors/budgets, oracle and failure decomposition | `task3_inference/offline.py` |
| Question-averaged orders and paired question bootstrap | `task3_inference/offline.py` |
| Length/quartile/correctness associations and tokenwise fusion ablation | `task3_inference/offline.py` |
| Same fixed live revision, native prompts, shared initial response and actual costs | `task3_inference/live.py` |
| Development-only stopping thresholds, fixed-32 fallback, held-out evaluation | `task3_inference/adaptive.py` |
| Stopping distributions and retrospective early-stop failures | `task3_inference/adaptive.py` |
| Learning, benchmark, signal, cost and stopping figures | `pa3/plots.py` |
| Retrospective disagreement/revision/early-stop cases with selection rules | `scripts/make_examples.py` |
| Output completeness audit | `pa3/audit.py` |

Still requires real assets and execution: physical MPS/GPU smoke qualification,
exact rollout-schema mapping and README
verification, original qualitative YAML schema adaptation if needed, official
scorer source pinning, actual complete experiments, independent
label inspection, manual qualitative annotations, and the student-written report.
