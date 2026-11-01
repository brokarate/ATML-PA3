# Report evidence checklist — November 1, 2026

Technical inventory only. The student must write all report language,
interpretation and analysis under the manual's AI policy.

| Task / required evidence | Source of measurements | Current status |
| --- | --- | --- |
| 1: combined losses, learning rates, L2-SP penalty/total loss | `checkpoints/task1/*/train.jsonl` | All five full training histories saved |
| 1: seven conditions, four benchmarks, raw/primary scores and baseline changes | `results/task1/*/{gsm8k,retention}.json` | Complete prescribed splits checked; `results/validation/task1-evaluation-review.json` |
| 1: extracted answers, actual generation tokens and truncation flags | `results/task1/*/gsm8k_samples.jsonl` | 1,319 records checked per condition |
| 1: displacement, subspace interference, update magnitude, fixed 32+32 activation probes | `results/task1/*/diagnostics.json`, `data/task1/probe_cache.pt`, `manifests/task1/probe_ids.json` | Six checkpoint diagnostics completed |
| 1: parameters, time, memory, checkpoint size, method overhead | `checkpoints/task1/*/resources_1250.json` | Saved for all five runs; sampled memory is a lower bound |
| 1: 168 qualitative responses, fixed 42 annotations, aligned excerpts | `results/task1/qualitative/*.jsonl` | Original linked YAML acquired and verified; generation and human annotations pending |
| 2: losses, mixtures, skipped updates/scales, stage boundaries | `checkpoints/task2/train.jsonl` | All 4,500 updates saved; stages 2/3 had no skipped updates; initial four scale-overflow skips retained |
| 2: caption/answer development losses and image-swap penalties | `checkpoints/task2/development.jsonl` | All prescribed development checkpoints saved, including updates 0, 2,250, 3,750 and 4,500 |
| 2: both test suites, four image conditions, question types, pair diagnostics, empty answers, partner following, NLL and paired bootstrap intervals | `results/task2/{predictions.jsonl,summary.json}` | Final evaluation pending after update 4,500 |
| 2: paired images, shared question, references and predictions | `data/coco/val2014`, selected IDs, final predictions; `pa3 plots --images data/coco` | Images/IDs saved; predictions pending; illustrative pairs selected by sorted pair ID independently of correctness |
| 2: feature caching cost, training/generation time and stage parameter counts | `data/features/resources.json`, `checkpoints/task2/resources_*.json`, `results/task2/resources.json` | Feature cache and all training resource records saved; generation costs await final evaluation |
| 2: peak memory by stage | Revised trainer records separate `stage_peak_*` values alongside cumulative peaks, with `stage_memory_complete` | Verified all three stage-end resources have stage_memory_complete=true. Original Stage 1 peak preserved across resume; stages 2/3 measured separately. MPS peaks remain sampled lower bounds |
| 3: accuracy/oracle/cost curves, selectors and failure decomposition | `results/task3/offline/*_{summary.csv,per_question.jsonl,question_means.csv,accuracy_differences.json}` | Logging implemented; supplied data/README/schema inspection and execution pending |
| 3: length/correctness associations, confidence/vision relationships, tokenwise fusion and token trajectories | `results/task3/offline/*_{signals,rollout_scores,trajectories}.jsonl` | Logging implemented; execution pending |
| 3: 50-question live comparisons, tokens, runtime, memory, truncation/extraction and revision transitions | `results/task3/live/{calls,per_question}.jsonl` | Logging implemented; original README/model revision and question mapping must be verified before execution |
| 3: development-only policy search, frozen thresholds, held-out accuracy/cost, stopping counts and retrospective errors | `results/task3/adaptive/*` | Logging implemented; execution pending |
| 3: aligned selection/refinement/stopping examples | Raw records plus `scripts/make_examples.py` | Execution and human review pending |

`pa3 plots --images data/coco` regenerates figures and data tables from the saved
measurements without training. Task 1 diagnostics may be presented as a table,
as explicitly allowed by the manual. `pa3 audit` checks output presence and some
counts; it does not establish correctness or replace review of original assets.

Do not remove original images before saving paired illustrative image panels and
completing independent label inspection. Do not remove checkpoints before all
qualitative generation and diagnostics finish. After submission, remove large
archives/datasets/features and unnecessary checkpoints under the user's cleanup
requirement, while retaining code, source manifests, measurements, figures and
the student-written report.

## Reporting export verification

The plotting command exports the stored numbers without generating report prose:

- Task 1: loss and learning-rate figures; regularization; benchmark and baseline-change figures/tables; diagnostics/resources tables; all qualitative responses and the required annotation subset.
- Task 2: loss/development figures with stage boundaries; stage and feature-caching resources; final per-type results; pair-bootstrap intervals and correct-image advantages; aligned image panels.
- Task 3: budget and token-cost curves; method/failure tables; paired accuracy-difference intervals; signal associations and trajectories; frozen policies/development search; held-out adaptive intervals and stopping distributions; live means/totals and revision-transition counts.

Validation on November 1: 53 tests passed, 2 accelerator-only tests skipped.
A separate temporary synthetic-data check exercises both datasets' offline and
adaptive reporting plus live cost/truncation/extraction/revision exports. These
fixture checks are not experimental evidence. The actual output-presence/field
audit is saved as `results/validation/report-evidence-audit.json`; unexecuted
Task 3 runs, unfinished evaluations and human annotations remain pending there.
