# PA3 implementation audit — 27 October 2026

This is a technical implementation checklist, not report prose or experimental
interpretation. Authority: the original `ATML-PA3.pdf`, pages 1–15.

Status: all three tasks received a source-level comparison against the manual.
The checks below describe implemented behavior, not completed experiments or
certification of every runtime path. The active Task 1 training run was not
interrupted or changed during this audit.

## Task 1 — manual pages 2–6

| Requirement | Implementation checked |
| --- | --- |
| Independent original Qwen3-0.6B starts, thinking disabled, seed 6304 | `common/__init__.py`, `task1_finetuning/data.py`, `train.py`, `model.py` |
| First 40,000 eligible MetaMathQA examples in source order; empty/overlength/GSM8K-test overlap exclusions; no answer truncation/shuffle | `task1_finetuning/data.py` |
| Assistant-response plus end-of-turn supervision; prompt/padding masked; effective-batch valid-token normalization | `common.supervised`, `task1_finetuning/loss.py`, `train.py` |
| One epoch, 1,250 updates, microbatch 2, accumulation 16, effective batch 32, maximum 2,048 | `task1_finetuning/train.py` |
| AdamW betas .9/.999, eps 1e-8, decay .01 excluding biases and dimensions <2; cosine, 38 warmup, norm clip 1 | `common.optimizer_groups`, `common.lr_factor`, `train.py` |
| FP16 autocast, dynamic scaler, FP32 trainables/states, checkpointing, unscale before clipping, skipped-update logs | `common/device.py`, `train.py`, `common/adapters.py` |
| Full FT LR 2e-5; L2-SP same LR and unaveraged unique-parameter half squared-distance penalty, lambda .001; fixed reference and overhead logs | `task1_finetuning/config.py`, `train.py` |
| LoRA ranks/alphas 4/8 and 16/32, LR 2e-4, dropout .05, seven target projections, random A/zero B, frozen backbone/biases | `common/adapters.py`, `config.py` |
| OPLoRA rank 16/alpha 32/k128; matched unprojected initialization; exact FP32 SVD, factorized projection, projected export, basis/setup/export overhead | `common/adapters.py`, `common/device.py`, `train.py` |
| Recovery at 312/625/937/1250; model, optimizer, scheduler, scaler, RNG and data position | `task1_finetuning/train.py` |
| Seven quantitative/qualitative conditions; four full benchmark splits; zero-shot, GSM non-thinking chat/greedy/cap1024, other stock completion scoring | `evaluate.py`, `scripts/task1.sh` |
| Fresh-backbone adapter merge and merged/unmerged logit check | `task1_finetuning/model.py` |
| FP32 down_proj diagnostics at 0/13/27, rho128, eta, fixed 32+32 last-prompt-token activations and R(z); full-model unique-parameter displacement | `task1_finetuning/diagnostics.py` |
| All 24 original prompts x7, greedy cap512; first/last plain question across formats/conditions marked for 42 human annotations | `task1_finetuning/evaluate.py`; original YAML still required |
| Learning, resources, benchmark baseline changes and diagnostic outputs | `train.py`, `diagnostics.py`, `pa3/plots.py` |

MPS implementation choices: right-pad to 512/1024/2048 without regrouping or
reordering; added labels -100 and attention zero; checkpoint output-head loss
chunks; release unused cache. These preserve real tokens and the objective.
The four-update bucket diagnostic passed on the user's Mac, including long
examples. Full 1,250-update stability and the other four methods are not yet
runtime-verified. Exact SVD on MPS is explicitly performed on CPU in FP32.

## Task 2 — manual pages 6–10

| Requirement | Implementation checked |
| --- | --- |
| Original Qwen3, thinking disabled; frozen SigLIP base-patch16-224; final 196x768 features in spatial order, final LN exactly once, deterministic preprocessing, FP16 cache | `task2_vlm/data.py`, `model.py` |
| Linear768→1024/GELU/Linear1024→1024, visual tokens inside user message before question, matching masks/positions; connector magnitude calibrated on training data only | `task2_vlm/model.py` |
| Caption training 6,000 images x3; caption dev96; generated presence/counting training pools and evaluation24/64 pairs | `task2_vlm/data.py`; equal presence/counting allocation is an implementation choice within the combined counts |
| Natural training1536 pairs with640/256/640 types, dev32 with12/8/12, test96 with36/24/36; >=8/10 agreement, different normalized answers, original consensus targets, all ten references retained | `task2_vlm/data.py` |
| train2014 vs val2014; globally image-disjoint train/dev/test; seeded IDs and fixed mismatch excluding both source images | `task2_vlm/data.py` |
| Question80/caption64/answer16 content caps; response/EOT loss only; token mean within example then effective-batch mean | `task2_vlm/model.py`, `train.py` |
| Same model through2250/1500/750 updates; batch8 accumulation2; connector LR1e-3/1e-4/5e-5; LoRA absent/2e-5/1e-5; decay.01, clip1, checkpointing, per-stage5% warmup/cosine floor10% | `task2_vlm/train.py` |
| Two caption passes; stage2 draws .5/.25/.25; stage3 .7/.1/.1/.1; each draw a full pair or two captions; no enforced mixture quotas | `task2_vlm/train.py` |
| Attention-only LoRA16/32/dropout.05 in stages2–3; no original trainables; FP16 backbone/AMP and FP32 trainables/states | `common/adapters.py`, `task2_vlm/model.py`, `train.py` |
| Checkpoint every250, stage weight exports, full resume/sampler/stage/RNG state; final model4500 | `task2_vlm/train.py`, `evaluate.py` |
| Dev0/every500/stage endpoints; caption loss and correct/partner/no-image answer NLL excluding EOT; image-swap penalty | `task2_vlm/train.py`, `model.py` |
| Four final controls; original references fixed; no-image removes visual tokens and rebuilds masks/positions; greedy16/batch8 | `task2_vlm/model.py`, `evaluate.py` |
| Normalized grounding EM by type, official-normalized VQA soft score by type, pair accuracy/identical/empty/NLL/time/partner-following; paired4000 bootstrap and correct-image advantages | `common/vqa.py`, `task2_vlm/evaluate.py` |
| Training/dev plots, stage resource counts, raw references/generations, complementary example selection | `train.py`, `evaluate.py`, `pa3/plots.py` |
| Optional49-token repeat:2x2 spatial mean, fresh run, otherwise same recipe | `model.py`, CLI `--compressed`; optional, not required |

Not yet verified against real assets: actual COCO/VQAv2 preparation, original
official VQA scorer checkout, independent label inspection, feature caching,
three-stage accelerator training/resume/generation. Task 1's MPS qualification
does not establish that Task 2's different batch/sequence workload will fit.

## Task 3 — manual pages 10–15

| Requirement | Implementation checked |
| --- | --- |
| Original course parquet joins on dataset/sample_id/rollout_id, one-to-one joins, token-list alignment/finiteness, 1000 questions x32 | `task3_inference/data.py`; actual schema still must be mapped from supplied README |
| Lexicographic IDs then default_rng6304 permutation,200dev/800heldout, first50heldout MATH live, ten saved nested orders | `task3_inference/data.py` |
| Supplied extracted answers; math-verify equivalence, released MathVista type/choice/precision normalization and option letters; one incorrect NO_ANSWER group retained | `task3_inference/scoring.py`, `data.py` |
| Gold-free grouping/selection/stopping; correctness stored separately; scorer version/source hashes | `data.py`, `scoring.py`, `common.environment` |
| Budgets1/2/4/8/16/32; FirstAns/SC/highest-logp/highest-K/K-RWC; VLM vision and VisConf highest/RWC; earliest tie, average tied ranks | `scoring.py`, `offline.py` |
| Normalized attention-group totals; current-prefix pooled KL linear Q.05/Q.95 calibration, clipped confidence, V*C and tokenwise fusion | `task3_inference/scoring.py` |
| FirstAns first-candidate cost, others all consumed retained tokens; oracle/distinct groups/failure decomposition; average orders within question before paired1000 bootstrap | `task3_inference/offline.py` |
| Within-question length/score/correctness associations, top/bottom quartiles, shortest control, confidence–vision correlation and fixed trajectories | `offline.py`, `pa3/plots.py` |
| Live original Qwen2.5-3B revision from README, FP16/noquant/batch1, native template/system/instruction, .7/.8/topk20/repetition1 | `task3_inference/live.py`; exact README verification remains blocked |
| 3072-long/3x1024parallel/shared1024+2048revision; independent seeds, SC selection, revised answer always returned, no gold feedback | `task3_inference/live.py` |
| Raw calls, actual prefill/output/runtime/memory, extraction/truncation, shared cost charged to both, revision transitions | `task3_inference/live.py` |
| Adaptive4/8/16/32, both return SC, unique nonmissing agreement, quality winner match/margin, prefix-only calibration | `task3_inference/scoring.py` |
| Dev-only tau/gamma grid, within1pp fixed32 target, always32 fallback, mincost/highaccuracy/smallerthreshold ties, freeze before heldout | `task3_inference/adaptive.py` |
| Heldout costs/counts/stopping distributions/wrong early stops/fixed32 recovery, gold only retrospective; deterministic illustrative selection rules | `adaptive.py`, `scripts/make_examples.py`, `pa3/plots.py` |

Real-data certification is blocked by missing original rollout parquets and
README.MD, the exact field map, released MathVista metadata/normalizer and their
versions, original MATH problems and model revision, and confirmation of the
README's exact balanced-brace extraction convention. No replacement course
rollouts, prompts or guessed model revision are supplied.

## Corrections made during this audit

1. Benchmark completeness now validates unique documents separately per harness
   answer filter. The pinned harness logs GSM8K under both strict and flexible
   filters; counting all rows as questions would incorrectly reject a full run.
2. GSM8K output now records actual generation token IDs/counts and termination
   information through a harness mixin; it no longer infers truncation by
   re-tokenizing postprocessed text. Stock prompts, stops, filters and metrics
   remain in the harness. The compact file retains the flexible-extraction row
   for every question; the full harness result retains all filters.
3. Task 3 source manifests now include the original README, schema, MathVista
   metadata and normalizer hashes. Live outputs record README/problem hashes.
4. Prepared candidate records retain gold-free answer-group IDs, and adaptive
   outputs record all consumed candidate IDs for traceability.

## Validation and remaining work

- Local tests: **50 passed, 2 skipped**. Formatting, static undefined-name checks,
  and patch whitespace checks passed for changed files.
- Tests include actual tiny Qwen loss/gradient/padding equivalence, adapter
  projection/merge, visual generation positions, normalization/scoring formulas,
  keyed parquet joining, exact seeded splits, prefix isolation, held-out-label
  independence of development policy choice, harness filter accounting and raw
  generation trace behavior. Synthetic fixtures are not experiment results.
- MPS tests skipped inside the tool process because it cannot access physical
  MPS; user-Terminal evidence is separate. MPS memory peaks are sampled lower
  bounds, not exact CUDA allocator peaks; that measurement limitation must be
  disclosed when reporting resource comparisons.
- Update October 31: retrieved the original `first_plot_questions.yaml` from
  the embedded page-3 GitHub link. Verified all 24 user prompts, eight JSON system
  instructions and original order survive loading unchanged. Source SHA256 is
  recorded in `manifests/task1/qualitative-source.json`. The 168-response run and
  forty-two student annotations remain outstanding.
- Task 2 requires its real data/scorer and independent label inspection. Task 3
  requires the original assets listed above. These are not solved by merely
  launching training commands.
- Real training, evaluation, resume qualification and complete outputs remain
  necessary. The source audit and file-presence audit do not certify their future
  correctness or the absence of every possible bug.
- Git currently has the implementation uncommitted. The public repository must
  receive the code/configs/IDs/environment/results, excluding raw data and large
  checkpoints, before submission.
- The student must independently write the required eight-page NeurIPS report,
  include the repository link at the end of the abstract, add citations, and
  submit the PDF/accessibly linked repository through LMS as instructed. No
  report language, interpretation or analysis is generated here.
