# PA3 evidence and student report

All required experimental data, graph exports and human-review records for
Tasks 1–3 are saved. The full requirement-to-file mapping is
[manual/report-evidence.md](../manual/report-evidence.md).

- [figures/](figures/): 50 figures, each in PDF and PNG, plus CSV tables of their supporting measurements and additional required statistics.
- [Task 1 annotation subset](figures/task1_annotation_subset.csv): all 42 ratings completed; full aligned responses in `task1_qualitative.csv`.
- [Task 2 final results](figures/task2_final_results.csv): both suites, all conditions, question types, pair metrics and bootstrap intervals.
- [Task 2 label review](figures/task2_label_review.csv): 12 independently reviewed image labels with notes.
- [Task 3 examples](../results/task3/examples/): selection, stopping and live refinement comparisons.
- [Evidence checks](../results/validation/report-evidence-audit.json): 70/70 pass.
- [Export verification](../results/validation/final-evidence-audit.json): inventory, counts and source checks.

Regenerate plots/tables with `pa3 plots --images data/coco` and aligned Task 3
examples with `python -m scripts.make_examples`.

The assignment prohibits generative AI from writing any part of the PDF report,
including its language, interpretation and analysis. Write those yourself.
The inventory is a technical checklist, not a completed report or a certification
that the report/submission requirements have been satisfied.
