# 001 — local task specification

Executor: Codex. Configuration revision: 1. Stage: implementation, experiments, reporting.

Reproduce HUGE and implement explicitly named zero-label adaptations of HIMVH and SD-FMM. Inspect primary papers and pinned official source first. Reuse exact existing financial split files, training/calibration sample indices and graph cohorts. Never read training/calibration labels, select models using label metrics, tune to test outcomes, or construct fake account relationships. Keep transaction and account graph comparisons separate. Preserve old runs. Fix configurations before financial outcome inspection. Use seeds 42/43/44, original POT and independent calibration q99, AP, trapezoid PR-AUC, ROC-AUC, unadjusted positive Precision/Recall/F1/FPR and confusion counts. Save checkpoints, scores, hashes, source revisions, reload checks and run budgets. Run meaningful numerical and label-independence checks plus CQ qualification before formal compute; spend $0. Independently replay metrics and verify exact test cohorts before generating a Chinese report and pushing to SansaZheng/Financial_unsuperviesdAD. State incompatible tasks and departures from authors' architectures/objectives honestly. No paper venue or manuscript requested.

Inputs: docs/zero_label_protocol_CN.md; results/kronos_stable_20261116; external exact caches in ../zero_label_20261114 and ../kronos_matched_20261115.
Outputs: implementation package, tests, frozen protocol, per-seed metrics, verification, updated docs/financial_ad_comparison_CN.md.
Success: all eligible method/task/seed runs verified, no missing or manufactured outcomes, report reproducible from records, remote Git refs verified.
