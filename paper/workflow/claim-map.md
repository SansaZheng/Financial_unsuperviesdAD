# Bounded claim map

| ID | Supported statement | Evidence | Boundary |
|---|---|---|---|
| C1 |30 new runs,60 operation records completed and independently verified | results/recent_zero_20261117/complete.json, verification.json | Other old methods reused; not30 runs per method/dataset. |
| C2 | Same evaluation cohorts and shared preprocessed row observations | verification.json, input_files_sha256.json | Native relations, derived representations, context length and pretraining budgets differ. Not identical full model input tensors. |
| C3 | No new port/adaptation is best by mean AP or either mean F1 | findings.json, best_methods.csv, combined summaries | Only the specified eligible tasks/methods, three seeds, fixed budgets; no global-SOTA or significance claim. |
| C4 | Real training/calibration labels not used for fitting or thresholds | fit APIs, test_recent.py absent-label end-to-end cases, saved gates | Synthetic targets/rank guidance are self-generated; byte hashes of labels are integrity checks. Training can contain anomalies. |
| C5 | SD-FMM core forward architecture matches author's pinned weights on the checked univariate cases | gpu_qualification.json | Multivariate inputs, synthetic generator/objective, splits and schedules adapted. Not a replication of original supervised scores. |
| C6 |54 older recent-method records have matching inputs and aggregate consistency | historical_recent_reference.json, historical_recent_summary.csv | Prior published verification retained; full old models/scores not independently replayed now. |
| C7 | Threshold failure alone cannot explain very low AP |30/30 POT success, threshold_audit.csv, AP tables | AP is threshold-independent. Specific causal failure mechanisms remain hypotheses without ablations. |

All paths above are relative to results/recent_zero_20261117 unless stated otherwise. No pending positive-effect, new-method-superiority, original-paper-reproduction or deployment claim is made.
