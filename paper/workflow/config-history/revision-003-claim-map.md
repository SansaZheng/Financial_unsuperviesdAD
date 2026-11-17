# Bounded claim map — revision2

| ID | Current supported statement | Evidence | Boundary |
|---|---|---|---|
| R1 | Completed174-config matrix and frozen role/cm/ge primaries | docs/recent_repair_protocol_CN.md; finad_repair.experiment.planned | verification.json establishes174 complete.156 refers to final analysis fits, excluding superseded/diagnostic compute. |
| R2 |92 tests and final CPU/GPU qualification pass | cq_cpu_tests.txt; gpu_qualification.json; tests/test_repair.py | Tests check specified invariants, not financial superiority. |
| R3 | Specific training-only adaptation counterexamples are repaired | train_only_audit.json; dedicated gradient/constraint/ego tests | Paired ablations are reported; grouped changes do not isolate every subcomponent. |
| R4 |All174 configs pass source/artifact/gate, original sampling, normalizer, CPU checkpoint and full metric/threshold replay | verification.json | Same common observations/cohorts; method-specific contexts and representations differ. |
| R5 | No real train/calibration labels used in fitting or thresholds | fit APIs, absent-label pipeline tests, save-before-label gates | Artificial intervention targets remain; unlabeled source data can be anomalous. |
| R6 | New source/core settings fixed before viewing newly fitted financial outcomes | frozen protocol snapshot; cq_job_specs.json; workflow events | Previous test results motivated development; this is retrospective. |
| R7 | Oversized/cancelled technical batches are not evidence of poor detection | execution.json; superseded_reload_gate_failure.txt | No cancelled-job status/trust override or score-based selection. |

All short evidence filenames above refer to results/recent_repair_20261117. Financial effects and best-method counts are complete in findings.json / paired_effects.csv / README_CN.md. SD primaryAP improves5/6,HUGE1/2,HIM0/2; no primary wins AP or eitherF1. Posthoc fields are evaluation-only and do not change primaries. Publication/CI complete: analysis734f50d on main/default, Python3.10/3.12 success. Original paper claims and globalSOTA are not claimed. Revision1 claim map preserved in config-history/revision-1-claim-map.md.

## Revision3 — roleweight evidence

| Claim | Evidence | Limits |
|---|---|---|
| 48 new HUGE fits,96 fresh HIM score configurations,12 frozen parents | roleweight_20261117/per_seed.json; run_manifests.json; execution.json | No new HIM network training; method-specific graph contexts persist. |
| Same101 inputs, original cohorts and metrics, all144 pass independent checks | roleweight_20261117/verification.json; parent lineage and score gates | GPU/CPU scatter tolerance disclosed; no blind test. |
| HUGE primary AP regresses2/2;HIM primary improves2/2;neither wins main pool | findings.json; summary.csv; combined_*; best_methods.csv | Relative to matched prior-primary anchors; no significance/SOTA claim. |
| Field calibration reduces scale domination; alignment compression harms Bank graph | paired_effects.csv; posthoc_field_diagnostics.csv; posthoc_role_geometry.csv | Diagnostic labels evaluation-only; dimension and geometry changes not separately causal. |

All short filenames in this section are inresults/roleweight_20261117.
