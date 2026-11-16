# Current research task

- Executor: Codex / CQ. Prompts: [implementation](prompts/001-recent-zero.md), [verification/reporting](prompts/002-verify-report.md).
- Stage: science, independent verification and report generation completed; publication CI / branch update pending.
- Venue: none. Topic: zero-real-label financial anomaly detection. Route: recent-zero-20261117. No manuscript or author list requested.
- Final scientific source: bff38e70725f67092feface4c6f9753cf76f8569.
- CPU gate j-1117-9bee95 / GPU gate j-1117-c75129 / full job j-1117-527aac all done.
- Verified:30 runs /60 operation records /101 exact inputs /350 combined summary rows.76 local tests pass.14 dedicated tests include absent training/calibration label files. All comparison figures inspected.
- Outcome: no new variant has highest mean AP, original-POT F1 or q99-F1; do not generalize to original supervised papers or globally optimized methods.
- Outputs: [Chinese report](../../results/recent_zero_20261117/README_CN.md), [main report](../../docs/financial_ad_comparison_CN.md), [evidence audit](evidence-audit.md), [claim map](claim-map.md), [per-run index](results.csv).
- Resources: $0 paid compute. All model/score files remain in CQ job archive and sibling recent_zero_20261117/cq_results/j-1117-527aac workspace.
- Next: publish verified code and results to main and default branch, verify remote refs and CI; no additional training or test-driven tuning.
