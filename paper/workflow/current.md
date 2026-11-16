# Current research task — completed

- Executor: Codex / CQ. Prompts: [implementation](prompts/001-recent-zero.md), [verification/reporting](prompts/002-verify-report.md).
- Status: completed. Code, verified results and reports published to main and default codex/classic-financial-ad; work branch also retained.
- Published analysis commit: c76649c3663d782c4ca354a67de535a6deab27f7. Completion-record commits only update workflow/publication metadata.
- CI: [Python3.10 and3.12 both passed](https://github.com/SansaZheng/Financial_unsuperviesdAD/actions/runs/43978530585). Local76 tests pass.
- Scientific source: bff38e70725f67092feface4c6f9753cf76f8569. CPU j-1117-9bee95 / GPU j-1117-c75129 / full j-1117-527aac all done.
- Verified30 new runs,60 operating records,101 exact inputs,350 combined summaries;14 dedicated tests include absent train/calibration labels. Figures visually inspected.
- Outcome: no new variant wins mean AP or either mean F1 on its eligible tasks. Original supervised-paper performance is not inferred from these adaptations.
- Outputs: [Chinese report](../../results/recent_zero_20261117/README_CN.md), [main report](../../docs/financial_ad_comparison_CN.md), [evidence audit](evidence-audit.md), [claim map](claim-map.md), [per-run index](results.csv), [execution proof](../../results/recent_zero_20261117/execution.json).
- Resources: $0 paid compute. Models and scores remain in CQ job archive and sibling recent_zero_20261117/cq_results/j-1117-527aac workspace; use cq fetch j-1117-527aac.
- Venue: none. Topic: zero-label financial anomaly detection. Route: recent-zero-20261117. No manuscript or author list requested.
- Outstanding required work: none. No further training or test-driven tuning scheduled.
