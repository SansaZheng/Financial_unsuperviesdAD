# Stage005 — role and field score adaptation

Executor: current primary agent, no delegation. Config revision3, route roleweight-20261117, venue none. User asks to improve HUGE/HIMVH based on prior ablations, preserve zero-label/cohorts/metrics, rerun comparisons/ablations and push Chinese report.

Implement exactly [frozen protocol](../../../docs/roleweight_protocol_CN.md). Preserve all previous source and results. Add train-only role feature adapters and unlabeled calibration-only field scorers; test numerical invariants, constants/ties, serialization and baseline identity. Pin source before computing new outcomes, CPU gate then actual-shape CQ GPU gate and 48 fresh HUGE fits; reuse12 immutable HIM checkpoints for96 scoring configurations. Independently verify before reporting; report mixed results and retrospective test reuse. No paid compute. Stage006 will verify/report all144 configs and288 operating records. Main primaries fixed dual and cm_tail; do not select by test outcomes.
