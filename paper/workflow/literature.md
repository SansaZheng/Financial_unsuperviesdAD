# Verified implementation sources

User selected these three methods; no broader literature-search or manuscript task was requested.

| Method | Primary source | Version | Task / evidence boundary |
|---|---|---|---|
| HUGE | [AAAI2025 paper](https://arxiv.org/html/2502.13308v3), [author code](https://github.com/CampanulaBells/HUGE-GAD) | bdf8d2aec6455b48d7322c0713e31d2f6fecc1b7 | Originally label-free, transductive graph fraud detection. Financial port changes graph features, preprocessing, ranking batch and inductive transfer; does not reproduce original six datasets. |
| HIMVH | [WWW2026 paper](https://arxiv.org/html/2601.11073v3) | arXiv v3, 2026-07-09; DOI 10.1145/3774904.3792979 | Supervised / semi-supervised multi-view transaction hypergraph. No linked official implementation found. Eq.1–18 independently implemented; real-label paths removed and objective replaced by denoising reconstruction. |
| SD-FMM | [IPM2026 paper](https://doi.org/10.1016/j.ipm.2026.104961), [author code](https://github.com/ys-Dai/SD-FMM) | 0d1a2158ee849fdc31a127bd049e622174f5c893 | Official training code reads publisher anomaly labels, extracts label-defined fragments and selects using validation F1. This financial zero-real-label adaptation uses self-generated endpoint wavelet corruption instead. |

Code fingerprints: [upstream record](../../docs/recent_zero_upstream.json). Detailed departures and frozen budgets: [protocol](../../docs/recent_zero_protocol_CN.md). Possible failure mechanisms in the report are hypotheses, not author-proven or experimentally isolated causal conclusions.
