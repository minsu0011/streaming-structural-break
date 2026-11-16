# Research design and admission gates

This independent implementation uses no prior financial project code, weights, data, domain features, participant solutions or leaderboard tuning.

`configs/research.json` freezes seed, split algorithm, thread limits and promotion thresholds before any competition score is observed. `configs/candidates.json` freezes a small model ladder and A/B/C/D family ablations. No seed shopping, row-random split, seal evaluation or reduced-target training is permitted.

`scripts/run_research.py` checks actual raw acquisition, audit, empirical label alignment, scorer parity, causal prefix parity, and a populated immutable split lock. Exact duplicate series/history flagged by the audit require review; the runner refuses to silently let duplicates cross folds. IDs map to roles by SHA256; a separate hash maps development IDs to one of five folds. The hash namespace and integer canonicalization are recorded in the split lock.

The cache uses float32 features and uint8 labels/sampling flags, with 65,536-row shards per fold. Labels are generated outside the state API. Both historical and online observations are converted to float32 during feature extraction to match the official wire format. Cache invalidation includes feature, loader, label, split and baseline implementations plus raw file hashes. IDs and online total length never enter the feature matrix.

Baseline and supervised CV scores are both computed after float32 prediction conversion, matching the official socket's quantization and tie behavior.

All validation points are scored. S0 uses all training rows; S1 uses every fourth time step; S2 caps each training series at 256 evenly spaced observations. These training sampling decisions may inspect training-series length; inference may not. All scalers and estimators fit only on the other four development folds. A validation-label mutation test confirms predictions remain unchanged when only held-out labels change.

The baseline ladder is constant, exact official EWMA, multiscale mean, CUSUM, variance, tail, dependence, and a fixed equal-family evidence combination. The official finite-input EWMA arithmetic is reproduced exactly from the pinned notebook. Its tanh can round to 1 in finite precision; documentation prose claiming non-saturation is not treated as a mathematical guarantee.

Supervised candidates are logistic regression, ridge decision with sigmoid, small HistGradientBoosting, LightGBM and XGBoost. Ridge scores are not claimed to be calibrated probabilities. Training defaults to four CPU threads. GPU hardware was independently probed; research remains CPU-capable. HistGB and LightGBM have restricted numeric binary-tree exporters with native prediction parity tests and fail-closed format checks. The original estimator is retained for audit and comparison. Model artifacts verify both feature and predictor implementation hashes on load.

Primary comparison is median five-fold TS-AUC, then worst fold and stability, with causal/deterministic/streaming hard gates first. Tiny gains do not justify severe worst-fold loss or latency growth; thresholds are frozen in the config. The automated runner saves complete CV evidence but intentionally does not perform unreviewed promotion. A named development candidate requires all requested gates and real scores. No champion exists while authentication is blocked.

Diagnostics include time buckets, post-break evidence age with negative controls, and metadata slices. A no-break-only slice has no positive-negative pairs; its AUC is reported as undefined and score/false-alarm summaries are used instead. Total online length and tau fractions are diagnostic metadata only. No true DGP labels are invented.

Current family selection reduces learner input columns; the shared V1 engine still calculates all 57 features. Ablation latency therefore does not yet represent a pruned feature kernel. Family usefulness, redundancy and gains remain unknown without official development data.
