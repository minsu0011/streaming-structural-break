# Frozen real-development research design

The authoritative feature settings are `configs/features.json`. `FeatureConfig` has no independent numeric defaults. Alphas are derived from the configured scales. All state kernels receive parameters, alphas and lags explicitly; inference does not derive them from targets, IDs or future online metadata. Feature and cache identities include the canonical configuration digest; cache identity also includes the real split digest.

Raw immutable identity is filename, byte size and SHA256. File mtime is informational. All six canonical raw files are independently authenticated official downloads; no locked original setup file was modified and no ACL was changed.

Grouping policy was fixed before any score. Historical exact hashes, normalized hashes and short prefix hashes precede indexed near-pair confirmation on full historical vectors. Eligible equal-length series are indexed with cKDTree, using a seeded projection only to generate candidates. Similar coarse statistics alone do not merge groups. Connected components use their minimum integer ID as representative. Role/fold hashes retain seed 20260908. The actual 10,000-group allocation is immutable; no seed shopping or post-score grouping changes are permitted.

Raw metadata inspection occurs before split allocation as requested; it includes break counts and tau distributions for the data audit. After allocation, evaluation readers select development IDs before yielding any features or labels. No seal predictions or seal performance are calculated. This distinction is recorded explicitly rather than claiming that even pre-allocation raw tau metadata was never read.

Sampling names changed before the first real model score. Version 1 was synthetic scaffolding: S1 every fourth step, S2 cap 256. Version 2 is:

- S0: all training rows.
- S1: deterministic uniform cap 256 per training series.
- S2: deterministic cap 32 per series within each fixed time bucket.
- S3: all rows; each positive has weight n_negative(t), each negative n_positive(t), using only training folds. Mean-one normalization leaves total per-time contribution proportional to positive-negative pair count.
- S3_sqrt: the one predeclared tempered weighting family, square root of the same opposite-class count, then mean-one normalization.

No validation labels are used in standardization, feature removal, sampling or weights. Constant-column removal is fitted on each training matrix separately. Diagnostic feature summaries may describe the whole DEV pool but do not select columns globally.

The first ladder consists of B0–B7, followed by five small ordinary-row model families. Subsequent capped and time-stratified sampling, pair-aware weighting and family ablations retain the same folds. Adaptive refinements change one configuration field or one shallow model parameter at a time. Candidate identities are deduplicated; each hypothesis, complete fold score and failure is persisted. The run restores its original wall-clock deadline when resumed.

Promotion gates fixed before supervised scores: median gain at least 0.001 and worst-fold loss at most 0.002. A latency increase over 5x requires a median gain at least 0.01. Serialization and live CPU latency are measured for every fitted fold. All candidates are DEV candidates, with at most primary, robust backup and fast backup retained for deployment review. Final selection also compares the strongest statistical baseline.

Real-world inference timing includes feature update and single-row prediction after warm-up. Historical initialization, JIT compilation, model loading and official socket IPC are reported separately. Fold timings are subject to shared CPU load; the final candidates receive isolated reruns.

Reduced targets have two permitted diagnostic score uses only: official EWMA sanity, then frozen primary smoke. Any reduced result cannot cause a model, feature or hyperparameter change. LOCAL_FINAL_SEAL stays unopened for performance evaluation throughout this session.
