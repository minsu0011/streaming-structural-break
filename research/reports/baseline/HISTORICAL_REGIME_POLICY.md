# Historical regime stress policy

This policy is fixed before NEXT candidate scores. Use the original 8,000 DEV IDs only. Compute H-only raw mean/median, log SD/MAD, skewness, log fourth standardized moment, robust tail rates, ACF at lags 1/2/4/8, AR4/8 coefficients and innovation variance/correlation, four coarse spectral fractions, and log H length. No online value, tau, target, model prediction or seal observation defines a cluster.

Fit each descriptor's mean/SD on these H-only descriptors, set zero SD to one, standardize and clip to [-8,8]. Run deterministic single-thread KMeans with K=5, n_init=10, seed=202609095, algorithm=lloyd. Freeze assignments and transformation/centers with SHA256 before stress model scores. Label clusters canonically by lexicographic center coordinates.

Train on four regimes and validate on the fifth, using the same grouping for every evaluated candidate. This global DEV historical clustering is transductive and diagnostic; the original 5-fold CV remains the selection authority. Regime scores compare only within-regime pairs, so they are not an additive decomposition of the full score. Report sample sizes, pair counts and worst regime, and preserve small-group limitations.
