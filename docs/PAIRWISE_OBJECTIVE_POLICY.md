# Within-time sampled pairwise objective

Two budgets are fixed before any pairwise candidate score: 8 and 32 sampled negatives per positive. Pairs are constructed only inside each training fold's current online step. The seed is 20260908 plus the validation fold number, fixed for reproducibility. No held-out labels, ID encodings or future-length features are used.

For each positive-negative training pair, the loss is log(1+exp(-(s_positive-s_negative))). A positive row samples K negatives with replacement from its time group. Each sampled pair receives weight n_negative(t)/K; therefore its expected weighted contribution equals that time group's full positive-negative pair loss. A common normalization sets total incoming row weight to the number of training rows. The derivatives passed to LightGBM are the exact gradient and diagonal second derivative of the sampled loss; tree boosting uses that diagonal curvature approximation. This avoids creating an O(N squared) pair matrix.

Raw scores are converted through sigmoid only for bounded inference and official float32 output. No pointwise probability-calibration claim is made. Comparisons use the same historical feature representation, fixed development folds and small tree settings as the classification reference. Validation rows remain unsampled.

Implementation contract follows the [official LightGBM custom-objective API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html), whose inputs are raw model margins and whose outputs are per-row gradient and Hessian arrays.
