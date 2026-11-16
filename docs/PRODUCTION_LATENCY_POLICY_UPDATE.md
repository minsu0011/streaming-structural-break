# Verified inference paths in research comparisons

The original single-row XGBoost interface took about 100 microseconds per point.
The separate compact exporter reproduced all 4,014,405 DEV float32 predictions
for the real raw-feature reference. Fused inference and shared historical setup
also reproduced the complete calibrated HistGB development OOF predictions.

After the AR input milestone, subsequent small-tree candidates are compared
using the actual deployable fused path. Each fold retains its native timing and
checks 1,000 real-prefix predictions for float32 equality before timing the new
path. XGBoost additionally compares exported predictions on every validation row.
The median gain, worst-fold-loss and latency-ratio promotion thresholds stay fixed.
Neither features, fitted model weights, validation predictions nor split changes
are part of this runtime optimization.

Earlier candidate timings remain their original measurements. A verified
optimized timing is an additional artifact, not a retroactive claim about the
native run. Final fast-backup shortlisting uses the best verified production path
available for each eligible model, followed by isolated timing. This avoids
discarding a compact XGBoost model because of its older Python API overhead.

Historical initialization and socket overhead remain separate from point timing.
Final reports include observed initialization cost and an amortized estimate.
