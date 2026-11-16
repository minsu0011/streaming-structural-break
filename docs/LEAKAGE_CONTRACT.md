# Causal feature and validation contract

`StreamingFeatureState(historical)` computes reference parameters once. `update_and_get(point)` consumes exactly the current scalar; its API accepts no tau, target, dataset ID or online length. Offline replay uses this same update call in chronological order.

No feature can depend on future points, final length, breakpoint position, break indicator, target counts, ID ordering, other fold targets or reduced labels. Time counters represent observations already seen only. IDs appear only in joins and deterministic split allocation.

Reference mean/scale, quantiles, tail rates, lag moments and AR coefficients are estimated from historical data only. Models/scalers fit only on development training folds. Validation rows are never reused for fitting. The final seal has no feature extraction or predictions in this session.

Hard gates: suffix mutation leaves prefix features and predictions unchanged; guarded one-pass iterators reject length access, replay and read-ahead; offline/live parity; state isolation across series; deterministic replay and serialization parity. Constant and nonfinite inputs have explicit numerical policies and cannot silently produce NaN predictions.
