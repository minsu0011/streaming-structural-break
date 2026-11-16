# Historical marginal rank transformation

The MAD input-normalization milestone improved mean DEV TS-AUC by 0.0056, while
historical-null feature calibration had already improved comparability across
heterogeneous tails and dependence. Test one additional input mechanism: map
each value through a piecewise-linear approximation to its historical marginal
normal score, then reuse the existing causal feature engine and MAD historical
feature calibration. This tests distribution comparability beyond affine scale.

The anchor is `M4_lightgbm_ABCD_S3_e3da86d90151`, with input median/MAD scaling and
scales 5,20,160. Keep that anchor's model, sampling, feature families and parameters.
Exactly one feature representation candidate is admitted initially. Its mapping
policy is frozen in `configs/rank_gaussian.json` before any rank-transform score.

All knots come from historical values only. Current online values take a binary
search and interpolation through 257 knots, followed by ordinary incremental
feature updates. Neither online final length nor suffix values determine the
mapping. No target is used to fit the transform. Historical ranks are an
in-sample reference, so finite-sample null estimation bias remains possible.

Endpoint clipping intentionally loses tail magnitude beyond the historical
0.1% and 99.9% quantiles. It can hurt scale/tail shifts; rejection is an informative
outcome. This remains a separate representation and does not replace older caches.
