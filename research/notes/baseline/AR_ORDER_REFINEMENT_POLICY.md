# AR lag-order refinement after the AR(4) milestone

The historical AR(4) input increased mean DEV TS-AUC from 0.59414 to 0.60804.
The order-4 feature source and existing model hashes remain unchanged. A separate
module defines prospective order-2 and order-8 candidates, with all other input,
model, feature, sampling and weighting settings inherited from the frozen AR(4)
anchor `M4_ar_residual_input_ABCD_S3_9f68be316ea3`.

Both orders are fixed in `configs/ar_order_variants.json` before either score.
Order 2 tests variance reduction and lower input cost; order 8 tests whether
remaining predictable dependence needs more lag context. No continuous order
search or score-dependent ridge formula is admitted. The order-4 branch of the
new implementation is tested against the original for exact equality.

Coefficients remain historical-only and fixed online; per-point work and state
are O(order). Omitting the first `order` historical residuals is an initialization
choice and never omits online predictions. Final seal and reduced-score policy
remain unchanged.
