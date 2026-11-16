# Historical AR(4) residual input

Dependence remains the strongest standalone family, but a full marginal rank
transform reduced mean DEV score from 0.59414 to 0.59137 and lost tail magnitude.
Test a temporal transformation that preserves innovation magnitude: estimate a
regularized AR(4) predictor from the historical segment, then feed its residuals
into the same 57-feature bank and historical MAD calibration.

The comparison anchor remains `M4_lightgbm_ABCD_S3_e3da86d90151`. Preserve model,
families, S3 weights and feature settings; change only the input representation.
`configs/ar_residual_input.json` freezes order, ridge fraction and coefficient
norm bound before this candidate is scored. One initial candidate is allowed.

Coefficients use historical values only and remain fixed online. The predictor
stores four past normalized observations. Residual history omits the first four
observations lacking complete lag context. Every online observation retains a
prediction; no future suffix, final online length, ID or break label enters the
filter. Validation labels are used only by the existing external scorer.

The method can reduce useful low-frequency evidence, and AR(4) is an approximation
for arbitrary dependence. A worse score will be retained as a rejected hypothesis.
