# Mechanism signature diagnostic policy

These post-hoc DEV signatures are approximate empirical descriptions, not true data-generating-process labels. They are prohibited in model fitting, feature generation, train sampling, prediction and deployment.

For a DEV break series, take the last 128 observations before tau from historical + the causal online prefix, and the first 128 online observations starting at tau. Require at least 32 observations on each side. No-break series are NO_BREAK_CONTROL; insufficient post-break support is WEAK_OR_AMBIGUOUS.

Compute absolute mean difference divided by pre SD (threshold 0.5), absolute log post/pre SD ratio (0.4), maximum absolute ACF difference at lags 1/2/4/8 (0.2), and change in the fraction exceeding 2.5 pre median/MAD robust scales (0.08). Divide each effect by its fixed threshold. If the largest is below one, assign WEAK_OR_AMBIGUOUS. If at least two exceed one and largest <1.5 times second largest, assign MIXED. Otherwise assign the largest effect's MEAN_DOMINANT, SCALE_DOMINANT, DEPENDENCE_DOMINANT or TAIL_DOMINANT category. Ties use that listed order.

Mean/scale shifts can alter the tail signature; dependence changes can affect sample moment uncertainty. These confounds remain explicit limitations. Fracture scores use all no-break control series plus the chosen break-signature series; same-time comparisons include negative pre-break rows. Show break count and pair support. Categories with fewer than 100 break series cannot trigger the predeclared mechanism promotion gate.
