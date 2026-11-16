# D004: diagnose router objective alignment after menu closure

No new router, fit, split, threshold or promotion is allowed. The finite scientific
menu remains closed. This is a B4 evaluation-only maintenance diagnostic.

R2 optimized a weighted probability log-loss surrogate for the blend. R3 learned
whether C205 had lower individual log loss than J102 on strictly inner-OOF
predictions. Neither objective is the official same-time pair-ranking TS-AUC.
The diagnostic tests whether the observed outer-held-out outputs improve their
probability-loss surrogate while losing official ranking quality, or whether
both metrics deteriorate. No training rows are scored to support validation.

Use the existing original five-fold held-out outputs of J102, C205, BL80 and all
four registered routers. Within each outer validation set, compute W3-weighted
and unweighted log loss with fixed clipping [1e-7, 1-1e-7], weighted Brier score,
and recheck the already recorded official TS-AUC. These loss diagnostics treat
scores as probabilities, although calibration is not established. They never
replace official TS-AUC or change admission thresholds.

Reconstruct each positive observation's same-time pair credit against all
negative controls. The summed credit difference divided by pair count must equal
the official score difference exactly up to 1e-14. Report the fraction of positive
pair exposure whose own log loss improves while its relative pair credit falls,
and the converse. These are exposure-weighted positive-row diagnostics, not counts
of uniquely identified pair flips or causal attribution of individual trees.

For R2 and R3 only, reload the frozen gate JSON and the original outer-validation
base feature matrix. Require byte-exact existing router predictions, and report
weight ranges, ordinary/W3-weighted mean weights, and the fraction below 0.2.
Also report the held-out W3 fraction on which C205's standalone log loss is lower.
Do not fit or score a new gate, read inner training arrays, or open independent
data. A disagreement between metrics is evidence of surrogate misalignment for
these fixed outputs; it does not prove a particular causal repair will succeed.
