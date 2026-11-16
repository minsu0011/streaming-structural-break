# One model-family follow-up on AR(8)

The preregistered AR(8) LightGBM candidate improved all five folds over AR(4):
mean 0.611350 versus 0.608044. Its median gain was only 0.000821, below the fixed
0.001 promotion threshold, so the primary stayed unchanged.

This supports one bounded follow-up: retain the exact AR(8) feature cache,
ABCD families and S3 training rows/weights, and replace the small LightGBM learner
with the previously defined small HistGB learner. No fold, input-order policy,
score threshold, feature normalization or tuning grid changes. The policy is
written before this follow-up fit in `AR8_HISTGB_FOLLOWUP_POLICY.json`.
