# NEXT2 observed-age normalization and nested alarm study

The frozen J102 Gaussian null simulation alarms on 32.23% of 512 sequences
by 16,384 observations at a threshold calibrated for 1,024 observations and
nominal alpha 0.05. This motivates a secondary alarm diagnostic.

Before computing its results, freeze beta in {0, 0.5, 1} and
q(p,t)=p/[p+(1-p)(1+t/128)^beta]. The detector probability p remains an input;
this is not a time-only predictor. At each common observed time t, the map is
strictly increasing, so exact ranks, ties and official TS-AUC are unchanged.
Keep float64 outputs to avoid introducing float32 rounding ties. Check score
parity numerically. This transformation is not a new performance candidate.

Recalibrate each transformation separately using only the already fixed,
independent 1,024-observation synthetic calibration cohorts. Compare eventual
null alarms at 1,024/4,096/16,384 and first clean post-break detection, including
the existing late tau=512 stratum. Use all exponents; do not choose the most
favorable one after observing results or claim an anytime error bound.

For real DEV, reuse exactly the five nested train/calibration/outer partitions
in configs/next_alarm_calibration_policy.json, including historical group
admission. Refit J102 and C205 on each training subset. Freeze the 80/20 blend
of these same-subset models. For each model/transform/alpha, calculate the
threshold using only null series in the calibration subset, write it to disk,
and only then predict the outer fold. Never refit after calibrating. Report
the secondary smaller-model TS-AUC separately from the primary full-CV scores.
Use alpha 0.01 and 0.05 with the same finite-sample strict-exceedance rank.

These DEV folds have already informed adaptive research. Conditional alarm
results are not untouched generalization evidence or proof of exchangeability.
The study selects no deployment alarm threshold and opens no seal.
