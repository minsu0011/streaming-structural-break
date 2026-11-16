# D005: matched-state interventions for the existing O1 outlier cohort

This is counterexample characterization after scientific-menu closure. It cannot
fit, tune, rank, select or promote a new model. Use the original frozen BL80
shared runtime and the exact existing O1 generator, 512 paired seeds, original
1,024-point horizon and original injection schedules 128/256/512. Do not extend
the suite, change its observations, invent a break label or evaluate TS-AUC on
these null series. Reconstruct both frozen component predictions and BL80 at
every point against the existing original O1 files before accepting any result.

The existing NEXT2 probability/age-expiry trajectories establish persistence
but do not identify which detector state sustains it. This follow-up performs
explicit internal-state interventions with a matched no-outlier donor. Historical
calibration, trained trees, feature definitions and future observed inputs stay
fixed. No hybrid is an eligible inference model or a performance candidate.

At post-injection offsets 0, 127 and 255, use all 512 original pairs and evaluate
remaining common offsets through 511. At offset 511, use only the already
availability-defined 342 pairs with original onset 128/256 and evaluate offsets
512 through 767. Keep these two cohorts separate in every table and interval.
No extension is generated for the unavailable onset-512 subgroup.

The six disjoint recurrent blocks are:

1. Complete legacy calibrated ARCH/ABCD branch state.
2. Shared new-branch historical AR filter's online ring/counter.
3. Shared conditional-variance recurrence state.
4. J102 serial evidence recurrence.
5. C205 lag ring/counter.
6. C205 EWMA/CUSUM/Bayes evidence recurrence.

Copy one block from the matched control into the outlier state, and separately
copy the outlier block into the control state. Replay exactly the same future
observations. Also test unchanged deep clones and complete six-block transfers
in both directions: all their future predictions must reproduce the respective
original trajectories bit-for-bit. Check array non-aliasing for mutable copies,
equal H-only constants, unchanged trained arrays, pre-injection equality and
unchanged numeric state size. Swaps start after the anchor's update; the first
evaluated output is the next observation, so stale output work arrays cannot
count as an intervention result.

Report per-offset mean probability contrasts and fixed window means using
boundaries 1/8/32/128/256/512/768 clipped to each anchor's available future.
Also report occupancy and any crossing at the original BL80 O1 thresholds for
alpha 0.05 and 0.01; do not recalibrate thresholds. Compare each single-block
hybrid with its starting state and matched opposite state. Use 2,000 common
whole-pair bootstrap draws with seed 2026091071 separately for each fixed cohort.
Intervals are descriptive, conditional on the frozen model/cohort/calibration,
and do not correct adaptive hypothesis choice or multiplicity.

Do not add single-block effects as if they were independent contributions. A
hybrid may be unreachable under ordinary unmodified input history; its effect
identifies only this intervention in the stated deterministic state graph. It
does not prove a unique causal feature effect in real DEV, establish a remedy,
or justify replacing the fixed inference state. Any source, paired trajectory,
clone, complete-transfer, aliasing or constant-state mismatch stops the diagnostic
and preserves the failure. No new scientific fit follows from these results.
