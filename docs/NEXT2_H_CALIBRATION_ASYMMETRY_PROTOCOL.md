# Fixed H-only calibration diagnostic

Registered before computing these summaries. This follow-up explains the already observed ARCH/tail scale-down weakness; it is not a model-selection experiment.

Use all 1,000 pre-existing partition-0 null calibration seeds in each of N0/N1/N2/N3, unchanged H=2,048 and burn-in=1,024. Generate only one unused online point, never read any competition row or model prediction. Apply the unchanged J102 new-branch H-only AR8 fit, ARCH1 normalization and historical raw-energy/forecast replay. Compare the reconstructed calibration with `CompactPairState` for the first seed of every background.

For raw energy `r=e_centered^2/v_H-1`, `r>=-1`. If H calibration is mean `m` and positive SD `s`, then standardized energy has pointwise lower bound `(-1-m)/s`, subsequently clipped below at -12 by the evidence updater. This algebraic floor says nothing by itself about eventual tree probabilities, power, or the independent forecast channel. Its magnitude can be smaller in H samples whose energy SD is dominated by extreme values.

Summarize H channel mean and SD, this negative floor, fitted ARCH alpha, maximum standardized H energy, fraction of H energy values clipped at +12, and the share of the centered energy sum-of-squares in the largest one percent of H positions. Report across-seed q01/q05/median/q95/q99; these are descriptive quantiles, not confidence intervals. The top-one-percent measure uses `ceil(.01*n)` samples and is computed before detector clipping.

There are zero new learner fits, scores, candidate rankings or threshold changes. The 4,000 historical AR/ARCH calibrations are fixed runtime initialization operations. Keep only per-seed synthetic parameter summaries and aggregates, with generator and source hashes. Do not infer a competition DGP or causality from these post-hoc summaries.
