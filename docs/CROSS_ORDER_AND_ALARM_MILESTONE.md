# DEV milestone: fixed AR4 / AR8 average

This is an intermediate research checkpoint, not the final handoff or an independent test result.

The prospectively fixed 50:50 average of `M4_ar_residual_input_ABCD_S3_9f68be316ea3` and `M3_ar_order8_ABCD_S3_242216e87a34` passed the original promotion gates. Its experiment ID is `M7_cross_order_AR4_LGB_AR8_HGB_equal`. No stacking weights were fitted or searched. Each child probability is converted to float32, the two values are averaged in float64, and the result is converted to float32.

| Fold | DEV TS-AUC |
|---|---:|
| 0 | 0.6057469121023034 |
| 1 | 0.6278833674453808 |
| 2 | 0.6099585400267096 |
| 3 | 0.6164640827673485 |
| 4 | 0.5964105295150426 |

Mean 0.611292686371357, median 0.6099585400267096, worst 0.5964105295150426. Relative to the previous AR4 primary, median improved by 0.0019511241 and the worst fold improved by 0.0027222999. The mean is not the highest observed among all research candidates; the frozen promotion gates govern the primary role.

The two historical AR states are independent and their fitted parameters remain fixed online. Each computes the original 57 features, so the combined representation has 114 columns. The streaming API retains only the two fixed-size states. Independent five-fold refitting reproduced all 4,014,405 float32 OOF predictions exactly. Complete live OOF replay also passed. The guarded API test covered 16,056 real online observations and 96 modified suffixes, including no length query, no materialization, no replay, no read-ahead, and no inference writes.

The original two-call path measured 8.884479 microseconds per point for an isolated one-million-point run. The single compiled dispatch measured 8.488400 microseconds per point in a separate isolated run. A stronger alternating comparison of nine paired 100,000-point runs gave a median reference/single-call speed ratio of 1.061978; all nine pairs favored the single-call path. These timings exclude historical initialization and IPC. Full final-artifact measurements remain required after final refitting.

## Actual first-alarm diagnostic

`scripts/diagnose_first_alarms.py` freezes 1% and 5% training-null tail fractions. For each outer fold, its model predicts only no-break TRAIN series to set a strict-threshold crossing rule from their series maxima. Held-out predictions and outcomes do not choose the threshold. Train predictions are in-sample; this is not a formal false-alarm guarantee. The competition model continues to emit its unchanged continuous probability. This diagnostic does not select a new model or change its outputs.

| Model | Training tail target | Held-out no-break any-alarm rate | Valid detected break rate | Detected-only median delay |
|---|---:|---:|---:|---:|
| Official EWMA | 1% | 1.0355% | 1.1663% | 176 |
| Official EWMA | 5% | 5.0049% | 3.7018% | 127.5 |
| Fixed AR4/AR8 average | 1% | 1.3807% | 4.9696% | 133 |
| Fixed AR4/AR8 average | 5% | 5.7199% | 13.1846% | 142.5 |

A premature alarm counts as failure to detect that later break. At the 5% training target, premature alarms occurred in 2.7383% of break series for the average. Only 0.0518% of break series with at least five post-break observations had a valid alarm within age 0–4. The median delays above condition on detected cases and omit undetected/censored series; they must be read beside detection rates. Available online horizons vary across series. These are post-selection DEV diagnostics, not independent performance guarantees.

The first-alarm results substantiate the weak early discrimination already seen in the time and break-age TS-AUC slices. A higher ranking score does not establish a usable low-delay alarm system.

## Further bounded hypotheses

The historical-context experiments were rejected: adding six static historical statistics to AR4 ABCD yielded mean 0.601113; context alone yielded 0.491926. AR8 ordinary-row and square-root weighting yielded 0.595549 and 0.605864, below the original S3 reference.

Output clipping and conditional-variance input experiments have their own prospective policies. The nonlinear historical audit used all 8,000 DEV histories without opening outcome tables. AR8 squared-residual lag-1 correlation exceeded 0.1 in 17.225% of histories, motivating exactly two ARCH(1) learner comparisons. No result of those comparisons is asserted in this milestone.

`LOCAL_FINAL_SEAL_OPENED = false`. Reduced scored uses remain 1/2 and neither reduced scalar has been used for model selection. The final freeze, second permitted official diagnostic, final audit and Desktop ZIP are reserved for the original final buffer.
