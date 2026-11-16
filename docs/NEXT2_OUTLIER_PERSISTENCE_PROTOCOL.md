# Single-observation persistence diagnostic

This is a descriptive follow-up registered after the original O1 aggregate false-alarm results were available. It does not nominate or fit another competition candidate, change the generator, select a threshold, or change any promotion gate.

Reuse every completed original fold-0 component model (25) and the five all-DEV reference/finalist models. Keep these fitted-model populations explicitly separate. Reuse the original 512 matched control/outlier pairs, the locked onset cycle 128/256/512, and the original independently calibrated alpha 0.05 and 0.01 thresholds. O1 has no persistent break and all labels remain null.

Examine the common 512-point post-injection window. Report paired probability differences at offsets -1, 0, 1, 7, 31, 127, 255 and 511. Separately report threshold occupancy and any-crossing probability in [0,8), [8,32), [32,128), and [128,512). A late-window crossing is a persistence diagnostic; it is not a first-alarm detection success. Existing false alarms before injection are not removed from these unconditional summaries.

Use 2,000 whole-pair bootstrap draws with seed 2026090963, shared across every model and window. Intervals are conditional on the fitted model, the fixed calibration sample and this synthetic DGP. They do not include training, threshold-estimation or adaptive-selection uncertainty. Quantities for different models or windows are correlated. Retain all results, including failed real-DEV candidates; no selection follows this diagnostic.

Before summarizing, verify every input prediction hash, that all stored O1 labels are null, exact pre-injection prediction equality, and that the generator produces identical historical and post-injection observations apart from the single injected observation. This distinguishes continued detector state effects from a persistent intervention in the latent DGP. It does not by itself identify which internal feature causes the persistence.
