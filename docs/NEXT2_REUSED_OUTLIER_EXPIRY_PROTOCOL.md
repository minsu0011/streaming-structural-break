# Reused ablations at the outlier age-expiry boundary

Extend the already specified outlier age-expiry calculation to the separately registered B101, B102 and V007 original fold-0 models once their fixed O1 trajectories are complete. This extension is recorded before reading their new O1 values. Use exactly the same 342 onset-128/256 pairs, offsets 0..767, interval checkpoints, boundary contrasts, 2,000 draws and seed 2026090967 as the original age-expiry protocol.

Compare these three original fold-0 models with the original fold-0 Y001/J102/B104 results, keeping all-DEV fits separate. The diagnostic asks whether removing GLR memory, removing Bayes evidence, or shortening the age grid changes persistence beyond the original rolling age windows. Model retraining uncertainty and adaptive-selection uncertainty remain excluded. These are existing trained ablations; no additional fits, generator changes, threshold choice or candidate promotion are allowed. All O1 labels stay null.
