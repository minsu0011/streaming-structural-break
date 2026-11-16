# D007: independent reconstruction of the fixed conditional bootstrap

This is a numerical audit of already completed results. Freeze all twelve D002
full-CV candidates plus BL80, all original five folds, all original 2,000 paired
whole-series bootstrap draws and their seeds. Do not create new folds, models,
predictions, screens, selection thresholds or an alternative inferential claim.

For each original held-out fold and candidate, independently enumerate ordered
positive-series/negative-series pairs at common online times. Accumulate integer
matrices N and D: N contains twice the number of strict wins plus ties; D contains
the number of eligible point pairs. Each bootstrap score is

    (w.T @ N @ w) / (2 * w.T @ D @ w)

where w is the original whole-series multiplicity vector. Use 0.5 for a zero
denominator, matching the frozen scorer contract. Matrices contain diagnostic
pair counts only and need not be retained; never export real point predictions
or real row labels. Use bounded chunks for matrix products and float64 only
where all integer products/sums remain exactly representable below 2**53.

The new implementation must not call PreparedAUC or the existing weighted rank
kernel. It may load frozen OOF files and metadata through the DEV-only guard and
regenerate the unchanged multiplicity draws. Verify the pair enumeration on
small hand-computable fixtures with ties, unequal horizons, empty classes,
duplicate multiplicities, row-order changes and all-zero weights before use.

Require all 2001 scores per fold (observed plus bootstrap) to agree with existing
committed fold arrays when available, and every candidate's five-fold contrast
vector to agree with D002 within absolute 1e-14. Reconstruct all individual
percentile intervals, positive-delta frequencies and the already committed
joint conditional envelope. Do not reinterpret matching numbers as independent
statistical validation: adaptive hypothesis creation and prior DEV reuse remain
unaccounted for. A mismatch is an audit failure, not permission to tune or replace
the reported metric.
