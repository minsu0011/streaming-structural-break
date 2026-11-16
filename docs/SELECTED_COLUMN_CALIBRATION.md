# Historical calibration of fitted tree columns

The 128-history AR4 stage profile attributed 3.670 ms of 6.773 ms mean
initialization to median/MAD calculations. The primary fold-0 trees use 27 of
the 48 historically calibrated columns; the B-family candidate uses nine.

The production fast initializer now intersects the historical calibration mask
with actual exported tree split columns. It computes the original median and
MAD for that intersection, preserving their values exactly. Other columns have
zero center, unit scale and a false calibration mask. No fitted tree split can
read those unused calibrated values. If no tree column requires calibration,
the historical calibration replay is omitted entirely.

Training features, historical fitting policies, caches and learner parameters
are unchanged. The normal bundle remains the full-feature reference. The
streaming feature kernel still computes all 57 online features. This change
reduces initialization and unnecessary transformation work; it does not claim
per-family pruning of the online feature kernel.

The AR4 primary and AR8 HistGB each reproduced all 4,014,405 original DEV OOF
values exactly with the new fast initializer. The B-family candidate also
passed a complete DEV replay. The full test suite passed 135 tests. Paired
alternating-order initialization comparisons on 128 real histories gave
2.016x speedup for AR4 and 2.362x for its B-family candidate relative to the
original full-reference fused path. These timings had concurrent validation
load; final isolated latency remains authoritative.

The implementation hash changed, so old deployment descriptors fail closed
until regenerated. Prior fused verification reports are preserved under
`artifacts/real/fused/history/before_selected_column_calibration/`.
