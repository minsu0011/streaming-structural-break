# NEXT3 exact shared inference arithmetic

This engineering hypothesis preserves the scientific J102/C205 models, fixed 80/20 weight, all selected feature values, all evidence recurrences and component float32 rounding. Frozen canonical artifacts remain available and unchanged.

The frozen shared runtime spends more measured time updating features than evaluating trees (NEXT2 medians 6.1 and 3.5 microseconds in separately dispatched calls). Precompute the same H-dependent serial Bayes arithmetic already tested in NEXT2, and additionally precompute fixed-age C205 Bayes constants. The C205 precomputation and their integration into the shared runtime are new. The earlier J-only version did not meet its 10% engineering threshold; that negative result is retained.

No fast-math, changed multiplication/division order, evidence clipping change, new truncation, altered ages, threshold quantization or tree refit is allowed. The additional serial tables occupy 1,200 private numeric bytes per series. The 320-byte dependence table is read-only and shared across streams. Numeric memory accounting must explicitly handle the nested argument layout and avoid counting model arrays as private state.

Before timing, compare every selected J/C feature and final output across all 8,000 DEV series using the all-DEV artifact and original OOF models. Then check all existing held-out predictions from the three alternate partitions and five leave-H-regime-out models. These are runtime parity checks; no new quality score or fit is generated. Save exact fingerprints and source-bound receipts.

After parity passes, run three paired isolated timing trials against the frozen shared runtime in the same session. Alternate execution order: baseline/new, new/baseline, baseline/new. Retain all runs. Reuse the old fixed DEV timing IDs and stream buffer, with three real-prefix repeats and an actual scalar 10,000,000-point continuous stream per variant per trial. Per-point and H initialization distributions include median, p95 and p99. The continuous stream must exactly match the old frozen fingerprint and keep bounded state.

The prospective engineering replacement threshold is a median new/baseline direct 10M time ratio at most 0.90, each paired trial faster, median point latency no worse, median H initialization no more than 5% slower, and private state no more than twice the frozen shared reference. Full parity, actual online API tests, serialization and fresh reproduction remain required before final recommendation. A faster result below this threshold is reported as an unselected measured alternative. Timing is conditional on this host; OS power and unrelated system load are not controlled.

The maximum four scientific candidate roles are unchanged by an exact runtime alternative. No new performance or generation claim follows from this work.
