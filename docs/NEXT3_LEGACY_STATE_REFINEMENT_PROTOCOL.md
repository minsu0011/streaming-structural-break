# D006: refine the legacy branch in the original O1 state diagnostic

D005 has completed. At anchors 0/127/255, transferring the legacy branch
changes mean BL80 probability over offsets [256,512) by approximately 0.115
in either direction. At anchor 511 the separate original 342-pair cohort also
retains legacy and J102 serial effects. This prospective follow-up asks which
legacy recurrent substate carries the observed intervention response.

Use exactly the original D005 model, 512 paired O1 seeds, observations, horizons,
four anchors, two directions, window boundaries and frozen alarm thresholds.
Keep the 342-pair late-availability cohort separate. No new fit, candidate,
threshold, online rule, TS-AUC, seed, horizon or quality comparison is allowed.

The seven disjoint legacy interventions are AR filter ring/counter, ARCH previous
squared state, standardized lag ring/counter, all sparse EWMA moment cells,
positive CUSUM, negative CUSUM, and running mean. Historical constants and
deterministic age/weight counters remain equal. Transfer the named array or
counter slice from the matched opposite trajectory after the anchor update.
Copying both CUSUM states separately is a diagnostic contrast, not a candidate
ablation. EWMA cells remain one registered block; no cell search is authorized.

Before accepting results, reconstruct every original J102/C205/BL80 prediction,
verify clone replay, and verify that transferring all seven fine blocks exactly
reproduces the existing complete-legacy-branch transfer in both directions.
Require non-aliasing of recurrent arrays, unchanged model arrays and state size,
and exact deterministic/H-only constants. Compare aggregate complete-legacy
trajectories and windows with D005's committed files. Any mismatch stops the run.

Report the same probability mean, threshold occupancy and any-crossing windows
with 2,000 whole-pair draws and seed 2026091071. Effects are conditional on this
observed synthetic state graph, can interact, and must not be added or interpreted
as generalizable feature importance. No remedy follows without a separate future
scientific protocol. The scientific menu remains closed in NEXT3.
