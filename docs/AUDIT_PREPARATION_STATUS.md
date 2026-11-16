# Transition to final DEV artifact verification

Model research closed after 249 complete candidates. The bounded adaptive queue
was exhausted; the last historical variance-lag admission rule was not met.
`artifacts/real/MODEL_RESEARCH_CLOSURE.json` records the reason and the exact
research-state hash. The original 10-hour session start, final deadline and
16:04:52 UTC diagnostic window were preserved.

The user's specification permits remaining time to be spent on meaningful
robustness and reproduction when further experiments are not justified. Final
DEV artifact preparation therefore starts before the last 45 minutes. This is
an audit-only transition, not a shortened session or a changed score gate.
`ResearchBudget.can_start()` now refuses new models after that transition.

`run_finalization_real.py --prepare-only` measures the fixed shortlist, freezes
up to three roles, refits only DEV, independently refits/replays the selections
and stops before reduced target access, final score disclosure or Desktop ZIP
creation. A direct early call of the primary reduced-diagnostic entry point was
also confirmed to refuse execution before authentication/data access, with the
quota ledger unchanged at 1. The final diagnostic remains blocked until the
original final buffer. The final handoff will supersede this progress note.

| Frozen role | Candidate | DEV mean | DEV median | DEV worst |
|---|---|---:|---:|---:|
| Primary | M7_cross_order_AR4_LGB_AR8_HGB_equal | 0.611293 | 0.609959 | 0.596411 |
| Robust backup | M3_histgb_AR8_ARCH1_ABCD_S3_1cb5e51ac7ae | 0.612049 | 0.606125 | 0.598774 |
| Fast backup | M4_lightgbm_ABCD_S3_122668fc55e8 | 0.607055 | 0.604984 | 0.595586 |

The isolated 10-million-point CV fold-0 selection measurements were 5.388313,
4.050697 and 3.523853 microseconds per point respectively. These are selection
measurements; the final all-DEV artifacts receive their own measurements.

Across all 8,000 DEV histories, three alternating runtime passes covered
24,086,430 point predictions over the primary and robustness models. Every
prediction matched its saved OOF value. The median pass cost including
historical initialization was 18.031647 microseconds per point for the primary
and 10.111532 for the robustness candidate. Historical initialization accounted
for 68.7855% and 58.5472% of those measured compute totals respectively. Input
loading, initial model loading/JIT and socket transport were outside these
intervals. The primary's median per-history initialization was 5.924 ms and
its 99th percentile was 12.521 ms. These use original fold models, not the
subsequent all-DEV refits.

Full actual-artifact parity is separately checked by
`verify_final_streaming_real.py`: the final all-DEV model's batch prediction on
historical-only cached features is compared with its actual guarded `infer()`
over every DEV raw input. No target column is read and no training-performance
score is computed. The primary additionally runs through a fresh-process
official socket path. This complements full OOF CV replay, which uses different
fold models.

`reproduce_final_model_from_raw.py` permits another independent all-DEV refit
in a new directory after extracting a source archive. It preserves the
packaged original artifact and compares all reconstructed feature rows, again
without scoring training performance or opening reduced/seal performance.

`RESEARCH_STATE.json` is the frozen model-research checkpoint. Its reduced-use
counter describes that checkpoint; the current authoritative quota is always
`REDUCED_TEST_USAGE_LEDGER.json`. Updating the model checkpoint after closure
would break its archived identity, so final operational status is recorded in
the finalization state, session state and final handoff instead.
