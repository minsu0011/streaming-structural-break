# Held-out router probability loss and competition ranking

The fixed D004 diagnostic uses the already completed original outer-validation
outputs. It does not evaluate training rows, fit another gate, change a weight,
or create another scientific selection rule. All five fold TS-AUC values match
their existing receipts. The sum of positive-observation pair-credit changes,
divided by the same-time pair count, reconstructs every fold score difference
within 1e-14.

| Existing recipe | Mean TS-AUC difference from BL80 | Mean W3 log-loss difference | Interpretation |
| --- | ---: | ---: | --- |
| J102 alone | -0.001636663 | -0.000766471 | Better probability loss does not recover the ensemble's ranking gain |
| R1 evidence ratio | -0.001028870 | -0.000866738 | Probability-loss improvement accompanies ranking deterioration |
| R1 score gap | +0.000109181 | -0.001076397 | Both improve slightly; fixed scientific promotion gates still fail |
| R2 nested logistic | -0.001158418 | -0.000768415 | Its probability surrogate improves while the competition metric falls |
| R3 nested depth 2 | -0.000629823 | -0.000546794 | The same disagreement occurs for the small tree gate |

Lower log loss is better. W3 is the frozen same-time opposite-class weighting,
applied here only as a held-out diagnostic. Probability clipping is fixed at
1e-7 for numerical evaluation and does not change any competition prediction.
These probabilities have not been established as calibrated posterior risks.

All observed outer-validation R2 weights are below the frozen C205 weight 0.2.
Fold means are 0.042561, 0.000025635, 0.072884, 0.056716, and 0.063314. Thus the
learned gate often moves strongly toward the J102-only direction, which itself
has lower W3 log loss and lower TS-AUC than BL80. R3 emits exactly 0.1 on every
validation row of folds 0–3. Fold 4 has weights 0.1 or 0.2 and mean 0.111023.
This held-out evidence is distinct from the final all-DEV R3 gate, whose leaves
all emit 0.1.

The result supports objective disagreement as a concrete diagnostic explanation
for these failures. It does not isolate a unique causal mechanism, prove that a
ranking-loss replacement will work, or authorize a new ranking-model experiment
after the finite menu is closed. The exact cross-fit construction and the
scientific rejection decisions remain unchanged.

Primary receipts: `artifacts/next3/router_metric_alignment/RESULTS.json`,
`SUMMARY.csv`, `FOLD_METRIC_ALIGNMENT.csv`, and `HELDOUT_GATE_WEIGHTS.csv`.
The immutable protocol is `docs/NEXT3_ROUTER_METRIC_ALIGNMENT_PROTOCOL.md` and
the prospective registration is `artifacts/next3/hypotheses/D004_ROUTER_METRIC_ALIGNMENT.json`.
