# NEXT3 causal router protocol

Frozen J102/C205 80/20 is R0. Only current-point component scores and already causal local evidence enter the router. Mechanism signatures, tau, ID, fold and future length are prohibited inputs.

R1_SCORE_GAP uses weights 0, 0.1, 0.2, 0.3 at the fixed C205−J102 cutoffs −0.1, 0, +0.1. R1_EVIDENCE_RATIO uses maximum absolute dependence EWMA8/32 across four lags divided by one plus maximum absolute raw-energy/forecast EWMA8/32. Dependence strength below 1 falls back to zero specialist weight; fixed ratio cutoffs are 1, 2, 4. Both preserve float32 component inputs and final float32 blending.

Stage S reuses exactly the original 3,000 screen IDs, folds 0 and 4, and matched frozen base-model OOF predictions. Mean delta versus 80/20 must be at least −0.001 and each fold delta at least −0.004. Only passing heuristics proceed to original full five-fold OOF. These heuristics fit no meta-model and do not inspect validation targets to set parameters.

Any later logistic/depth-2/residual combiner must first generate grouped inner base OOF entirely inside outer training. Original outer validation series and historical groups are absent from every inner/base/router training set. Simply training a gate on existing full five-fold OOF and scoring those same rows is forbidden. Learned menus require their own immutable exact formula/solver registration before results.

Candidate quality is measured against frozen 80/20, with J102 retained as a secondary scientific comparator. Conditional bootstrap does not adjust repeated DEV adaptation. Mechanism/age diagnostics are evaluation-only; no oracle gate is nominated. Failed screen receipts remain preserved.
