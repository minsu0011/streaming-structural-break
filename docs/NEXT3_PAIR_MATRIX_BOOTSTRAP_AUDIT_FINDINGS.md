# NEXT3 conditional bootstrap: independent numerical audit

The independent pair-matrix implementation reproduced every completed candidate's
original conditional bootstrap contrast exactly. This supports the arithmetic of
the reported evidence. It does not provide independent validation data or remove
adaptive DEV selection uncertainty.

The audit covered frozen BL80 and all twelve NEXT3 full-CV recipes, including
full-admission failures. It reconstructed 65 fold vectors, each with the observed
sample and the same original 2,000 whole-series resamples: 130,065 scores in total.
It enumerated 8,669,272,690 eligible ordered point pairs. Its recorded runtime was
44.43 seconds and its sampled peak process RSS was 495,198,208 bytes; these are
diagnostic computation measurements, not inference benchmarks.

For the 45 candidate/fold combinations with previously committed standalone
bootstrap arrays, all 90,045 numbers matched exactly. For all twelve candidates,
all 24,012 five-fold contrast entries matched D002 exactly. Four candidates had
no prior standalone full-fold forensic bootstrap arrays; their complete vectors
were reconstructed, their observed fold scores were checked, and their averaged
contrast vectors were checked against D002. Do not describe those missing prior
arrays as independently matched fold by fold.

The new code explicitly counts twice-wins and eligible pairs for every ordered
series pair, then evaluates the quadratic forms under the original multiplicity
vectors. It does not use the cached ranking kernel for reconstruction. Integer
count bounds guarantee exact float64 representation before division. Fifteen
tests passed, including a literal duplicated-row oracle, ties, unequal lengths,
zero weights, single-class cases, row permutation and invalid timeline rejection.

The joint conditional critical value reconstructed as 2.831791373275449, matching
the committed value within floating-point tolerance. Individual percentile
intervals and the fixed joint envelope also matched. The I2 lower bound under
the joint envelope remains negative; no full-CV candidate reaches the fixed
performance effect threshold. The numerical audit introduces no new fit,
candidate, bootstrap seed, fold, promotion rule or generalization claim.

Evidence:

- `artifacts/next3/pair_matrix_audit/run_v1/RESULTS.json`
- `artifacts/next3/pair_matrix_audit/run_v1/FOLD_AUDIT.csv`
- `artifacts/next3/pair_matrix_audit/run_v1/CONDITIONAL_SUMMARY_RECONSTRUCTED.csv`
- `artifacts/next3/pair_matrix_audit/IMPLEMENTATION_LOCK.json`
- `artifacts/next3/maintenance/PAIR_MATRIX_UNIT_TESTS.xml`

The primary uncertainty remains the previously reused, adaptively explored
8,000-series DEV set. All four independent seals remain unopened.
