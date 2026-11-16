# Descriptive age-window expiry follow-up

The completed fixed 512-point O1 persistence diagnostic found nonzero mean probability differences at offset 511 for J102, Y001, B104 and their blend. This follow-up is registered after that observation; it cannot select or promote a model.

Use the existing original O1 prediction files for the same five all-DEV and 25 original CV0 models. Restrict to the 342 seed pairs whose locked injection onset is 128 or 256, because both groups have at least 768 observed post-injection points. The excluded onset-512 group lacks that future observed window; selection uses only the original schedule, never predictions or outcomes. Do not extend or regenerate the suite.

Report the entire matched-control mean probability-difference curve at offsets 0 through 767, and conditional paired-bootstrap intervals at offsets 127,128,255,256,511,512,639,767. Also report the paired change in the intervention-minus-control score difference from 127 to 128, 255 to 256 and 511 to 512. Use 2,000 whole-pair draws with seed 2026090967, common across every model and offset.

These boundaries correspond to existing age-grid windows. An observed score change at a boundary does not alone identify one feature's causal contribution: trained trees combine all evidence and the old ABCD branch. In particular, removing GLR memory does not remove Bayes age sums, CUSUM or all old memory. Intervals condition on frozen fitted models and this synthetic subgroup, omit adaptive-selection uncertainty, and are not competition validation. Keep all null labels null and retain all model results.
