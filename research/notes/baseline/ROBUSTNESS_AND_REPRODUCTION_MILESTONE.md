# DEV robustness and reproduction checkpoint

This is a post-selection diagnostic checkpoint. No seal performance, reduced
scalar, model parameters, promotion gates or group membership changed.

The research registry contains 249 complete five-fold candidates. The two
historical quantile remappings and two magnitude-dependence feature candidates
failed the existing primary gates. The best magnitude-augmented LightGBM mean
was 0.612021, but its median was 0.609503 and worst fold was 0.594278. The fixed
AR4/AR8 primary remains mean 0.611293, median 0.609959, worst 0.596411. The ARCH
HistGB candidate is the current robustness-backup choice under the original
policy, pending the final freeze.

## Raw reconstruction from an extracted source archive

`SRC_REAL_SNAPSHOT_20260908T130730Z.zip`, SHA256
`0562b1691867184b499fabdc24a24e2ff110a493af1b4d3a838d9f10d2257745`,
was extracted into a separate directory and executed with a separate Python
environment. The source-only check passed 161 tests, dependency consistency and
official scorer parity. Later source additions require another final archive
check; this snapshot does not verify source changes made after its timestamp.

The extracted copy used a workspace-contained junction to the unchanged
canonical raw data. It contained no original feature caches or OOF arrays.
Raw-to-feature reconstruction and independent five-fold fitting then passed:

| Candidate | OOF observations | Changed float32 predictions | Rebuilt cache bytes identical | Refit model bytes identical |
|---|---:|---:|---|---|
| Fixed AR4 LightGBM / AR8 HistGB average | 4,014,405 | 0 | All 5 folds | No |
| AR8 ARCH(1) HistGB | 4,014,405 | 0 | All 5 folds | All 5 folds |

Without the original OOF arrays, the comparison uses packaged original fold
models on newly reconstructed raw features. Equality of the cache file bytes
separately confirms the input reconstruction. The cross-order serialized model
bytes differ despite exact prediction equality; numerical determinism and
serialized byte identity are separate claims. Detailed receipts are in
`artifacts/real/cv_reproduction/standalone_archive_rebuild/`.

## Historical cohorts

Thirteen slices were specified before this diagnostic was scored: all DEV plus
three bins each for historical length, lag-one correlation, excess kurtosis and
AR8 historical squared-innovation lag-one correlation. These attributes alone
determine membership. No tau, label, online length or prediction is used to
form a cohort. This is still a post-selection diagnostic, not a prospectively
independent test.

| Historical structure | DEV series | Primary mean TS-AUC | Worst fold |
|---|---:|---:|---:|
| Length below 2,000 | 2,062 | 0.626660 | 0.616917 |
| Length 2,000 to below 3,500 | 2,901 | 0.603374 | 0.579732 |
| Length at least 3,500 | 3,037 | 0.608569 | 0.595445 |
| Squared AR8 residual ACF1 below 0.05 | 5,552 | 0.619904 | 0.597280 |
| Squared AR8 residual ACF1 0.05 to below 0.2 | 1,955 | 0.587927 | 0.576895 |
| Squared AR8 residual ACF1 at least 0.2 | 493 | 0.598257 | 0.566729 |

The final row has fewer than 100 series in at least one fold and is flagged for
small support. Each score excludes cross-cohort positive/negative pairs, so
these scores cannot be added to reconstruct the overall TS-AUC.

A paired whole-group bootstrap uses 400 replicates, preserves frozen folds and
uses the same multiplicities for all three fixed candidates. The ARCH HistGB
minus primary overall mean difference is +0.000757 with a marginal 95%
percentile interval [-0.001320, +0.002645]. There is no clear general superiority
from that conditional diagnostic. The short-history difference is +0.003970
[+0.000339, +0.007622], but intervals are neither simultaneous across cohorts
nor adjusted for adaptive selection. All results, including weak and uncertain
cohorts, are retained in `artifacts/real/historical_cohorts/`.

## Streaming engineering

The full real-input socket audit uses the unchanged pinned `Remote` transport
and double-protection code over actual Windows loopback sockets. A producer
and consumer thread run in the same process. This checks the real `infer()`
generator across multiple series, not just a prepared detector per series. It
also reverses a fixed prefix of series to check state reset. No target table is
opened and no score is computed. This does not consume reduced quota and is
not an additional authenticated `crunch_tools.test()` call.

Before the resident-state optimization, the primary matched all 4,014,405 OOF
values, with zero differences under reverse-series checks. The measured local
socket path took 272.536 seconds (67.889 microseconds per point). The guarded
direct API path took 86.375 seconds. These measurements include model loading,
readiness, historical initialization and result collection, but exclude raw
input loading. Other diagnostics shared the machine during part of the run.
The 678.895-second 10-million-point socket value is an extrapolation from real
DEV lengths, not a measured 10-million-point wire run or a cloud SLA.

The `deep_size` engineering helper now walks nested tuples/lists and retained
NumPy owners without counting shared objects twice. Older deep-size totals
under-counted some tuple-held state. Explicit `state_array_bytes` reports were
separate and remain valid. Final measurements use the corrected helper and
also report RSS, since native extension allocations are not fully visible to
Python object traversal.

The resident-state prototype keeps the original arguments in one typed object
and invokes the same arithmetic kernel. Seven paired 100,000-point probes
gave a median 1.5645 speed ratio with exact float32 predictions. This probe
alone is not production validation. Numba's documented native field storage
motivates the implementation; actual speed and numerical behavior are measured
locally. See the [official jitclass documentation](https://numba.readthedocs.io/en/stable/user/jitclass.html).

The integrated resident path passed all 4,014,405 DEV OOF predictions, 16,056
guarded real-prefix observations and 96 modified suffixes. Nine isolated paired
runs gave a median tuple/resident ratio of 1.556398. A measured continuous
10-million-point run took 54.391 seconds (5.439142 microseconds per point).
Explicit online arrays remained 4,858 bytes and reachable Python size remained
536,961 bytes. Peak RSS did not rise above the measured starting RSS. These
are the CV fold-0 model's engineering measurements; final all-DEV artifact
measurements remain a separate required step.

The resident primary and ARCH HistGB also passed complete OOF replays with
`NUMBA_CPU_NAME=generic` and `PYTHONHASHSEED=17`. This changes the compiled CPU
target on the same Windows host; it is not Linux execution or an independent
hardware test. Receipts are under `artifacts/real/fused/generic_cpu_hashseed17/`.

The unchanged official scorer's full `check()` and `score()` functions were
executed on every saved DEV OOF observation of the primary, ARCH HistGB and
ARCH LightGBM. Each candidate covered 4,014,405 observations and all five fold
scores had maximum difference zero against both the local scorer and the
saved research scores. Only I/O loaders were replaced with in-memory frames;
no reduced data or seal performance was involved.

A fresh-process socket audit separately verified all 4,014,405 primary OOF
predictions. Total time including five fresh process starts was 285.240 seconds
(71.054 microseconds per point). Startup to ready took 8.08–8.43 seconds per
process; after-ready processing, including per-history initialization and
wire transfer, cost 60.349 microseconds per point. Proportional extrapolation
of whole-job time also scales those five cold starts, so it must not be read
as one continuous consumer's 10-million-point latency. This remains a local
Windows protocol test, not cloud deployment.

## Further variance-lag hypothesis admission

Before considering another variance-lag model, an outcome-blind audit checked
all 8,000 DEV histories after the fixed ARCH(1) transform. The admission rule
was recorded before that diagnostic: at least 10% of histories must have
post-transform winsorized squared-residual ACF2 above 0.1. The observed fraction
was 8.5%, below the rule. No ARCH(2) model was fitted or scored, and the
threshold was not revised. Full historical summaries are under
`artifacts/real/arch_historical_residual_audit/`.
