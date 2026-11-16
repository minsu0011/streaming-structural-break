# Deployment engineering status

`main.py` exposes the official two-function API and `INFER_PARALLELISM=1`. During
research, fold models and their scores live under `artifacts/models/real` and
`artifacts/real/RESEARCH_STATE.json`. Once final selection is frozen,
`configs/deployment.json` directs `train()` to refit that specification using only
the frozen DEV IDs. It skips seal entries before inspecting their arrays or labels
and rejects unknown, missing or duplicate IDs. Before that configuration exists,
the default is the explicitly labeled B7 engineering fixture.

`infer()` loads and hashes artifacts before readiness, warms JIT code in memory,
yields `None`, then emits one Python float for each incoming observation. It does
not write files. Numba caching is disabled. State size depends on representation
and lag configuration; it remains constant as the online stream grows. Historical
initialization, first compilation, point latency and amortized cost are reported
separately by `scripts/benchmark_real_candidate.py`.

Supported fitted paths are raw features, historical mean/SD or median/MAD
calibration, optional finite-prefix variance correction, historical AR(2/4/8)
residual inputs with calibrated features, AR8 ARCH(1) variance-standardized
inputs, and the fixed AR4 LightGBM / AR8 HistGB 50:50 average. Numeric HistGB, LightGBM and scalar
binary XGBoost have verified compact/fused inference paths. Source hashes for
fused kernels and optional initialization optimizations are checked in the
deployment descriptor. Research rank-Gaussian, historical-quantile and
magnitude-dependence inputs remain exploratory artifacts; they are not silently routed through a
different production representation.

The unchanged official `_run_with_double_protection()` and `Remote` transport
were exercised over TCP loopback with synthetic fixtures and all 8,000 DEV
series for the primary and ARCH HistGB models. The real-input audit opens no
target table and computes no score. Real-prefix model
contract reports additionally reject future-length queries, indexing, iterator
replay, read-ahead, and Python/OS file writes. Selected full DEV replays compare
every float32 prediction against the untouched saved OOF array.

The permitted authenticated `crunch_tools.test()` diagnostics are one baseline
and one final frozen primary run, each on 100 reduced series / 50,983
predictions. The baseline scalar is withheld until final freeze. Read
`REDUCED_TEST_USAGE_LEDGER.json` for the authoritative completed usage count and
report paths. There are no further scored diagnostics after both are used, and
no cloud submission has occurred.

An additional full runner check executes the unchanged official `run()` and `execute()` functions with synthetic parquet in temporary directories and a new Python subprocess for each command. Training, parallelism discovery, inference, the official 10% determinism rerun and official prediction-index/dtype checks all pass. It produces 2,400 correctly indexed predictions. This substitutes only the local execution context and tracing; authentication and the cloud data sandbox remain untested. `scripts/verify_offline_runner.py` reproduces this check without reading competition data.

Explicit AR-order bundles preserve order 2 or 8 through save/load, final DEV-only
training, native inference, fused inference and shared historical initialization.
The AR8 HistGB candidate passed all 4,014,405 OOF values in both independent CV
refitting and full fused live replay. Its stronger worst fold did not change the
precommitted primary promotion threshold. Final backup refits have an explicit
BACKUP status and use the same unopened-seal training guard as PRIMARY.

Windows official runner forces one inference worker. Linux fork behavior and multiworker memory have not been measured. Kernel/library portability and wall-clock timing must be checked in the actual Crunch environment. Current packages are frozen in `requirements-lock.txt`; an installed local version is not evidence of an approved cloud package version.

The cross-order production predictor keeps the exact original kernel arguments
in a Numba typed resident object. Model files, feature calculations and the
float32 averaging rule are unchanged. The resident object's class is compiled
in memory before readiness and is neither serialized nor cached to disk. Its
source bytes are part of the cross-order production hash. The original tuple
call remains available for numerical and paired timing comparisons. See
`docs/ROBUSTNESS_AND_REPRODUCTION_MILESTONE.md` and the current `LATENCY_REAL.json`
for the distinction between model-only, initialization and socket measurements.

Predictions are generated in native order and matched to `(id,time)` keys by the official runner. Local TS-AUC frame scoring sorts `(id,time)` before the official online `cumcount` convention. The default statistical bundle and compact models are not calibrated posterior probability estimates; they output valid ranking scores in [0,1].

Do not upload the entire research workspace. It includes caches, virtual environments and local reports. Prepare a small source bundle containing `main.py`, `src/`, the training source/configs and the reviewed model artifact, with compatible requirements. Credentials under `.crunchdao` are never included. No upload was performed.
