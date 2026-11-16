# Frozen official competition contract

Competition: `structural-break-real-time` (2026 real-time edition, not the 2025 known-boundary competition).

Authority: [official documentation](https://docs.crunchdao.com/competitions/competitions/structural-break-real-time), [official repository](https://github.com/crunchdao/competitions/tree/ddb0d2240aa885965a0cb5d16df04dae70c66679/competitions/structural-break-real-time). Source snapshots and SHA256: `vendor/official/SOURCE_MANIFEST.json`.

* Training receives `(dataset_id, historical, online, tau_index)`; runner maps stored `-1` to `None`.
* Inference receives `(historical, online_iterator)` without ID or target. Both outer and inner iterables are single-pass. Load artifact before initial `yield None`; consume exactly one point and emit exactly one finite float in [0,1]. No future point, horizon, cross-series state, disk writes, or fitting from test targets.
* Parquet input has MultiIndex `(id,time)` and `value`, `period` columns. Historical period is 1, online period is 2. Index time is not necessarily zero-based online time.
* Historical reference is fully available, with no break. Online has at most one break. Documented tau is zero-based within online; labels are positive from tau inclusive.
* Official scorer joins on `(id,time)` and computes per-series `cumcount()` online time in current row order. It skips single-class steps, weights AUC by positive-negative pair count, and returns 0.5 when all weights vanish. The project sorts `(id,time)` before either scorer: unsorted raw scorer input is order-sensitive.
* Official check rejects invalid index/column types, NaN/Inf and mismatched/duplicate row keys. The project additionally enforces [0,1], as required by the documented interface.
* Runner requires Crunch CLI >=11.7.0. Windows runner forces parallelism=1; Linux cloud can fork workers. Start with 1.
* Official transport converts historical, online points and output to float32. Test cloud transport rounding separately from direct Python predictions.

Documented dataset sizes are 10,000 training series and 100 reduced test series. These are **official descriptions, not counts observed locally**. Exact raw schemas, counts and label alignment remain subject to an empirical audit after acquisition.

Reduced labels may be used for schema/alignment contract verification, plus at most two scored diagnostics: initial official baseline and final candidate smoke. Never tune on them. The local final seal remains unevaluated.
