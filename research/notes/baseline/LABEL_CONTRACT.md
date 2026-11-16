# Label alignment

Status: **DOCUMENTED; EMPIRICAL OFFICIAL-DATA CHECK PENDING**.

The [official real-time documentation](https://docs.crunchdao.com/competitions/competitions/structural-break-real-time#break-position) calls tau a zero-indexed online position. The pinned quickstarter says the ideal score is 1 from tau onward. `labels_from_tau(T, tau)` implements `target[t] = int(t >= tau)`; `None` and stored `-1` mean all-zero target.

The official runner loads `tau_index` from `y_train_index.parquet` and splits periods 1/2. `(id,time)` absolute index values must not be compared directly to tau: use the zero-based position inside each online segment.

Tests cover tau=0, interior tau, tau=T-1, no-break, invalid indices, and label monotonicity. These establish implementation correctness under the documented convention; they do **not** substitute for actual reduced-label alignment. That check is explicitly pending while competition data is unavailable. No real-data supervised ranking is allowed before verifying the contract.
