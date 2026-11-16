# E003: load only the dependencies required by frozen packed inference

Prospective B8 engineering hypothesis, following E002's completed cost gates.
This is the third and final major runtime-maintenance hypothesis family in NEXT3:
precomputed arithmetic, packed nodes, and inference-only dependency loading.
There is one lean candidate, based on the selected PACKED_SHARED E002 variant.
No scientific fit, feature change, tree change, optional precompute switch or new
quality-based selection is allowed.

Source inspection found that the canonical bundle loader reaches training/cache
modules through bundle classes, candidate I/O and fitting imports. Those imports
are unnecessary for evaluating already trained numeric trees. The lean entry will
validate the unchanged canonical envelope and exact frozen 80/20 feature contract
without constructing training bundle classes. It will retain the E002 packed
forest representation and the same H/current-point arithmetic.

Where protected runtime files combine numeric kernels with training-dependent
classes, copy only an explicit finite list of required definitions into a new
NEXT3 numeric module. A build manifest binds every complete original source hash
and extracted definition hash. Preserve expression order and function bodies;
changing only disk-cache decorators to cache=False is permitted. Original source
files are not edited. Reuse existing pure numeric leaf modules directly. Do not
hide training imports through runtime monkeypatches or import stubs.

Before measuring the new candidate, freeze its generated sources, classless
decoder, minimal metadata contract, public entry and measurement scripts. Tests
must reject malformed envelopes, invalid graph/index bounds and mismatched
feature/source contracts, and must reproduce canonical trees and real streaming
features/predictions. A lean inference process must not import lightgbm, sklearn,
pandas, pyarrow or joblib, including during the first warmed None. Training may
import them lazily when the public train function is explicitly called.

The primary test uses three paired fresh subprocesses per variant in orders
[E002, LEAN], [LEAN, E002], [E002, LEAN]. Record import time, time through first
warmed None, process RSS after that None, peak working set where available, and
module names. All trials remain visible. Acceptance requires either median
post-warm RSS at most 80% of E002 with every pair smaller and median first-None
time at most 110%, or median first-None time at most 80% with post-warm RSS at most
100%. The cold probe uses fixed synthetic H only and touches no online iterator.

Also require unchanged private numeric state, unchanged prepared numeric model
size, exact every-point DEV/OOF/ALT/H equivalence, actual lazy read-only public API
equivalence, and three paired direct 10M trials with median time ratio at most
1.05 versus E002. No other project Python work runs during isolated measurements.
JIT time and process-memory boundaries are reported explicitly. Lower imported
module count alone does not pass the cost gate.

Any acceptance remains provisional until final fresh pinned raw-data training
reproduction and final audit. If a fixed gate fails, retain E002 and end this
runtime hypothesis family. No threshold relaxation or extra dependency variants.
