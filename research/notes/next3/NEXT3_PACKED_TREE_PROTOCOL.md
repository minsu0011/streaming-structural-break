# E002: exact packed tree deployment representation

This is an engineering experiment under B8. It does not fit a learner, select a
new scientific recipe, or change any existing canonical file. Registration is
prospective: no packed implementation has been timed or scored at registration.

The compact tree evaluator currently retains separate int32 feature/child arrays,
float64 threshold/value arrays, and missing-direction arrays. A binary node uses
only a threshold or a leaf value. E002 will represent each node with a 16-byte
aligned record: int16 feature/left/right, uint8 missing direction, one reserved
byte, and float64 threshold-or-value. Roots use int16. Conversion rejects values
outside the representation bounds and malformed/cyclic trees. The feature test,
NaN direction, tree order, addition order, sigmoid casts and 80/20 blend arithmetic
remain exactly the same. Unused original tree arrays will not be retained by the
prepared deployment object; original scientific model files remain unchanged.

The finite menu has two variants: PACKED_SHARED, using the original shared
feature update, and PACKED_PRECOMPUTED, using the previously tested E001 feature
update. This is one packing hypothesis with a two-level, fixed feature-runtime
factor. No node-order, dtype, quantization, weight, tree-depth or compiler sweep.
Runtime sources and verification/benchmark scripts will be hashed in a separate
implementation lock before real DEV equivalence checks or timing.

Primary engineering outcome: at least 25% reduction in retained prepared/model
numeric bytes, compared with frozen shared performance, with no more than 5%
median direct 10M time regression. Alternatively, at least 10% median direct 10M
time reduction and no increase in prepared/model bytes qualifies. In both cases:
three paired direct 10M trials per variant, median point latency within 5%, median
H initialization within 10%, private state within 1.10 times baseline, and fixed
state size are required. All real DEV, original OOF, fixed ALT/H, edge-case,
interleaved-stream and continuous-stream predictions must be exactly equal.

The fixed trial orders are [FROZEN_SHARED, PACKED_SHARED, PACKED_PRECOMPUTED],
[PACKED_PRECOMPUTED, FROZEN_SHARED, PACKED_SHARED], and
[PACKED_SHARED, PACKED_PRECOMPUTED, FROZEN_SHARED]. All trials are retained. No
other project Python work may run during measurement; process isolation is
checked before each phase and each million points. A measurement overlap is
reported and prevents acceptance of that evidence, rather than being hidden.

If both variants qualify, choose PACKED_SHARED unless PACKED_PRECOMPUTED has at
least 5% smaller median direct runtime than PACKED_SHARED. This rule gives the
smaller state and simpler feature update priority when latency is similar.
Failure ends this finite experiment; thresholds will not be relaxed.

Risks include int16 overflow, malformed graph traversal, structured-array
alignment/compiler overhead, accidental floating-point reassociation, retained
duplicate buffers, source-hash omissions, and host timing variation. Tests and
explicit retained-allocation inventories address these risks. JIT compilation is
reported separately. Acceptance is provisional until the public lazy read-only
API, raw-data reproduction, protected-file checks and final audit also pass.

The primary scientific metric remains official TS-AUC. Exact prediction parity
inherits existing scores and uncertainty; no new DEV score improvement is claimed.
