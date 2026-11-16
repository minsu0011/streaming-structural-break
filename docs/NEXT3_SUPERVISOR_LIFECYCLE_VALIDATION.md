# Lifecycle validation of the NEXT3 maintenance supervisor

The run has already encountered queue registration and audit-schema errors.
This maintenance check adds explicit process-lifecycle evidence without touching
the live research worker or its accumulated accounting.

The test suite starts only its own short, hidden subprocesses. Each child sends
a readiness line and waits for a pipe input. A controlled exit code 23 verifies
that recovery retains the old attempt and records failure. A still-running child
with a matching creation timestamp must block duplicate ownership; a reused PID
with a different timestamp must never be adopted or terminated. A simulated
1,801-second inactivity age verifies that the hang path terminates only the test's
owned child and saves its diagnostic. No real half-hour wait or synthetic credit
is added to the project clock.

Separate deterministic clock/resource tests cover idle intervals, active worker
intervals, recent agent work, suspension gaps exceeding two heartbeat periods,
RAM at 85%, disk at the 10 GiB limit, and a terminal session with pending work.
They use temporary state files and isolated module bindings; the live supervisor
continues to record its own actual observed activity.

Static review of the last case found that a COMPLETE session could still launch
one eligible queued task before returning its terminal status. Add an explicit
non-COMPLETE condition to worker launch. Preserve the previous complete source
at `artifacts/next3/maintenance/supervisor_terminal_guard_fix/original_maintenance.py`.
This changes lifecycle termination only, with no feature, learner, model, metric,
or accumulated accounting change. The running older supervisor will be replaced
at a verified idle boundary after the isolated benchmarks and queued checks.

This document describes the checks and correction; it is not a claim that they
have already passed. Their actual full-suite XML and replacement receipt remain
the authorities. Intentionally induced test-child failures are distinguished
from project experiment failures.
