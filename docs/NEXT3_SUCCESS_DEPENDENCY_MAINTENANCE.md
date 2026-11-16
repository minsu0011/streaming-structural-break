# Successful-worker prerequisites

The source-copy dry run M609 checked all 12,571 copied files successfully, but
its earlier `requires` prerequisite only required a pytest XML file to exist.
The M608 report existed even though one Windows newline fixture had failed.
No final archive or scientific acceptance was made. The production copy helper
preserved bytes correctly; the fixture was fixed to write explicit bytes, and
all 17 targeted tests subsequently passed. Both original and corrected reports
remain available.

New queue records can additionally use `depends_on_success`, an explicit list of
previously registered job IDs. The scheduler requires each named job's actual
state to be COMPLETE with integer exit code zero. A missing state, running job,
failed attempt or quarantined job holds the dependent job pending, while other
eligible work can proceed. An artifact file still must satisfy `requires` when
that field is present. File existence and successful execution are distinct.

Prerequisites must be registered first, so immutable queue records cannot create
forward references, self dependencies or cycles. Unknown IDs, duplicate IDs and
unsafe spellings are rejected before publication. Existing queue records retain
their original semantics; they are not rewritten retrospectively. Scientific
workers also retain their own stricter checks of result status, exact source
hashes and complete unit reports.

The targeted scheduler/lifecycle suite passed 44 tests before full-suite review.
The live supervisor must be replaced only after that full review, at an observed
idle boundary with matching PID/create-time identity and no owned child. Its
previous source, clock accounting, failure receipts and queue remain preserved.
The final activation receipt, when present, is
`artifacts/next3/maintenance/success_dependency_guard/ACTIVATION.json`.
