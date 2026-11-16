"""Queue post-change checks without overlapping the isolated E003 measurements."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, read_json, atomic_json, digest, utc_now, progress
from src.next3.maintenance import register_job


def main():
    register_job({'id': 'F410_FULL_TESTS_AFTER_LEAN', 'kind': 'maintenance',
        'argv': ['-m', 'pytest', '-q', '--junitxml=artifacts/next3/maintenance/FULL_TESTS_CHECKPOINT_4.xml'],
        'description': 'Full suite after lean runtime, legacy-metadata validation, and final-reproduction selection changes',
        'requires': ['artifacts/next3/engineering/lean_shared/direct/RESULTS.json'], 'retry_safe': False})
    register_job({'id': 'F411_EXPORT_AFTER_LEAN', 'kind': 'maintenance',
        'argv': ['scripts/export_next3_progress.py'],
        'description': 'Refresh evidence inventory distinguishing actual failures, registration bugs and superseded pending jobs',
        'requires': ['artifacts/next3/maintenance/FULL_TESTS_CHECKPOINT_4.xml'], 'retry_safe': False})
    progress('Post-lean full verification queued', 'New full suite and evidence export run after isolated direct timing; no extra model fits')


if __name__ == '__main__': main()
