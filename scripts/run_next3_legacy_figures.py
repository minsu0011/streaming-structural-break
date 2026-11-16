"""D006 reporting in the unchanged isolated figure environment."""
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, utc_now


if __name__ == '__main__':
    script = ROOT/'scripts/report_next3_legacy_state_transfer.py'
    subprocess.run([str(ROOT/'.figure-venv/Scripts/python.exe'), '-u', str(script)],
                   cwd=ROOT, check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    atomic_json(BASE/'figures/legacy_state_transfer_v1/LAUNCH.json', {
        'status': 'PASS', 'created_utc': utc_now(), 'source_sha256': digest(script),
        'global_packages_modified': False}, immutable=True)
