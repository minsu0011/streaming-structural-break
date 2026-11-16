"""Use the existing separately pinned figure environment, under the owned worker."""
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, utc_now


def main():
    python = ROOT/'.figure-venv/Scripts/python.exe'
    if not python.is_file(): raise RuntimeError('Existing pinned figure environment is required')
    script = ROOT/'scripts/plot_next3_scientific_menu.py'
    subprocess.run([str(python), '-u', str(script)], cwd=ROOT, check=True,
                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    atomic_json(BASE/'figures/scientific_menu_v1/LAUNCH.json', {'status': 'PASS', 'created_utc': utc_now(),
        'python': str(python), 'driver_sha256': digest(script), 'global_packages_modified': False,
        'wrapper_sha256': digest(Path(__file__))}, immutable=True)


if __name__ == '__main__': main()
