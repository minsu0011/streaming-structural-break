"""Render only complete, hash-bound all-fold stress evidence."""
from pathlib import Path
import argparse
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress
from src.next5.scheduler import register_job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true')
    args = parser.parse_args()
    lock_path = BASE / 'analysis_figures/STRESS_POLICY_LOCK.json'
    if args.register:
        atomic_json(lock_path, {'created_utc': utc_now(), 'sources_sha256': {
            name: digest(ROOT / name) for name in ('scripts/run_next5_stress_figures.py', 'scripts/render_next5_stress_figures.py')},
            'plots': ['All 19 full candidates on 15 genuine-change scenarios',
                'X2 versus matched BL80 long-null rates over all five fixed folds',
                'P1/R2/X2 impulse amplitude decomposition at the 11 frozen lags'],
            'requires_complete_97_artifact_summary': True, 'requires_cohort_provenance_repair': True,
            'environment': 'Existing .figure-venv; no installation and no training',
            'scope': 'Scalar CSV/JSON evidence only; display means/ranges of fixed artifacts, no independent-fold intervals',
            'new_fits': 0, 'new_predictions': 0, 'new_seed_draws': 0}, immutable=True)
        register_job({'id': 'FIGURES_ALL_FITTED_STRESS', 'kind': 'validation', 'priority': 93,
            'depends_on_success': ['SYN_ALL_FITTED_SUMMARY', 'IMPULSE_AFFINE_DECOMPOSITION'],
            'argv': ['scripts/run_next5_stress_figures.py'],
            'description': 'Standalone all-fold scale counterexamples, long-null rates and impulse interpretation figures',
            'retry_safe': False})
        progress('Complete fitted-stress figures registered', 'Use full fixed coverage and original scalar results only')
        return
    policy = read_json(lock_path)
    for name, sha in policy['sources_sha256'].items():
        assert digest(ROOT / name) == sha
    progress('Render complete fitted-stress evidence', 'Fixed-artifact scores, null alarm horizons and impulse decomposition')
    subprocess.run([str(ROOT / '.figure-venv/Scripts/python.exe'), '-B', '-u',
        str(ROOT / 'scripts/render_next5_stress_figures.py')], cwd=ROOT, check=True,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    assert read_json(BASE / 'analysis_figures/stress_v1/RESULTS.json')['status'] == 'PASS'
    progress('Complete fitted-stress figures rendered', 'Three PNGs and a standalone three-page PDF, all source tables hash-bound')


if __name__ == '__main__':
    main()
