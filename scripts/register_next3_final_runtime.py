"""Freeze the final verification protocol without starting a new runtime search."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now


def main():
    freeze = read_json(ROOT/'configs/next3/SCIENTIFIC_ROLE_FREEZE.json')
    assert freeze['scientific_role_count'] == 2 and not freeze['new_scientific_search_authorized']
    paths = ('scripts/verify_next3_final_runtime.py', 'src/next3/final_runtime_probe.py',
             'configs/next2/RUNTIME_MEASUREMENT_LOCK.json', 'configs/next3/SCIENTIFIC_ROLE_FREEZE.json')
    atomic_json(BASE/'final_audit/FINAL_RUNTIME_PROTOCOL.json', {'status': 'REGISTERED_FINAL_VERIFICATION',
        'created_utc': utc_now(), 'source_sha256': {name: digest(ROOT/name) for name in paths},
        'fixed_order': ['DISTILLED', 'REFERENCE_PERFORMANCE', 'PERFORMANCE'],
        'points_per_entry': 10_000_000, 'block_points': 10000, 'maximum_selected_reference_ratio': 1.5,
        'H_and_buffer': 'Exactly the old locked 128 DEV timing probes: first H and first 10000 concatenated online values.',
        'roles': 'Two freshly raw-reproduced selected roles plus the unchanged old performance reference for the same-protocol timing comparison.',
        'selection': 'No candidate generation, parameter changes or new runtime family. E003 three-pair selection remains fixed.',
        'acceptance': 'Exact previous 10M float32 fingerprints, current-point causal protocol, finite unrounded probability range, fixed numeric state and <=1.5x same-protocol performance reference time.',
        'time_boundary': 'Only after 11h10 wall and separate final-audit start, with completed actual fresh raw reproduction.',
        'measurements_started': False, 'new_fits': 0, 'new_scientific_candidates': 0,
        'interpretation': 'Generator timing includes validation overhead and one H initialization; does not replace pure-predictor trials or cold-RSS measurements.',
        'script_sha256': digest(Path(__file__))}, immutable=True)
    progress('Final runtime verification protocol frozen', 'Two selected raw-reproduced roles and old performance reference will each run exactly 10M public API points during final audit; no new runtime family')
    print('NEXT3 FINAL RUNTIME PROTOCOL REGISTERED; NOT EXECUTED', flush=True)


if __name__ == '__main__': main()
