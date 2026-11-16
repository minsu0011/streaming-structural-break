"""Qualified G2 coordinates using verified, unchanged G1 current outputs."""
from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
import numpy as np
from src.next7.common import BASE, save, read_json, digest, csv_write, utc_now, activity, require_guard, note
from src.next7.null_growth import (historical_admission, new_admission_args, admission_replay,
    QualifiedState, qualified_features, QUALIFIED_COLUMNS)
from src.next7.features import NAMES
from src.next7.calibration import load_calibration, cutoff
from src.next6.data import source
from scripts.next7.evaluate_g2_synthetic import check_lock


def main():
    guard = require_guard()
    check_lock()
    assert read_json(BASE / 'G2_SCREEN_GATE.json')['continue_full']
    calibration = load_calibration()
    previous = BASE / 'cache/g1'
    old_manifest = read_json(previous / 'MANIFEST.json')
    assert old_manifest['status'] == 'COMPLETE'
    for name, sha in old_manifest['files_sha256'].items():
        assert digest(previous / name) == sha
    dest = BASE / 'cache/g2'
    if (dest / 'MANIFEST.json').exists():
        print('G2 RP cache complete; preserve')
        return
    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError('Partial G2 cache requires an explicit repair receipt')
    files = ['scripts/next7/prepare_g2_rp.py', 'src/next7/null_growth.py', 'src/next7/features.py']
    hashes = {n: digest(ROOT / n) for n in files}
    save('G2_RP_FEATURE_IMPLEMENTATION_LOCK.json', {'status': 'FROZEN', 'created_utc': utc_now(),
        'source_sha256': hashes, 'G2_protocol_sha256': digest(BASE / 'G2_PROTOCOL_LOCK.json'),
        'G1_cache_manifest_sha256': digest(previous / 'MANIFEST.json'),
        'before_G2_RP_scores': True})
    dest.mkdir(parents=True, exist_ok=True)
    with activity('G2 H-only qualified RP features', 'FEATURE_RESEARCH', 'F4_NULL_GROWTH_CORRECTION'):
        data = source()
        count = int(data.series['online_offsets'][-1])
        raw = np.load(previous / 'state.npy', mmap_mode='r', allow_pickle=False)
        qualified = np.lib.format.open_memmap(dest / 'qualified.npy', mode='w+', dtype=np.float64, shape=(count, 15))
        features = np.lib.format.open_memmap(dest / 'head_features.npy', mode='w+', dtype=np.float64, shape=(count, 10))
        calibrations, checks, hrows = [], [], []
        for index, sid in enumerate(data.ids):
            h, o, _ = data.get(int(sid))
            a, b = data.series['online_offsets'][index:index + 2]
            cut = cutoff(calibration, len(h), 0, 1)
            cal = historical_admission(h, core_cut=calibration['core_cut'], scale_cut=cut)
            out = admission_replay(raw[a:b], new_admission_args(cal))
            assert np.isfinite(out).all() and np.all(np.diff(out[:, 30]) >= 0)
            assert not np.any((out[:, 40] > 0) & (out[:, 18] == 0))
            np.testing.assert_array_equal(out[:, :30], raw[a:b])
            qualified[a:b] = out[:, 30:]
            features[a:b] = qualified_features(out)
            calibrations.append({'dataset_id': int(sid), **cal})
            hrows.append({'dataset_id': int(sid), 'H_length': len(h),
                'reference_prefix_length': cal['reference_prefix_length'], 'pseudo_blocks': cal['pseudo_blocks'],
                'reference_valid': cal['reference_valid'], 'support_reason': cal['support_reason'],
                **{f'cut_blocks{g}': v for g, v in zip(cal['grid_blocks'], cal['admission_cuts'])}})
            if index % 64 == 0:
                split = min(len(o), max(1, len(o) // 2))
                state = QualifiedState(h, core_cut=calibration['core_cut'], scale_cut=cut)
                actual = np.array([state.predict_one(x) for x in o[:split]])
                np.testing.assert_array_equal(actual, out[:split])
                checks.append({'dataset_id': int(sid), 'prefix_points': split, 'scalar_exact': True})
            if index % 400 == 399:
                qualified.flush()
                features.flush()
                print(utc_now(), 'G2 RP', index + 1, '/6400', flush=True)
        qualified.flush()
        features.flush()
        del qualified, features, raw
        save('cache/g2/H_CALIBRATIONS.json', calibrations)
        csv_write('H_CALIBRATION_RP_G2.csv', hrows)
        save('audit/G2_RP_PREFIX_SPOT_CHECKS.json', {'status': 'PASS', 'created_utc': utc_now(),
            'selection': 'sorted RP index modulo64==0', 'series': len(checks), 'results': checks})
        save('cache/g2/MANIFEST.json', {'status': 'COMPLETE', 'created_utc': utc_now(), 'series': 6400,
            'rows': count, 'qualified_columns': QUALIFIED_COLUMNS, 'head_feature_names': NAMES,
            'files_sha256': {p.name: digest(p) for p in dest.iterdir() if p.is_file()},
            'source_sha256': hashes, 'G1_cache_manifest_sha256': digest(previous / 'MANIFEST.json'),
            'G2_implementation_sha256': digest(BASE / 'G2_IMPLEMENTATION_LOCK.json'),
            'research_split_sha256': guard.split.digest, 'tau_or_future_length_passed_to_state': False,
            'outer_score_computed': False, 'current_G1_outputs_identical': True,
            'large_cache_excluded_from_handoff': True})
    guard.check_state()
    note('G2 RP causal features ready', 'Fit only the preregistered qualified-memory correction', 'F5_CAUSAL_NESTED_INTEGRATION')


if __name__ == '__main__':
    main()
