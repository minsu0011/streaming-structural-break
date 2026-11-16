"""Fresh current-point-only replay of both immutable NEXT2 public entries."""
from pathlib import Path
import sys
import argparse
import importlib
import hashlib
import time
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, read_json, digest, progress, utc_now
from src.next.data import prepare_dev
from scripts.verify_next2_shared_public_api import Datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=['distilled', 'performance'], required=True)
    args = parser.parse_args()
    out = BASE/'preflight'/args.role
    if (out/'RESULTS.json').exists():
        raise RuntimeError('Preserve completed replay; do not overwrite')
    selection = read_json(ROOT/'DEV_CANDIDATES_NEXT2.json')[args.role]
    folder = ROOT/selection['model_directory']
    assert digest(folder/'model.sbnext2') == selection['canonical_model_sha256']
    reference = ROOT/'artifacts/next2'/(
        'canonical_api_all_dev/DISTILLED/RESULTS.json' if args.role == 'distilled'
        else 'shared_public_api_all_dev/PERFORMANCE/RESULTS.json')
    expected = read_json(reference)['output_fingerprint_sha256']
    progress('Frozen public replay', args.role, phase='PREFLIGHT')
    data = prepare_dev(ROOT)
    source = Datasets(data, range(len(data.ids)))
    entry = importlib.import_module(Path(selection['entry']).stem)
    stream = entry.infer(source, folder)
    assert source.accessed == 0
    assert next(stream) is None and source.accessed == 0
    fingerprint = hashlib.sha256()
    tick = time.monotonic()
    count = 0
    offsets = data.series['online_offsets']
    for j in range(len(data.ids)):
        values = np.empty(int(offsets[j+1]-offsets[j]), dtype=np.float32)
        for k in range(len(values)):
            value = next(stream)
            assert type(value) is float and np.isfinite(value) and 0 <= value <= 1
            assert source.index == j and source.current.consumed == k+1
            values[k] = value
        fingerprint.update(values.tobytes())
        count += len(values)
        if (j+1) % 500 == 0:
            progress('Frozen public replay', f'{args.role}: {j+1}/8000 DEV series', phase='PREFLIGHT')
            print('NEXT3 FROZEN REPLAY', args.role, j+1, flush=True)
    try:
        next(stream)
    except StopIteration:
        pass
    else:
        raise AssertionError('Extra output after stream')
    assert fingerprint.hexdigest() == expected
    data.guard.check_state()
    atomic_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(),
        'role': args.role, 'series': len(data.ids), 'points': count,
        'output_fingerprint_sha256': fingerprint.hexdigest(), 'expected_fingerprint_sha256': expected,
        'reference_receipt_sha256': digest(reference), 'canonical_model_sha256': digest(folder/'model.sbnext2'),
        'script_sha256': digest(Path(__file__)), 'elapsed_seconds': time.monotonic()-tick,
        'first_none_before_data': True, 'one_current_point_per_output': True,
        'future_length_and_suffix_inaccessible': True, 'seal_rows': 0,
        'new_learner_fits': 0, 'quality_score_computed': False}, immutable=True)
    print('NEXT3 FROZEN REPLAY PASS', args.role, count, flush=True)


if __name__ == '__main__':
    main()
