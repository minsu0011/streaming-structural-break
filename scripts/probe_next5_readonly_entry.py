"""Cold import and tiny synthetic batch parity; no real observations or fits."""
from pathlib import Path
import argparse
import hashlib
import importlib
import os
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now
from src.next5.stream_contracts import Datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cid = 'E2_SPARSE_START_MIXTURE'
    policy = read_json(BASE / 'runtime_validation/API_POLICY_LOCK.json')
    descriptor = ROOT / policy['candidates'][cid]['descriptors']['0']
    records = []
    for number, length in enumerate((256, 512)):
        h = (np.sin(np.arange(length) * .31) + .2 * np.cos(np.arange(length) * .17)).astype(np.float32)
        online = (.6 * np.sin(np.arange(96) * .47) + .1 * number).astype(np.float32)
        records.append((number, h, online))
    active, writes = [True], []
    def audit(event, arguments):
        if not active[0]:
            return
        write = event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'os.chmod', 'os.truncate', 'os.link', 'os.symlink')
        if event == 'open':
            path, mode, flags = arguments
            write = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if write and isinstance(path, (str, bytes, os.PathLike)) and os.path.normcase(os.fsdecode(path)) == os.path.normcase(os.devnull):
                return
        if write:
            writes.append(event)
            raise RuntimeError('Cold NEXT5 adapter attempted persistent mutation: ' + event)
    sys.addaudithook(audit)
    outputs = []
    try:
        entry = importlib.import_module('src.next5.readonly_streaming')
        source = Datasets(records)
        stream = entry.infer(source, descriptor)
        assert source.accessed == 0 and next(stream) is None and source.accessed == 0
        for sid, historical, online in records:
            actual = np.empty(len(online), dtype=np.float64)
            for index in range(len(online)):
                actual[index] = next(stream)
                assert source.sid == sid and source.current.consumed == index + 1
            outputs.append(actual)
        try:
            next(stream)
        except StopIteration:
            pass
        else:
            raise AssertionError('Extra output after all current observations')
        from src.next.readonly_bootstrap import cache_requests_suppressed
        suppressed = cache_requests_suppressed()
        assert suppressed and not writes
    finally:
        active[0] = False
    from src.next5.synthetic import load_reference, load_candidate, predict_model, reference_predictions, combine_candidate
    references, _ = load_reference('full', 0)
    model, registration, _ = load_candidate(cid, 'full', 0)
    maximum = 0.
    exact = True
    for (_, historical, online), actual in zip(records, outputs):
        baseline = reference_predictions(references, historical, online)
        expected = combine_candidate(predict_model(model, historical, online), baseline['BL80'], baseline['C205'], registration)
        maximum = max(maximum, float(np.max(np.abs(actual - expected))))
        assert np.isfinite(actual).all() and np.all((actual >= 0) & (actual <= 1))
        exact = exact and np.array_equal(actual, expected)
    assert maximum <= 1e-15
    atomic_json(Path(args.output), {'created_utc': utc_now(), 'status': 'PASS', 'candidate': cid,
        'cold_audit_enabled_before_entry_import': True, 'entry': 'src.next5.readonly_streaming',
        'persistent_filesystem_write_attempts': len(writes), 'suppressed_project_cache_requests': len(suppressed),
        'first_none_before_dataset_access': True, 'current_observation_contract': True,
        'synthetic_fixture_series': len(records), 'synthetic_fixture_points': sum(map(len, outputs)),
        'batch_probability_max_abs_difference': maximum, 'batch_probability_bitwise_equal': bool(exact),
        'prediction_sha256': hashlib.sha256(np.concatenate(outputs).tobytes()).hexdigest(),
        'real_observations_or_labels_read': False, 'new_fits': 0,
        'scope': 'Deterministic unit fixture only; not a new scientific DGP or quality evaluation',
        'source_sha256': digest(Path(__file__))}, immutable=True)
    print('NEXT5 READONLY COLD ENTRY PASS; original batch probabilities matched', flush=True)


if __name__ == '__main__':
    main()
