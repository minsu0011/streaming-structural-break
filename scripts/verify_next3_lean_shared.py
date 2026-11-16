"""Actual streaming equivalence for the locked lean variant on DEV only."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys
import argparse
import hashlib
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, read_json, atomic_json, digest, utc_now, progress
from src.next3.lean_shared_runtime import PreparedLeanSharedPredictor, source_hash, accounting
from src.next3.lean_model import loads as lean_loads, feature_contract
VARIANTS = ('LEAN_PACKED_SHARED',)


def PreparedPackedSharedPredictor(model, *, variant):
    if variant != 'LEAN_PACKED_SHARED': raise ValueError('One fixed E003 variant')
    contract = feature_contract()
    return PreparedLeanSharedPredictor(lean_loads(dumps(model), contract=contract), contract=contract)

from src.next2.shared_blend_runtime import PreparedSharedBlendPredictor
from src.next2.blend import FixedBlendBundle, source_hash as blend_hash
from src.next2.canonical_model import load, dumps, loads
from src.next2.evaluation import load_full_oof
from src.next2.io import model_class
from src.next.data import prepare_dev

J = 'J102_SERIAL_VARIANCE_512'; C = 'C205_DEPENDENCE'; B = 'BL2_C205_DEPENDENCE_80'
OUT = BASE/'engineering/lean_shared'


def checked_implementation():
    path = OUT/'IMPLEMENTATION_LOCK_CORRECTION_001.json'; lock = read_json(path)
    assert lock['runtime_sha256'] == source_hash()
    for name, sha in lock['source_sha256'].items():
        assert digest(ROOT/name) == sha, name
    return digest(path)


def checked_model(folder, name, candidate, sources):
    receipt = folder/(name+'.json'); record = read_json(receipt)
    assert record['status'] == 'PASS'
    path = folder/(name+'.joblib')
    assert digest(path) == record['files_sha256'][path.name]
    sources[path.relative_to(ROOT).as_posix()] = digest(path)
    sources[receipt.relative_to(ROOT).as_posix()] = digest(receipt)
    return model_class(candidate).load(path)


def original(data, stage, lock_sha):
    out = OUT/stage; out.mkdir(parents=True, exist_ok=True)
    identity = {'stage': stage, 'runtime_sha256': source_hash(), 'implementation_lock_sha256': lock_sha,
                'dev_manifest_sha256': digest(data.directory/'MANIFEST.json'), 'variants': list(VARIANTS)}
    if (out/'RESULTS.json').exists():
        report = read_json(out/'RESULTS.json')
        assert report['status'] == 'PASS' and report['identity'] == identity
        return
    models = []; sources = {}; expected = None
    if stage == 'all_dev':
        path = ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1/model.sbnext2'
        models.append(load(path)); sources[path.relative_to(ROOT).as_posix()] = digest(path)
        receipt = ROOT/'artifacts/next2/canonical_api_all_dev/PERFORMANCE/RESULTS.json'
        frozen = read_json(receipt); assert frozen['status'] == 'PASS'
        expected_hash = frozen['output_fingerprint_sha256']; sources[receipt.relative_to(ROOT).as_posix()] = digest(receipt)
    else:
        jc, _, _, _, js = load_full_oof(data, J)
        cc, _, _, _, cs = load_full_oof(data, C)
        _, _, _, expected, bs = load_full_oof(data, B)
        sources = {J: js, C: cs, B: bs}; expected_hash = hashlib.sha256(expected.tobytes()).hexdigest()
        for fold in range(5):
            jp = ROOT/'artifacts/next/experiments'/J/'full'/f'fold_{fold}.joblib'
            cp = ROOT/'artifacts/next2/experiments'/C/'full'/f'fold_{fold}.joblib'
            model = FixedBlendBundle(model_class(jc).load(jp), model_class(cc).load(cp), .8, blend_hash())
            serialized = dumps(model); restored = loads(serialized); assert dumps(restored) == serialized
            models.append(restored)
    policy = {'identity': identity, 'expected_fingerprint': expected_hash, 'sources': sources}
    if (out/'POLICY.json').exists(): assert read_json(out/'POLICY.json') == policy
    else: atomic_json(out/'POLICY.json', policy, immutable=True)
    engines = {v: [PreparedPackedSharedPredictor(m, variant=v) for m in models] for v in VARIANTS}
    references = [PreparedSharedBlendPredictor(m) for m in models]
    ho, oo = data.series['historical_offsets'], data.series['online_offsets']
    hashes = {v: hashlib.sha256() for v in VARIANTS}; sizes = {v: [] for v in VARIANTS}
    probes = {v: [] for v in VARIANTS}; began = time.perf_counter(); compared = 0
    for j, sid in enumerate(data.ids):
        fold = 0 if stage == 'all_dev' else data.guard.split.assignment(int(sid))[1]
        h = data.historical[ho[j]:ho[j+1]]; o = data.online[oo[j]:oo[j+1]]
        left = references[fold].new_state(h)
        states = {v: engines[v][fold].new_state(h) for v in VARIANTS}
        actual = {v: np.empty(len(o), dtype=np.float32) for v in VARIANTS}
        before = {v: state.state_array_bytes for v, state in states.items()}
        for k, point in enumerate(o):
            a = left.predict_one(point)
            for variant, state in states.items():
                b = state.predict_one(point)
                j_equal = np.array_equal(left.output.view(np.uint32), state.output.view(np.uint32))
                c_equal = np.array_equal(left.shared.dependence_output.view(np.uint32), state.shared.dependence_output.view(np.uint32))
                if a != b or not j_equal or not c_equal:
                    atomic_json(out/'FAILURE.json', {'status': 'FAIL', 'identity': identity, 'variant': variant,
                        'dataset_id': int(sid), 'online_index': k, 'prediction_exact': a == b,
                        'primary_features_exact': j_equal, 'complement_features_exact': c_equal}, immutable=True)
                    raise AssertionError('Packed prediction or live feature parity failed')
                actual[variant][k] = b
        for variant, state in states.items():
            if expected is not None: np.testing.assert_array_equal(actual[variant], expected[oo[j]:oo[j+1]])
            hashes[variant].update(actual[variant].tobytes()); sizes[variant].append(state.state_array_bytes)
            assert before[variant] == state.state_array_bytes
            if j in (0, len(data.ids)//2, len(data.ids)-1):
                mem, _, _ = accounting(state)
                probes[variant].append({'historical_points': len(h), **mem})
        compared += len(o)*(len(left.output)+24)
        if (j+1) % 500 == 0:
            progress('Packed shared exact '+stage, f'{j+1} DEV series, one fixed lean variant', experiment='E003_LEAN_SHARED')
            print('NEXT3 LEAN EXACT', stage, j+1, round(time.perf_counter()-began, 1), flush=True)
    data.guard.check_state()
    records = []
    for variant in VARIANTS:
        assert hashes[variant].hexdigest() == expected_hash
        records.append({'variant': variant, 'series': len(data.ids), 'points': len(data.online),
            'prediction_fingerprint_sha256': hashes[variant].hexdigest(), 'selected_feature_entries_compared': compared,
            'prediction_max_abs_diff': 0., 'selected_feature_max_abs_diff': 0.,
            'state_bytes_min': min(sizes[variant]), 'state_bytes_max': max(sizes[variant]),
            'numeric_memory_probes': probes[variant]})
    atomic_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'identity': identity,
        'variants': records, 'elapsed_seconds': time.perf_counter()-began, 'new_fits': 0, 'seal_rows': 0,
        'in_sample_quality_score_computed': False, 'heldout_oof_checked': expected is not None,
        'policy_sha256': digest(out/'POLICY.json')}, immutable=True)


def alternate(data, lock_sha):
    out = OUT/'alternate'; out.mkdir(parents=True, exist_ok=True)
    splits = ('ALT_1', 'ALT_2', 'ALT_3', 'LEAVE_ONE_H_REGIME_OUT')
    identity = {'stage': 'alternate', 'runtime_sha256': source_hash(), 'implementation_lock_sha256': lock_sha,
                'dev_manifest_sha256': digest(data.directory/'MANIFEST.json'), 'variants': list(VARIANTS), 'splits': list(splits)}
    if (out/'RESULTS.json').exists():
        r = read_json(out/'RESULTS.json'); assert r['status'] == 'PASS' and r['identity'] == identity
        return
    meta = data.rows(); ho, oo = data.series['historical_offsets'], data.series['online_offsets']
    jc = read_json(ROOT/'artifacts/next/experiments'/J/'RESULTS.json')['candidate']
    cc = read_json(ROOT/'artifacts/next2/experiments'/C/'RESULTS.json')['candidate']
    index = {int(sid): j for j, sid in enumerate(data.ids)}; records = []; began = time.perf_counter()
    for split in splits:
        for fold in range(5):
            name = f'fold_{fold}'; folder = out/split; folder.mkdir(parents=True, exist_ok=True); sources = {}
            left = ROOT/'artifacts/next/robustness/candidates'/J/split
            right = ROOT/'artifacts/next2/robustness'/C/split
            expected_folder = ROOT/'artifacts/next2/robustness'/B/split
            model = FixedBlendBundle(checked_model(left, name, jc, sources),
                                     checked_model(right, name, cc, sources), .8, blend_hash())
            expected_receipt = expected_folder/(name+'.json'); r = read_json(expected_receipt); assert r['status'] == 'PASS'
            sources[expected_receipt.relative_to(ROOT).as_posix()] = digest(expected_receipt)
            for relative, sha in r.get('component_sources_sha256', {}).items(): assert digest(ROOT/relative) == sha
            for suffix in ('.npy', '_global_rows.npy'):
                path = expected_folder/(name+suffix); assert digest(path) == r['files_sha256'][path.name]
                sources[path.relative_to(ROOT).as_posix()] = digest(path)
            expected = np.load(expected_folder/(name+'.npy'), allow_pickle=False)
            coordinates = np.load(expected_folder/(name+'_global_rows.npy'), allow_pickle=False)
            row_ids = meta['dataset_id'][coordinates]
            starts = np.r_[0, np.flatnonzero(row_ids[1:] != row_ids[:-1])+1]; ids = row_ids[starts]
            assert len(ids) == len(np.unique(ids)); data.guard.admit(ids, purpose='Exact packed held-out ALT/H parity')
            serialized = dumps(model); model = loads(serialized); assert dumps(model) == serialized
            key = {**identity, 'split': split, 'fold': fold, 'source_sha256': sources,
                   'canonical_model_sha256': hashlib.sha256(serialized).hexdigest()}
            receipt = folder/(name+'.json')
            if receipt.exists():
                previous = read_json(receipt); assert previous['status'] == 'PASS' and previous['identity'] == key
                records.append(previous); continue
            engines = {v: PreparedPackedSharedPredictor(model, variant=v) for v in VARIANTS}
            hashes = {v: hashlib.sha256() for v in VARIANTS}; cursor = 0
            for sid in ids:
                j = index[int(sid)]; h = data.historical[ho[j]:ho[j+1]]; o = data.online[oo[j]:oo[j+1]]
                np.testing.assert_array_equal(coordinates[cursor:cursor+len(o)], np.arange(oo[j], oo[j+1]))
                for variant, engine in engines.items():
                    state = engine.new_state(h); before = state.state_array_bytes
                    p = np.asarray([state.predict_one(v) for v in o], dtype=np.float32)
                    np.testing.assert_array_equal(p, expected[cursor:cursor+len(o)])
                    assert state.state_array_bytes == before; hashes[variant].update(p.tobytes())
                cursor += len(o)
            expected_sha = hashlib.sha256(expected.tobytes()).hexdigest()
            assert cursor == len(expected) and all(h.hexdigest() == expected_sha for h in hashes.values())
            report = {'status': 'PASS', 'identity': key, 'created_utc': utc_now(), 'series': len(ids), 'points_per_variant': cursor,
                      'prediction_max_abs_diff': 0., 'prediction_fingerprint_sha256': expected_sha, 'new_fits': 0, 'seal_rows': 0}
            atomic_json(receipt, report, immutable=True); records.append(report)
            progress('Packed shared ALT/H exact', f'{split} fold {fold}, one fixed lean variant', experiment='E003_LEAN_SHARED')
            print('NEXT3 LEAN ALTERNATE', split, fold, round(time.perf_counter()-began, 1), flush=True)
    assert len(records) == 20 and sum(r['points_per_variant'] for r in records) == 4*len(data.online)
    data.guard.check_state()
    atomic_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'identity': identity,
        'model_fits_checked_per_variant': 20, 'series_replays_per_variant': sum(r['series'] for r in records),
        'points_per_variant': sum(r['points_per_variant'] for r in records), 'prediction_max_abs_diff': 0.,
        'elapsed_seconds': time.perf_counter()-began, 'new_fits': 0, 'seal_rows': 0,
        'receipts_sha256': {p.relative_to(out).as_posix(): digest(p) for p in out.glob('*/fold_?.json')}}, immutable=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', choices=['all_dev', 'oof', 'alternate'], required=True)
    args = parser.parse_args(); lock_sha = checked_implementation(); data = prepare_dev(ROOT)
    if args.stage == 'alternate': alternate(data, lock_sha)
    else: original(data, args.stage, lock_sha)


if __name__ == '__main__': main()
