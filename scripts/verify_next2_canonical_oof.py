"""Every held-out DEV prediction after canonical roundtrip and actual runtime pruning."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys
import json
import hashlib
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.data import prepare_dev
from src.next2.evaluation import load_full_oof
from src.next2.io import model_class
from src.next2.blend import FixedBlendBundle
from src.next2.canonical_model import dumps, loads, source_hash as format_hash
from src.next2.used_feature_runtime import PreparedUsedPredictor, source_hash as used_hash
from src.next2.runtime import prepare, source_hash as compact_hash
from src.next2.session import checkpoint
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    data = prepare_dev(ROOT)
    candidates = ['J102_SERIAL_VARIANCE_512', 'Y001_LONG_VARIANCE_ARCH1',
                  'B104_LONG_VARIANCE_NO_GLR_MEMORY', 'BL2_C205_DEPENDENCE_80']
    ho, oo = data.series['historical_offsets'], data.series['online_offsets']
    for eid in candidates:
        candidate, original, metadata, expected, sources = load_full_oof(data, eid)
        out = ROOT / 'artifacts/next2/canonical_oof' / eid
        out.mkdir(parents=True, exist_ok=True)
        blend = eid.startswith('BL2_')
        identity = {'candidate': eid, 'oof_sources': sources, 'format_sha256': format_hash(),
                    'runtime_sha256': compact_hash() if blend else used_hash(),
                    'script_sha256': sha256(Path(__file__)), 'dev_manifest_sha256': sha256(data.directory/'MANIFEST.json')}
        if (out/'RESULTS.json').exists():
            r = json.loads((out/'RESULTS.json').read_text())
            assert r['status'] == 'PASS' and r['identity'] == identity
            print('NEXT2 CANONICAL OOF RETAINED', eid, flush=True)
            continue
        write_json(out/'POLICY.json', identity)
        engines, canonical_hashes, features = [], {}, []
        generation = 'next2' if blend else 'next'
        for fold in range(5):
            path = ROOT / f'artifacts/{generation}/experiments' / eid / 'full' / f'fold_{fold}.joblib'
            model = FixedBlendBundle.load(path) if blend else model_class(candidate).load(path)
            content = dumps(model)
            restored = loads(content)
            assert dumps(restored) == content
            canonical_hashes[str(fold)] = hashlib.sha256(content).hexdigest()
            engine = prepare(restored) if blend else PreparedUsedPredictor(restored)
            engines.append(engine)
            features.append(len(restored.primary.names)+len(restored.complement.names) if blend else len(engine.names))
        tick = time.perf_counter()
        fingerprint = hashlib.sha256()
        sizes = []
        for j, sid in enumerate(data.ids):
            fold = data.guard.split.assignment(int(sid))[1]
            state = engines[fold].new_state(data.historical[ho[j]:ho[j+1]])
            observed = data.online[oo[j]:oo[j+1]]
            actual = np.asarray([state.predict_one(point) for point in observed], dtype=np.float32)
            np.testing.assert_array_equal(actual, expected[oo[j]:oo[j+1]])
            fingerprint.update(actual.tobytes())
            sizes.append(state.state_array_bytes)
            if (j+1) % 1000 == 0:
                print('NEXT2 CANONICAL OOF EXACT', eid, j+1, round(time.perf_counter()-tick, 1), flush=True)
        assert fingerprint.hexdigest() == hashlib.sha256(expected.tobytes()).hexdigest()
        write_json(out/'RESULTS.json', {'status': 'PASS', 'identity': identity, 'created_utc': utc_now(),
            'series': len(data.ids), 'points': len(expected), 'canonical_fold_sha256': canonical_hashes,
            'materialized_features_per_fold': features, 'state_bytes_min': min(sizes), 'state_bytes_max': max(sizes),
            'prediction_max_abs_diff': 0., 'prediction_fingerprint_sha256': fingerprint.hexdigest(),
            'runtime': 'compact_v1' if blend else 'tree_used_v1', 'elapsed_seconds': time.perf_counter()-tick,
            'training_fits': 0, 'seal_rows': 0, 'all_five_heldout_models_checked': True})
        checkpoint('Canonical full OOF exact ' + eid, ['Isolated runtime profile', 'Report role gates'],
                   artifacts=[(out/'RESULTS.json').relative_to(ROOT).as_posix()])
        print('NEXT2 CANONICAL OOF PASS', eid, flush=True)


if __name__ == '__main__':
    main()
