"""Check all selected features and outputs of the shared exact blend runtime."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys
import json
import time
import hashlib
import argparse
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next.data import prepare_dev
from src.next2.evaluation import load_full_oof
from src.next2.canonical_model import load, dumps, loads
from src.next2.io import model_class
from src.next2.blend import FixedBlendBundle, source_hash as blend_hash
from src.next2.runtime import prepare
from src.next3.precomputed_shared_runtime import PreparedPrecomputedSharedPredictor as PreparedSharedBlendPredictor, source_hash
from src.next2.shared_blend_runtime import PreparedSharedBlendPredictor as FrozenSharedPredictor
from src.next3.runtime_memory import accounting
from src.next3.artifacts import progress
def checkpoint(task,details):progress(task,"; ".join(details))
from src.next3.artifacts import atomic_json as write_json, digest as sha256, utc_now, BASE, read_json
J = 'J102_SERIAL_VARIANCE_512'; C = 'C205_DEPENDENCE'; B = 'BL2_C205_DEPENDENCE_80'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', choices=['all_dev', 'oof'], required=True); args = parser.parse_args()
    hypothesis=read_json(BASE/'hypotheses/E001_PRECOMPUTED_SHARED.json')
    assert source_hash()==hypothesis['runtime_sha256']
    for name,expected_source in hypothesis['source_sha256'].items():assert sha256(ROOT/name)==expected_source
    data = prepare_dev(ROOT); out = ROOT/'artifacts/next3/engineering/precomputed_shared'/args.stage; out.mkdir(parents=True, exist_ok=True)
    identity = {'stage': args.stage, 'runtime_sha256': source_hash(), 'script_sha256': sha256(Path(__file__)),
        'protocol_sha256': sha256(ROOT/'docs/NEXT3_PRECOMPUTED_SHARED_PROTOCOL.md'),
        'dev_manifest_sha256': sha256(data.directory/'MANIFEST.json')}
    expected = None; sources = {}; models = []
    if args.stage == 'all_dev':
        path = ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_compact_v1/model.sbnext2'
        models.append(load(path)); sources[path.relative_to(ROOT).as_posix()] = sha256(path)
        receipt = ROOT/'artifacts/next2/canonical_api_all_dev/PERFORMANCE/RESULTS.json'
        previous = json.loads(receipt.read_text()); assert previous['status'] == 'PASS'
        expected_hash = previous['output_fingerprint_sha256']; sources[receipt.relative_to(ROOT).as_posix()] = sha256(receipt)
    else:
        jc, _, _, _, j_sources = load_full_oof(data, J)
        cc, _, _, _, c_sources = load_full_oof(data, C)
        _, _, _, expected, b_sources = load_full_oof(data, B)
        sources = {J: j_sources, C: c_sources, B: b_sources}
        expected_hash = hashlib.sha256(expected.tobytes()).hexdigest()
        for fold in range(5):
            jp = ROOT/'artifacts/next/experiments'/J/'full'/f'fold_{fold}.joblib'
            cp = ROOT/'artifacts/next2/experiments'/C/'full'/f'fold_{fold}.joblib'
            model = FixedBlendBundle(model_class(jc).load(jp), model_class(cc).load(cp), .8, blend_hash())
            content = dumps(model); restored = loads(content); assert dumps(restored) == content
            models.append(restored)
    identity['sources'] = sources; identity['expected_prediction_fingerprint'] = expected_hash
    if (out/'POLICY.json').exists(): assert json.loads((out/'POLICY.json').read_text()) == identity
    else: write_json(out/'POLICY.json', identity)
    if (out/'RESULTS.json').exists():
        r = json.loads((out/'RESULTS.json').read_text()); assert r['status'] == 'PASS' and r['identity'] == identity
        print('NEXT3 PRECOMPUTED SHARED PROOF RETAINED', args.stage, flush=True); return
    engines = [PreparedSharedBlendPredictor(m) for m in models]
    references = [FrozenSharedPredictor(m) for m in models]
    ho, oo = data.series['historical_offsets'], data.series['online_offsets']
    began = time.perf_counter(); digest = hashlib.sha256(); features = 0; sizes = []; probes = []
    for j, sid in enumerate(data.ids):
        fold = 0 if args.stage == 'all_dev' else data.guard.split.assignment(int(sid))[1]
        h = data.historical[ho[j]:ho[j+1]]; o = data.online[oo[j]:oo[j+1]]
        left = references[fold].new_state(h); right = engines[fold].new_state(h)
        actual = np.empty(len(o), dtype=np.float32)
        used = engines[fold].primary.used
        for k, point in enumerate(o):
            a, b = left.predict_one(point), right.predict_one(point)
            j_exact = np.array_equal(left.output.view(np.uint32), right.base.output.view(np.uint32))
            c_exact = np.array_equal(left.shared.dependence_output.view(np.uint32), right.base.shared.dependence_output.view(np.uint32))
            if a != b or not j_exact or not c_exact:
                write_json(out/'FAILURE.json', {'status': 'FAIL', 'identity': identity, 'dataset_id': int(sid),
                    'online_index': k, 'prediction_exact': a == b, 'primary_features_exact': j_exact, 'complement_features_exact': c_exact})
                raise AssertionError('Precomputed shared changed selected feature or prediction')
            actual[k] = b
        if expected is not None: np.testing.assert_array_equal(actual, expected[oo[j]:oo[j+1]])
        digest.update(actual.tobytes()); features += len(o)*(len(used)+24); sizes.append(right.state_array_bytes)
        if j in (0, len(data.ids)//2, len(data.ids)-1):
            mem, _, _ = accounting(right); probes.append({'dataset_id': int(sid), 'historical_points': len(h), **mem})
        if (j+1) % 500 == 0:
            checkpoint('NEXT3 precomputed shared exact '+args.stage,[str(j+1)+' DEV series verified'])
            print('NEXT3 PRECOMPUTED SHARED EXACT', args.stage, j+1, round(time.perf_counter()-began, 1), flush=True)
    assert digest.hexdigest() == expected_hash
    data.guard.check_state()
    write_json(out/'RESULTS.json', {'status': 'PASS', 'identity': identity, 'created_utc': utc_now(),
        'series': len(data.ids), 'points': len(data.online), 'selected_feature_entries_compared': features,
        'prediction_max_abs_diff': 0., 'selected_feature_max_abs_diff': 0., 'prediction_fingerprint_sha256': digest.hexdigest(),
        'materialized_tree_inputs': [len(e.primary.names)+24 for e in engines],
        'state_array_bytes_min': min(sizes), 'state_array_bytes_max': max(sizes), 'numeric_memory_probes': probes,
        'elapsed_seconds': time.perf_counter()-began, 'new_fits': 0, 'seal_rows': 0,
        'in_sample_quality_score_computed': False, 'heldout_oof_checked': expected is not None})
    checkpoint('Shared exact blend runtime '+args.stage, ['Complete alternate/H proof', 'One isolated registered benchmark'])


if __name__ == '__main__': main()
