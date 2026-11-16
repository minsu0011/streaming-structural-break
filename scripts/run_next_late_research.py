"""Score only preregistered late blends after hour eight; never fit components."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
from numba import set_num_threads
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.data import prepare_dev
from src.next.late_evaluation import assert_late_time, lookup, combine_fold, components
from src.next.late_blend import LateBlendBundle
from src.next.paired_blend import SameBankBlendBundle
from src.next.bootstrap import PreparedAUC, paired_summary
from src.next.bootstrap_integrity import load_reference_samples
from src.next.references import verify_reference_sources
from src.next.progress import checkpoint
from src.validation.group_bootstrap import draw_group_multiplicities
from src.validation.research import summarize
from src.scoring.ts_auc import ts_auc
from src.utils.artifacts import write_json, sha256, utc_now
from scripts.run_next_research import fractures, export_tables, screen_reference


def scope_result(data, candidate, scope, policy):
    assert_late_time(ROOT)
    eid = candidate['experiment_id']
    out = ROOT/'artifacts/next/experiments'/eid/scope
    out.mkdir(parents=True, exist_ok=True)
    identity = {'candidate': candidate, 'scope': scope, 'component_training': 'No additional fitting; exact matching frozen component fits only',
        'split_sha256': data.guard.split.digest, 'seal_lock_sha256': data.guard.lock_sha,
        'policy_sha256': sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'),
        'implementation_lock_sha256': sha256(ROOT/'configs/next_late_implementation_lock.json')}
    path = out/'POLICY.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != identity:
        raise RuntimeError('Late scope policy changed')
    if not path.exists():
        write_json(path, identity)
    metadata = data.rows()
    scores, bootstrap_scores, baseline_scores, cv, fracture_rows = [], [], [], [], []
    folds = policy['screen']['folds'] if scope == 'screen' else list(range(5))
    for fold in folds:
        tick = time.perf_counter()
        saved_path = out/f'fold_{fold}.json'
        if saved_path.exists():
            saved = json.loads(saved_path.read_text(encoding='utf-8'))
            if saved['status'] != 'PASS' or any(sha256(out/name) != digest for name, digest in saved['files_sha256'].items()):
                raise RuntimeError('Completed late fold changed')
            if any(sha256(ROOT/name) != digest for name, digest in saved['component_sources'].items()):
                raise RuntimeError('A frozen late component changed')
            prediction = np.load(out/f'fold_{fold}.npy', allow_pickle=False)
            rows = np.load(out/f'fold_{fold}_global_rows.npy', allow_pickle=False)
        else:
            model, prediction, rows, sources = combine_fold(data, candidate, scope, fold)
            expected = metadata['fold'] == fold
            if scope == 'screen':
                expected &= np.isin(metadata['dataset_id'], policy['screen']['ids'])
            np.testing.assert_array_equal(rows, np.flatnonzero(expected))
            model.save(out/f'fold_{fold}.joblib')
            cls = LateBlendBundle if 'late_blend' in candidate else SameBankBlendBundle
            loaded = cls.load(out/f'fold_{fold}.joblib')
            probes = np.unique(metadata['dataset_id'][rows])[:2]
            for sid in probes:
                historical, points, _ = data.get(sid)
                state = loaded.make_state(historical)
                live = np.asarray([loaded.predict_one(state.update(point)) for point in points], dtype=np.float32)
                np.testing.assert_array_equal(live, prediction[metadata['dataset_id'][rows] == sid])
            np.save(out/f'fold_{fold}.npy', prediction, allow_pickle=False)
            np.save(out/f'fold_{fold}_global_rows.npy', rows, allow_pickle=False)
            files = [f'fold_{fold}.joblib', f'fold_{fold}.npy', f'fold_{fold}_global_rows.npy']
            write_json(saved_path, {'status': 'PASS', 'created_utc': utc_now(), 'component_sources': sources,
                'files_sha256': {name: sha256(out/name) for name in files},
                'engineering': {'live_cached_probe_max_abs_diff': 0., 'live_probe_series': len(probes), 'new_fits': 0},
                'seal_rows': 0, 'reduced_new_uses': 0})
        y, t, ids = (metadata[key][rows] for key in ('target', 'time_online', 'dataset_id'))
        score = ts_auc(y, prediction, t)
        scores.append(score)
        reps = policy['bootstrap']['screen_replicates'] if scope == 'screen' else policy['bootstrap']['final_replicates']
        code, counts = draw_group_multiplicities(ids, reps, policy['bootstrap']['seed']+fold)
        samples = PreparedAUC.create(y, prediction, t, code, counts.shape[1]).evaluate(counts)
        np.testing.assert_allclose(samples[0], score, rtol=0, atol=1e-14)
        bootstrap_scores.append(samples)
        if scope == 'full':
            baseline_scores.append(load_reference_samples(ROOT, 'PRIMARY', fold))
            fracture_rows.extend(fractures({key: value[rows] for key, value in metadata.items()}, prediction, data, eid, fold))
        else:
            base = np.load(ROOT/'artifacts/next/screen_reference'/f'fold_{fold}.npy', allow_pickle=False)
            baseline_scores.append(PreparedAUC.create(y, base, t, code, counts.shape[1]).evaluate(counts))
        cv.append({'experiment_id': eid, 'scope': scope, 'fold': fold, 'ts_auc': score,
            'validation_rows': len(rows), 'validation_series': len(np.unique(ids)), 'seal_rows': 0})
        print('LATE FIXED BLEND', eid, scope, fold, score, round(time.perf_counter()-tick, 1), flush=True)
    matrix = np.asarray(bootstrap_scores).T
    np.save(out/'bootstrap_scores.npy', matrix, allow_pickle=False)
    summary = summarize(scores) if scope == 'full' else {'fold_scores': scores, 'mean': float(np.mean(scores)), 'median': float(np.median(scores)), 'worst_fold': min(scores)}
    result = {'status': 'COMPLETE', 'scope': scope, **summary,
        'paired_bootstrap': paired_summary(np.asarray(baseline_scores).T, matrix),
        'cv_rows': cv, 'fractures': fracture_rows, 'new_fits': 0,
        'created_utc': utc_now(), 'seal_rows': 0, 'reduced_new_uses': 0}
    write_json(out/'RESULTS.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', required=True)
    parser.add_argument('--candidate')
    args = parser.parse_args()
    assert_late_time(ROOT)
    set_num_threads(2)
    verify_reference_sources(ROOT)
    data = prepare_dev(ROOT)
    policy = json.loads((ROOT/'ROBUSTNESS_SPLIT_LOCK.json').read_text(encoding='utf-8'))
    requested = json.loads((ROOT/args.queue).read_text(encoding='utf-8'))
    screen_reference(data, policy)
    reference = json.loads((ROOT/'artifacts/next/REFERENCE_RESTORATION.json').read_text(encoding='utf-8'))['references']['PRIMARY']
    for item in requested['candidates']:
        eid = item['experiment_id']
        if args.candidate and args.candidate != eid:
            continue
        candidate, queue, relative = lookup(ROOT, eid)
        if Path(args.queue) != Path(relative) or candidate != item:
            raise RuntimeError('Requested queue does not match frozen late definition')
        out = ROOT/'artifacts/next/experiments'/eid
        if (out/'RESULTS.json').exists():
            saved = json.loads((out/'RESULTS.json').read_text(encoding='utf-8'))
            if saved['candidate'] != candidate:
                raise RuntimeError('Completed late definition changed')
            print('LATE RETAINED', eid, saved['status'], flush=True)
            continue
        start = time.perf_counter()
        screen = scope_result(data, candidate, 'screen', policy)
        result = {'candidate': candidate, 'screen': screen, 'created_utc': utc_now(),
            'status': 'SCREEN_FAIL', 'reason': 'Fixed screen admission failed; no full combined score inspected',
            'feature_hash': sha256(ROOT/'configs/next_late_implementation_lock.json'),
            'time_gate_evidence': assert_late_time(ROOT)}
        if screen['paired_bootstrap']['estimate_mean_delta'] >= policy['screen']['minimum_mean_delta_for_full']:
            full = scope_result(data, candidate, 'full', policy)
            delta = full['mean']-reference['mean']
            result.update(full=full, delta_vs_primary=delta)
            passed = (delta >= policy['promotion']['mean_delta_min'] and full['median'] >= reference['median']
                and full['worst_fold']-reference['worst'] >= policy['promotion']['worst_delta_min']
                and full['paired_bootstrap']['probability_delta_positive'] >= policy['promotion']['bootstrap_probability_min'])
            left, right, _ = components(candidate)
            means = [reference['mean'] if eid2 == 'PRIMARY' else json.loads((ROOT/'artifacts/next/experiments'/eid2/'RESULTS.json').read_text(encoding='utf-8'))['full']['mean'] for eid2 in (left, right)]
            result['marginal_mean_gain_vs_best_component'] = full['mean']-max(means)
            result['marginal_gate_pass'] = result['marginal_mean_gain_vs_best_component'] >= queue.get('marginal_mean_gain_vs_best_component_min', -float('inf'))
            result.update(status='ROBUSTNESS_PENDING' if passed else 'FULL_FAIL',
                reason='Primary gates passed; fixed robustness, marginal-complexity and engineering gates remain' if passed else 'One or more frozen primary gates failed')
        result['elapsed_seconds'] = time.perf_counter()-start
        write_json(out/'RESULTS.json', result)
        export_tables()
        checkpoint(ROOT, eid, [eid], ['late robustness', 'runtime verification'], artifacts=[str((out/'RESULTS.json').relative_to(ROOT))], best=max(reference['mean'], result.get('full', {}).get('mean', 0.)))
        print('LATE RESULT', eid, result['status'], result.get('full', {}).get('mean'), result.get('marginal_gate_pass'), flush=True)


if __name__ == '__main__':
    main()
