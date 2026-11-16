"""Fit 60 unique three-fold models; select using four inner folds per outer fold."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
from pathlib import Path
import gc
import json
import sys
import time
import numpy as np
import pandas as pd
from numba import set_num_threads
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.data import prepare_dev
from src.next.diagnostic_training import DiagnosticTraining
from src.next.nested_menu import unique_training_tasks, select_inner_candidate
from src.next.bootstrap import PreparedAUC, paired_summary
from src.next.bootstrap_integrity import load_reference_samples
from src.next.references import load_reference_fold, verify_reference_sources
from src.next.candidate_io import load_full_oof
from src.next.progress import checkpoint
from src.validation.group_bootstrap import draw_group_multiplicities
from src.validation.research import summarize
from src.scoring.ts_auc import ts_auc
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    started = time.perf_counter()
    policy_path = ROOT/'configs/next_nested_menu_policy.json'
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    for relative, digest in policy['files_sha256'].items():
        if sha256(ROOT/relative) != digest:
            raise RuntimeError('Nested-menu implementation changed: '+relative)
    for eid, digest in policy['candidate_reports_sha256'].items():
        if sha256(ROOT/'artifacts/next/experiments'/eid/'RESULTS.json') != digest:
            raise RuntimeError('Nested-menu candidate definition changed')
    data = prepare_dev(ROOT)
    verify_reference_sources(ROOT)
    if data.guard.split.digest != policy['split_sha256'] or data.guard.lock_sha != policy['seal_lock_sha256']:
        raise RuntimeError('Nested-menu split or seal identity changed')
    out = ROOT/'artifacts/next/nested_menu_cv'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'RESULTS.json').exists():
        result = json.loads((out/'RESULTS.json').read_text(encoding='utf-8'))
        if any(sha256(out/name) != digest for name, digest in result['files_sha256'].items()):
            raise RuntimeError('Completed nested-menu evidence changed')
        print('NESTED MENU retained PASS')
        return
    cells = []
    for eid in policy['candidates']:
        study = DiagnosticTraining(data, eid)
        folder = out/eid
        folder.mkdir(exist_ok=True)
        identity = {'candidate': eid, 'policy_sha256': sha256(policy_path), 'sources': study.sources}
        if (folder/'IDENTITY.json').exists() and json.loads((folder/'IDENTITY.json').read_text(encoding='utf-8')) != identity:
            raise RuntimeError('Nested-menu feature/OOF sources changed')
        if not (folder/'IDENTITY.json').exists():
            write_json(folder/'IDENTITY.json', identity)
        metadata = study.metadata
        for task in unique_training_tasks():
            left, right = task['held_out_folds']
            pair = folder/f'exclude_{left}_{right}'
            pair.mkdir(exist_ok=True)
            path = pair/'RESULTS.json'
            if path.exists():
                record = json.loads(path.read_text(encoding='utf-8'))
                if record['status'] != 'PASS' or any(sha256(pair/name) != digest for name, digest in record['files_sha256'].items()):
                    raise RuntimeError('Completed nested training task changed')
                cells.extend(record['held_out_cells'])
                continue
            tick = time.perf_counter()
            train = np.concatenate([np.flatnonzero(metadata['fold'] == k) for k in task['training_folds']])
            valid = np.concatenate([np.flatnonzero(metadata['fold'] == k) for k in (left, right)])
            model, prediction, detail = study.fit(train, valid)
            parity = study.validate_saved_predictions(model, pair/'model.joblib', valid, prediction)
            # Verify an entire series from the second excluded fold as well.
            second = valid[metadata['fold'][valid] == right][0]
            sid = int(metadata['dataset_id'][second])
            historical, points, _ = data.get(sid)
            loaded = study.load(pair/'model.joblib')
            state = loaded.make_state(historical)
            update = state.update if study.candidate is not None else state.update_and_get
            live = np.asarray([loaded.predict_one(update(point)) for point in points], dtype=np.float32)
            np.testing.assert_array_equal(live, prediction[metadata['dataset_id'][valid] == sid])
            np.save(pair/'prediction.npy', prediction, allow_pickle=False)
            np.save(pair/'global_rows.npy', valid, allow_pickle=False)
            pair_cells = []
            for held in (left, right):
                mask = metadata['fold'][valid] == held
                score = ts_auc(metadata['target'][valid][mask], prediction[mask], metadata['time_online'][valid][mask])
                pair_cells.append({'candidate': eid, 'outer_fold': right if held == left else left,
                    'inner_validation_fold': held, 'ts_auc': score, 'training_folds': task['training_folds'],
                    'training_series': len(np.unique(metadata['dataset_id'][train])),
                    'inner_validation_series': len(np.unique(metadata['dataset_id'][valid][mask]))})
            names = ['model.joblib', 'prediction.npy', 'global_rows.npy']
            record = {'status': 'PASS', 'created_utc': utc_now(), 'candidate': eid, 'task': task,
                'held_out_cells': pair_cells, 'engineering': detail, **parity,
                'both_excluded_fold_live_probes_equal': True, 'elapsed_seconds': time.perf_counter()-tick,
                'files_sha256': {name: sha256(pair/name) for name in names}, 'seal_rows': 0, 'reduced_new_uses': 0}
            write_json(path, record)
            cells.extend(pair_cells)
            print('NESTED MENU FIT', eid, left, right, [cell['ts_auc'] for cell in pair_cells], round(record['elapsed_seconds'], 1), flush=True)
            del model, loaded, prediction
            gc.collect()
        del study
        gc.collect()
        pd.DataFrame(cells).to_csv(out/'INNER_CELLS.csv', index=False)
        checkpoint(ROOT, eid+'/nested-menu', [eid+' ten unique nested fits'], ['remaining nested menu', 'finalist engineering'],
            artifacts=[str((out/'INNER_CELLS.csv').relative_to(ROOT))], best=.611292686371357)
    selection = []
    for outer in range(5):
        scores = {eid: [cell['ts_auc'] for cell in sorted(cells, key=lambda cell: cell['inner_validation_fold']) if cell['candidate'] == eid and cell['outer_fold'] == outer] for eid in policy['candidates']}
        winner = select_inner_candidate(scores)
        selection.append({'outer_fold': outer, 'selected_candidate': winner, 'inner_scores': scores,
            'inner_mean_scores': {eid: float(np.mean(values)) for eid, values in scores.items()}})
    # Freeze all five choices before accessing selected outer OOF values here.
    selection_path = out/'INNER_SELECTIONS.json'
    if selection_path.exists() and json.loads(selection_path.read_text(encoding='utf-8'))['selections'] != selection:
        raise RuntimeError('A reproduced inner selection changed')
    if not selection_path.exists():
        write_json(selection_path, {'status': 'LOCKED', 'created_utc': utc_now(), 'selections': selection,
            'policy_sha256': sha256(policy_path), 'outer_scores_used_by_choice': False})
    bootstrap_policy = json.loads((ROOT/'ROBUSTNESS_SPLIT_LOCK.json').read_text(encoding='utf-8'))['bootstrap']
    metadata = data.rows()
    outer_rows, samples, reference = [], [], []
    set_num_threads(2)
    for choice in selection:
        fold, eid = choice['outer_fold'], choice['selected_candidate']
        rows = np.flatnonzero(metadata['fold'] == fold)
        if eid in ('PRIMARY', 'HIGHEST_MEAN'):
            coords, prediction, _ = load_reference_fold(ROOT, data.guard, eid, fold)
            for key in ('dataset_id', 'time_online', 'target'):
                np.testing.assert_array_equal(coords[key].to_numpy(), metadata[key][rows])
        else:
            _, _, _, all_prediction, _ = load_full_oof(data, eid)
            prediction = all_prediction[rows]
        ids, target, times = (metadata[key][rows] for key in ('dataset_id', 'target', 'time_online'))
        code, counts = draw_group_multiplicities(ids, bootstrap_policy['final_replicates'], bootstrap_policy['seed']+fold)
        values = PreparedAUC.create(target, prediction, times, code, counts.shape[1]).evaluate(counts)
        np.testing.assert_allclose(values[0], ts_auc(target, prediction, times), rtol=0, atol=1e-14)
        samples.append(values)
        reference.append(load_reference_samples(ROOT, 'PRIMARY', fold))
        outer_rows.append({'outer_fold': fold, 'selected_candidate': eid, 'outer_ts_auc': values[0]})
    pd.DataFrame(outer_rows).to_csv(out/'OUTER_DIAGNOSTIC.csv', index=False)
    np.save(out/'selected_outer_bootstrap.npy', np.asarray(samples).T, allow_pickle=False)
    names = ['INNER_CELLS.csv', 'INNER_SELECTIONS.json', 'OUTER_DIAGNOSTIC.csv', 'selected_outer_bootstrap.npy']
    write_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'unique_new_fits': 60,
        'inner_held_out_score_cells': len(cells), 'selected_outer': outer_rows,
        'secondary_outer_summary': summarize([row['outer_ts_auc'] for row in outer_rows]),
        'conditional_selected_outer_bootstrap': paired_summary(np.asarray(reference).T, np.asarray(samples).T),
        'policy_sha256': sha256(policy_path), 'elapsed_seconds': time.perf_counter()-started,
        'files_sha256': {name: sha256(out/name) for name in names}, 'seal_rows': 0, 'reduced_new_uses': 0,
        'limitation': policy['nomination_limitation']+' The reported bootstrap conditions on the fixed inner choices and fitted OOF; it does not incorporate selection or fitting uncertainty.',
        'primary_cv_authority_changed': False, 'new_model_candidate_created': False})
    print('NESTED MENU COMPLETE', outer_rows, flush=True)


if __name__ == '__main__':
    main()
