"""Recompute selected original OOF with the unchanged official scorer.

Only during the additional audit. This does not fit models, read seal rows,
change selection, or evaluate a new dataset.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.data import prepare_dev
from src.next.candidate_io import load_full_oof
from src.next.research_clock import ResearchClock
from src.scoring.ts_auc import ts_auc, score_frames
from scripts.verify_scorer import load_official, official_score
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    clock = ResearchClock(ROOT).snapshot()
    if not clock['minimum_research_met'] or not clock['final_audit_started']:
        raise RuntimeError('This audit requires completed research and explicit final-audit start')
    out = ROOT/'artifacts/next/final_scoring'
    if (out/'RESULTS.json').exists():
        raise RuntimeError('Final selected scoring audit is immutable')
    data = prepare_dev(ROOT)
    selection_path = ROOT/'artifacts/next/final_selection/RESULTS.json'
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    official = load_official()
    rows, files = [], {}
    start = time.perf_counter()
    for selected in selection['candidates']:
        eid = selected['experiment_id']
        candidate, result, metadata, prediction, sources = load_full_oof(data, eid)
        if candidate != selected['candidate']:
            raise RuntimeError('Selected OOF candidate differs from its frozen selection')
        for fold in range(5):
            chosen = np.flatnonzero(metadata['fold'] == fold)
            train_ids = np.unique(metadata['dataset_id'][metadata['fold'] != fold])
            valid_ids = np.unique(metadata['dataset_id'][chosen])
            data.guard.partition(train_ids, valid_ids)
            y, p, t = metadata['target'][chosen], prediction[chosen], metadata['time_online'][chosen]
            ids = metadata['dataset_id'][chosen]
            # Arbitrary monotone nonzero absolute coordinates exercise the
            # official within-series cumcount convention. They are never
            # described as the original absolute timestamps.
            index = pd.MultiIndex.from_arrays([ids, t.astype(np.int64)+10000+3*ids.astype(np.int64)], names=['id', 'time'])
            pred_frame = pd.DataFrame({'prediction': p}, index=index).sort_index()
            target_frame = pd.DataFrame({'target': y}, index=index).sort_index()
            expected = selected['fold_scores'][fold]
            direct = ts_auc(y, p, t)
            canonical = score_frames(pred_frame, target_frame)
            reference = official_score(official, pred_frame, target_frame)
            shuffled = score_frames(pred_frame.sample(frac=1., random_state=202609090+fold), target_frame.sample(frac=1., random_state=202609100+fold))
            maximum = max(abs(value-expected) for value in (direct, canonical, reference, shuffled))
            if maximum > 1e-12:
                raise RuntimeError('Selected official score parity failed')
            rows.append({'candidate': eid, 'role': selected['role'], 'fold': fold, 'points': len(chosen),
                'series': len(valid_ids), 'frozen_score': expected, 'direct_rank_score': direct,
                'canonical_frame_score': canonical, 'official_scorer_score': reference,
                'independently_shuffled_local_score': shuffled, 'maximum_absolute_error': maximum,
                'train_validation_id_overlap': 0, 'train_validation_historical_group_overlap': 0, 'seal_rows': 0})
            for name, digest in sources.items():
                path = ROOT/'artifacts/next/experiments'/eid/'full'/('POLICY.json' if name == 'policy_sha256' else name)
                if sha256(path) != digest:
                    raise RuntimeError('OOF audit payload changed')
                files[path.relative_to(ROOT).as_posix()] = digest
            print('FINAL OFFICIAL OOF SCORE PASS', eid, fold, reference, flush=True)
    data.guard.check_state()
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'SELECTED_OFFICIAL_SCORES.csv', index=False)
    for relative in ('vendor/official/scoring.py', 'src/scoring/ts_auc.py', 'scripts/verify_scorer.py',
            'src/next/candidate_io.py', 'src/next/guard.py', 'src/next/data.py', 'scripts/audit_next_final_scoring.py'):
        files[relative] = sha256(ROOT/relative)
    files[(out/'SELECTED_OFFICIAL_SCORES.csv').relative_to(ROOT).as_posix()] = sha256(out/'SELECTED_OFFICIAL_SCORES.csv')
    write_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'clock_at_start': clock,
        'selected_candidates': len(selection['candidates']), 'selected_original_folds': len(rows),
        'official_code_executed': True, 'scorer_io_only_monkeypatched': True,
        'maximum_absolute_error': max(row['maximum_absolute_error'] for row in rows), 'tolerance': 1e-12,
        'selection_sha256': sha256(selection_path), 'evidence_files_sha256': files,
        'elapsed_seconds': time.perf_counter()-start, 'seal_rows': 0, 'reduced_new_uses': 0,
        'coordinate_scope': 'DEV dataset IDs with synthetic monotone absolute time offsets; exact original online ordinal positions. No new raw observation or label rows are created.'})


if __name__ == '__main__':
    main()
