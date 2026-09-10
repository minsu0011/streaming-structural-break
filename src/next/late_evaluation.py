"""Audited component-only access for the two frozen late-combination queues.

No model is fitted here. Every real prediction combination is time-gated.
Original single-bank scoring and forensic sources remain unchanged.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.next.research_clock import ResearchClock, ResearchTimeError
from src.next.candidate_io import model_class, validate_sources
from src.next.references import load_reference_fold, REFERENCES
from src.next.late_blend import LateBlendBundle, combine_scores
from src.next.paired_blend import SameBankBlendBundle
from src.models.representations import bundle_class
from src.scoring.ts_auc import ts_auc
from src.utils.artifacts import sha256

QUEUES = ('configs/next_late_combination_candidates.json', 'configs/next_late_pair_candidates.json')


def assert_late_time(root):
    snapshot = ResearchClock(root).snapshot()
    if snapshot['research_elapsed_seconds'] < 28800:
        raise ResearchTimeError('Real combined predictions cannot be formed before research hour eight')
    return snapshot


def implementation_files():
    return ('src/next/late_evaluation.py', 'src/next/late_blend.py', 'src/next/paired_blend.py',
        'src/next/paired_variance_runtime.py', 'scripts/run_next_late_research.py',
        'scripts/robustness_next_late.py', 'scripts/forensic_next_late.py',
        'scripts/forensic_next_candidate.py', 'scripts/run_next_research.py', 'src/next/candidate_io.py',
        'src/next/bootstrap.py', 'src/next/research_clock.py')


def validate_lock(root):
    root = Path(root)
    lock = json.loads((root/'configs/next_late_implementation_lock.json').read_text(encoding='utf-8'))
    if lock['status'] != 'LOCKED':
        raise RuntimeError('Late implementation has not been frozen')
    for relative, digest in lock['files_sha256'].items():
        if sha256(root/relative) != digest:
            raise RuntimeError('Late implementation or queue changed: '+relative)
    return lock


def lookup(root, eid):
    root = Path(root)
    validate_lock(root)
    for relative in QUEUES:
        queue = json.loads((root/relative).read_text(encoding='utf-8'))
        for candidate in queue['candidates']:
            if candidate['experiment_id'] == eid:
                return candidate, queue, relative
    raise KeyError('Candidate is outside the two frozen late grids')


def components(candidate):
    if 'late_blend' in candidate:
        item = candidate['late_blend']
        return item['primary'], item['partner'], item['primary_weight']
    return candidate['left'], candidate['right'], candidate['left_weight']


def _checked(root, path):
    result = json.loads(path.read_text(encoding='utf-8'))
    if result['status'] != 'PASS':
        raise RuntimeError('A frozen component fold did not pass')
    sources = {str(path.relative_to(root)): sha256(path)}
    for name, digest in result['files_sha256'].items():
        item = path.parent/name
        if sha256(item) != digest:
            raise RuntimeError('Frozen component bytes changed: '+str(item))
        sources[str(item.relative_to(root))] = digest
    return result, sources


def component_fold(data, eid, scope, fold, split_name=None):
    """Return model, float32 scores, exact global coordinates, and byte receipts."""
    root = data.guard.root
    metadata = data.rows()
    sources = {}
    if eid == 'PRIMARY' and scope == 'full':
        frame, prediction, model = load_reference_fold(root, data.guard, eid, fold)
        rows = np.flatnonzero(metadata['fold'] == fold)
        for name in ('dataset_id', 'time_online', 'target'):
            np.testing.assert_array_equal(frame[name].to_numpy(), metadata[name][rows])
        oof = root/'data/processed/oof'/REFERENCES[eid]/f'fold_{fold}.npy'
        model_path = root/'artifacts/models/real'/REFERENCES[eid]/f'fold_{fold}.joblib'
        # The original reference bootstrap integrity lock pins these OOF bytes.
        integrity = json.loads((root/'artifacts/next/reference_bootstrap/REPRODUCED_INTEGRITY_LOCK.json').read_text(encoding='utf-8'))
        if sha256(oof) != integrity['reference_oof_sha256'][str(oof.relative_to(root))]:
            raise RuntimeError('Original reference OOF differs from the reproduced integrity lock')
        sources.update({str(oof.relative_to(root)): sha256(oof), str(model_path.relative_to(root)): sha256(model_path)})
    elif eid == 'PRIMARY' and scope == 'screen':
        folder = root/'artifacts/next/screen_reference'
        _, sources = _checked(root, folder/'RESULTS.json')
        frame = pd.read_parquet(folder/f'fold_{fold}_coordinates.parquet')
        policy = json.loads((root/'ROBUSTNESS_SPLIT_LOCK.json').read_text(encoding='utf-8'))
        rows = np.flatnonzero((metadata['fold'] == fold) & np.isin(metadata['dataset_id'], policy['screen']['ids']))
        for name in ('dataset_id', 'time_online', 'target'):
            np.testing.assert_array_equal(frame[name].to_numpy(), metadata[name][rows])
        model = bundle_class('cross_order_ar4_ar8_equal').load(folder/f'fold_{fold}.joblib')
        prediction = np.load(folder/f'fold_{fold}.npy', allow_pickle=False)
    else:
        if scope == 'stress':
            kind = 'reference' if eid == 'PRIMARY' else 'candidates'
            folder = root/'artifacts/next/robustness'/kind/eid/split_name
        else:
            folder = root/'artifacts/next/experiments'/eid/scope
        _, sources = _checked(root, folder/f'fold_{fold}.json')
        prediction = np.load(folder/f'fold_{fold}.npy', allow_pickle=False)
        row_path = folder/f'fold_{fold}_global_rows.npy'
        if row_path.exists():
            rows = np.load(row_path, allow_pickle=False)
        elif eid == 'PRIMARY' and scope == 'stress':
            # Reference stress coordinates preserve the same original-fold order.
            frame = pd.read_parquet(folder/f'fold_{fold}_coordinates.parquet')
            ordered = np.concatenate([np.flatnonzero(metadata['fold'] == k) for k in range(5)])
            rows = ordered[np.isin(metadata['dataset_id'][ordered], frame.dataset_id.unique())]
        else:
            raise RuntimeError('Component global-row coordinates are missing')
        if eid == 'PRIMARY':
            model = bundle_class('cross_order_ar4_ar8_equal').load(folder/f'fold_{fold}.joblib')
            frame = pd.read_parquet(folder/f'fold_{fold}_coordinates.parquet')
            for name in ('dataset_id', 'time_online', 'target'):
                np.testing.assert_array_equal(frame[name].to_numpy(), metadata[name][rows])
        else:
            report_path = root/'artifacts/next/experiments'/eid/'RESULTS.json'
            report = json.loads(report_path.read_text(encoding='utf-8'))
            if report['status'] == 'BUG' or report.get('full', {}).get('status') != 'COMPLETE':
                raise RuntimeError('Late partner lacks valid completed full OOF')
            validate_sources(report['candidate'])
            model = model_class(report['candidate']).load(folder/f'fold_{fold}.joblib')
            sources[str(report_path.relative_to(root))] = sha256(report_path)
    if prediction.dtype != np.float32 or prediction.shape != (len(rows),) or not np.isfinite(prediction).all():
        raise RuntimeError('Invalid component prediction array')
    if len(np.unique(rows)) != len(rows) or (rows < 0).any() or (rows >= len(metadata['target'])).any():
        raise RuntimeError('Invalid or repeated component coordinates')
    data.guard.admit(metadata['dataset_id'][rows], purpose='late fixed component predictions')
    return model, prediction, rows, sources


def combine_fold(data, candidate, scope, fold, split_name=None):
    assert_late_time(data.guard.root)
    left_id, right_id, weight = components(candidate)
    left, a, rows, sources = component_fold(data, left_id, scope, fold, split_name)
    right, b, right_rows, right_sources = component_fold(data, right_id, scope, fold, split_name)
    np.testing.assert_array_equal(rows, right_rows)
    sources.update(right_sources)
    if left_id == 'PRIMARY':
        from src.next.late_blend import source_hash
        model = LateBlendBundle(left, right, weight, source_hash())
        model.validate()
    else:
        model = SameBankBlendBundle.create(left, right, weight)
    prediction = combine_scores(a, b, weight)
    return model, prediction, rows, sources


def load_full_oof(data, eid):
    """Explicit adapter for the frozen forensic engine; all sources are pinned."""
    root = data.guard.root
    assert_late_time(root)
    candidate, queue, relative = lookup(root, eid)
    folder = root/'artifacts/next/experiments'/eid
    original = json.loads((folder/'RESULTS.json').read_text(encoding='utf-8'))
    if original['candidate'] != candidate or original.get('full', {}).get('status') != 'COMPLETE' or original['status'] == 'BUG':
        raise RuntimeError('Late candidate needs valid completed five-fold OOF')
    metadata = data.rows()
    full = np.full(len(metadata['target']), np.nan, dtype=np.float32)
    sources = {'late_implementation_lock_sha256': sha256(root/'configs/next_late_implementation_lock.json'),
        'late_queue_sha256': sha256(root/relative), 'late_io_source_sha256': sha256(Path(__file__)),
        'late_forensic_adapter_sha256': sha256(root/'scripts/forensic_next_late.py')}
    for fold in range(5):
        record, digests = _checked(root, folder/'full'/f'fold_{fold}.json')
        for path, digest in record['component_sources'].items():
            if sha256(root/path) != digest:
                raise RuntimeError('Late component provenance changed')
        rows = np.load(folder/'full'/f'fold_{fold}_global_rows.npy', allow_pickle=False)
        np.testing.assert_array_equal(rows, np.flatnonzero(metadata['fold'] == fold))
        values = np.load(folder/'full'/f'fold_{fold}.npy', allow_pickle=False)
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise RuntimeError('Invalid late OOF values')
        np.testing.assert_allclose(ts_auc(metadata['target'][rows], values, metadata['time_online'][rows]), original['full']['fold_scores'][fold], rtol=0, atol=1e-14)
        full[rows] = values
        sources.update(digests)
    if not np.isfinite(full).all():
        raise RuntimeError('Incomplete late OOF')
    return candidate, original, metadata, full, sources
