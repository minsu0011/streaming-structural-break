"""Explicit full-DEV binary fit for reviewed registered sequential banks.

The earlier variance final fitter and its trained artifacts remain untouched.
Training order, weighting, constant removal, and learner export are identical.
"""
from pathlib import Path
import hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from src.next.engine import EngineConfig, feature_hash
from src.next.predictor import NextBundle, predictor_hash, arch_contract
from src.next.extensions import names_and_groups
from src.next.registered_extensions import RegisteredExtendedBundle
from src.next.candidate_io import validate_sources
from src.next.fitting import weights_from_training, fitting_hash
from src.features.config import CONFIG
from src.features.streaming import feature_names, FEATURE_FAMILIES
from src.models.learners import make_estimator
from src.models.compact_trees import export_lightgbm


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+fitting_hash().encode()).hexdigest()


def training_feature_names(candidate):
    validate_sources(candidate)
    expected = {'log_energy': 'log_energy_evidence', 'serial_variance': 'serial_variance_evidence'}
    kind = candidate.get('extension', {}).get('kind')
    if kind not in expected or candidate.get('extension_registration') != {'kind': kind, 'module': expected[kind]}:
        raise ValueError('Only the explicitly reviewed registered binary banks are supported')
    if candidate.get('base') != 'arch8' or candidate.get('base_family_selection') != 'ABCD':
        raise ValueError('The unchanged old ARCH8 51-column ABCD bank is required')
    config = CONFIG.variant(normalization='median_mad', scales=(5, 20, 160))
    old = ['arch8__'+name for name, family in zip(feature_names(config), FEATURE_FAMILIES) if family in 'ABCD']
    names, groups = names_and_groups(candidate['extension'])
    keep = set(candidate.get('keep_names', names))
    if not keep.issubset(names):
        raise ValueError('Unregistered final feature selection')
    new = [name for name, group in zip(names, groups) if name in keep and group in candidate['groups']]
    if not new:
        raise ValueError('A registered sequential bank is required')
    return tuple(old+new)


def fit_all_development(design, metadata, candidate, guard, names):
    names = tuple(names)
    if names != training_feature_names(candidate):
        raise RuntimeError('Registered all-DEV feature identity differs')
    ids = np.asarray(metadata['dataset_id'])
    admitted = guard.admit(np.unique(ids), purpose='registered all-DEV final/refit training')
    if set(admitted) != set(guard.dev_ids):
        raise RuntimeError('Exact complete DEV identity coverage required')
    times, target = np.asarray(metadata['time_online'], dtype=np.int32), np.asarray(metadata['target'], dtype=np.uint8)
    if len(ids) != len(times) or len(ids) != len(target) or len(ids) != len(design) or design.shape[1] != len(names):
        raise ValueError('Registered all-DEV feature/metadata dimensions differ')
    by_id = {sid: guard.split.assignment(int(sid))[1] for sid in admitted}
    fold_code = np.fromiter((by_id[int(sid)] for sid in ids), dtype=np.int8, count=len(ids))
    order = np.lexsort((times, ids, fold_code))
    ordered_ids, ordered_time = ids[order], times[order]
    starts = np.r_[True, ordered_ids[1:] != ordered_ids[:-1]]
    if np.any(ordered_time[starts] != 0) or np.any(np.diff(ordered_time)[~starts[1:]] != 1):
        raise ValueError('Every training series must have unique complete chronological rows')
    x, y = np.asarray(design[order], dtype=np.float32), target[order]
    if not np.isfinite(x).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError('Finite features and both training classes required')
    if candidate['learner'] != 'M4_lightgbm' or candidate['weighting'] not in ('W0', 'W1', 'W2', 'W3') or candidate.get('ranking_objective'):
        raise ValueError('Registered final fitting supports only the frozen binary LightGBM recipes')
    columns = np.flatnonzero(np.ptp(x, axis=0) > 1e-12).astype(np.int32)
    feature_digest = hashlib.sha256(memoryview(x).cast('B')).hexdigest()
    x = x[:, columns]
    weight = weights_from_training(y, ordered_time, ordered_ids, candidate['weighting'])
    estimator = make_estimator(candidate['learner'], candidate['params'])
    with threadpool_limits(limits=4):
        estimator.fit(x, y, sample_weight=weight)
    compact = export_lightgbm(estimator)
    probe = x[:min(len(x), 2048)]
    with threadpool_limits(limits=1):
        native = estimator.booster_.predict(probe, num_threads=1)
    np.testing.assert_allclose(compact.predict(probe), native, rtol=0, atol=1e-12)
    config = EngineConfig(**candidate['engine_config'])
    base = NextBundle(config, names, columns, compact, estimator, candidate['learner'], feature_hash(config), predictor_hash(), arch_contract())
    model = RegisteredExtendedBundle.wrap_registered(base, candidate['extension'], candidate['extension_registration'])
    return model, {'training_series': len(admitted), 'training_rows': len(y), 'training_positive_rows': int(y.sum()),
        'training_order': 'Original fold, then dataset ID, then online time', 'feature_matrix_sha256': feature_digest,
        'ordered_ids_sha256': hashlib.sha256(np.asarray(ordered_ids, dtype=np.int32).tobytes()).hexdigest(),
        'ordered_targets_sha256': hashlib.sha256(y.tobytes()).hexdigest(),
        'training_constant_columns_removed': len(names)-len(columns), 'retained_columns': len(columns), 'weighting': candidate['weighting'],
        'native_compact_max_abs_diff_bound': 1e-12, 'source_sha256': source_hash(),
        'validation_partition': 'None: explicit complete-DEV refit after candidate definition; no fabricated CV score.',
        'seal_rows_fitted': 0, 'reduced_new_usage': 0}
