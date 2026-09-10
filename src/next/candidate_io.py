"""Audited input and model access for post-screen candidate diagnostics."""
from pathlib import Path
import json
import numpy as np
from src.next.engine import EngineConfig, feature_hash
from src.next.cache import load_design
from src.next.design_selection import select_reference_families, selection_hash
from src.next.extensions import load_extended_design, ExtendedBundle, extension_hash
from src.next.registered_extensions import RegisteredExtendedBundle, register_extension, registration_hash
from src.next.predictor import NextBundle, predictor_hash
from src.next.fitting import fitting_hash, fit_binary
from src.next.ranking_objectives import implementation_hash as ranking_hash, fit_ranking
from src.scoring.ts_auc import ts_auc
from src.utils.artifacts import sha256


def model_class(candidate):
    return RegisteredExtendedBundle if candidate.get('extension_registration') else ExtendedBundle if candidate.get('extension') else NextBundle


def validate_sources(candidate):
    if candidate.get('extension_registration'):
        if candidate['registration_source_sha256'] != registration_hash():
            raise RuntimeError('Registered extension adapter source changed')
        register_extension(candidate['extension'], candidate['extension_registration'])
    if candidate.get('extension') and candidate['extension_frozen_hash'] != extension_hash(candidate['extension']):
        raise RuntimeError('Preregistered extension source changed')
    if candidate.get('ranking_objective') and candidate['ranking_source_sha256'] != ranking_hash():
        raise RuntimeError('Preregistered ranking source changed')


def candidate_design(data, candidate, scope='full'):
    validate_sources(candidate)
    if candidate.get('extension'):
        return load_extended_design(data, candidate, scope)
    result = list(load_design(data, config=EngineConfig(**candidate['engine_config']), scope=scope,
        groups=tuple(candidate['groups']), base=candidate['base'], keep_names=candidate.get('keep_names')))
    if candidate.get('base_family_selection'):
        result[0], result[2] = select_reference_families(result[0], result[2], candidate['base_family_selection'])
        result[3]['base_family_selection_code_sha256'] = selection_hash()
    return tuple(result)


def fit_candidate(data, candidate, design, metadata, names, train, valid):
    validate_sources(candidate)
    common = dict(guard=data.guard, config=EngineConfig(**candidate['engine_config']), names=names)
    if candidate.get('ranking_objective'):
        model, prediction, detail = fit_ranking(design, metadata, train, valid, objective=candidate['ranking_objective'], **common)
    else:
        model, prediction, detail = fit_binary(design, metadata, train, valid,
            learner=candidate['learner'], weighting=candidate['weighting'], params=candidate['params'], **common)
    if candidate.get('extension_registration'):
        model = RegisteredExtendedBundle.wrap_registered(model, candidate['extension'], candidate['extension_registration'])
    elif candidate.get('extension'):
        model = ExtendedBundle.wrap(model, candidate['extension'])
    return model, prediction, detail


def load_full_oof(data, eid):
    root = data.guard.root
    out = root/'artifacts/next/experiments'/eid
    report = json.loads((out/'RESULTS.json').read_text(encoding='utf-8'))
    if report['status'] == 'BUG' or report.get('full', {}).get('status') != 'COMPLETE':
        raise ValueError('Candidate needs valid completed primary five-fold OOF')
    candidate = report['candidate']
    validate_sources(candidate)
    policy = json.loads((out/'full/POLICY.json').read_text(encoding='utf-8'))
    expected = {'candidate': candidate, 'feature_hash': feature_hash(EngineConfig(**candidate['engine_config'])),
        'fitting_hash': fitting_hash(), 'predictor_hash': predictor_hash(), 'split_sha256': data.guard.split.digest,
        'seal_lock_sha256': data.guard.lock_sha, 'policy_sha256': sha256(root/'ROBUSTNESS_SPLIT_LOCK.json')}
    for key, value in expected.items():
        if policy[key] != value:
            raise RuntimeError(f'Full OOF identity changed: {key}')
    metadata = data.rows()
    prediction = np.full(len(metadata['target']), np.nan, dtype=np.float32)
    sources = {'policy_sha256': sha256(out/'full/POLICY.json')}
    for fold in range(5):
        folder = out/'full'
        result = json.loads((folder/f'fold_{fold}.json').read_text(encoding='utf-8'))
        if result['status'] != 'PASS':
            raise RuntimeError('Invalid completed OOF fold')
        for name, digest in result['files_sha256'].items():
            if sha256(folder/name) != digest:
                raise RuntimeError('OOF prediction, model or coordinate bytes changed')
            sources[name] = digest
        rows = np.load(folder/f'fold_{fold}_global_rows.npy', allow_pickle=False)
        np.testing.assert_array_equal(rows, np.flatnonzero(metadata['fold'] == fold))
        values = np.load(folder/f'fold_{fold}.npy', allow_pickle=False)
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise RuntimeError('OOF prediction dtype or finiteness changed')
        prediction[rows] = values
        score = ts_auc(metadata['target'][rows], values, metadata['time_online'][rows])
        np.testing.assert_allclose(score, report['full']['fold_scores'][fold], rtol=0, atol=1e-14)
    if not np.isfinite(prediction).all():
        raise RuntimeError('Incomplete OOF coverage')
    return candidate, report, metadata, prediction, sources
