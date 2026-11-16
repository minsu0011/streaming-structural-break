"""Primitive-fit accounting and membership/source chronology, without new fits."""
from pathlib import Path
from collections import Counter
from datetime import datetime
import argparse
import csv
import hashlib
import re
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress
from src.next5.guard import ResearchGuard
from src.next5.data import admitted_npz
from src.next5.model import FeatureModel
from src.next5.stability import assignments, reference_root
from src.next5.scheduler import register_job


def plan():
    records = []
    registrations = sorted((BASE / 'registrations').glob('*.json'))
    assert len(registrations) == 20
    for path in registrations:
        cid = path.stem
        screen = read_json(BASE / 'experiments' / cid / 'screen/RESULTS.json')
        for scope in ('screen', 'full'):
            if scope == 'full' and not screen['screen_pass']:
                closed = BASE / 'experiments' / cid / 'full'
                assert read_json(closed / 'SKIPPED.json')['new_fits'] == 0
                assert not list(closed.glob('**/FIT_START.json'))
                continue
            for fold in ([0, 4] if scope == 'screen' else range(5)):
                records.append((BASE / 'experiments' / cid / scope / f'fold_{fold}',
                                path, 'candidate_' + scope, 'PRIMARY', scope, fold))
    for kind in ('J102', 'C205', 'B2_augmented_J', 'D2_expert'):
        for scope in (('screen', 'full') if kind in ('J102', 'C205') else ('full',)):
            for fold in ([0, 4] if scope == 'screen' else range(5)):
                records.append((BASE / 'references/fits/PRIMARY' / scope / kind / f'fold_{fold}',
                    BASE / 'references/registrations' / ('REF_' + kind + '.json'),
                    'primary_reference', 'PRIMARY', scope, fold))
    records.append((BASE / 'negative_controls/label_permutation_v1',
                    BASE / 'references/registrations/REF_NEGATIVE_LABEL_PERMUTATION.json',
                    'negative_control', 'PRIMARY', 'full', 0))
    policy = read_json(BASE / 'stability/POLICY_LOCK.json')
    for split in ('ALT_A', 'ALT_B', 'H_EXCLUSION'):
        for kind in ('J102', 'C205'):
            for fold in range(5):
                records.append((reference_root(kind, split) / f'fold_{fold}',
                    BASE / 'references/registrations' / ('REF_' + kind + '.json'),
                    'stability_reference', split, 'full', fold))
        for cid in policy['diagnostic_nominees']:
            for fold in range(5):
                records.append((BASE / 'stability/candidates' / cid / split / f'fold_{fold}',
                    BASE / 'registrations' / (cid + '.json'), 'stability_candidate', split, 'full', fold))
    assert len(records) == 295 and len({r[0] for r in records}) == 295
    return records


def register():
    policy = read_json(BASE / 'stability/POLICY_LOCK.json')
    dependencies = [f'STAB_{cid}_{split}' for cid in policy['diagnostic_nominees']
                    for split in ('ALT_A', 'ALT_B', 'H_EXCLUSION')]
    register_job({'id': 'AUDIT_PRIMITIVE_FIT_INVENTORY', 'kind': 'validation', 'priority': 80,
        'depends_on_success': dependencies, 'argv': ['scripts/audit_next5_fit_inventory.py', '--tag', 'research_v1'],
        'description': 'Verify all 295 primitive fit receipts, preregistration chronology, immutable model bytes and exact held-out group membership',
        'retry_safe': False})
    progress('Primitive-fit evidence audit queued', 'Count leaf fits once, reconstruct held-out IDs and check source, receipt and model metadata')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag')
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--final-audit', action='store_true')
    args = parser.parse_args()
    if args.register:
        register()
        return
    if not args.tag or not re.fullmatch('[A-Za-z0-9_]+', args.tag):
        raise ValueError('Safe immutable audit tag required')
    session = read_json(BASE / 'SESSION.json')
    if args.final_audit:
        assert session.get('audit_started_utc')
    phase = 'FINAL_AUDIT' if args.final_audit else 'RESEARCH'
    out = BASE / ('final_audit' if args.final_audit else 'integrity_audits') / args.tag / 'fit_inventory'
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / 'START.json', {'created_utc': utc_now(), 'phase': phase,
        'source_sha256': digest(Path(__file__)), 'new_fits': 0}, immutable=True)
    records = plan()
    roots = [BASE / n for n in ('experiments', 'references/fits', 'stability', 'negative_controls')]
    observed = {p.parent for root in roots for p in root.rglob('FIT_START.json')}
    assert observed == {r[0] for r in records}, 'Unexpected or missing primitive fit start'
    models = {p.parent for root in roots for p in root.rglob('model.joblib')}
    assert models == observed, 'A fitted artifact lacks a unique primitive fit receipt'
    guard = ResearchGuard()
    pointer = read_json(BASE / 'data/RESEARCH_POINTER.json')
    data_dir = ROOT / pointer['directory']
    assert data_dir.resolve().is_relative_to((ROOT / 'data/next5').resolve())
    assert digest(data_dir / 'MANIFEST.json') == pointer['manifest_sha256']
    manifest = read_json(data_dir / 'MANIFEST.json')
    assert digest(data_dir / 'series.npz') == manifest['files_sha256']['series.npz']
    series = admitted_npz(data_dir / 'series.npz', guard, 'ResearchPool ID/length membership audit')
    ids = series['dataset_id']
    assert ids.dtype == np.int32 and len(ids) == 6400
    lengths = np.diff(series['online_offsets'])
    split_lock = read_json(ROOT / 'NEXT5_SPLIT_LOCK.json')
    mappings = {'PRIMARY': {int(i): int(f) for i, f in split_lock['mappings']['PRIMARY'].items()}}
    mappings.update({s: assignments(s, guard) for s in ('ALT_A', 'ALT_B', 'H_EXCLUSION')})
    parts = {}
    for split, mapping in mappings.items():
        fold_vector = np.asarray([mapping[int(i)] for i in ids])
        for scope in ('full', 'screen'):
            eligible = np.ones(len(ids), dtype=bool) if scope == 'full' else np.isin(ids, split_lock['screen_ids'])
            for fold in range(5):
                train = eligible & (fold_vector != fold)
                valid = eligible & (fold_vector == fold)
                train_ids, valid_ids = ids[train], ids[valid]
                guard.partition(train_ids, valid_ids)
                parts[(split, scope, fold)] = {
                    'training_rows': int(lengths[train].sum()), 'validation_rows': int(lengths[valid].sum()),
                    'training_series': int(train.sum()), 'validation_series': int(valid.sum()),
                    'training_ids_sha256': hashlib.sha256(np.sort(train_ids).tobytes()).hexdigest(),
                    'heldout_ids_sha256': hashlib.sha256(np.sort(valid_ids).tobytes()).hexdigest(),
                }
    rows = []
    sources = {}
    reg_cache = {}
    for index, (directory, reg_path, category, split, scope, fold) in enumerate(records):
        start = read_json(directory / 'FIT_START.json')
        result = read_json(directory / 'RESULTS.json')
        if reg_path not in reg_cache:
            registration = read_json(reg_path)
            reg_cache[reg_path] = registration
            for field in ('execution_source_sha256', 'registration_source_sha256'):
                for name, sha in registration.get(field, {}).items():
                    assert digest(ROOT / name) == sha, name
                    if name in sources:
                        assert sources[name] == sha
                    sources[name] = sha
        registration = reg_cache[reg_path]
        reg_sha = digest(reg_path)
        assert start['registration_sha256'] == reg_sha
        assert datetime.fromisoformat(registration['created_utc']) < datetime.fromisoformat(start['created_utc'])
        assert datetime.fromisoformat(start['created_utc']) < datetime.fromisoformat(result['created_utc'])
        if category == 'candidate_full':
            screening = read_json(directory.parent.parent / 'screen/RESULTS.json')
            assert screening['screen_pass'] is True
            assert datetime.fromisoformat(screening['created_utc']) < datetime.fromisoformat(start['created_utc'])
        training = result['training'] if category == 'negative_control' else result
        for name, value in parts[(split, scope, fold)].items():
            if name != 'heldout_ids_sha256':
                assert training[name] == value, (directory, name)
        assert start['training_rows'] == training['training_rows']
        assert start['validation_rows'] == training['validation_rows']
        assert training['new_fits'] == 1 and result['new_fits'] == 1
        assert training['shadow_rows'] == training['seal_rows'] == 0
        assert training['training_only_columns_weights'] is True
        assert training['validation_labels_used_in_fit'] is False
        assert training['native_compact_max_abs_difference'] <= 1e-12
        model_sha = digest(directory / 'model.joblib')
        expected_sha = result['files_sha256']['model.joblib'] if category == 'negative_control' else result['model_sha256']
        assert model_sha == expected_sha
        model = FeatureModel.load(directory / 'model.joblib')
        assert model.module_name == registration['feature_module']
        assert model.settings == registration['feature_settings']
        assert model.feature_sha256 == registration['feature_sha256']
        assert model.model_source_sha256 == registration['model_source_sha256']
        for name, value in model.training.items():
            assert training[name] == value, (directory, name)
        parameters = model.estimator.get_params()
        for name, value in registration['parameters']['learner_params'].items():
            assert parameters[name] == value, (directory, name)
        prediction_sha = digest(directory / 'predictions.npz')
        expected_prediction = (result['files_sha256']['predictions.npz'] if category == 'negative_control'
            else result.get('prediction_file_sha256', result.get('predictions_file_sha256')))
        assert prediction_sha == expected_prediction
        rows.append({'fit_directory': directory.relative_to(ROOT).as_posix(), 'category': category,
            'experiment_id': registration['experiment_id'], 'split': split, 'scope': scope, 'fold': fold,
            'registration_utc': registration['created_utc'], 'fit_started_utc': start['created_utc'],
            'fit_completed_utc': result['created_utc'], 'registration_sha256': reg_sha,
            'model_sha256': model_sha, 'prediction_file_sha256': prediction_sha,
            **parts[(split, scope, fold)], 'new_fits': 1,
            'fit_start_sha256': digest(directory / 'FIT_START.json'), 'fit_result_sha256': digest(directory / 'RESULTS.json')})
        if index % 10 == 0:
            progress('Primitive-fit source and membership audit', str(index + 1) + '/295 verified',
                     phase=phase, counts_as_research=not args.final_audit)
            print('NEXT5 FIT INVENTORY', index + 1, '/295', flush=True)
    counts = dict(Counter(r['category'] for r in rows))
    assert counts == {'candidate_screen': 40, 'candidate_full': 95, 'primary_reference': 24,
                      'negative_control': 1, 'stability_reference': 30, 'stability_candidate': 105}
    with (out / 'PRIMITIVE_FITS.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    guard.check_state()
    atomic_json(out / 'RESULTS.json', {'created_utc': utc_now(), 'status': 'PASS', 'phase': phase,
        'primitive_research_learner_fits': 295, 'categories': counts, 'aggregate_receipts_double_counted': False,
        'unit_fixture_and_H_nuisance_estimates_counted_as_research_learner_fits': False,
        'all_preregistered_before_fit': True, 'all_exact_training_group_memberships': True,
        'all_model_source_and_learner_parameters_unchanged': True, 'all_prediction_file_hashes_verified': True,
        'real_point_predictions_decoded_by_this_audit': False, 'shadow_or_seal_rows_read': 0,
        'models_without_fit_receipts': 0, 'fit_receipts_without_models': 0, 'new_fits': 0,
        'table_sha256': digest(out / 'PRIMITIVE_FITS.csv'), 'registered_execution_sources_sha256': sources,
        'source_sha256': digest(Path(__file__)), 'research_split_sha256': guard.split.digest,
        'shadow_lock_sha256': guard.shadow_lock_sha,
    }, immutable=True)
    progress('Primitive-fit inventory verified', '295 distinct real learner fits; exact RP group boundaries and immutable source/model receipts',
             phase=phase, counts_as_research=not args.final_audit)


if __name__ == '__main__':
    main()
