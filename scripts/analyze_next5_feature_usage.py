"""Training-tree feature usage, with exact original feature coordinates.

These are descriptive learner statistics, not held-out importance, causal
attribution, an ablation experiment, or a model selection criterion.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress
from src.next5.model import FeatureModel
from src.next5.scheduler import register_job


def read_table(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_table(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bank(name):
    for prefix, label in (
        ('next5_gaussian_', 'Gaussian_scale'), ('next5_student5_', 'Student5_scale'),
        ('next5_semitail_', 'log_energy_semitail'), ('next5_energy_pit_', 'H_energy_PIT'),
        ('next5_cross_', 'explicit_bounded_products'), ('next5_two_timescale_', 'persistence_two_timescales'),
    ):
        if name.startswith(prefix):
            return label
    return 'other_registered_coordinates'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true')
    args = parser.parse_args()
    parent = BASE / 'feature_usage'
    if args.register:
        atomic_json(parent / 'POLICY_LOCK.json', {
            'created_utc': utc_now(), 'source_sha256': digest(Path(__file__)),
            'scope': 'All 295 existing ResearchPool learner artifacts from the verified primitive inventory',
            'statistics': 'Native training split counts and gains aligned through retained-column indices to registered feature names',
            'fold_summary': 'Fixed-fit mean/min/max and count used; overlapping training folds are not independent replicates',
            'model_scope': 'Statistics describe each new learner, not its fixed composite prediction or the BL80 share',
            'interpretation': 'Correlated features can exchange split gain. These statistics do not establish held-out incremental value or causal importance.',
            'new_fits': 0, 'real_rows_read': 0, 'selection_authority': False,
        }, immutable=True)
        register_job({'id': 'INFO_TREE_FEATURE_USAGE', 'kind': 'validation', 'priority': 80,
            'depends_on_success': ['AUDIT_PRIMITIVE_FIT_INVENTORY'],
            'argv': ['scripts/analyze_next5_feature_usage.py'],
            'description': 'Exact registered feature usage in 295 immutable learner trees; descriptive training statistics only',
            'retry_safe': False})
        progress('Tree feature usage protocol registered', 'Original retained-column mapping; no score-selected ablation or new model')
        return
    policy = read_json(parent / 'POLICY_LOCK.json')
    assert digest(Path(__file__)) == policy['source_sha256']
    inventory = BASE / 'integrity_audits/research_v1/fit_inventory'
    proof = read_json(inventory / 'RESULTS.json')
    assert proof['status'] == 'PASS' and proof['primitive_research_learner_fits'] == 295
    assert digest(inventory / 'PRIMITIVE_FITS.csv') == proof['table_sha256']
    fitted = read_table(inventory / 'PRIMITIVE_FITS.csv')
    out = parent / 'fixed_v1'
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / 'START.json', {'created_utc': utc_now(), 'new_fits': 0,
        'inventory_sha256': digest(inventory / 'RESULTS.json'), 'policy_sha256': digest(parent / 'POLICY_LOCK.json')}, immutable=True)
    rows, model_rows, bank_rows = [], [], []
    feature_groups = defaultdict(list)
    bank_groups = defaultdict(list)
    for i, record in enumerate(fitted):
        directory = ROOT / record['fit_directory']
        assert digest(directory / 'model.joblib') == record['model_sha256']
        model = FeatureModel.load(directory / 'model.joblib')
        columns = model.columns
        assert len(np.unique(columns)) == len(columns) and columns.min() >= 0 and columns.max() < len(model.names)
        gain = np.zeros(len(model.names), dtype=float)
        splits = np.zeros(len(model.names), dtype=np.int64)
        gain[columns] = model.estimator.booster_.feature_importance(importance_type='gain')
        splits[columns] = model.estimator.booster_.feature_importance(importance_type='split')
        assert np.isfinite(gain).all() and np.all(gain >= 0) and np.all(splits >= 0)
        assert gain.sum() > 0 and splits.sum() > 0
        share = gain / gain.sum()
        base = {k: record[k] for k in ('experiment_id', 'category', 'split', 'scope', 'fold')}
        banks = defaultdict(lambda: {'gain_share': 0., 'splits': 0, 'coordinates': 0, 'used_coordinates': 0})
        for coordinate, name in enumerate(model.names):
            group = bank(name)
            row = {**base, 'coordinate': coordinate, 'feature': name, 'bank': group,
                   'retained_by_training_column_filter': bool(coordinate in columns),
                   'training_gain': float(gain[coordinate]), 'training_gain_share': float(share[coordinate]),
                   'training_split_count': int(splits[coordinate]), 'used_by_tree': bool(splits[coordinate] > 0)}
            rows.append(row)
            key = tuple(base[k] for k in ('experiment_id', 'category', 'split', 'scope')) + (name,)
            feature_groups[key].append(row)
            banks[group]['gain_share'] += float(share[coordinate])
            banks[group]['splits'] += int(splits[coordinate])
            banks[group]['coordinates'] += 1
            banks[group]['used_coordinates'] += int(splits[coordinate] > 0)
        for group, values in sorted(banks.items()):
            row = {**base, 'bank': group, **values}
            bank_rows.append(row)
            key = tuple(base[k] for k in ('experiment_id', 'category', 'split', 'scope')) + (group,)
            bank_groups[key].append(row)
        assert abs(sum(v['gain_share'] for v in banks.values()) - 1.) < 1e-12
        model_rows.append({**base, 'model_sha256': record['model_sha256'], 'registered_coordinates': len(model.names),
            'retained_coordinates': len(columns), 'tree_used_coordinates': int(np.sum(splits > 0)),
            'total_split_count': int(splits.sum()), 'total_training_gain': float(gain.sum()),
            'top_single_coordinate_gain_share': float(share.max()),
            'effective_gain_coordinate_count': float(1. / np.sum(share ** 2)),
            'tree_count': int(model.estimator.booster_.num_trees())})
        if i % 25 == 0:
            progress('Read-only training-tree usage analysis', str(i + 1) + '/295 fixed learner artifacts; original feature names retained')
    feature_summary = []
    for key, group in sorted(feature_groups.items()):
        folds = [int(r['fold']) for r in group]
        assert len(folds) == len(set(folds))
        values = np.asarray([r['training_gain_share'] for r in group])
        feature_summary.append({**dict(zip(('experiment_id', 'category', 'split', 'scope', 'feature'), key)),
            'fitted_artifacts': len(group), 'artifacts_using_coordinate': sum(r['used_by_tree'] for r in group),
            'mean_training_gain_share': float(values.mean()), 'minimum_training_gain_share': float(values.min()),
            'maximum_training_gain_share': float(values.max()), 'no_independent_fold_interval': True})
    bank_summary = []
    for key, group in sorted(bank_groups.items()):
        values = np.asarray([r['gain_share'] for r in group])
        bank_summary.append({**dict(zip(('experiment_id', 'category', 'split', 'scope', 'bank'), key)),
            'fitted_artifacts': len(group), 'mean_training_gain_share': float(values.mean()),
            'minimum_training_gain_share': float(values.min()), 'maximum_training_gain_share': float(values.max()),
            'no_independent_fold_interval': True})
    tables = {'FEATURE_USAGE.csv': rows, 'MODEL_USAGE.csv': model_rows, 'BANK_USAGE.csv': bank_rows,
              'FEATURE_FIXED_FOLD_SUMMARY.csv': feature_summary, 'BANK_FIXED_FOLD_SUMMARY.csv': bank_summary}
    for name, values in tables.items():
        write_table(out / name, values)
    atomic_json(out / 'RESULTS.json', {'created_utc': utc_now(), 'status': 'PASS', 'fitted_artifacts': len(fitted),
        'feature_rows': len(rows), 'new_fits': 0, 'real_observations_or_labels_read': 0,
        'heldout_importance_claimed': False, 'causal_importance_claimed': False, 'selection_authority': False,
        'source_sha256': digest(Path(__file__)), 'policy_sha256': digest(parent / 'POLICY_LOCK.json'),
        'inventory_sha256': digest(inventory / 'RESULTS.json'),
        'files_sha256': {name: digest(out / name) for name in tables}}, immutable=True)
    progress('Fixed learner feature usage complete', '295 artifacts, all coordinates including unused columns; training statistics only')


if __name__ == '__main__':
    main()
