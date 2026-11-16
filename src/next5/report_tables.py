"""Traceable scalar exports; no model fitting, prediction loading or new draws."""
import csv
from statistics import median
from src.next5.artifacts import ROOT, BASE, digest, read_json

FAMILY_FILES = {
    'S': 'SCALE_UNIFICATION_RESULTS.csv', 'P': 'PERSISTENCE_RESULTS.csv',
    'W': 'WINDOWED_EVIDENCE_RESULTS.csv', 'E': 'EPROCESS_RESULTS.csv',
    'R': 'ROBUST_NULL_RESULTS.csv', 'M': 'MEAN_SPECIALIST_RESULTS.csv',
    'X': 'CROSS_FAMILY_RESULTS.csv',
}


class ScalarEvidence:
    def __init__(self):
        self.inputs = {}
        self.tables = {}
        self.pending = []

    def record(self, path):
        relative = path.relative_to(ROOT).as_posix()
        self.inputs[relative] = digest(path)
        return relative

    def json(self, path):
        self.record(path)
        return read_json(path)

    def csv(self, path):
        source = self.record(path)
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        return [{**r, 'source_file': source, 'source_sha256': self.inputs[source]} for r in rows]

    def add(self, name, rows):
        self.tables.setdefault(name, []).extend(rows)

    def write(self, out):
        results = {}
        for name, rows in self.tables.items():
            if not rows:
                raise ValueError('Expected populated table: ' + name)
            fields = list(dict.fromkeys(key for row in rows for key in row))
            with (out / name).open('x', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            results[name] = {'sha256': digest(out / name), 'rows': len(rows), 'columns': fields}
        return results


def primary_tables(evidence):
    policy = evidence.json(ROOT / 'configs/next5/RESEARCH_POLICY_LOCK.json')
    gate = policy['performance_gate']
    candidates, cv, boot = [], [], []
    for registration_path in sorted((BASE / 'registrations').glob('*.json')):
        registration = evidence.json(registration_path)
        cid, family = registration['experiment_id'], registration['family']
        screen_path = BASE / 'experiments' / cid / 'screen/RESULTS.json'
        screen = evidence.json(screen_path)
        assert screen['status'] == 'COMPLETE' and screen['scope'] == 'screen'
        full_path = BASE / 'experiments' / cid / 'full/RESULTS.json'
        full = evidence.json(full_path) if full_path.exists() else None
        if full is not None:
            assert screen['screen_pass'] is True and full['status'] == 'COMPLETE'
            result_path = full_path
            result = full
            uncertainty_path = BASE / 'bootstrap' / cid / 'RESULTS.json'
            uncertainty = evidence.json(uncertainty_path)
            assert uncertainty['full_result_sha256'] == digest(full_path)
            uncertainty_values = uncertainty['paired_all_time']
            failures = []
            for key in ('mean_delta', 'median_delta', 'worst_delta'):
                if result[key] < gate[key + '_min']:
                    failures.append(key + '_below_frozen_minimum')
            if uncertainty_values['probability_delta_positive'] < gate['bootstrap_probability_min']:
                failures.append('bootstrap_probability_below_frozen_minimum')
            if uncertainty_values['ci_2_5'] <= gate['ci_lower_strict_min']:
                failures.append('bootstrap_lower_not_above_frozen_minimum')
            assert (not failures) == uncertainty['performance_primary_and_bootstrap_admission']
            boot.append({'candidate': cid, 'family': family, 'split': 'PRIMARY',
                'score_population': 'ResearchPool_6400_grouped_5fold', **uncertainty_values,
                'median_of_folds_delta_definition': 'Median candidate fold scores minus median baseline fold scores, not median of matched fold differences',
                'conditional_fixed_OOF_not_selection_adjusted': True,
                'primary_and_bootstrap_gate_pass': not failures,
                'source_file': uncertainty_path.relative_to(ROOT).as_posix(),
                'source_sha256': digest(uncertainty_path)})
        else:
            assert screen['screen_pass'] is False
            result_path, result = screen_path, screen
            uncertainty_values = {}
            failures = ['SCREEN_FAIL_full_not_run']
        row = {'record_type': 'registered_candidate', 'candidate': cid, 'family': family,
            'scientific_question': registration['scientific_question'],
            'score_population': 'ResearchPool_6400_grouped_5fold' if full else 'ResearchPool_screen_2560_folds_0_4',
            'score_scope': result['scope'], 'screen_pass': screen['screen_pass'],
            'full_completed': full is not None,
            **{k: result[k] for k in ('mean', 'median', 'worst', 'mean_delta', 'median_delta',
                'worst_delta', 'minimum_fold_delta', 'positive_folds', 'standalone_mean', 'baseline_mean')},
            'primary_mean_delta_gate': gate['mean_delta_min'],
            'mean_delta_gate_margin': result['mean_delta'] - gate['mean_delta_min'] if full else '',
            'minimum_matched_fold_delta_floor_pass': result['minimum_fold_delta'] >= gate['worst_delta_min'] if full else '',
            'median_matched_fold_delta': median(f['delta'] for f in result['fold_results']),
            'median_matched_fold_delta_floor_pass': median(f['delta'] for f in result['fold_results']) >= gate['median_delta_min'] if full else '',
            'median_delta_definition': 'Median candidate fold scores minus median baseline fold scores; median_matched_fold_delta is the median matched fold difference',
            'worst_delta_definition': 'Minimum candidate fold score minus minimum baseline fold score; minimum_fold_delta is the minimum matched fold difference',
            'primary_gate_failure_reasons': ';'.join(failures),
            **{'paired_' + k: v for k, v in uncertainty_values.items()},
            'screen_learner_fits': screen['new_fits'], 'full_learner_fits': full['new_fits'] if full else 0,
            'source_file': result_path.relative_to(ROOT).as_posix(), 'source_sha256': digest(result_path),
            'registration_file': registration_path.relative_to(ROOT).as_posix(),
            'registration_sha256': digest(registration_path)}
        candidates.append(row)
        evidence.add(FAMILY_FILES[family], [row])
        for scope_result in [screen] + ([full] if full else []):
            for fold in scope_result['fold_results']:
                cv.append({'record_type': 'registered_candidate', 'candidate': cid, 'family': family,
                    'split': 'PRIMARY', 'scope': scope_result['scope'],
                    **{k: fold[k] for k in ('fold', 'TS_AUC', 'BL80_TS_AUC', 'delta', 'standalone_TS_AUC',
                        'training_series', 'validation_series', 'training_rows', 'validation_rows', 'model_sha256')},
                    'source_file': (BASE / 'experiments' / cid / scope_result['scope'] / 'RESULTS.json').relative_to(ROOT).as_posix()})
    assert len(candidates) == 20 and sum(r['full_completed'] for r in candidates) == 19
    assert len(cv) == 135 and len(boot) == 19
    evidence.add('MODEL_RESULTS_NEXT5.csv', candidates)
    evidence.add('CV_RESULTS_NEXT5.csv', cv)
    evidence.add('PAIRED_BOOTSTRAP_NEXT5.csv', boot)
    references = evidence.csv(BASE / 'references/comparisons/SUMMARY.csv')
    evidence.add('MATCHED_REFERENCE_RESULTS_NEXT5.csv', references)
    evidence.add('MATCHED_REFERENCE_CV_NEXT5.csv', evidence.csv(BASE / 'references/comparisons/FOLD_SCORES.csv'))
    evidence.add(FAMILY_FILES['S'], [
        {'record_type': 'S3_matched_reference_decomposition', 'family': 'S',
         'candidate': r['reference'], 'score_population': 'ResearchPool_6400_grouped_5fold', **r}
        for r in references if r['reference'] in ('BL80', 'B2', 'D2')])
    evidence.json(BASE / 'diagnostics/S3_SCALE_DECOMPOSITION/RESULTS.json')
    for path in sorted((BASE / 'forensic').glob('*/PAIRED_FRACTURES.csv')):
        for row in evidence.csv(path):
            for prefix, name in [('regime_', 'REGIME_RESULTS_NEXT5.csv'),
                ('mechanism_', 'MECHANISM_RESULTS_NEXT5.csv'), ('time_', 'TIME_FRACTURE_NEXT5.csv'),
                ('age_', 'BREAK_AGE_NEXT5.csv')]:
                if row['scope'].startswith(prefix):
                    evidence.add(name, [{'analysis_type': 'PRIMARY_fixed_OOF_slice', **row}])
    return candidates


def stability_tables(evidence, *, complete):
    policy = evidence.json(BASE / 'stability/POLICY_LOCK.json')
    for cid in policy['diagnostic_nominees']:
        for split in ('ALT_A', 'ALT_B', 'H_EXCLUSION'):
            directory = BASE / 'stability/candidates' / cid / split
            result = evidence.json(directory / 'RESULTS.json')
            assert result['status'] == 'COMPLETE' and len(result['fold_results']) == 5
            uncertainty_path = directory / 'BOOTSTRAP/RESULTS.json'
            if uncertainty_path.exists():
                uncertainty = evidence.json(uncertainty_path)
                assert uncertainty['status'] == 'PASS'
                interval = uncertainty['paired_all_time']
                design = uncertainty['bootstrap_design']
            else:
                evidence.pending.append(uncertainty_path.relative_to(ROOT).as_posix())
                if complete:
                    raise RuntimeError('Fixed stability uncertainty is not complete')
                interval, design = {}, 'PENDING; no interval imputed'
            folds = result['fold_results']
            values = [r['TS_AUC'] for r in folds]
            baseline = [r['BL80_TS_AUC'] for r in folds]
            delta = [r['delta'] for r in folds]
            row = {'analysis_type': 'matched_refit_H_exclusion' if split == 'H_EXCLUSION' else 'matched_refit_ALT',
                'candidate': cid, 'split': split, 'candidate_mean': sum(values) / 5,
                'baseline_mean': sum(baseline) / 5, 'mean_fold_delta': sum(delta) / 5,
                'minimum_fold_delta': min(delta), 'maximum_fold_delta': max(delta),
                'median_matched_fold_delta': median(delta),
                'median_of_folds_delta_definition': 'Median candidate fold scores minus median baseline fold scores, not median of matched fold differences',
                'positive_folds': sum(v > 0 for v in delta), **interval,
                'bootstrap_design': design, 'conditional_fixed_OOF_not_selection_adjusted': True,
                'primary_gate_can_be_overridden': False,
                'source_file': (directory / 'RESULTS.json').relative_to(ROOT).as_posix(),
                'source_sha256': digest(directory / 'RESULTS.json')}
            evidence.add('REGIME_RESULTS_NEXT5.csv' if split == 'H_EXCLUSION' else 'ALT_RESULTS_NEXT5.csv', [row])
            evidence.add('STABILITY_FOLD_RESULTS_NEXT5.csv', [
                {'analysis_type': row['analysis_type'], 'candidate': cid, 'split': split,
                 'source_file': row['source_file'], **{k: r[k] for k in (
                     'fold', 'TS_AUC', 'BL80_TS_AUC', 'delta', 'training_series',
                     'validation_series', 'training_rows', 'validation_rows', 'model_sha256')}} for r in folds])


def completed_stress_tables(evidence):
    directory = BASE / 'synthetic/fitted_summary/complete_v1'
    result = evidence.json(directory / 'RESULTS.json')
    assert result['status'] == 'PASS' and result['fitted_artifacts'] == 97
    provenance = evidence.json(directory / 'COHORT_PROVENANCE_REPAIR.json')
    assert provenance['status'] == 'PASS' and provenance['original_numeric_summary_result_sha256'] == digest(directory / 'RESULTS.json')
    mapping = {
        'COUNTEREXAMPLE_RESULTS_NEXT5.csv': ['COUNTEREXAMPLE_SCORES.csv', 'PAIRED_COUNTEREXAMPLES.csv',
            'FIXED_ARTIFACT_MEAN_TS_AUC.csv', 'COUNTEREXAMPLE_ALARMS.csv', 'PAIRED_ALARMS.csv'],
        'LONG_NULL_RESULTS_NEXT5.csv': ['LONG_NULL_RESULTS.csv', 'PAIRED_ALARMS.csv'],
        'OUTLIER_IMPULSE_NEXT5.csv': ['OUTLIER_RESPONSE_SUMMARY.csv', 'PAIRED_OUTLIER_RESPONSE.csv',
            'OUTLIER_LAG_RESPONSE.csv', 'FOLD_IMPULSE_HETEROGENEITY.csv'],
    }
    for output, inputs in mapping.items():
        for name in inputs:
            path = directory / name
            assert digest(path) == result['files_sha256'][name]
            rows = evidence.csv(path)
            if name == 'PAIRED_ALARMS.csv':
                kind = 'long_null' if output == 'LONG_NULL_RESULTS_NEXT5.csv' else 'counterexample'
                rows = [r for r in rows if r['cohort_kind'] == kind]
            evidence.add(output, [{'record_type': name.removesuffix('.csv'),
                'deployment_scope': 'fixed_CV_artifacts_only', **r} for r in rows])
    for name in ('CALIBRATION.csv', 'FEATURE_IMPULSE_RESPONSE.csv', 'FOLD_ALARM_HETEROGENEITY.csv'):
        path = directory / name
        assert digest(path) == result['files_sha256'][name]
        evidence.add('SYNTHETIC_' + name.removesuffix('.csv') + '_NEXT5.csv', evidence.csv(path))
