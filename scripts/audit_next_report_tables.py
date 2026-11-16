"""Reconcile final exported tables against immutable experiment evidence."""
from pathlib import Path
import csv
import json
import math
import statistics
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.research_clock import ResearchClock
from src.utils.artifacts import write_json, sha256, utc_now


def require(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    clock = ResearchClock(ROOT).snapshot()
    require(clock['minimum_research_met'] and clock['final_audit_started'], 'Final report audit requires the additional audit phase')
    out = ROOT/'artifacts/next/final_report_audit'
    require(not out.exists(), 'Preserve the first final table audit')
    files = {}
    def pin(path):
        path = ROOT/path
        files[path.relative_to(ROOT).as_posix()] = sha256(path)
        return path
    def read(path):
        return json.loads(pin(path).read_text(encoding='utf-8'))
    def table(path):
        with pin(path).open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))
    def near(actual, expected, name):
        if expected is None:
            require(actual == '', 'Fabricated value in '+name)
        else:
            require(actual != '' and math.isfinite(float(actual)) and abs(float(actual)-expected) <= 1e-13,
                'Exported numeric value differs: '+name)
    writer = read('artifacts/next/final_reports/RESULTS.json')
    require(writer['status'] == 'PASS', 'Final report writer incomplete')
    for relative, digest in writer['output_files_sha256'].items():
        require(sha256(pin(relative)) == digest, 'Final writer output changed')
    selection = read('artifacts/next/final_selection/RESULTS.json')
    evidence = {row['candidate']['experiment_id']: row for row in read(Path(selection['evidence_snapshot'])/'CANDIDATE_EVIDENCE.json')}
    exported = table('MODEL_RESULTS_NEXT.csv')
    models = {row['experiment_id']: row for row in exported}
    originals = {path.parent.name: read(path.relative_to(ROOT)) for path in (ROOT/'artifacts/next/experiments').glob('*/RESULTS.json')}
    require(len(models) == len(exported) == len(originals) == 142 and set(models) == set(originals) == set(evidence), 'Candidate coverage/uniqueness differs')
    require(sha256(pin('MODEL_RESULTS_NEXT.csv')) == sha256(pin('EXPERIMENT_LEDGER.csv')), 'Ledger and model table differ')
    expected_cv, expected_fractures = {}, {}
    for eid, original in originals.items():
        row, decision = models[eid], evidence[eid]
        require(row['original_stage_status'] == original['status'] and row['final_evidence_status'] == decision['assessed_status'], 'Stage/final status mismatch: '+eid)
        require(json.loads(row['failed_gates']) == decision['decision']['failed_gates'] and json.loads(row['pending_gates']) == decision['decision']['pending_gates'], 'Gate explanation mismatch: '+eid)
        require(row['experiment_report_sha256'] == sha256(ROOT/'artifacts/next/experiments'/eid/'RESULTS.json'), 'Export references wrong experiment bytes')
        full = original.get('full', {})
        valid = full.get('status') == 'COMPLETE' and original['status'] != 'BUG'
        scores = full.get('fold_scores', []) if valid else []
        for name, value in [('primary_cv_mean', statistics.mean(scores) if valid else None),
                ('primary_cv_median', statistics.median(scores) if valid else None),
                ('primary_cv_worst', min(scores) if valid else None)]:
            near(row[name], value, eid+'/'+name)
        screen = original.get('screen', {})
        near(row['screen_mean'], screen.get('mean'), eid+'/screen')
        near(row['screen_delta'], screen.get('paired_bootstrap', {}).get('estimate_mean_delta'), eid+'/screen_delta')
        for detail in screen.get('cv_rows', []):
            expected_cv[(eid, 'fixed_3000_series_screen_only', detail['fold'])] = (detail['ts_auc'], original['status'] != 'BUG')
        if valid:
            for fold, score in enumerate(scores):
                expected_cv[(eid, 'original_five_fold_primary_authority', fold)] = (score, True)
        for split in decision['supplemental'].get('robustness_splits', []):
            for fold, score in enumerate(split['fold_scores']):
                detail = read(Path('artifacts/next/robustness/candidates')/eid/split['split_name']/f'fold_{fold}.json')
                near(str(score), detail['ts_auc'], eid+'/'+split['split_name'])
                expected_cv[(eid, split['split_name'], fold)] = (score, True)
        fracture = Path('artifacts/next/forensic')/eid/'PAIRED_FRACTURES.csv'
        if (ROOT/fracture).exists():
            for item in table(fracture):
                key = (item['candidate'], item['scope'])
                require(key not in expected_fractures, 'Duplicate original fracture')
                expected_fractures[key] = item
    cv_rows = table('CV_RESULTS_NEXT.csv')
    seen = set()
    for row in cv_rows:
        key = row['experiment_id'], row['scope'], int(row['fold'])
        require(key in expected_cv and key not in seen, 'Unexpected or duplicate exported CV cell')
        score, admissible = expected_cv[key]
        near(row['ts_auc'], score, str(key))
        require(row['quality_admissible'] == str(admissible), 'BUG/screen authority changed in exported CV')
        require(int(row['seal_rows']) == 0, 'Exported CV contains seal rows')
        seen.add(key)
    require(seen == set(expected_cv), 'Missing original screen/full/stress CV cells')
    fractures = table('PAIRED_BOOTSTRAP_RESULTS.csv')
    require(len(fractures) == len(expected_fractures), 'Fracture coverage differs')
    seen = set()
    for row in fractures:
        key = row['candidate'], row['scope']
        require(key in expected_fractures and key not in seen, 'Unexpected or duplicate exported fracture')
        original = expected_fractures[key]
        for name in ('candidate_mean', 'reference_mean', 'estimate_mean_delta', 'ci_2_5', 'ci_97_5', 'probability_delta_positive', 'replicates'):
            near(row[name], float(original[name]), str(key)+'/'+name)
        require(row['supported'] == original['supported'] and row['reference'] == original['reference'], 'Fracture support/reference changed')
        seen.add(key)
    for name, prefix in [('TIME_FRACTURE_NEXT.csv', 'time_'), ('BREAK_AGE_NEXT.csv', 'age_'), ('MECHANISM_FRACTURE_RESULTS.csv', 'mechanism_')]:
        rows = table(name)
        expected = {key for key in expected_fractures if key[1].startswith(prefix)}
        actual = {(row['candidate'], row['scope']) for row in rows}
        require(len(rows) == len(actual) and actual == expected, 'Incomplete exported slice: '+name)
    for row in selection['candidates']:
        eid = row['experiment_id']
        require(models[eid]['final_evidence_status'] in ('NEW_GENERATION', 'PROMOTED'), 'Final role lacks promotion')
        near(models[eid]['primary_cv_mean'], row['primary_cv_mean'], row['role'])
        near(models[eid]['primary_cv_worst'], row['worst_fold'], row['role'])
    out.mkdir(parents=True)
    write_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'clock': clock,
        'checks': {'all_142_candidate_rows_and_statuses_match': True, 'screen_scores_not_exported_as_full_cv': True,
            'bug_scores_excluded_from_primary_authority': True, 'all_original_screen_full_and_stress_cells_match': True,
            'bootstrap_values_references_support_and_scopes_match': True, 'final_roles_match_exported_promoted_rows': True,
            'model_table_and_ledger_identical': True},
        'candidate_rows': len(models), 'cv_cells': len(cv_rows), 'paired_fracture_rows': len(fractures),
        'evidence_files_sha256': files, 'driver_sha256': sha256(Path(__file__)),
        'new_model_fits': 0, 'new_predictions': 0, 'new_quality_scores': 0, 'seal_rows': 0})
    print('FINAL EXPORTED TABLE AUDIT PASS', len(models), len(cv_rows), len(fractures), flush=True)


if __name__ == '__main__':
    main()
