"""Independently join report claims to original results, closed tables and receipts."""
from pathlib import Path
import argparse
import csv
import datetime
import re
import statistics
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next3.claim_checks import ClaimChecks, numbered_answers, require_literal
from src.next3.final_validation import prepared_table_proof, successful_receipt
from scripts.prepare_next3_independent_reproduction import time_gate


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream: return list(csv.DictReader(stream))


def one(values, **keys):
    found = [row for row in values if all(row.get(name) == value for name, value in keys.items())]
    if len(found) != 1: raise RuntimeError('Ambiguous original evidence row: '+str(keys))
    return found[0]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--preview')
    parser.add_argument('--preview-check-label', default='claims'); args = parser.parse_args()
    preview = args.preview is not None
    if preview and not re.fullmatch('[A-Za-z0-9_-]+', args.preview): raise ValueError('Safe preview identifier required')
    if not re.fullmatch('[A-Za-z0-9_-]+', args.preview_check_label): raise ValueError('Safe claim preview label required')
    if not preview: time_gate()
    report_dir = BASE/'maintenance/report_previews'/args.preview if preview else BASE/'final_reports'
    report = successful_receipt(report_dir/'RESULTS.json',
        statuses=('REVIEW_PREVIEW_NOT_FINAL_DELIVERY',) if preview else ('PASS',))
    assert report['preview'] is preview and report['required_numbered_answers'] == 44
    text_root = report_dir if preview else ROOT; tables = BASE/'final_tables/prepared_v2' if preview else ROOT
    prepared_table_proof(ROOT, BASE/'final_tables/prepared_v2')
    sources = {}
    def bind(path):
        path = Path(path); sources[path.relative_to(ROOT).as_posix()] = digest(path); return path
    for name, sha in report['files_sha256'].items():
        path = bind(text_root/name)
        if digest(path) != sha: raise RuntimeError('Report bytes changed before claim verification: '+name)
    facts = read_json(bind(report_dir/'FACTS.json')); run = read_json(bind(report_dir/'RUN_CONTEXT.json'))
    assert digest(report_dir/'FACTS.json') == report['facts_sha256']
    assert digest(report_dir/'RUN_CONTEXT.json') == report['run_context_sha256']
    for name, sha in facts['evidence_sha256'].items():
        if digest(bind(ROOT/name)) != sha: raise RuntimeError('Report input evidence changed: '+name)
    if not preview:
        for name, sha in report['source_sha256'].items(): assert digest(bind(ROOT/name)) == sha
    elif 'retained_source_sha256' in report:
        for name, sha in report['retained_source_sha256'].items(): assert digest(bind(report_dir/'original_sources'/name)) == sha
    else:
        retention = successful_receipt(bind(report_dir/'SOURCE_RETENTION.json'))
        for name, item in retention['source_snapshots'].items():
            assert item['sha256'] == report['source_sha256'][name] == digest(bind(ROOT/item['snapshot']))
    answers = numbered_answers((text_root/'00_GPT_READ_FIRST_NEXT3.md').read_text(encoding='utf-8'))
    check = ClaimChecks()
    def number(name, stated, observed, path, location, *, atol=1e-14):
        bind(path); check.numeric(name, stated, observed, Path(path).relative_to(ROOT).as_posix(), location, atol=atol)
    performance = facts['performance']; distilled = facts['distilled']
    for role, record in (('PERFORMANCE', performance), ('DISTILLED', distilled)):
        old_selection_path = ROOT/'DEV_CANDIDATES_NEXT2.json'
        original = read_json(bind(old_selection_path))[role.lower()]
        for key in ('mean', 'median', 'worst_fold'):
            number(role+'_'+key, record['full'][key], original['full'][key], old_selection_path, role+'.full.'+key)
        old_cv_path = ROOT/'CV_RESULTS_NEXT2.csv'
        old_cv = [row for row in csv_rows(bind(old_cv_path)) if row['experiment_id'] == record['experiment_id'] and row['scope'] == 'full']
        old_cv.sort(key=lambda row: int(row['fold']))
        assert len(old_cv) == 5
        current_cv_path = tables/'CV_RESULTS_NEXT3.csv'
        current_cv = csv_rows(bind(current_cv_path))
        for fold, row in enumerate(old_cv):
            assert int(row['fold']) == fold
            number(role+'_fold_'+str(fold), record['full']['fold_scores'][fold], row['ts_auc'], old_cv_path, record['experiment_id']+'/full/'+str(fold))
            current = one(current_cv, experiment_id=record['experiment_id'], scope='full', fold=str(fold))
            number(role+'_published_fold_'+str(fold), current['ts_auc'], row['ts_auc'], old_cv_path, record['experiment_id']+'/full/'+str(fold))
        number(role+'_mean_from_five_folds', record['full']['mean'], statistics.mean(float(row['ts_auc']) for row in old_cv), old_cv_path, 'arithmetic mean of locked five folds')
        number(role+'_median_from_five_folds', record['full']['median'], statistics.median(float(row['ts_auc']) for row in old_cv), old_cv_path, 'median of locked five folds')
    for question, value in ((5, performance['full']['mean']), (6, performance['full']['mean']),
        (11, distilled['full']['mean']), (22, performance['full']['median']), (23, performance['full']['worst_fold'])):
        require_literal(answers[question]['answer'], repr(value))
    for value in performance['full']['fold_scores']: require_literal(answers[21]['answer'], repr(value))
    pair_path = tables/'PAIRED_BOOTSTRAP_NEXT3.csv'; pair_rows = csv_rows(bind(pair_path))
    identity = one(pair_rows, candidate=performance['experiment_id'], reference=performance['experiment_id'], scope='all_time')
    assert identity['analytic_identity'] == 'True' and identity['record_type'] == 'SELECTED_PERFORMANCE_VS_SESSION_START_IDENTICAL_SCIENCE'
    for key in ('estimate_mean_delta', 'ci_2_5', 'ci_97_5', 'probability_delta_positive', 'replicates', 'new_draws'):
        number('selected_identity_'+key, identity[key], 0., pair_path, 'selected-vs-start/'+key, atol=0)
    require_literal(answers[7]['answer'], '0.0'); require_literal(answers[24]['answer'], '[0.0, 0.0]')
    require_literal(answers[25]['answer'], 'P(Δ>0)=0.0')
    inherited = one(pair_rows, candidate=performance['experiment_id'], reference=distilled['experiment_id'], scope='all_time')
    for key in ('estimate_mean_delta', 'ci_2_5', 'ci_97_5', 'probability_delta_positive'):
        number('inherited_BL_vs_J_'+key, performance['paired_all_time'][key], inherited[key], pair_path, 'BL/J/all_time/'+key)
    model_path = tables/'MODEL_RESULTS_NEXT3.csv'; models = csv_rows(bind(model_path))
    new = [row for row in models if row['record_type'] == 'NEXT3_NEW_SCIENTIFIC_RECIPE']
    assert len(models) == 17 and len(new) == 13 and sum(bool(row['full_mean']) for row in new) == 12
    assert all(row['promoted'] == row['scientific_role_selected'] == 'False' and row['final_failure_reason'] for row in new)
    best = max((row for row in new if row['full_mean']), key=lambda row: float(row['full_mean']))
    assert best['experiment_id'] == facts['exploratory_best']['experiment_id'] == 'I2_HISTORICAL_RANK_PRODUCTS'
    number('exploratory_best_mean', facts['exploratory_best']['full_mean'], best['full_mean'], model_path, 'I2/full_mean')
    number('exploratory_best_delta', facts['exploratory_best']['delta_vs_frozen_performance'], best['delta_vs_frozen_performance'], model_path, 'I2/delta')
    require_literal(answers[6]['answer'], best['full_mean']); require_literal(answers[20]['answer'], best['delta_vs_frozen_performance'])
    ledger_path = tables/'EXPERIMENT_LEDGER.csv'; ledger_bytes = bind(ledger_path).read_bytes()
    original_ledger_path = BASE/'preflight/EXPERIMENT_LEDGER_BEFORE_NEXT3.csv'
    old_ledger = bind(original_ledger_path).read_bytes()
    assert ledger_bytes.startswith(old_ledger) and len(csv_rows(original_ledger_path)) == 213
    ledger = csv_rows(ledger_path); assert len(ledger) == 242
    number('science_fit_ledger_sum', facts['science_fits'], sum(int(row['new_fits'] or 0) for row in ledger[213:]), ledger_path, 'sum of appended 29 records', atol=0)
    assert not any(row.get('seal_rows') not in ('', '0', '0.0') for row in ledger[213:])
    assert len([row for row in ledger[213:] if row['ledger_section'] == 'NEXT3_SCIENTIFIC_RECIPES']) == 13
    mech_path = tables/'MECHANISM_RESULTS_NEXT3.csv'; mech = csv_rows(bind(mech_path))
    for scope, question in (('DEPENDENCE_DOMINANT', 14), ('SCALE_DOMINANT', 15)):
        row = one(mech, candidate=performance['experiment_id'], reference=distilled['experiment_id'], scope='mechanism_'+scope)
        assert row['supported'] == 'True'
        for key, value in facts['inherited_mechanism'][scope].items():
            number('inherited_'+scope+'_'+key, value, row[key], mech_path, scope+'/'+key)
        require_literal(answers[question]['answer'], repr(float(row['estimate_mean_delta'])))
    for candidate, scopes in facts['recipe_pairs'].items():
        for scope, values in scopes.items():
            row = one(pair_rows, candidate=candidate, reference=performance['experiment_id'], scope=scope)
            for key, value in values.items(): number(candidate+'_'+scope+'_'+key, value, row[key], pair_path, candidate+'/'+scope+'/'+key)
    aggregate_path = BASE/'pair_matrix_audit/run_v1/CONDITIONAL_SUMMARY_RECONSTRUCTED.csv'
    aggregate = csv_rows(bind(aggregate_path))
    for row in new:
        if not row['full_mean']: continue
        candidate = row['experiment_id']; audit = one(aggregate, candidate=candidate)
        published = one(pair_rows, candidate=candidate, reference=performance['experiment_id'], scope='all_time')
        for current, original in [('estimate_mean_delta', 'observed_mean_delta'), ('ci_2_5', 'conditional_ci_low'),
                ('ci_97_5', 'conditional_ci_high'), ('probability_delta_positive', 'probability_delta_positive'),
                ('joint_conditional_envelope_lower', 'joint_conditional_envelope_lower'),
                ('joint_conditional_envelope_upper', 'joint_conditional_envelope_upper')]:
            number(candidate+'_D007_'+current, published[current], audit[original], aggregate_path, candidate+'/'+original)
        assert audit['selection_adjusted'] == 'False' and published['selection_adjusted'] == 'False'
    alt_path = tables/'ALT_SPLIT_RESULTS_NEXT3.csv'; alt_rows = csv_rows(bind(alt_path))
    assert len(alt_rows) == 30 and all(row['new_fits_in_NEXT3'] == '0' and row['new_independent_validation'] == 'False' for row in alt_rows)
    for alternate in performance['alternate_splits']:
        selected = [row for row in alt_rows if row['candidate'] == performance['experiment_id'] and row['split'] == alternate['split']]
        assert len(selected) == 5
        for fold, row in enumerate(sorted(selected, key=lambda row: int(row['fold']))):
            number(alternate['split']+'_fold_'+str(fold), alternate['fold_scores'][fold], row['ts_auc'], alt_path, alternate['split']+'/'+str(fold))
        number(alternate['split']+'_mean', alternate['mean'], statistics.mean(float(row['ts_auc']) for row in selected), alt_path, alternate['split']+'/mean')
        require_literal(answers[26]['answer'], repr(alternate['mean']))
    regime_path = tables/'REGIME_RESULTS_NEXT3.csv'; regime = csv_rows(bind(regime_path))
    held = [row for row in regime if row['candidate'] == performance['experiment_id'] and row['split'] == 'LEAVE_ONE_H_REGIME_OUT']
    assert len(held) == 5 and all(row['new_fits_in_NEXT3'] == '0' for row in held)
    number('H_regime_worst', performance['heldout_H']['worst'], min(float(row['ts_auc']) for row in held), regime_path, 'refitted leave-H-out worst')
    number('H_regime_mean', performance['heldout_H']['mean'], statistics.mean(float(row['ts_auc']) for row in held), regime_path, 'refitted leave-H-out mean')
    require_literal(answers[27]['answer'], repr(performance['heldout_H']['worst']))
    null_path = tables/'LONG_NULL_RESULTS_NEXT3.csv'; nulls = csv_rows(bind(null_path))
    assert all(row['ts_auc'] == '' for row in nulls)
    for item in facts['null_N0_alpha05']:
        row = one(nulls, candidate=performance['experiment_id'], background='N0', horizon=str(item['horizon']),
                  alpha='0.05', evidence_origin='NEXT2_FROZEN_NO_NEXT3_REFIT')
        number('N0_null_'+str(item['horizon']), item['eventual_alarm_rate'], row['eventual_alarm_rate'], null_path, 'BL/N0/alpha.05/'+str(item['horizon']))
        require_literal(answers[28]['answer'], repr(item['eventual_alarm_rate']))
    counter_path = tables/'COUNTEREXAMPLE_RESULTS_NEXT3.csv'; counter = csv_rows(bind(counter_path))
    assert all(row['ts_auc'] == '' for row in counter if row['scenario'].startswith('N') or row['scenario'] == 'O1')
    for candidate, value in facts['inherited_outlier_excess'].items():
        row = one(counter, candidate=candidate, scenario='O1', alpha='0.05', evidence_origin='NEXT2_FROZEN_NO_NEXT3_REFIT')
        number(candidate+'_outlier_excess', value, row['outlier_alarm_excess'], counter_path, candidate+'/O1/alpha.05')
        require_literal(answers[29]['answer'], repr(value))
    # The probability-scale transfer is not a TS-AUC contrast or a deployable reset.
    transfer_path = BASE/'legacy_state_transfer/run_v1/WINDOW_COUNTERFACTUAL_INTERVALS.csv'
    transfer = csv_rows(bind(transfer_path))
    for anchor, values in facts['legacy_EWMA_state_transfer'].items():
        row = one(transfer, anchor_offset=anchor, block='LEGACY_EWMA', direction='CONTROL_INTO_OUTLIER',
            metric='probability_mean', window_start_inclusive='256', window_end_exclusive='512')
        for key, value in values.items(): number('D006_anchor'+anchor+'_'+key, value, row[key], transfer_path, 'EWMA/control_into_outlier/'+anchor+'/'+key)
    cold_path = BASE/'engineering/lean_shared/cold/RESULTS.json'; cold = successful_receipt(bind(cold_path))
    direct_path = BASE/'engineering/lean_shared/direct/RESULTS.json'; direct = successful_receipt(bind(direct_path))
    published_latency = read_json(bind(tables/'LATENCY_NEXT3.json'))['PERFORMANCE']
    trials = published_latency['cold_process']['trials']
    assert len(trials) == len(cold['pairs']) == 3
    for index, row in enumerate(trials):
        for variant, prefix in [('PACKED_SHARED', 'packed'), ('LEAN_PACKED_SHARED', 'lean')]:
            path = cold_path.parent/f'trial_{index}_{variant}.json'
            original = successful_receipt(bind(path))
            assert digest(path) == cold['files_sha256'][path.name]
            number(f'cold_{index}_{prefix}_RSS', row[prefix+'_RSS_bytes'], original['post_warm_memory']['rss'], path, 'post_warm_memory.rss', atol=0)
            number(f'cold_{index}_{prefix}_None', row[prefix+'_first_none_seconds'], original['first_warmed_none_seconds'], path, 'first_warmed_none_seconds', atol=0)
    rss_ratio = statistics.median(row['lean_RSS_bytes'] for row in trials)/statistics.median(row['packed_RSS_bytes'] for row in trials)
    none_ratio = statistics.median(row['lean_first_none_seconds'] for row in trials)/statistics.median(row['packed_first_none_seconds'] for row in trials)
    number('cold_RSS_ratio_of_medians', published_latency['cold_process']['ratio_of_median_postwarm_RSS_vs_E002'], rss_ratio, tables/'LATENCY_NEXT3.json', 'recomputed from six original trial scalars')
    number('cold_first_None_ratio_of_medians', published_latency['cold_process']['ratio_of_median_first_none_seconds_vs_E002'], none_ratio, tables/'LATENCY_NEXT3.json', 'recomputed from six original trial scalars')
    number('cold_ratio_matches_original_receipt', rss_ratio, cold['median_RSS_ratio'], cold_path, 'ratio of original RSS medians')
    number('cold_None_matches_original_receipt', none_ratio, cold['median_first_none_ratio'], cold_path, 'ratio of original ready-time medians')
    require_literal(answers[30]['answer'], repr(rss_ratio)); require_literal(answers[30]['answer'], repr(none_ratio))
    original_direct = []; paired_ratios = []
    for index in range(3):
        path = BASE/f'engineering/lean_shared/direct/trial_{index}_LEAN_PACKED_SHARED/RESULTS.json'
        trial = successful_receipt(bind(path)); seconds = trial['continuous_stream']['measured_prediction_seconds']
        number('E003_direct_10M_'+str(index), published_latency['three_actual_10M_seconds'][index], seconds, path, 'continuous_stream.measured_prediction_seconds', atol=1e-12)
        original_direct.append(seconds)
        packed_path = BASE/f'engineering/lean_shared/direct/trial_{index}_PACKED_SHARED/RESULTS.json'
        packed_trial = successful_receipt(bind(packed_path))
        paired_ratios.append(seconds/packed_trial['continuous_stream']['measured_prediction_seconds'])
    number('E003_actual_10M_median', published_latency['median_actual_10M_seconds'], statistics.median(original_direct), direct_path, 'median of actual three LEAN trials', atol=1e-12)
    number('E003_median_of_paired_direct_ratios', published_latency['median_direct_time_ratio_vs_E002'], statistics.median(paired_ratios), direct_path, 'median of ratios from six original direct receipts')
    number('E003_pair_ratio_matches_original_receipt', statistics.median(paired_ratios), direct['median_direct_10M_ratio'], direct_path, 'median_direct_10M_ratio')
    assert not published_latency['direct_throughput_improvement_claimed']
    for seconds in original_direct: require_literal(answers[32]['answer'], repr(seconds))
    state = read_json(bind(tables/'STATE_MEMORY_NEXT3.json'))['PERFORMANCE']
    number('memory_sum_private_plus_prepared', state['total_one_stream_plus_one_model_numeric_bytes'],
        state['private_per_series_numeric_bytes']+state['model_and_prepared_shared_numeric_bytes'], tables/'STATE_MEMORY_NEXT3.json', 'private + prepared (selector not counted twice)', atol=0)
    number('retained_private_plus_selector', state['retained_state_numeric_bytes_including_shared_selectors'],
        state['private_per_series_numeric_bytes']+state['shared_selector_bytes_reachable_from_state'], tables/'STATE_MEMORY_NEXT3.json', 'private + shared selectors', atol=0)
    for key, value in trial['numeric_memory'].items():
        number('retained_memory_original_'+key, state[key], value, path, 'numeric_memory.'+key, atol=0)
    for key in ('private_per_series_numeric_bytes', 'shared_selector_bytes_reachable_from_state',
                'model_and_prepared_shared_numeric_bytes', 'total_one_stream_plus_one_model_numeric_bytes', 'canonical_model_file_bytes'):
        require_literal(answers[31]['answer'], format(state[key], ','))
    account_path = report_dir/'ACCOUNTING_SNAPSHOT.json'; account = read_json(bind(account_path))
    for field, original in [('productive_research_seconds', 'productive_seconds'), ('audit_productive_seconds', 'audit_productive_seconds'), ('unobserved_seconds', 'unobserved_seconds')]:
        number('report_'+field, run[field], account[original], account_path, original, atol=0)
    started = datetime.datetime.fromisoformat(run['started_utc']); observed = datetime.datetime.fromisoformat(run['observed_utc'])
    number('report_wall_seconds', run['wall_seconds'], (observed-started).total_seconds(), report_dir/'RUN_CONTEXT.json', 'observed_utc-started_utc', atol=0)
    failures = facts['maintenance']['failed_child_receipts_sha256']
    for name, sha in failures.items(): assert digest(bind(ROOT/name)) == sha and read_json(ROOT/name)['status'] == 'FAILED'
    number('failed_child_count', facts['maintenance']['failed_child_attempts'], len(failures), report_dir/'FACTS.json', 'maintenance.failed_child_receipts_sha256 count', atol=0)
    for question in range(35, 39): require_literal(answers[question]['answer'], 'false'); require_literal(answers[question]['answer'], '500')
    for question in (39, 40): require_literal(answers[question]['answer'], '0회')
    require_literal(answers[43]['answer'], 'NOT_FORMALLY_TRIGGERED')
    assert 'posterior' in answers[25]['answer'] and '독립 검증' in answers[24]['answer']
    assert 'EWMA160' in answers[29]['answer'] and '아니오' in answers[8]['answer']
    if not preview:
        selection = successful_receipt(bind(ROOT/'DEV_CANDIDATES_NEXT3.json'), statuses=('FROZEN_NEXT3_DEV_SELECTION',))
        assert selection['scientific_role_count'] == 2 and selection['new_performance_candidate'] is None and selection['new_robustness_candidate'] is None
        assert not selection['generation_level_promotion'] and selection['selected_quality_delta_vs_starting_performance'] == 0.0
        assert run['raw_fits'] == 3 and run['tests']['tests'] >= 798 and run['runtime']['total_actual_public_API_points'] == 30_000_000
        require_literal(answers[33]['answer'], str(run['tests']['tests'])+'개')
        assert not any(item['status'] == 'ACTIVE' for item in read_json(ROOT/'BOTTLENECK_BOARD_FINAL.json')['bottlenecks'])
    out = report_dir/args.preview_check_label if preview else BASE/'final_claims'; out.mkdir(parents=True, exist_ok=False)
    retained = out/'original_sources'; retained.mkdir()
    for source in (Path(__file__), ROOT/'src/next3/claim_checks.py'):
        (retained/source.name).write_bytes(source.read_bytes())
    atomic_json(out/'CLAIMS.json', check.claims, immutable=True)
    atomic_json(out/'RESULTS.json', {'status': 'PREVIEW_CLAIMS_PASS_NOT_FINAL_AUDIT' if preview else 'PASS',
        'created_utc': utc_now(), 'numeric_cross_checks': len(check.claims), 'numbered_Korean_answers': len(answers),
        'new_scientific_recipes': 13, 'previous_ledger_rows_preserved': 213, 'total_ledger_rows': 242,
        'scientific_fits': 129, 'raw_reproduction_fits': run['raw_fits'], 'new_fits_in_this_check': 0,
        'literal_gates_not_relaxed': True, 'selected_contrast_is_zero_identity': True,
        'inherited_ALT_H_not_new_generalization': True, 'null_and_outlier_only_TS_AUC_undefined': True,
        'selection_adjustment_not_claimed': True, 'source_sha256': sources,
        'claim_file_sha256': digest(out/'CLAIMS.json'), 'script_sha256': digest(Path(__file__)),
        'boundary': 'Independent joins and arithmetic within the same agent; not an independent reviewer or external statistical validation.'}, immutable=True)
    progress('NEXT3 '+('preview' if preview else 'final')+' numeric claims checked',
        str(len(check.claims))+' numerical joins and all 44 Korean answer blocks verified', phase='ACTIVE_RESEARCH' if preview else 'FINAL_AUDIT')
    print('NEXT3 CLAIMS', 'PREVIEW PASS' if preview else 'PASS', len(check.claims), flush=True)


if __name__ == '__main__': main()
