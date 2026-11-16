"""Publish only completed, time-gated NEXT5 evidence and Korean explanations."""
from pathlib import Path
import csv
import json
import re
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress
from src.next5.finalization import require_handoff_time
from src.next5.korean_reports import documents


def main():
    timing = require_handoff_time()
    evidence = {}
    def read(path, status=None):
        value = read_json(path)
        if status is not None:
            assert value['status'] == status, (path, value.get('status'))
        evidence[path.relative_to(ROOT).as_posix()] = digest(path)
        return value
    plan = read(BASE / 'final_audit/PLAN.json')
    for name, sha in plan['source_sha256'].items():
        assert digest(ROOT / name) == sha
    for identifier in plan['jobs']:
        result = read(BASE / 'jobs' / identifier / 'STATE.json', 'COMPLETE')
        assert result['exit_code'] == 0
    selection = read(ROOT / 'DEV_CANDIDATES_NEXT5.json', 'FROZEN_NEXT5_FINAL_ROLES')
    finalists = read(ROOT / 'FINALIST_MANIFEST_NEXT5.json', 'FROZEN_NEXT5_FINALISTS')
    assert not finalists['finalists'] and not selection['new_NEXT5_roles']
    assert digest(BASE / 'shadow/FINALIST_MANIFEST_NEXT5.json') == digest(ROOT / 'FINALIST_MANIFEST_NEXT5.json')
    audit_root = BASE / 'final_audit/final_v1'
    fits = read(audit_root / 'fit_inventory/RESULTS.json', 'PASS')
    tests = read(audit_root / 'full_tests/RESULTS.json', 'PASS')
    statistics = read(audit_root / 'frozen_statistics/RESULTS.json', 'PASS')
    synthetic_bytes = read(audit_root / 'synthetic_receipts/RESULTS.json', 'PASS')
    boundary = read(audit_root / 'frozen_boundary/RESULTS.json', 'PASS')
    graph = read(audit_root / 'execution_graph/RESULTS.json', 'PASS')
    assert graph['all_preserved_failures_resolved'] and not graph['pending_preserved_failure_resolutions']
    fold_definitions = read(BASE / 'integrity_audits/fold_summary_definitions_v1/RESULTS.json', 'PASS')
    assert fold_definitions['full_candidates'] == 19 and fold_definitions['all_full_candidates_fail_original_mean_floor']
    for name, sha in fold_definitions['inputs_sha256'].items():
        assert digest(ROOT / name) == sha
    assert digest(BASE / 'integrity_audits/fold_summary_definitions_v1/FOLD_SUMMARY_DEFINITIONS.csv') == fold_definitions['table_sha256']
    ledger = read(BASE / 'ledger/final_v1/RESULTS.json', 'PASS')
    portable = read(BASE / 'portable_verification/final_v1/RESULTS.json', 'PASS')
    tables = read(BASE / 'report_tables/final_v1/RESULTS.json', 'PASS_COMPLETE_SCALAR_TABLES')
    assert not tables['pending']
    latency = read(BASE / 'latency/fixed_v1/RESULTS.json', 'PASS')
    assert latency['actual_total_continuous_points'] == 270000000 and latency['all_repeat_fingerprints_identical']
    retained = read(BASE / 'retained_artifact_validation/research_v1/RESULTS.json', 'PASS')
    assert retained['series_per_role'] == 6400 and retained['real_points_per_role'] == 3203157
    assert all(r['batch_and_actual_API_bitwise_equal'] for r in retained['roles'].values())
    api_policy = read(BASE / 'runtime_validation/API_POLICY_LOCK.json')
    apis = {cid: read(BASE / 'runtime_validation/api' / cid / 'RESULTS.json', 'PASS') for cid in api_policy['candidates']}
    assert len(apis) == 7
    import_repair = read(BASE / 'maintenance/READONLY_IMPORT_REPAIR/RESOLVED.json', 'PASS')
    read(BASE / 'maintenance/READONLY_IMPORT_REPAIR/POLICY_LOCK.json')
    cold_adapter = read(BASE / 'maintenance/READONLY_IMPORT_REPAIR/COLD_PROBE.json', 'PASS')
    assert import_repair['successful_complete_replacement_API_jobs'] == 7
    assert cold_adapter['persistent_filesystem_write_attempts'] == 0 and cold_adapter['batch_probability_max_abs_difference'] <= 1e-15
    for cid in apis:
        adapter = read(BASE / 'runtime_validation/api' / cid / 'API_IMPORT_REPAIR.json', 'PASS')
        assert adapter['fresh_result_sha256'] == digest(BASE / 'runtime_validation/api' / cid / 'RESULTS.json')
    for record in apis.values():
        assert record['prefix_cases'] >= 300 and record['series'] == 6400 and record['points'] == 3203157
        assert record['persistent_filesystem_write_attempts'] == 0 and record['official_TS_AUC_parity']
        assert record['max_abs_probability_difference'] <= 1e-15
    stress = read(BASE / 'synthetic/fitted_summary/complete_v1/RESULTS.json', 'PASS')
    assert stress['fitted_artifacts'] == 97
    cohort_repair = read(BASE / 'synthetic/fitted_summary/complete_v1/COHORT_PROVENANCE_REPAIR.json', 'PASS')
    assert cohort_repair['original_numeric_summary_result_sha256'] == digest(BASE / 'synthetic/fitted_summary/complete_v1/RESULTS.json')
    read(BASE / 'maintenance/COHORT_PROVENANCE_REPAIR/REPAIR.json', 'REPAIRED_UNSTARTED_METADATA_AGGREGATION')
    decomposition = read(BASE / 'impulse_decomposition/complete_v1/RESULTS.json', 'PASS')
    direction = read(BASE / 'scale_direction_response/complete_v1/RESULTS.json', 'PASS')
    null_drift = read(BASE / 'null_drift/fixed_v1/RESULTS.json', 'PASS')
    assert null_drift['scenarios'] == 4 and null_drift['generating_seeds_per_scenario'] == 256
    assert null_drift['feature_bank_innovation_channels_exactly_equal'] and null_drift['new_predictions'] == 0
    stress_figures = read(BASE / 'analysis_figures/stress_v1/RESULTS.json', 'PASS')
    assert stress_figures['figures'] == 3 and stress_figures['fitted_coverage'] == 97
    for name, sha in stress_figures['files_sha256'].items():
        assert digest(BASE / 'analysis_figures/stress_v1' / name) == sha
    information = read(BASE / 'information/RESULTS.json', 'PASS')
    for name, sha in information['files_sha256'].items():
        assert digest(BASE / 'information' / name) == sha
    uncertainty = read(BASE / 'selection_uncertainty/RESULTS.json', 'PASS')
    negative = read(BASE / 'negative_controls/label_permutation_v1/RESULTS.json', 'PASS')
    read(BASE / 'feature_usage/fixed_v1/RESULTS.json', 'PASS')
    late = []
    for cid in ('P1_TWO_TIMESCALE_CONFIRMATION', 'P2_SOFT_PERSISTENCE_COUNT', 'P3_SHOCK_RECOVERY'):
        for fold in range(5):
            late.append(read(BASE / 'synthetic_late_break' / cid / ('fold_' + str(fold)) / 'RESULTS.json', 'PASS'))
    stability = []
    for cid in read(BASE / 'stability/POLICY_LOCK.json')['diagnostic_nominees']:
        for split in ('ALT_A', 'ALT_B', 'H_EXCLUSION'):
            stability.append(read(BASE / 'stability/candidates' / cid / split / 'BOOTSTRAP/RESULTS.json', 'PASS'))
    assert len(stability) == 21
    for path in sorted((BASE / 'failure_resolutions').glob('*.json')):
        resolution = read(path, 'RESOLVED_WITH_PRESERVED_FAILURE')
        assert digest(BASE / 'jobs' / resolution['job'] / 'STATE.json') == resolution['quarantined_state_sha256']
    from src.next5.failure_reports import completed_failure_rows
    failure_rows, failure_summary = completed_failure_rows(read)
    native_review_repair = read(BASE / 'maintenance/NATIVE_REVIEW_PUBLICATION_REPAIR/RESOLVED.json', 'PASS')
    out = BASE / 'final_publication/final_v1'
    out.mkdir(parents=True, exist_ok=False)
    published = {}
    def json_file(name, value):
        atomic_json(ROOT / name, value, immutable=True)
        published[name] = digest(ROOT / name)
    def copy(source, name):
        assert not (ROOT / name).exists(), 'Final publication never silently overwrites: ' + name
        shutil.copyfile(source, ROOT / name)
        assert digest(source) == digest(ROOT / name)
        published[name] = digest(ROOT / name)
    failure_name = 'FAILURE_CLASSIFICATION_NEXT5.csv'
    fields = list(dict.fromkeys(k for row in failure_rows for k in row))
    with (ROOT / failure_name).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failure_rows)
    published[failure_name] = digest(ROOT / failure_name)
    json_file('FAILURE_CLASSIFICATION_NEXT5.json', {**failure_summary, 'created_utc': utc_now(),
        'table_sha256': published[failure_name]})
    for name, record in tables['files'].items():
        source = BASE / 'report_tables/final_v1' / name
        assert digest(source) == record['sha256']
        copy(source, name)
    # The original pre-score lock files already exist; preserve their bytes.
    for name in ('NEXT5_PREFLIGHT.json', 'NEXT5_SHADOW_LOCK.json', 'NEXT5_SHADOW_GUARD.json',
        'NEXT5_SPLIT_LOCK.json', 'NEXT5_RANDOMNESS_LOCK.json', 'HYPOTHESIS_FAMILY_LOCK_NEXT5.json', 'CANDIDATE_BUDGET_NEXT5.json'):
        assert (ROOT / name).is_file()
        published[name] = digest(ROOT / name)
    common = {'created_utc': utc_now(), 'audit_started_utc': read_json(BASE / 'SESSION.json')['audit_started_utc'],
        'new_NEXT5_roles': [], 'Shadow_opened': False, 'LOCAL_SEAL_rows': 0,
        'access_counter_scope': 'Admitted returned rows, caches and application training/evaluation/analysis; Arrow scanner ID predicates and returned-ID checks were used. Internal Parquet page/row-group decoding was not independently instrumented.',
        'zero_internal_Parquet_decoding_claimed': False}
    json_file('SEAL_GUARD_NEXT5.json', {'status': 'PASS', **common, 'research_series': 6400,
        'shadow_series': 1600, 'seal_series': 2000, 'all_four_seals_unopened': True,
        'new_reduced_usage': 0, 'historical_reduced_usage': '2/2 retained',
        'new_public_submission': 0, 'new_cloud_submission': 0,
        'evidence': (audit_root / 'frozen_boundary/RESULTS.json').relative_to(ROOT).as_posix(),
        'guard_scope': 'Application admission and state checks; no OS sandbox claim'})
    json_file('PREFIX_INVARIANCE_NEXT5.json', {'status': 'PASS', **common,
        'family_minimum_prefix_cases': 100, 'representative_actual_API_cases_per_candidate': 300,
        'candidate_APIs': apis, 'retained_existing_role_APIs': retained,
        'read_only_import_adapter': cold_adapter, 'import_failure_recovery': import_repair,
        'whole_project_final_test_suite': tests,
        'future_suffix_final_length_and_current_point_traps': True})
    json_file('DETERMINISM_NEXT5.json', {'status': 'PASS', **common, 'candidate_APIs': apis,
        'representative_candidate_API_count': len(apis),
        'actual_candidate_API_coverage': 'Seven fixed family diagnostic nominees, each evaluated on all five PRIMARY ResearchPool held-out folds',
        'candidate_readonly_entry': 'src.next5.readonly_streaming',
        'unchanged_numeric_core': 'src.next5.streaming', 'import_failure_recovery': import_repair,
        'all_seven_nominee_primary_OOF_bitwise_equal': all(r['bitwise_float64_equal'] for r in apis.values()),
        'maximum_candidate_probability_difference': max(r['max_abs_probability_difference'] for r in apis.values()),
        'declared_probability_tolerance': 1e-15, 'declared_score_tolerance': 1e-14,
        'all_repeat_10M_fingerprints_identical': latency['all_repeat_fingerprints_identical'],
        'retained_batch_API_bitwise_equal': True, 'portable_fresh_processes': portable,
        'model_and_training_identity_audit': fits,
        'serialization_boundary': 'Exact model bytes checked against original fits; semantic prediction parity and separately generated joblib bytes are distinct quantities'})
    json_file('LATENCY_NEXT5.json', {'status': 'PASS', **common, 'fixed_next5_measurement': latency,
        'threefold_gate_scope': 'For each candidate, medians of three within-repeat FROZEN_PRIMARY ratios for scalar median, amortized real sequence and actual 10M time; p95/p99, H initialization and state bytes are reported separately, not included in all_ratios_within_3x.',
        'reading_guide': 'docs/NEXT5_RUNTIME_READING_GUIDE.md',
        'timing_isolation_scope': 'No competing workspace Python processes; operating-system activity and native metadata monitoring/reviews can still occur. All predefined blocks are retained.',
        'native_metadata_review_policy': read(BASE / 'bottlenecks/TIMING_REVIEW_POLICY_LOCK.json'),
        'native_metadata_review_v2_policy': read(BASE / 'bottlenecks/TIMING_REVIEW_POLICY_LOCK_V2.json'),
        'native_metadata_publication_recovery': native_review_repair,
        'native_bottleneck_reviews_during_timing': [read(path) for path in sorted((BASE / 'bottlenecks/history').glob('*.json'))
            if read_json(path).get('review_engine', '').startswith('Native PowerShell')],
        'block_records': [read(path) for path in sorted((BASE / 'latency/fixed_v1').glob('repeat_*.json'))],
        'old_D2_measurement': read(BASE / 'preflight/FROZEN_REFERENCES.json')['roles']['NEXT4_ROBUSTNESS']['direct_runtime_median_ratios'],
        'old_D2_measurement_scope': 'Historical NEXT4 timing of actual retained artifact; not a new NEXT5 timing run',
        'new_candidate_promotion': False})
    json_file('LEAKAGE_AUDIT_NEXT5.json', {'status': 'PASS', **common, 'boundary': boundary,
        'primitive_fit_inventory': fits, 'execution_graph': graph, 'negative_control': negative,
        'candidate_API_validation': apis, 'full_suite': tests,
        'forbidden_inference_inputs': ['dataset_id', 'tau', 'online_total_length', 'future_suffix'],
        'posthoc_mechanism_in_inference': False, 'Shadow_used_for_NEXT5_fit_or_selection': False,
        'historical_Shadow_used_as_DEV_before_NEXT5': True,
        'H_regime_classifier_historical_transductive': True, 'X_parent_selection_globally_adaptive_within_RP': True,
        'selection_bias_removed': False, 'application_guard_not_OS_sandbox': True})
    json_file('REPRODUCTION_NEXT5.json', {'status': 'PASS', **common,
        'candidate_readonly_entry': 'src.next5.readonly_streaming',
        'candidate_cold_import_probe': cold_adapter, 'candidate_full_API_recovery': import_repair,
        'fresh_new_role_raw_to_fit': 'NOT_REQUIRED_NO_NEW_NEXT5_ROLE',
        'fresh_raw_new_model_fit_performed': False, 'new_reproduction_fits': 0,
        'retained_actual_entry_RP_batch_stream_parity': retained, 'copied_raw_free_fresh_process_replay': portable,
        'all_original_PRIMARY_statistics_reproduced': statistics, 'synthetic_receipt_bytes_verified': synthetic_bytes,
        'prior_NEXT4_fresh_training_evidence_preserved_as_history': True})
    # Additional publication, audit synthesis and Korean text follow below.
    publish_documents(out, published, evidence, common, {
        'timing': timing, 'fits': fits, 'tests': tests, 'statistics': statistics, 'boundary': boundary,
        'graph': graph, 'ledger': ledger, 'portable': portable, 'latency': latency, 'apis': apis,
        'fold_definitions': fold_definitions,
        'failure_summary': failure_summary,
        'native_review_repair': native_review_repair,
        'retained': retained, 'stress': stress, 'decomposition': decomposition, 'direction': direction,
        'information': information, 'uncertainty': uncertainty, 'selection': selection, 'null_drift': null_drift,
        'stress_figures': stress_figures,
        'import_repair': import_repair, 'cold_adapter': cold_adapter,
        'synthetic_bytes': synthetic_bytes, 'late': late, 'stability': stability})


def publish_documents(out, published, evidence, common, context):
    def write_json(name, value):
        atomic_json(ROOT / name, value, immutable=True)
        published[name] = digest(ROOT / name)
    initial_path = BASE / 'bottlenecks/history/001_INITIAL.json'
    board_path = BASE / 'bottlenecks/BOARD.json'
    evidence[initial_path.relative_to(ROOT).as_posix()] = digest(initial_path)
    evidence[board_path.relative_to(ROOT).as_posix()] = digest(board_path)
    write_json('BOTTLENECK_BOARD_INITIAL_NEXT5.json', read_json(initial_path))
    write_json('BOTTLENECK_BOARD_FINAL_NEXT5.json', {**read_json(board_path),
        'final_snapshot_utc': utc_now(), 'research_work_complete': True,
        'new_finalists': 0, 'final_roles': list(context['selection']['roles']),
        'final_execution_and_scientific_failure_classification': context['failure_summary'],
        'earlier_failure_interpretation_field_scope': 'Preserved original preflight description; final counts also include the subsequent API import and native metadata publication repairs.',
        'interpretation': 'Priority inputs remain subjective. Open scientific questions continue in the next-session plan; no further NEXT5 fitting is authorized by this board.'})
    history = []
    for path in sorted((BASE / 'bottlenecks/history').glob('*.json')):
        record = read_json(path)
        evidence[path.relative_to(ROOT).as_posix()] = digest(path)
        for row in record['rows']:
            values = {k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()}
            history.append({'snapshot': path.name, 'snapshot_utc': record['created_utc'],
                'publication_status': 'PARTIAL_PUBLICATION_FAILED_PRESERVED' if path.name == '012_REVIEW.json' else 'PUBLISHED',
                'publication_recovery_receipt': 'artifacts/next5/maintenance/NATIVE_REVIEW_PUBLICATION_REPAIR/RESOLVED.json' if path.name in ('012_REVIEW.json', '013_REVIEW.json') else '',
                'review_reason': record.get('review_reason', 'Initial subjective priorities before family results'),
                **values, 'source_sha256': digest(path)})
    fields = list(dict.fromkeys(k for row in history for k in row))
    with (ROOT / 'BOTTLENECK_HISTORY_NEXT5.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    published['BOTTLENECK_HISTORY_NEXT5.csv'] = digest(ROOT / 'BOTTLENECK_HISTORY_NEXT5.csv')
    write_json('CANDIDATE_USAGE_NEXT5.json', {'status': 'FINAL_USAGE', **common,
        'families': 7, 'screen_candidates': 20, 'full_candidates': 19, 'Shadow_finalists': 0,
        'primitive_learner_fits': 295, 'fit_categories': context['fits']['categories'],
        'frozen_budget_sha256': digest(ROOT / 'CANDIDATE_BUDGET_NEXT5.json'),
        'all_finite_scientific_recipes_closed': True, 'new_deployment_roles': 0})
    write_json('RAW_DATA_REQUIRED_NEXT5.json', {'status': 'RAW_EXCLUDED_BY_DESIGN', **common,
        'expected_raw_file_sha256': context['boundary']['raw_byte_sha256'],
        'required_interpreter': 'Python 3.13.2', 'exact_pins': 'requirements-lock.txt',
        'ResearchPool_regeneration_script': 'scripts/prepare_next5_research_data.py',
        'retraining_boundary': 'Use separately supplied hash-matched raw files and frozen RP IDs/splits/registrations. Do not relax Shadow or LOCAL_SEAL admission. Finished jobs are not automatically rerun.',
        'raw_free_capability': 'Original retained BL80/B2/D2 inference and included fitted research models can be loaded with their source/metadata. Full held-out metrics require excluded raw/cache/point OOF.',
        'raw_data_download_or_external_submission_instructions_provided': False})
    docs = documents(context)
    docs['LIMITATIONS_NEXT5.md'] = '''# NEXT5 한계

새 primary 또는 NEXT5 역할은 없다. X2가 ResearchPool 최고였지만 원래 평균 증가분 기준 +0.0012에 미달했다. 미개봉 Shadow의 점수를 만들거나 일반화 성공/실패로 해석하지 않는다.

동결 구현의 worst_delta는 후보 최저 fold score에서 기준 모델 최저 fold score를 뺀 값이다. 사용자 요구를 각 대응 fold의 악화 한도로 읽으면 minimum_fold_delta가 더 엄격한 비교다. 두 정의와 각 통과 여부를 표에 분리했다. S4A는 전자의 하한을 통과하지만 최소 대응 Δ는 약 -0.0012384로 후자의 -0.001 한도에 미달한다. 원래 결과나 규칙을 사후 덮어쓰지 않았다. 모든 19개 full 후보가 이미 평균 개선 기준에 미달하므로 어느 정의에서도 finalist와 최종 역할은 바뀌지 않는다.

원래 median_delta 및 해당 bootstrap 구간도 각 모델 fold score 중앙값끼리의 차이다. 대응 fold 차이의 중앙값은 별도 열로 보고하며 원래 구간을 그 값의 CI로 부르지 않는다. P2의 중앙값 차이는 +0.0003344897이지만 대응 차이 중앙값은 -0.0001361490으로 하한 통과 여부가 다르다. X2는 각각 +0.0022104184와 +0.0010587485다. 이 차이 역시 평균 기준 미달과 미선정 결론을 바꾸지 않는다.

기존 8000 DEV를 장기간 반복 사용했다. NEXT5 Shadow 1600개는 이번 연구에서는 사용하지 않았지만 과거 DEV 탐색의 일부였다. X parent 선택은 전체 ResearchPool 결과를 본 뒤 수행했으므로 전체 후보 설계가 nested 검증된 것은 아니다. 2000 whole-series bootstrap과 centered max statistic은 고정 OOF에 조건부이며 과거 적응적 선택과 학습 불확실성을 제거하지 않는다.

H-regime 변환에는 과거 전체 DEV H로 fit한 역사적 transductive 요소가 있다. 이번 RP descriptor는 H만으로 새로 계산했다. Mechanism signature는 사후 해석이며 실제 생성 원인의 정답이 아니다. SCALE positive 106개, MEAN 16개, TAIL 64개이므로 작은 subset에서 확증을 주장하지 않는다.

H-exclusion은 크기가 1179/507/133/801/3780인 다섯 regime을 하나씩 제외하고 학습한다. 각 학습 크기는 5221/5893/6267/5599/2620이며, 후보와 기준 모델의 ID는 각 fold 안에서 일치한다. 보고 평균은 다섯 regime 차이에 동일 가중치를 둔 값이다. 배경 변화와 학습 표본 수 변화의 효과를 분리하거나 전체 RP pooled TS-AUC로 해석하지 않는다.

합성 결과는 동일한 고정 생성기와 seed에 조건부다. 97개 CV artifact는 훈련 fold가 겹치므로 독립 표본이 아니다. 평균 artifact TS-AUC는 ensemble prediction의 TS-AUC가 아니다. 별도 새 전체 RP/DEV 배포 모델은 만들지 않았다.

P3는 사전등록대로 빠른 peak와 ungated EWMA64를 비교했다. 사용자 예시의 confirmation이 있을 때만 증가하는 persistent 상태는 구현하지 않았으므로, 그 더 넓은 구조의 실패로 일반화하지 않는다. P1의 confirmation feature와 P2의 threshold occupancy는 별도 후보다. E1 역시 사전등록된 unreset 누적 mixture이며 최근 age 제한은 E2에만 적용했다. 고정 크기 배열과 제한된 evidence horizon은 다르다. E1과 E2는 별도 학습 모델이므로 window 제한만의 인과적 ablation으로 설명하지 않는다.

단발·반복 충격 뒤 probability excess 감소에는 0.9 BL80 혼합의 진폭 축소가 포함될 수 있다. 선형 분해는 recovery time을 분해하지 않는다. 첫 16연속 tolerance 충족은 영구 복구가 아니며, 2048 censoring과 recovered-only 통계를 구별한다. postbreak detection의 분모는 전체 변화 시계열이고 조건부 delay는 검출 성공한 사례만 포함한다.

임계값은 독립 null의 1024 horizon에서 정했다. 16384 eventual alarm에 같은 alpha 보장을 붙이지 않는다. H-estimated dependent likelihood는 ranking evidence이며 엄밀한 e-process/anytime validity를 입증하지 않았다.

N0–N3의 counterexample와 long_null은 같은 scenario/horizon이어도 생성 시드 partition이 다른 코호트다. 원래 paired alarm 표의 누락된 종류 열은 전수 집계 실행 전에 동결 반복 순서와 SHA 영수증으로 복원했다. 원래 폴드별 표와 모든 수치는 보존했다. 최종 paired alarm 비교에는 cohort_kind를 포함해야 하며, 수정 계약과 97개 표 보존 영수증을 함께 전달한다.

Late-break 진단은 기존 onset=512의 85개 seed만 사용한 사후 층화다. Scale 방향 진단과 impulse 진폭 분해도 결과를 본 뒤 추가한 해석 도구다. 새 후보나 gate를 만들지 않았으며 독립 확인으로 세지 않는다. Tree gain/split 사용량은 훈련 learner의 통계로, held-out 중요도나 인과적 제품항 기여가 아니다.

긴 무변화의 에너지·누적 likelihood 진단도 원래 N0–N3 각 256개 경로의 사후 해석이다. 누적 Gaussian LR와 clipped standardized innovation 에너지 합의 산술 관계를 확인한다. 양의 working likelihood 비율은 학습 모델의 알람 확률과 다르고, 그 자체로 false alarm의 인과적 기여를 분리하지 않는다.

원래 D2 실제 파일의 NEXT4 반복 이상치 악화는 남아 있다. SCALE 보조 역할 유지가 전체 성능·반복 충격에 대한 새 보장이 아니다. 원래 D2 latency는 역사적 측정으로 표시했다.

애플리케이션 admission guard와 파일·실행 증거를 검증했지만 OS sandbox 증명이나 외부 계정의 모든 활동 감사는 아니다. 같은 파일의 raw byte hashing과 seal 관측/label decoding은 다르다. 이번 작업은 reduced/public/cloud 제출을 하지 않았다.

Shadow/SEAL 0행과 미개봉 표시는 허용된 반환 행, 캐시, 훈련·평가·특징 분석 및 guard 상태 기준이다. 원래 raw reader는 ID 조건을 Arrow scanner에 전달하고 반환된 ID를 확인했다. Parquet 내부 page/row-group decoder를 별도로 계측하지 않았으므로 내부 디코딩까지 0이었다는 독립 검증을 주장하지 않는다. 이 더 강한 경계 요구와 분석에 사용된 행의 검사는 구별해야 한다.

새 NEXT5 role이 없어 새 raw→fit 재현은 해당하지 않는다. 기존 역할의 RP batch/stream parity와 raw-free 복사본의 새 프로세스 검증을 새 raw 재학습이라고 부르지 않는다. 최종 감사의 관측 시간과 실측 범위는 FINAL_AUDIT_NEXT5.json을 따른다.

실제 API 검사 두 건은 원래 core import의 Numba 캐시 디렉터리 생성 시도를 차단하고 추론 전에 실패했다. 뒤따를 미실행 다섯 건을 보존하고, 기존 read-only bootstrap을 먼저 적용하는 별도 진입점으로 7개 검사를 새 경로에서 수행했다. 원래 수학·모델·descriptor·prefix·쓰기 차단 검사는 보존했다. 원래 실패와 미실행 상태는 quarantine으로 남기고 성공 영수증을 연결했다. 작업 관리자 일시 정지 구간은 연구 시간으로 소급 인정하지 않았다.
'''
    docs['NEXT_SESSION_PLAN_NEXT5.md'] = '''# 다음 세션의 세 가지 연구 병목

1. 분산 감소 신호가 H/ARCH/heavy-tail 배경 변화에 따라 무너지는 원인을 우선한다. 이번 고정 feature와 prediction 방향 진단을 읽고, 특성의 반응과 학습기의 점수 반응을 구분한다. 다음 세션의 새로운 구조 가설은 결과를 보기 전에 등록하며, 이번 S/W/X의 seed·threshold·weight 미세 탐색을 다시 열지 않는다.

2. 반복 이상치 이후의 잔류 신호, long-null 오경보, 늦게 발생한 진짜 변화 검출을 함께 다룬다. 진폭 감소만으로 회복을 주장하지 않는다. 이번 97개 artifact의 fold 차이, 고정 calibration horizon, censoring과 support를 출발 자료로 사용한다. P3의 ungated EWMA와 구별되는 confirmation-only 상태, E1의 누적형과 구별되는 제한된 evidence horizon의 가설은 다음 세션에서 새 검증 계획 아래 사전등록해야 한다. 이번 결과를 그 미실행 구조의 반증으로 사용하지 않는다.

3. 독립적인 신호와 작은 평균 증가분을 구별할 검증력을 설계한다. 기존 Shadow의 과거 DEV 이력을 유지하며 새로운 독립 데이터로 취급하지 않는다. 향후 parent 선택/feature 설계까지 포함한 검증 절차를 먼저 정하고, 같은 데이터 재분할 결과만으로 일반화를 선언하지 않는다.

현재 BL80 primary와 D2 SCALE 보조 역할을 유지한다. NEXT5 finalist는 0개, 새 역할은 없다. LOCAL SEAL 4개와 reduced/public/cloud 금지 상태를 보존한다. 실제 개봉이나 제출을 이 계획에서 실행하지 않는다. 전체 ZIP의 source manifest와 마지막 Desktop 영수증을 먼저 확인한다.
'''
    docs['SEAL_OPEN_RECOMMENDATION_NEXT5.md'] = '# LOCAL SEAL 권고\n\nNO. 새 PRIMARY 승격 기준 통과나 Shadow 확인이 없다. 작은 winner gap과 반복 DEV 선택 위험이 남아 있다. A/B/C/D 500개씩 총 2000개는 모두 미개봉으로 유지했다.\n'
    docs['PUBLIC_SUBMISSION_RECOMMENDATION_NEXT5.md'] = '# Public/cloud 제출 권고\n\n이번 NEXT5에서는 제출하지 않았다. 사용자 금지 조건에 따라 reduced 새 사용 0, public 0, cloud 0이다. 새 NEXT5 role이나 generation을 선언하지 않았으며 기존 primary를 유지한다.\n'
    for name, text in docs.items():
        with (ROOT / name).open('x', encoding='utf-8', newline='') as stream:
            stream.write(text)
        published[name] = digest(ROOT / name)
    answers = [int(v) for v in re.findall(r'^(\d+)\. \*\*', docs['00_GPT_READ_FIRST_NEXT5.md'], re.MULTILINE)]
    assert answers == list(range(1, 48))
    from src.next5.handoff_payload import REQUIRED
    missing = [name for name in REQUIRED if name != 'FINAL_AUDIT_NEXT5.json' and not (ROOT / name).is_file()]
    assert not missing, missing
    for name in ('EXPERIMENT_LEDGER.csv', 'DEV_CANDIDATES_NEXT5.json', 'FINALIST_MANIFEST_NEXT5.json', 'SHADOW_RESULTS_NEXT5.csv'):
        published[name] = digest(ROOT / name)
    tests, boundary, statistics = (context[k] for k in ('tests', 'boundary', 'statistics'))
    checks = {
        'observed_research_at_least_36000_seconds': context['timing']['research_minimum_pass'],
        'separate_observed_audit_at_least_3000_seconds': context['timing']['separate_audit_minimum_pass'],
        'all_required_final_audit_jobs_completed': True,
        'full_suite_without_failures_errors_or_skips': tests['failures'] == tests['errors'] == tests['skipped'] == 0,
        'all_295_primitive_fits_reconciled': context['fits']['primitive_research_learner_fits'] == 295,
        '2470_PRIMARY_fold_scopes_reproduced': statistics['candidate_fold_scopes'] == 2470,
        '95_actual_official_score_comparisons': statistics['official_full_fold_comparisons'] == 95,
        'all_original_bootstrap_draw_values_reproduced': statistics['all_original_draw_arrays_bitwise_equal'],
        'fold_summary_targets_disambiguated_without_gate_changes': context['fold_definitions']['source_statistics_preserved'],
        'all_97_fitted_synthetic_artifacts_completed': context['stress']['fitted_artifacts'] == 97,
        'all_3536_synthetic_cohort_hashes_verified': context['synthetic_bytes']['cohort_files'] == 3536,
        'all_21_fixed_ALT_H_uncertainty_results': len(context['stability']) == 21,
        'all_15_P_late_break_artifacts': len(context['late']) == 15,
        'original_four_null_cohorts_energy_identity_verified': context['null_drift']['scenarios'] == 4 and context['null_drift']['feature_bank_innovation_channels_exactly_equal'],
        'causal_current_point_APIs_verified': all(r['prefix_suffix_and_final_length_invariance'] for r in context['apis'].values()),
        'read_only_import_recovery_complete': context['import_repair']['successful_complete_replacement_API_jobs'] == 7,
        'native_metadata_publication_recovery_verified': context['native_review_repair']['successful_recovery_verified'] and context['native_review_repair']['all_prior_scientific_row_values_and_best_result_preserved'],
        'all_preserved_failure_resolutions_complete': context['graph']['all_preserved_failures_resolved'],
        'retained_actual_entries_match_batch_oracle': all(r['batch_and_actual_API_bitwise_equal'] for r in context['retained']['roles'].values()),
        '270M_actual_continuous_updates_and_repeated_fingerprints': context['latency']['actual_total_continuous_points'] == 270000000 and context['latency']['all_repeat_fingerprints_identical'],
        'raw_free_copied_payload_replay': context['portable']['status'] == 'PASS',
        'raw_models_environment_and_old_deliveries_verified': boundary['status'] == 'PASS',
        'staged_changes_preserved': boundary['staged_patch_byte_equal'],
        'old_session_states_preserved': boundary['prior_session_states_byte_equal'],
        'original_ledger_bytes_preserved': context['ledger']['original_bytes_preserved_as_exact_prefix'],
        'all_four_seals_unopened': boundary['all_seals_unopened'],
        'Shadow_unopened_without_invented_scores': boundary['Shadow_status']['status'] == 'CLOSED' and boundary['Shadow_status']['evaluations_completed'] == 0,
        'no_forced_new_role': not context['selection']['new_NEXT5_roles'],
        'all_47_Korean_answers_present': answers == list(range(1, 48)),
        'all_required_pre_archive_reports_present': not missing,
    }
    assert all(value is True for value in checks.values())
    write_json('FINAL_AUDIT_NEXT5.json', {'status': 'PASS', **common, 'checks': checks,
        'observed_timing': context['timing'], 'checks_completed_utc': utc_now(),
        'audit_within_60_minutes_at_publication': require_handoff_time()['audit_within_60_minutes_at_check'],
        'tests': tests, 'fit_count': 295, 'source_preservation': boundary,
        'statistical_definition_audit': context['fold_definitions'],
        'design_scope_limitations': [
            'P3 compares peak and ungated slow EWMA; confirmation-only state was not implemented',
            'E1 remains prospectively registered cumulative evidence; recent-age restriction is applied in E2',
            'X2 uses H-relative PIT rather than a new standalone static H-context predictor'],
        'selected_roles': context['selection']['roles'], 'new_NEXT5_roles': [],
        'fresh_new_role_raw_training': 'NOT_REQUIRED_NO_NEW_NEXT5_ROLE',
        'evidence_sha256': evidence, 'published_files_sha256_excluding_this_audit': dict(published),
        'source_manifest_and_archive_verification': 'Generated and verified by the gated package step; external Desktop delivery receipt is authoritative for final archive hashes.',
        'source_sha256': digest(Path(__file__))})
    atomic_json(out / 'RESULTS.json', {'created_utc': utc_now(), 'status': 'PASS',
        'files_sha256': published, 'evidence_sha256': evidence, 'Korean_answers': 47,
        'source_sha256': digest(Path(__file__)), 'research_and_audit_time_gates_passed': True,
        'Desktop_archive_created_by_this_script': False}, immutable=True)
    progress('NEXT5 final Korean evidence published', '47 answers, full scalar reports, original primary retained and unopened guards', phase='DELIVERY')
    print('NEXT5 FINAL REPORTS PUBLISHED', len(published), 'files; 47 Korean answers', flush=True)


if __name__ == '__main__':
    main()
