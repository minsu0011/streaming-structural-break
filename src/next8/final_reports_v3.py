"""Final handoff v3: completed-work clock, actual scalar census, guarded test v3."""
import hashlib
import json
import math
import sys
from src.next8.common import ROOT, BASE, atomic, read_json, digest, utc_now, union_seconds
from src.next8.research_reports import table, csv_rows, number, interval, verified
from src.next8.delivery_payload import REQUIRED_REPORTS


def merge_intervals(intervals):
    result = []
    for start, stop in sorted(intervals):
        assert math.isfinite(start) and math.isfinite(stop) and stop >= start
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], stop)
        else:
            result.append([start, stop])
    return result


def time_contract(state, admission, *, minimum_research=36000., minimum_audit=3600.):
    research = merge_intervals(state['intervals'])
    audit = merge_intervals(state['audit_intervals'])
    assert admission['status'] == 'PASS' and admission['productive_seconds_observed'] >= minimum_research
    admitted = admission['admitted_epoch']
    assert all(stop <= admitted for _, stop in research), 'Research may not continue accumulating after audit admission'
    assert all(start >= admitted for start, _ in audit), 'Audit may not predate admission'
    productive_seconds, audit_seconds = union_seconds(research), union_seconds(audit)
    assert productive_seconds >= minimum_research and audit_seconds >= minimum_audit
    return {'status': 'PASS', 'productive_seconds_observed': productive_seconds,
            'separate_audit_seconds_observed': audit_seconds,
            'research_minimum_seconds': minimum_research, 'separate_audit_minimum_seconds': minimum_audit,
            'productive_hours': productive_seconds / 3600., 'audit_hours': audit_seconds / 3600.,
            'research_interval_union': research, 'audit_interval_union': audit,
            'research_audit_do_not_overlap': True, 'parallel_workers_not_summed': True,
            'idle_polling_sleep_and_unobserved_gaps_not_credited': True,
            'clock_source': 'Existing scheduler: observed child CPU progress, completed work and finite controller work leases',
            'controller_work_lease_max_seconds': 150, 'observation_interval_credit_cap_seconds': 60,
            'accounting_limits': 'Conservative application work-evidence clock, not a hardware-level human-attention measurement',
            'as_of_epoch': state['last_epoch'], 'audit_admitted_epoch': admitted}


def final_model_decisions(models, latency):
    records = []
    for comparison in models['comparisons']:
        if comparison['reference'] != 'M0' or comparison['candidate'] not in ('M1', 'M2', 'M3', 'M4', 'M5', 'M6'):
            continue
        method = comparison['candidate']
        runtime = next(r for r in latency['rows'] if not r['early_only'] and r['H_bin'] == 'ALL' and r['method'] == method)
        complete_runtime = (runtime['status'] == 'MEASURED' and runtime['measured_series'] == runtime['available_series'])
        runtime_pass = complete_runtime and runtime['runtime_ratio_gate'] is True
        numeric = comparison['historical_numeric_promotion_gate'] is True
        eligible = numeric and runtime_pass
        records.append({**comparison, 'historical_numeric_promotion_gate': numeric,
            'runtime_measurement_complete_for_the_fixed_sample': complete_runtime,
            'runtime_ratio_gate': runtime['runtime_ratio_gate'],
            'runtime_evidence_sufficient_for_promotion': runtime_pass,
            'causality_lineage_determinism_core_reproduction_passed': True,
            'final_promotion_criteria_satisfied': eligible, 'promotion_final': eligible,
            'promotion_status': 'CRITERIA_SATISFIED_ROLE_ASSIGNMENT_NOT_EXECUTED' if eligible else 'NO_PROMOTION',
            'new_primary_role_assigned': False,
            'final_publication_ZIP_reproduction_record_is_external': True})
    assert len(records) == 6
    return records


def center_note(normal, early):
    return ('자기 이력 해석 보완: D1이 D0(현재값 유지)보다 좋아도 독립 잡음의 중심 회귀를 배제하지 못한다. '
        '같은 동결 예측을 H 중심 F[t+h]=0과 추가 비교했다. Holm36을 통과한 D1 MSE 증분은 '
        f"정상 {normal['supported_self_history_increment_feature_horizons']}/36, "
        f"early {early['supported_self_history_increment_feature_horizons']}/36 feature×horizon이다. "
        '0은 H40 중앙값이며 최적 조건부 평균이라는 보장은 없다. 원래 D0 대비 판정과 B/C 판정을 바꾸지 않았다. '
        'SELF_HISTORY_CENTER_CONTROL_NEXT8.csv와 SELF_HISTORY_BASELINE_LIMIT_KO.md를 함께 읽는다.')


def next_plan(science, decisions, normal_center):
    eligible = [r['candidate'] for r in decisions if r['final_promotion_criteria_satisfied']]
    return f'''# NEXT8 이후 연구 계획

현재 판정은 `{science['scientific_status_subject_to_final_integrity_audit']}`다. 원래 BL80/D2/J102 파일과 역할을 보존했다. 새 승격 기준 충족 후보는 {json.dumps(eligible, ensure_ascii=False)}이며 역할 파일을 자동 변경하지 않았다. 다음 세션은 별도 사용자 지시를 받은 뒤 새로운 run으로 시작한다.

1. 먼저 이번 최종 감사, TRAIN 탈락 사유, 같은 입력 대조와 자기 이력 중심값 감사를 읽는다. 빈 TRAIN 집합에서 M1–M5가 절편이 된 경우에는 이 선택 절차의 한계로 해석한다. 이번 outer 결과로 피처 목록·상태 경계·가중치·부모를 다시 맞추지 않는다.
2. 다음 연구의 우선 가설은 방향 반전의 유무와 검정력 부족을 분리하는 것이다. 이론상 식별 가능한 효과 크기, 독립 시계열 수, 시간 초반·짧은 H의 양성/음성 지원을 prospective simulation으로 사전등록한다. 결과를 본 뒤 이번 95% CI·실용적 반전 제외 기준을 낮추지 않는다.
3. 자기 과거 증분은 지속 예측뿐 아니라 TRAIN에서 고정한 H-only/무조건 평균과 비교한다. 이번 중심값 감사에서 정상 {normal_center['supported_self_history_increment_feature_horizons']}/36이 통과했지만, 중앙값0은 MSE 최적 평균이 아니다. 새 평균 기준을 검증하려면 별도 사전등록과 학습 제외 경로가 필요하다. 같은 원시 충격의 여러 필터 응답을 독립 증거로 세지 않는다.
4. 선택 절차 때문에 피처 집합이 비었다면, 다음의 같은 입력 일반 모델 비교는 소수의 사전 고정 원시 피처 메뉴로 별도로 설계할 수 있다. 이는 이번 M2/M3의 사후 대체 모델이 아니라 새로운 adaptive 개발 가설이다. 후보 수를 늘려 이번 실패를 지우지 않는다.
5. 조건부 방향(B)이 재현된 경우에만 그 인증 단위와 실제 실행 헤드의 동일성을 확인한다. 움직임 상태(A), 조건부 정답 방향(B), 최종 TS-AUC(C)의 경로가 다르면 JOINT 인증을 옮기지 않는다. 기권·UNKNOWN과 관측 전 방향 선택 범위를 그대로 유지한다.
6. 새로운 split으로 Shadow/Seal을 RP에 되돌리지 않는다. Shadow 1,600 및 Seal 2,000, reduced/public/cloud 값은 계속 미개봉이다. 이번 RP는 과거부터 adaptive하게 사용한 개발 자료이므로 독립 hidden 확인이라고 부르지 않는다.

중단된 동일 run의 재개는 원래 프로젝트에서 `runtime/TASK_QUEUE.json`, 작업별 TERMINAL/부분 결과, `CONTROLLER_CHECKPOINT_NEXT8.json`과 scheduler 소유 PID를 먼저 확인한다. 같은 살아 있는 scheduler를 중복 실행하지 않는다. 정지 요청이 없고 소유 프로세스가 끝난 미완료 run에서만 ` .\\.venv\\Scripts\\python.exe -B scripts\\next8\\run_next8_autonomous.py `로 남은 등록 작업을 재개한다. 완료한 연구·score를 다시 실행하지 않는다. 초기화 스크립트는 이번 최초 생성용이며 기존 locator/source를 덮어쓰는 재개 도구가 아니다.

raw 없는 ZIP의 실행 재현은 `REPRODUCE_NEXT8_KO.md`를 따른다. 전체 재학습에는 원래 RP 캐시와 보호된 원본/분할/원래 fold별 reference가 필요하며 이번 ZIP에는 넣지 않았다.
'''


def reproduction_document(closure):
    dependencies = '\n'.join(name + '==' + version for name, version in closure['package_versions'].items())
    return f'''# NEXT8 raw 없는 실행 재현

이 패키지는 실제 저장한 정상/early 10개 절차의 M0–M6와 원래 canonical BL80/D2/J102 공개 infer 경로를 포함한다. 원본 RP와 real point OOF는 포함하지 않는다. synthetic 669개 경로, scalar 예측 685,056개를 기존 fixture와 비교한다. 각 연구 경로에서 scalar/chunk, 73시점 pickle 복원, 미래 suffix 변경 후 prefix 일치를 검사한다. canonical 9개 경로는 기존 NEXT7 fixture와 bitwise 비교한다.

ZIP을 새 폴더에 풀고 그 폴더를 현재 작업 경로로 둔다. 이번 검증은 원래 프로젝트의 기존 `.venv` 인터프리터만 재사용했으며 원래 source/model/data/cache의 누락 파일 fallback은 차단했다. 출력 경로는 존재하지 않는 새 폴더여야 한다.

```powershell
$next8Python = 'C:\\Users\\minsu\\Documents\\Structural Break\\.venv\\Scripts\\python.exe'
& $next8Python -B scripts/next8/reproduce_next8_portable.py --output-directory 'C:\\Users\\minsu\\Desktop\\NEXT8_REPLAY_NEW' --deny-original-root 'C:\\Users\\minsu\\Documents\\Structural Break'
```

이미 존재하는 출력 폴더는 덮어쓰지 않는다. `REPRODUCTION_DONE.json`의 PASS와 input/model/source hash, 경로 수, 원본 접근 차단을 확인한다. 예제 출력 폴더가 이미 있으면 다른 새 이름을 쓴다. 과거 fixture 생성 옵션 `--create-fixture`는 검증 명령이 아니며 배포된 fixture를 새 값으로 대체하지 않는다.

실제 검증 Python: `{sys.version.splitlines()[0]}`. 아래는 기존 환경에서 읽은 버전이며 이 작업에서 새 설치를 하지 않았다. 새로운 OS/라이브러리/하드웨어의 bitwise 재현까지 보장한 것은 아니다.

```text
{dependencies}
```

전체 연구 코드와 테스트는 보존했다. 원래 프로젝트에서 기존 및 NEXT8 전체 테스트를 실제 실행한 결과는 FINAL_AUDIT/DETERMINISM 기록을 따른다. raw 없는 ZIP에서 기존 전체 테스트를 그대로 실행하면 과거 실험 자산을 요구하는 테스트가 있으므로 전체 suite의 독립 재학습 재현이라고 주장하지 않는다.

원본 RP를 사용한 전체 재학습은 별개다. RP6400 고정 cache, 원래 group/fold metadata, BL80의 original outer/inner 제외 학습 모델, 원래 feature matrix와 source authority, hash가 필요한 입력이다. NEXT8_DATA_ACCESS_AUDIT와 BASELINE_PROVENANCE_NEXT8.json이 그 출처를 기록한다. 원본값 및 큰 feature/OOF 캐시는 제외했다. 등록된 각 작업의 script·arguments·gates·hash는 REGISTRATION_MANIFEST_NEXT8.json에 보존한다. 기존 run에서 initializer를 반복 실행하거나 완료 score를 덮어쓰지 않는다. 새 재학습에는 새 namespace 설정이 필요하며 이번 세션에서 별도의 두 번째 전면 재학습을 했다고 주장하지 않는다.

첫 core ZIP을 실제 추출·실행한 증거는 REPRODUCTION_NEXT8.json에 포함된다. 최종 전체 ZIP도 별도로 새로 추출해 같은 모델 재현을 실행한다. 그 ZIP 자신의 SHA와 최종 replay 결과는 자기 포함 해시 순환을 피하려고 ZIP 외부 DELIVERY_RECEIPT 및 바탕화면 20파일 묶음의 18번 JSON에 기록한다.
'''


def build_final_reports():
    out = BASE / 'final_reports'
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), 'Existing final publication retained; explicit repair required'
    core = read_json(BASE / 'CORE_ARCHIVE_REPRODUCTION_NEXT8.json')
    assert core['status'] == 'PASS' and core['all_frozen_synthetic_vector_bits_equal']
    assert digest(ROOT / core['archive_path']) == core['archive_sha256']
    assert digest(ROOT / core['reproduction_receipt_path']) == core['reproduction_receipt_sha256']
    visual = read_json(BASE / 'FIGURES/VIEW_REVIEW_NEXT8.json')
    assert visual['status'] == 'PASS' and visual['figure_count'] == 8
    for name, sha in visual['files_sha256'].items():
        assert digest(BASE / 'FIGURES' / name) == sha
    science = verified('evidence_synthesis', 'SCIENTIFIC_SYNTHESIS.json')
    models = verified('model_summary', 'NORMAL.json')
    latency = verified('execution_summary', 'LATENCY_NEXT8.json')
    fits = verified('execution_summary', 'FIT_ACCOUNTING_NEXT8.json')
    center = [verified('center_control', mode + '.json') for mode in ('NORMAL', 'EARLY')]
    computation = verified('computation_audit', 'COMPUTATION_AUDIT_NEXT8.json')
    leakage = verified('computation_audit', 'LEAKAGE_AUDIT_NEXT8.json')
    determinism = verified('computation_audit', 'DETERMINISM_COMPUTATION_NEXT8.json')
    assert computation['status'] == leakage['status'] == determinism['status'] == 'PASS'
    scalar = read_json(BASE / 'FULL_SCALAR_CAUSAL_AUDIT_NEXT8.json')
    assert scalar['status'] == 'PASS' and scalar['series'] == 6400 and scalar['paths'] == 89600
    assert digest(BASE / scalar['case_table']) == scalar['case_table_sha256']
    scalar_proof = {'receipt': 'FULL_SCALAR_CAUSAL_AUDIT_NEXT8.json', 'sha256': digest(BASE / 'FULL_SCALAR_CAUSAL_AUDIT_NEXT8.json'), 'summary': scalar}
    leakage = {**leakage, 'whole_RP_actual_scalar_causality_census': scalar_proof}
    determinism = {**determinism, 'whole_RP_actual_scalar_causality_census': scalar_proof}
    compatibility = read_json(BASE / 'audit/FINAL_TEST_COMPATIBILITY_V3_NEXT8.json')
    assert compatibility['status'] == 'COMPLETE' and len(compatibility['test_only_adaptations']) == 8
    closure_path = BASE / 'runtime/FINAL_SOURCE_CLOSURE_NEXT8.json'
    closure = read_json(closure_path)
    for name, sha in closure['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    admission = read_json(BASE / 'runtime/FINAL_AUDIT_ADMISSION.json')
    assert admission['session_sha256'] == digest(BASE / 'SESSION_NEXT8.json')
    from src.next8.strict_work_time import final_work_time_contract
    clock = final_work_time_contract(admission)
    decisions = final_model_decisions(models, latency)
    eligible = [r['candidate'] for r in decisions if r['final_promotion_criteria_satisfied']]
    read_note = center_note(*center)
    original = read_json(BASE / 'science_reports/DONE.json')
    for name, sha in original['files_sha256'].items():
        source = BASE / 'science_reports' / name
        assert digest(source) == sha
        if name in ('00_READ_FIRST_NEXT8_KO.md', 'FEATURE_EXPLANATION_CARDS_KO.md', 'MODEL_COMPARISON_NEXT8.csv'):
            continue
        atomic(out / name, source.read_text(encoding='utf-8-sig'))
    model_rows = csv_rows(BASE / 'science_reports/MODEL_COMPARISON_NEXT8.csv')
    for row in model_rows:
        if row['early_only'] == 'False' and row['reference'] == 'M0' and row['candidate'] in {r['candidate'] for r in decisions}:
            row.update(next(r for r in decisions if r['candidate'] == row['candidate']))
    table(out / 'MODEL_COMPARISON_NEXT8.csv', model_rows)
    read_first = (BASE / 'science_reports/00_READ_FIRST_NEXT8_KO.md').read_text(encoding='utf-8-sig')
    prefix = (f"최종 계산·계보·보호 파일·전체 테스트·core ZIP 재현 감사 PASS. "
              f"관측 연구 {number(clock['productive_hours'])}시간, 별도 감사 {number(clock['audit_hours'])}시간. "
              f"승격 기준 충족 후보 {json.dumps(eligible, ensure_ascii=False)}. 기존 canonical 역할과 파일은 유지했다.\n\n" + read_note + '\n\n')
    read_first = read_first.replace('이 문서는 연구 결과 조립본이다.', '동결 연구 본문은 science_reports에 보존했으며 이 전달본에 최종 감사와 중심값 대조를 추가했다.')
    read_first = read_first.replace('별도 최종 실행·시간·재현 감사가 끝나기 전 승격하지 않는다.', '최종 승격 여부는 MODEL_COMPARISON_NEXT8.csv의 final_promotion_criteria_satisfied를 따른다.')
    atomic(out / '00_READ_FIRST_NEXT8_KO.md', prefix + read_first)
    cards = (BASE / 'science_reports/FEATURE_EXPLANATION_CARDS_KO.md').read_text(encoding='utf-8-sig')
    for feature in science['features']:
        j = feature['feature']
        rows = [r for r in center[0]['rows'] if r['feature'] == j and r['primary_claim']]
        if not rows:
            continue
        title = f"## F{j:03d} — {feature['name']}"
        extra = '; '.join(f"h={r['horizon']}: H 중심 MSE−D1 MSE {interval(r)}, "
                          f"Holm36 지지={r['self_history_increment_beyond_fixed_H_center']}" for r in rows)
        cards = cards.replace(title, title + '\n\n추가 자기 이력 기준: ' + extra + '. 중앙값 기준을 넘는 결과와 모든 H-only 평균 기준을 넘는 결과는 같지 않다.')
    atomic(out / 'FEATURE_EXPLANATION_CARDS_KO.md', read_note + '\n\n' + cards)
    for name in ('NEXT8_PROTOCOL_LOCK.json', 'NEXT8_FEATURE_BUDGET.json', 'NEXT8_MODEL_MENU_LOCK.json'):
        atomic(out / name, read_json(BASE / name))
    for name in ('SELF_HISTORY_CENTER_CONTROL_NEXT8.csv', 'SELF_HISTORY_BASELINE_LIMIT_KO.md'):
        receipt = read_json(BASE / 'center_control/DONE.json')
        assert digest(BASE / 'center_control' / name) == receipt['files_sha256'][name]
        atomic(out / name, (BASE / 'center_control' / name).read_text(encoding='utf-8-sig'))
    reviews = [read_json(p) for p in sorted((BASE / 'bottleneck_reviews').glob('REVIEW_*.json'))]
    table(out / 'BOTTLENECK_HISTORY_NEXT8.csv', reviews)
    jobs = [read_json(p) for p in (BASE / 'ledger').glob('*.json')]
    table(out / 'EXPERIMENT_LEDGER_NEXT8.csv', sorted(jobs, key=lambda r: r['started_epoch']))
    queue = read_json(BASE / 'runtime/TASK_QUEUE.json')
    tasks = []
    for item in queue:
        path = BASE / 'runtime/queue' / item['id'] / 'TERMINAL.json'
        tasks.append({'registration': item, 'terminal': read_json(path) if path.exists() else None})
    atomic(out / 'REGISTRATION_MANIFEST_NEXT8.json', {'status': 'FROZEN', 'timestamp': utc_now(), 'tasks': tasks,
        'publication_task_may_be_currently_running': True, 'queue_tasks_not_independent_scientific_candidate_count': True})
    atomic(out / 'TIME_ACCOUNTING_NEXT8.json', clock)
    atomic(out / 'FIT_ACCOUNTING_NEXT8.json', fits)
    atomic(out / 'NEXT8_DATA_ACCESS_AUDIT.json', {'status': 'PASS', 'timestamp': utc_now(),
        'preflight': read_json(BASE / 'PREFLIGHT_NEXT8.json'),
        'final_data_boundary': verified('final_integrity', 'DATA_BOUNDARY_INTEGRITY.json'),
        'training_lineage_audit_sha256': digest(BASE / 'TRAINING_LINEAGE_AUDIT_NEXT8.json'),
        'new_Shadow_Seal_reduced_public_cloud_value_usage': 0,
        'only_RP6400_observations_used_for_real_feature_fit_prediction_score_bootstrap': True,
        'membership_metadata_and_raw_byte_hashes_are_not_observation_value_analysis': True,
        'application_guard_not_OS_information_flow_proof': True})
    atomic(out / 'LEAKAGE_AUDIT_NEXT8.json', leakage)
    atomic(out / 'LATENCY_NEXT8.json', latency)
    atomic(out / 'REPRODUCTION_NEXT8.json', {'status': 'PASS', 'core_archive_actual_replay': core,
        'final_publication_archive_is_replayed_separately_after_this_report_is_packed': True,
        'final_ZIP_own_hash_and_replay_proof_are_external_DELIVERY_RECEIPT_and_Desktop_file18': True,
        'original_RP_retraining_is_a_distinct_unperformed_second_full_run': True})
    atomic(out / 'DETERMINISM_NEXT8.json', {**determinism, 'portable_actual_core_replay': core,
        'portable_new_process_and_public_entry_bitwise_proof_pending_separate_archive_receipt': False,
        'final_publication_archive_replay_receipt_external': True})
    final = {'status': 'PASS', 'timestamp': utc_now(), 'scientific_status': science['scientific_status_subject_to_final_integrity_audit'],
        'scope': 'Final computational, scientific-report, source, core-reproduction and observed-time audit; final ZIP self-check is external',
        'time_contract': clock, 'computational_audit': computation, 'core_reproduction': core,
        'figure_visual_review_sha256': digest(BASE / 'FIGURES/VIEW_REVIEW_NEXT8.json'),
        'model_decisions': decisions, 'promotion_criteria_satisfied_candidates': eligible,
        'canonical_BL80_D2_J102_roles_changed': False, 'Shadow_Seal_still_closed': True,
        'final_publication_ZIP_must_pass_actual_fresh_replay_before_Desktop_delivery': True,
        'source_closure_sha256': digest(closure_path), 'A_B_C_claims_kept_separate': True,
        'D0_persistence_vs_H_center_limitation_included': True, 'adaptive_RP_not_independent_hidden_confirmation': True,
        'whole_RP_actual_scalar_causality_census': scalar_proof,
        'final_test_v3_compatibility_sha256': digest(BASE / 'audit/FINAL_TEST_COMPATIBILITY_V3_NEXT8.json'),
        'original_final_test_failure_evidence_preserved': 'repairs/FINAL_TEST_V2_FAILED'}
    atomic(out / 'FINAL_AUDIT_NEXT8.json', final)
    handoff = [
        '# NEXT8 GPT 전달', '',
        f"연구 `{BASE.name}`. 판정 **{final['scientific_status']}**. 원래 BL80 RP mean TS-AUC는 0.6203652100698587이다. "
        f"최종 승격 기준 충족 후보: {json.dumps(eligible, ensure_ascii=False)}. 기존 BL80/D2/J102 역할과 byte authority를 보존했다.", '',
        f"실질 연구 {number(clock['productive_seconds_observed'])}초, 이후 별도 감사 {number(clock['separate_audit_seconds_observed'])}초. "
        '완료된 실제 계산·검증 작업 구간의 합집합이다. 종료 알림·controller lease·작업 사이의 대기와 기록되지 않은 controller 구현 시간은 보수적으로 제외하고 worker 시간을 중복 합산하지 않았다. ZIP 마지막 생성 시간까지 소급해 이 숫자를 부풀리지 않는다.', '',
        'A–J 질문과 exact 모델 수치는 00_READ_FIRST_NEXT8_KO.md에 모두 있다. 카드의 14개 항목은 각 피처의 실제 역할·조건·대조·지원·fallback을 설명한다. '
        '사전 고정 상세 12개에 포함됐다는 사실을 가변 후보 진입이나 인증이라고 부르지 않는다.', '',
        '검증 구조: 기존 RP6400 grouped outer5, outer TRAIN 내부 inner3, meta 입력의 상태·선택 절차까지 제외 학습. '
        '새 정규화는 H60 fit→같은 엔진 H40 calibration→Online 연속이다. canonical full-H 정규화는 바꾸지 않았다. '
        'early는 학습 정답·보조 target·모든 mapping을 t≤199로 제한하고 다른 시계열 t≥400을 평가했다.', '',
        f"실제 저장 receipt 기준 새 estimator fit 호출 수는 {fits['total_recorded_new_estimator_training_calls']}이다. "
        '이는 screening 수천 회, 회귀·분류·상수·전이 확률 추정 등을 포함하는 계산량이며 독립 과학 후보 수가 아니다. '
        '최종 classifier 가족은 M1–M6이고 M6 가중치는 학습하지 않았다. alias 및 미실행은 별도 기록한다.', '',
        'outer 공개 후 방법 진단의 합성 estimator fit 6,852회는 별도 집계했다. 고정 점수 rank-CI 진단은 fit이 없다. '
        '이 수를 실제 RP fit이나 독립 확인 실험으로 합산하지 않는다. 실제 H 엔진의 세 대조 실험과 한계는 '
        'RAW_ENGINE_GATE_RESULTS_NEXT8_KO.md 및 bottleneck_reviews/REVIEW_FINAL_STATISTICAL_CONTROLS.json에 있다.', '',
        read_note, '',
        '빈 TRAIN 집합이면 M1–M5는 절편으로 축소된다. 이 결과로 일반적인 tabular/graph 모델의 표현력을 부정하지 않는다. '
        '넓은 CI가 방향 반전을 배제하지 못한 경우와 실제 방향 반전 지지를 구분한다. DYNAMICS_ONLY도 B/C 비유의성만으로 부여하지 않았다.', '',
        f"전체 RP replay {computation['unique_RP_series']}개 시계열, {computation['online_points']}개 관측. "
        f"전체 테스트: {json.dumps(computation['tests'], ensure_ascii=False)}. "
        'all153 피처와 원래 full-H J/C 피처는 bitwise 비교하고, saved float32 헤드는 eps와 exact 여부를 별도로 기록했다. '
        '실제 scalar API도 RP6400 전체에서 정상·early M0–M6의 89,600개 경로, 원래 scalar 예측 44,844,198개를 검증했다. '
        'pickle 복원·분할 입력·미래 구간 변경 후 prefix 일치를 포함한다. 반복 재생을 독립 예측으로 중복 합산하지 않았다. '
        'FULL_SCALAR_CAUSAL_AUDIT_NEXT8.json과 모든 사례를 보존한 CSV를 참조한다.', '',
        'raw 없는 core ZIP에서 실제 frozen 669개 경로의 685,056개 예측을 새 프로세스로 재생했다. 최종 전체 ZIP도 별도로 새로 풀어 실행한다. '
        '전체 ZIP 자기 SHA와 최종 replay 증거는 바탕화면 18번 JSON 및 프로젝트 외부 receipt에 있다. 원본 RP를 요구하는 재학습과 혼동하지 않는다.', '',
        '모든 frozen 구현/정책·등록·명시적 수리·검증 증거는 전체 ZIP의 artifacts/next8 아래에 보존한다. '
        '전체 RP 커널 재생에는 v2를 사용했다. 첫 전체 테스트의 환경 오류 8건과 로그/XML은 repairs/FINAL_TEST_V2_FAILED에 보존했다. '
        '테스트 전용 v3는 기존 6개 테스트의 실제 2스레드 요청을 허용하고, 합성 테스트 2개의 저장 기준 폴더를 해당 임시 폴더로 격리한다. '
        '원래 테스트의 수치 assertion과 예측 소스는 유지했고 전체 테스트를 v3로 다시 실행했다. '
        '새로운 사후 아이디어는 NEXT_SESSION_PLAN_NEXT8.md의 다음 연구 가설이며 이번 outer 후보를 다시 맞추지 않는다.', '']
    atomic(out / '01_GPT_HANDOFF_NEXT8.md', '\n\n'.join(handoff))
    atomic(out / 'NEXT_SESSION_PLAN_NEXT8.md', next_plan(science, decisions, center[0]))
    atomic(out / 'REPRODUCE_NEXT8_KO.md', reproduction_document(closure))
    atomic(out / 'REQUIREMENTS_NEXT8_REPLAY.txt', '\n'.join(n + '==' + v for n, v in closure['package_versions'].items()) + '\n')
    atomic(out / 'SOURCE_MANIFEST_NEXT8.json', {'status': 'FROZEN', 'timestamp': utc_now(),
        'source_sha256': closure['source_sha256'], 'package_versions': closure['package_versions'],
        'source_closure_sha256': digest(closure_path),
        'portable_model_dependency_manifest_sha256': digest(BASE / 'PORTABLE_RUNTIME_MANIFEST_NEXT8.json'),
        'final_report_sha256': {p.name: digest(p) for p in out.iterdir() if p.is_file()},
        'source_manifest_self_hash_excluded': True,
        'ZIP_complete_member_hash_manifest_is_PAYLOAD_CONTENT_MANIFEST_NEXT8_json': True})
    assert set(REQUIRED_REPORTS) <= {p.name for p in out.iterdir() if p.is_file()}
    atomic(out / 'DONE.json', {'status': 'COMPLETE', 'timestamp': utc_now(),
        'required_reports': list(REQUIRED_REPORTS),
        'files_sha256': {p.name: digest(p) for p in out.iterdir() if p.is_file()},
        'no_scientific_selection_or_fit_changed': True})
    return out
