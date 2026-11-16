"""Finalize Korean handoff after completed research and actual final audits."""
from pathlib import Path
import sys
import argparse
import csv
import json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now,csv_write

def main(core_tag):
    from src.next9.work_time import snapshot
    from src.next9.queue import identity_alive
    clock=snapshot();assert clock['productive_seconds']>=36000 and clock['audit_seconds']>=3600,clock
    required=['SELF_CADENCE_DONE_NEXT9.json','SAMPLING_PHASE_DONE_NEXT9.json','EAGER_SCHEMA_CACHE_QUALIFICATION_NEXT9.json','INDEPENDENT_TS_AUC_AUDIT_NEXT9.json','FAILED_ATTEMPT_RECONCILIATION_NEXT9.json','FITTED_MODEL_PARAMETER_AUDIT_NEXT9.json','STATE_TRANSPORT_DONE_NEXT9.json','LOGLOSS_SIGN_REPAIR_AUDIT_NEXT9.json','FAST_HEADS_QUALIFICATION_NEXT9.json','SELECTED_ENGINE_QUALIFICATION_NEXT9.json','JOINT_SOLVER_AUDIT_NEXT9.json','EXECUTION_AUDIT_NEXT9.json','LEAKAGE_AUDIT_NEXT9.json','DETERMINISM_NEXT9.json',
        'INTEGRATED_RP_FINAL_AUDIT_NEXT9.json','MOVEMENT_EQUATION_FINAL_AUDIT_NEXT9.json',
        'final_integrity/FINAL_PROTECTED_NEXT9.json','tests/FINAL_TEST_SUITE_NEXT9.json',
        'FIGURES/ALL_TEN_VISUALLY_INSPECTED_NEXT9.json']
    for name in required:assert read_json(BASE/name)['status']=='PASS',name
    reproduction_path=BASE/'delivery'/core_tag/'REPRODUCTION_DONE.json'
    reproduction=read_json(reproduction_path);assert reproduction['status']=='PASS'
    assert digest(ROOT/reproduction['archive_path'])==reproduction['archive_sha256']
    closure=read_json(BASE/'runtime/FINAL_SOURCE_CLOSURE_NEXT9.json')
    for name,sha in closure['source_sha256'].items():assert digest(ROOT/name)==sha,name
    owner=read_json(BASE/'runtime/SCHEDULER_OWNER.json');assert not identity_alive(owner['pid'],owner['process_created'])
    with activity('Finalize Korean results, transparent failures and GPT handoff after actual final audits','FINAL_AUDIT','HANDOFF'):
        science=read_json(BASE/'SCIENTIFIC_RESULTS_NEXT9.json');cert=read_json(BASE/'VARIABLE_CERTIFICATION_NEXT9.json');latency=read_json(BASE/'LATENCY_NEXT9.json')
        assert latency['status'] in ['PASS','FAIL']
        promotion=bool(science['C']['promotion_score_gates']['score_gates_pass'] and latency['status']=='PASS')
        signals=[r for r in cert['certified_rules'] if r['research_signal_science_gate']]
        status='NEXT9_PROMOTION_CANDIDATE' if promotion else ('VARIABLE_ORIENTATION_SUPPORTED_NO_MODEL_GAIN' if cert['certified_variable_count'] else 'NO_MOVEMENT_VARIABLE_FEATURE')
        atomic(BASE/'REPRODUCTION_NEXT9.json',dict(**reproduction,source_receipt_sha256=digest(reproduction_path),
            final_report_archive_replay='The final payload is built after this full raw-free core replay and audited again by reproduce_next9_archive_v10 --final. That archive hash and execution receipt are external to its immutable bytes.'))
        atomic(BASE/'FINAL_TIME_ACCOUNTING_NEXT9.json',dict(status='PASS',timestamp=utc_now(),
            productive_seconds=clock['productive_seconds'],audit_seconds=clock['audit_seconds'],
            minimum_productive_seconds=36000,minimum_separate_audit_seconds=3600,
            completed_intervals=clock['successful_work_intervals'],excluded_counts=clock['excluded_counts'],
            idle_sleep_monitor_parent_wait_excluded=True,overlapping_workers_union_not_sum=True,
            cutoff='Conservative completed-work snapshot before this final report and subsequent final ZIP replay/delivery; later work is additional.',
            admission_sha256=digest(BASE/'runtime/FINAL_AUDIT_ADMISSION_NEXT9.json')))
        history=[]
        for path in sorted((BASE/'bottleneck_reviews').glob('REVIEW_*.json')):
            record=read_json(path);history.append({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in record.items()})
        csv_write('BOTTLENECK_HISTORY_NEXT9.csv',history)
        ledger=[]
        for path in (BASE/'ledger').glob('*.json'):
            record=read_json(path);assert digest(path)==digest(BASE/'runtime/jobs'/path.name)
            ledger.append({**record,'wall_seconds':record['ended_epoch']-record['started_epoch'],'receipt_sha256':digest(path)})
        csv_write('EXPERIMENT_LEDGER_NEXT9.csv',sorted(ledger,key=lambda r:r['started_epoch']))
        record=dict(status='PASS',timestamp=utc_now(),scientific_status=status,
            scientific_hypothesis_success_separate_from_audit_success=True,certified_variable_count=cert['certified_variable_count'],
            certified_cross_edges=science['cross_edge_count'],promotion_candidate=promotion,canonical_roles_changed=False,
            canonical_roles=dict(primary='BL80',robustness='D2',distilled='J102'),
            research_role='MOVEMENT_VARIABLE_SIGNAL' if signals else None,research_signal_features=[r['feature'] for r in signals],
            runtime_status=latency['status'],runtime_failure_prevents_promotion=latency['status']=='FAIL',
            uncertainty='RP was repeatedly used for adaptive development; bootstrap conditions on fitted procedures, not independent hidden confirmation.',
            optimizer_repair_sha256=digest(BASE/'repairs/JOINT_SOLVER_CONTINUATION_REPAIR_NEXT9.json'),
            numerical_repair='Unsigned-label negative logloss was invalid. All6457 supported internal panels reconstructed in float64 with the same predictions and bootstrap; candidate selection independently unchanged and reverified. Corrected outer loss applied before outer orientation execution. Legacy membership and runtime schema repairs separately recorded.',
            namespace_exception='One existing pytest nodeid cache was written earlier; no prior byte restoration claimed. Protected7573files and dirty Git patches verified unchanged.',
            protected_integrity_sha256=digest(BASE/'final_integrity/FINAL_PROTECTED_NEXT9.json'),
            source_closure_sha256=digest(BASE/'runtime/FINAL_SOURCE_CLOSURE_NEXT9.json'),
            time_accounting_sha256=digest(BASE/'FINAL_TIME_ACCOUNTING_NEXT9.json'),
            core_reproduction_sha256=digest(reproduction_path),
            evidence_sha256={name:digest(BASE/name) for name in required},
            exact20_desktop_delivery_pending_final_archive_replay=True,final_delivery_receipt_external=True,
            all_scientific_selection_frozen_before_outer_results=True,SHADOW_ACCESS=0,SEAL_ACCESS=0,
            reduced_public_cloud_new_usage=0,owned_scheduler_stopped=True)
        atomic(BASE/'FINAL_AUDIT_NEXT9.json',record)
        elapsed=f"실제 연구 {clock['productive_seconds']/3600:.3f}시간, 별도 최종 감사 {clock['audit_seconds']/3600:.3f}시간 이상. 대기와 겹친 작업 시간은 중복 집계하지 않았다."
        lead=f"# NEXT9 결과 먼저 읽기\n\n**{status}**\n\n{elapsed}\n\nA 후보 {science['A']['eligible_outerTRAIN_candidates']}개, 인증 가변 피처 {cert['certified_variable_count']}개, cross edge {science['cross_edge_count']}개. Primary는 BL80다. 감사 PASS와 과학 가설의 성공은 구분한다.\n\n"
        lead+='실제 최종 전달은 바탕화면 핵심20개 폴더와 대응 ZIP, 전체 raw-free ZIP으로 구성한다. 전체 ZIP을 새 폴더에서 다시 실행한 확인과 최종 ZIP 해시는 프로젝트의 DESKTOP_DELIVERY_NEXT9.json에 남긴다. ZIP 안에 자기 자신의 해시를 기록했다고 주장하지 않는다.\n\n'
        lead+='예외 기록: 기존 pytest 캐시 1개 갱신, float32 경계 로그손실 계산 오류와 복구, 처음 잘못 구성한 합성 반례 및 인프라 수정을 숨기지 않았다. 기존 보호 파일·모델·Git 변경 사항은 해시 검사를 통과했다.\n\n'
        lead+='중요한 수치 수정: 이전 orientation_selection 내부 로그손실은 uint8 부호 오류로 무효다. orientation_logloss_corrected 자료가 최종 근거이며, 6,457개 지원 패널을 재계산해 선택 불변을 확인했다. TS-AUC·방향성·학습 모델은 변하지 않았다. LOGLOSS_SIGN_REPAIR_AUDIT_NEXT9.json에 전체 검사가 기록돼 있다.\n\n'
        lead+='?? ????? ??: ?? ?? ??? ?????????? FITTED_MODEL_PARAMETER_AUDIT_NEXT9.json, ?? ?? ??? ??? ??? ??? FAILED_ATTEMPT_RECONCILIATION_NEXT9.json? ??. EARLY ?? ??? ?? ??? ?? ?? ??? STATE_TRANSPORT_FINDING_NEXT9_KO.md? ???, ?? ?? ??? ???? ???? ???.\n\n'
        lead+=(BASE/'SCIENTIFIC_READ_FIRST_NEXT9_KO.md').read_text(encoding='utf-8')
        atomic(BASE/'00_READ_FIRST_NEXT9_KO.md',lead)
        with (BASE/'MODEL_COMPARISON_NEXT9.csv').open(encoding='utf-8-sig',newline='') as f:models=list(csv.DictReader(f))
        table='| 학습 | 모델 | TS-AUC | Δ vs BL80 | 95% CI |\n|---|---|---:|---:|---|\n'
        for r in models:table+=f"| {r['mode']} | {r['method']} | {r['mean_TS_AUC']} | {r['mean_delta_vs_BL80']} | [{r['ci_low']}, {r['ci_high']}] |\n"
        handoff=f'''# GPT 전달용 NEXT9 연구 기록

프로젝트: `{ROOT}`
실행: `{BASE.relative_to(ROOT)}`
과학 판정: **{status}**. {elapsed}

## 반드시 유지할 해석

A는 자기 움직임의 보조 예측, B는 같은 현재 F에서 과거 상태에 따른 target 방향 반전, C는 held-out 모델 점수 이득이다. A 성공을 B/C 성공으로 바꾸지 않는다. 후보는 break label 사용 전에153전체의 자기 예측력으로 동결했다. 20개 nested절차, 원래 group5fold, 별도 EARLY 학습을 유지했다.

현재 F[t]는 pre-state에 들어가지 않는다. 마지막16F의 차분·run·slope·reversal로 상태를 정하고, H<160은 UNKNOWN이다. GLR 선택 창 길이는 true break age가 아니다. H 길이·의존성·꼬리는 series 배경이며 within-series switching으로 부르지 않는다.

운영 H-center는 H 정규화 뒤 중앙값0이다. MSE에 최적인 TRAIN 평균 상수와 구분하며 CENTER_DECOMPOSITION_SUMMARY_NEXT9.csv가 그 분해다. 자기 예측력은 ARCH/EWMA/GLR 기억일 수 있고 인과 메커니즘 증거가 아니다.

## 결과

{table}

인증 B={cert['certified_variable_count']}, cross edge={science['cross_edge_count']}. 승격 후보 gate={promotion}; runtime={latency['status']}. canonical primary/robustness/distilled는 BL80/D2/J102 그대로다. 새 연구 signal 피처: {[r['feature'] for r in signals]}.

B는 TRAIN에서 고른 같은 pair/방향이 NORMAL 및 EARLY 각각4/5fold, 시간2개·H영역2개에서 반복되어야 한다. O1 고정 비선형과 state-only를 모두 이겨야 한다. 2,000draw의 p값 해상도를 고려해 각 사전고정 domain 안에서 전체 A×8pair Holm을 적용했다. domain 전체 동시 보장이라고 쓰지 않는다.

M6 대 TABULAR만 항상 동일 입력이다. M6=M5일 때만 M5 대 TABULAR도 동일 입력이다. 비유의는 동등성 증거가 아니다. B가0이라서 M4/M5의 조건부 분기가 비는 것은 전이 모델 용량 실험의 실패가 아니다.

## 재현과 제한

실제 사용한 Python·라이브러리 버전은 REPRODUCTION_ENVIRONMENT_NEXT9.json과 REPRODUCTION_REQUIREMENTS_NEXT9.txt에 기록했다. 패키지를 새로 설치하지 않았으며 외부 배포 가능 여부까지 검증한 것은 아니다.

입력 스키마 캐시를 첫 호출 전에 초기화한 v6 loader로 replay 객체 해시를 검증했다. 모든 10개 절차에서 학습 파라미터와 예측 불변을 별도 검사했다. TS-AUC는 정렬 순위와 다른 명시적 관측 쌍 계산으로 모든 도메인·2000회 bootstrap을 교차 검증했다.

자기 전이 보조 모델은 NORMAL·EARLY의 전체 6,406,314개 관측에서 현재 F를 받기 전에 순차 발행하여 별도 검증했다. 이 경로는 B=0인 이번 M5에 활성화하지 않았다. 전체 시점과 기존 stride8 비교는 SELF_CADENCE_FINDING_NEXT9_KO.md, 표본 위상 진단은 SAMPLING_PHASE_FINDING_NEXT9_KO.md에 있다. 두 분석은 사후 진단이며 새 인증이나 튜닝에 사용하지 않았다.

현재 공개 추론 진입점은 `scripts/next9/infer_next9_v5.py`, loader는 `src/next9/runtime_loading_v5.py`다. 전체 ZIP 재현은 `scripts/next9/reproduce_next9_archive_v10.py`를 사용한다. 과거 버전은 실패와 수정의 증거로 함께 보존했다.

RP6400에서 NORMAL/EARLY 전체 scalar replay와 반대 series 순서, state 저장/복구, 독립 TS-AUC 대조를 실행했다. 별도 최종 감사에서 실제 M7 전용 경로도 전체 RP로 검증했다. 모든153피처의18,360개 frozen ridge 학습 방정식·held-out group MSE를 새 fit 없이 재검산했다.

raw-free ZIP은168개 고정 합성 시계열×10학습절차×9모델을 새 추출 폴더에서 실제 실행한다. 최초 fixture와 역순·중단 없는 재현은 bitwise 일치해야 한다. NEXT9 공개 generator168경로와 기존 canonical 공개 generator9경로도 검사한다. 원래 작업 폴더의 모델·데이터로 되돌아가는 읽기는 막고 기존 venv의 패키지만 재사용한다.

원본 관측값, 전체 실제 OOF 배열, feature/state cache, Shadow/Seal 값, credentials, 이전 ZIP은 전달물에 넣지 않는다. 작은 NPZ는 합성 fixture 또는 group bootstrap이다. 포함된 group parquet는 ID/role/fold 메타데이터다.

RP는 여러 세션에서 반복 개발에 사용되었다. independent hidden confirmation이 아니다. Shadow/Seal 관측값·정답·점수 사용0, reduced/public/cloud 신규 사용0.

## 실패와 수정

`repairs/`와 실패 로그가 authority다. M2/M3의 첫250회 최적화는 미수렴이었다. outer 점수를 열기 전에 TRAIN 경사 잔차로 진단하고 C=.1·동일 입력·동일 목적함수에서 한 차례 최대5,000회 추가 최적화를 허용했다. 수렴한 기존 모델은 재학습하지 않았고, 이전 모델·예측은 보존했다. 최종 전체 수렴 증거는 JOINT_SOLVER_AUDIT_NEXT9.json에 있다. float32 epsilon 하한 표현, queue label gate adapter, 기존 figure 환경 선택, ARCH 반례 안정성, domain 다중검정 해상도, T2 첫시점 빈 batch, 로그손실 상한 반올림, M7 shared/dedicated 경로의 M5 입력 정밀도 문제를 구분해 기록했다. 원래 파일은 보존하고 가능한 경우 새 버전/작업 ID를 사용했다. 기존 pytest cache1개 쓰기는 명시적 namespace 예외다.

## 다음 작업

`NEXT_SESSION_PLAN_NEXT9.md`, `FEATURE_CARDS_NEXT9_KO.md`의17항목, `BOTTLENECK_HISTORY_NEXT9.csv`를 읽는다. 상태·문턱·parent를 outer 점수를 보고 바꾸거나 zero edge를 억지로 채우지 않는다. 새 탐색은 다음 세션에서 별도 사전등록한다.

최종 immutable archive의 해시와 실제 재현 receipt는 archive 밖의 `DESKTOP_DELIVERY_NEXT9.json` 및 `delivery/*/REPRODUCTION_DONE.json`에 있다. 이 파일 작성 이후 최종 ZIP의 전체 실제 replay와 전달 검사가 추가로 수행된다.
'''
        atomic(BASE/'01_GPT_HANDOFF_NEXT9.md',handoff)
        from src.next9.delivery_payload import REQUIRED_REPORTS
        assert all((BASE/name).is_file() for name in REQUIRED_REPORTS)
        atomic(BASE/'FINAL_REPORTS_DONE_NEXT9.json',dict(status='COMPLETE',timestamp=utc_now(),
            required_files_sha256={name:digest(BASE/name) for name in REQUIRED_REPORTS},
            final_archive_actual_replay_and_desktop_delivery_are_remaining=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--core-tag',required=True);main(p.parse_args().core_tag)
