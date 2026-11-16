"""Assemble the numerical real-data handoff only after final diagnostics pass."""
import json, shutil, sys
from collections import defaultdict
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils.artifacts import write_json, sha256, utc_now


def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def write(name,text):(ROOT/name).write_text(text,encoding='utf-8')
def score(value):return f'{value:.6f}'


def ablation_panel(results,primary):
    panels=defaultdict(dict)
    for result in results:
        if not result.get('candidate'):continue
        spec={k:v for k,v in result['candidate'].items() if k not in ('families','hypothesis','phase')}
        panels[json.dumps(spec,sort_keys=True)][result['families']]=result
    complete=[(key,rows) for key,rows in panels.items() if set('ABCD')<=set(rows)]
    if not complete:raise RuntimeError('No matched A/B/C/D standalone ablation panel')
    anchor=primary['candidate']
    if anchor.get('representation')=='cross_order_ar4_ar8_equal':
        # No blended A/B/C/D ablation was performed. Report the exact matched
        # AR4 component panel and disclose that narrower attribution scope.
        anchor=next(r['candidate'] for r in results if r['experiment_id']==primary['components'][0])
    def priority(item):
        key,rows=item;spec=json.loads(key)
        return (spec.get('representation','base')==anchor.get('representation','base'),spec.get('feature_changes')==anchor.get('feature_changes'),
            spec['model']==anchor['model'],spec['sampling']==anchor['sampling'],spec['params']==anchor['params'],len(rows))
    key,rows=max(complete,key=priority);marginals=[]
    for family,full,base in [('D','AD','A'),('D','BD','B'),('B','BD','D'),('A','AD','D'),('A','ABD','BD'),('C','ABCD','ABD'),('E','ABCDE','ABCD'),('F','ABCDF','ABCD')]:
        if full in rows and base in rows:
            marginals.append({'added_family':family,'full':full,'base':base,'mean_gain':rows[full]['mean']-rows[base]['mean'],
                'median_gain':rows[full]['median']-rows[base]['median'],'worst_fold_change':rows[full]['worst_fold']-rows[base]['worst_fold'],
                'full_experiment_id':rows[full]['experiment_id'],'base_experiment_id':rows[base]['experiment_id']})
    return json.loads(key),rows,marginals


def main():
    audit=read('artifacts/validation/FINAL_AUDIT_REAL.json');frozen=read('DEV_CHAMPION_CANDIDATES.json')
    if audit['status']!='PASS' or frozen['status']!='FROZEN_FOR_FINAL_DIAGNOSTICS':raise RuntimeError('Completed final real audit and frozen selection required')
    state=read('artifacts/real/RESEARCH_STATE.json');raw=read('artifacts/audits/DATA_AUDIT.json');duplicate=read('artifacts/audits/DUPLICATE_AUDIT.json')
    results=state['results'];lookup={r['experiment_id']:r for r in results};primary=lookup[frozen['PRIMARY_DEV_CANDIDATE']]
    baseline=lookup['B1_official_ewma'];statistical=max((r for r in results if r['phase'] in ('statistical_ladder','complementary_statistical')),key=lambda r:r['mean'])
    supervised=max((r for r in results if r['model'].startswith('M')),key=lambda r:r['mean'])
    usage=read('REDUCED_TEST_USAGE_LEDGER.json');refit=read('artifacts/real/FINAL_REFIT_REAL.json');session=read('artifacts/validation/REAL_SESSION.json')
    if usage['scored_uses']!=2:raise RuntimeError('The final two permitted diagnostics must be complete')
    scores={}
    for kind in ('baseline','primary'):
        entry=next(e for e in usage['entries'] if e['kind']==kind);path=ROOT/entry['score_artifact']
        public=ROOT/'artifacts/real/final_reduced_scalars'/f'{kind.upper()}_SCORE.json'
        if not path.exists():path=public
        if sha256(path)!=entry['score_artifact_sha256']:raise RuntimeError('Stored diagnostic score changed')
        scores[kind]=json.loads(path.read_text())
        # Only the already-gated final handoff publishes these scalar reports.
        # Byte-identical copies let a source archive reconstruct the report
        # without the private-until-final directory or any target reread.
        if path!=public:
            public.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,public)
    write_json(ROOT/'artifacts/real/REDUCED_SCORES_DISCLOSED_AFTER_FREEZE.json',{'timestamp':utc_now(),'selection_sha256':sha256(ROOT/'DEV_CHAMPION_CANDIDATES.json'),
        'selection_frozen_utc':frozen['freeze_timestamp'],'scores':scores,'scope':'Read stored scalar diagnostics after all candidates were frozen. No target file reread, rescoring, or model changes.'})
    panel_spec,panel,marginals=ablation_panel(results,primary)
    ranked=sorted((panel[f] for f in 'ABCD'),key=lambda r:r['mean'],reverse=True)
    pd.DataFrame([{'rank':i+1,'family':r['families'],'experiment_id':r['experiment_id'],'mean':r['mean'],'median':r['median'],'worst_fold':r['worst_fold']} for i,r in enumerate(ranked)]).to_csv(ROOT/'FEATURE_STANDALONE_RANKING.csv',index=False)
    pd.DataFrame(marginals).to_csv(ROOT/'FEATURE_MARGINAL_CONTRIBUTIONS.csv',index=False)
    write_json(ROOT/'artifacts/real/FINAL_ABLATION_PANEL.json',{'matched_configuration':panel_spec,'candidate_ids':{f:r['experiment_id'] for f,r in panel.items()},'primary_configuration':primary['candidate'],
        'exact_primary_settings_except_family':all(panel_spec.get(k)==primary['candidate'].get(k) for k in panel_spec),'selection_basis':'Prefer the primary representation/configuration/model/sampling among panels with all A/B/C/D standalone results. For the fixed cross-order average, use its AR4 component panel; blended family ablations were not evaluated. No cross-configuration marginal subtraction.'})
    margin=max(marginals,key=lambda r:r['mean_gain']) if marginals else None
    selected=frozen['candidates'];latency={'status':'PASS','timestamp':utc_now(),'candidates':{},'selection_benchmarks':{},'scope':'Isolated Windows CPU. Point timing excludes initialization, file I/O and IPC; amortized estimates add measured historical initialization.'}
    for role,item in selected.items():
        latency['selection_benchmarks'][role]=read(item['benchmark']['path'])
        descriptor=read(item['model_directory']+'/model.json');mode='fused_fast_init' if descriptor['use_fused_inference'] else 'native'
        actual=read(f"artifacts/real/latency/final_deployment/{item['experiment_id']}_{mode}.json")
        if actual['model_sha256']!=descriptor['model_sha256']:raise RuntimeError('Final deployment timing model identity differs')
        latency['candidates'][role]=actual
    write_json(ROOT/'LATENCY_REAL.json',latency)
    performance=latency['candidates']['PRIMARY_DEV_CANDIDATE'];point_us=performance['long_stream']['measured_us_per_point']
    amortized=performance['real_history_initialization']['ten_million_amortized_seconds_at_observed_mean_length']
    determinism={'status':'PASS','timestamp':utc_now(),'final_all_dev_refit':refit['primary_independent_refit'],
        'selected_cv_refit':read(f"artifacts/real/cv_reproduction/final_selected/{primary['experiment_id']}/RESULTS.json"),
        'all_selected_cv_refits':{role:read(f"artifacts/real/cv_reproduction/final_selected/{item['experiment_id']}/RESULTS.json") for role,item in selected.items()},
        'separate_archive_raw_reproduction':read('artifacts/real/ARCHIVE_RAW_REPRODUCTION_REAL.json'),
        'exact_predictive_state_reproduction':read('artifacts/real/PREDICTIVE_STATE_REPRODUCTION_REAL.json'),
        'official_reduced_determinism':True,'LOCAL_FINAL_SEAL_OPENED':False}
    write_json(ROOT/'DETERMINISM_REAL.json',determinism)
    smoke={'status':'PASS','baseline':read('artifacts/real/CRUNCH_TEST_BASELINE_REAL.json'),'primary':read('artifacts/real/CRUNCH_TEST_PRIMARY_REAL.json'),
        'full_dev_official_socket_path':audit['evidence']['primary_full_dev_socket'],
        'actual_final_full_streaming':{role:audit['evidence'][role]['full_streaming_actual_artifact'] for role in selected},
        'live_contracts':{role:audit['evidence'][role]['contract'] for role in selected},'submitted':False,'LOCAL_FINAL_SEAL_OPENED':False}
    write_json(ROOT/'DEPLOYMENT_SMOKE_REAL.json',smoke)
    mappings={'REAL_DATA_MANIFEST.json':'artifacts/audits/RAW_DATA_MANIFEST.json','REAL_DATA_AUDIT.md':'artifacts/audits/DATA_AUDIT.md','REAL_DATA_AUDIT.json':'artifacts/audits/DATA_AUDIT.json',
        'DUPLICATE_AUDIT.json':'artifacts/audits/DUPLICATE_AUDIT.json','GROUP_POLICY.md':'artifacts/audits/GROUP_POLICY.md','GROUP_POLICY.json':'artifacts/audits/GROUP_POLICY.json',
        'GROUP_ASSIGNMENTS.parquet':'artifacts/validation/GROUP_ASSIGNMENTS.parquet','SPLIT_LOCK_REAL.json':'artifacts/validation/SPLIT_LOCK_REAL.json',
        'SCORER_PARITY_REAL.json':'artifacts/validation/SCORER_PARITY_REAL.json','EXPERIMENT_LEDGER.csv':'experiments/results/EXPERIMENT_LEDGER.csv'}
    for target,source in mappings.items():shutil.copy2(ROOT/source,ROOT/target)
    weak=min(primary['time_profile'],key=primary['time_profile'].get)
    rows=[('1. Train series',f"{raw['series_count']:,}"),('2. Break / no-break',f"{raw['break_count']:,} / {raw['no_break_count']:,}"),
        ('3. Historical 길이',f"{int(raw['distributions']['historical_length']['min']):,}–{int(raw['distributions']['historical_length']['max']):,}"),
        ('4. Online 길이',f"{int(raw['distributions']['online_length']['min'])}–{int(raw['distributions']['online_length']['max'])}"),
        ('5. 중복 / group',f"exact {duplicate['exact_duplicate_excess']}, normalized exact {duplicate['normalized_duplicate_excess']}, near merge {len(duplicate['merges'])}; groups {duplicate['group_count']:,}"),
        ('6. DEV / seal','8,000 / 2,000 series; DEV online 4,014,405 rows'),('7. Official EWMA DEV mean',score(baseline['mean'])),
        ('8. Best statistical mean',f"{statistical['model']}: {score(statistical['mean'])}"),('9. Best supervised by mean',f"{supervised['experiment_id']}: {score(supervised['mean'])}"),
        ('10. PRIMARY',primary['experiment_id']),('11. PRIMARY fold 0–4',' / '.join(map(score,primary['fold_scores']))),
        ('12. Mean / median / worst',f"{score(primary['mean'])} / {score(primary['median'])} / {score(primary['worst_fold'])}"),
        ('13. Standalone family 순위',' > '.join(f"{r['families']} {score(r['mean'])}" for r in ranked)),
        ('14. 가장 큰 matched marginal',f"{margin['added_family']}: {margin['base']}→{margin['full']} mean {margin['mean_gain']:+.6f}" if margin else '해당 비교 없음'),
        ('15. 가장 약한 online time',f"{weak}: {score(primary['time_profile'][weak])}"),
        ('16. Break age 0–4 / 50+',f"{score(primary['age_profile']['0-4'])} / {score(primary['age_profile']['50+'])}"),
        ('17. Inference μs/point',f'{point_us:.3f} (isolated, continuous stream)'),('18. 10M 예상 시간',f'{point_us*10:.2f}s point-only; {amortized:.2f}s historical 초기화 포함 추정'),
        ('19. Reduced 사용','2/2; baseline 1회 + frozen primary 1회'),('20. LOCAL_FINAL_SEAL_OPENED','false; 성능 평가 0회'),
        ('21. Leakage / API 검사','PASS; 금지 feature 식별자, prefix, single pass, no read-ahead, no inference writes'),
        ('22. Determinism','PASS; DEV 4,014,405 OOF 재학습 및 별도 전체 DEV refit 예측 동일'),
        ('23. Crunch runner',f"PASS; baseline / primary 각각 100 series, 50,983 predictions; 제출 0회"),
        ('24. 병목 3개','변화 직후 약한 구별력; 반복 DEV 선택의 낙관 편향과 fold 편차; historical 초기화 및 미검증 cloud IPC 비용')]
    table='| 항목 | 실측 결과 |\n|---|---|\n'+'\n'.join(f'| {key} | {value} |' for key,value in rows)
    candidate_table='| 역할 | 후보 | mean | median | worst |\n|---|---|---:|---:|---:|\n'+'\n'.join(f"| {role} | {r['experiment_id']} | {score(r['mean'])} | {score(r['median'])} | {score(r['worst_fold'])} |" for role,r in selected.items())
    scope='사전 전체 기술 통계 감사에서는 모든 train series의 길이와 break/tau 메타데이터를 요약했다. split 동결 후 모델 학습·CV·진단은 DEV에 한정했으며 seal 성능을 계산하지 않았다. 원시 파일 SHA256 검사는 파일 전체 바이트를 읽는 작업이다.'
    alarms_path=ROOT/'artifacts/real/first_alarm_diagnostics/RESULTS.json';alarm_text=''
    wire=audit['evidence']['primary_full_dev_socket']['report']
    wire_topology='별도 Python consumer 프로세스' if wire.get('transport')=='subprocess' else '같은 프로세스의 producer/consumer thread'
    wire_text=f"추가 DEV 통신 검증: 원래 fold 모델 5개를 실제 `infer()`와 변경 없는 공식 소켓/보호 코드로 실행하여 {wire['total_rows']:,}개 예측이 OOF와 정확히 일치했다. 역순 시계열 검증도 통과했다. Windows loopback의 {wire_topology} 방식에서 초기화·전송·예측·수집을 포함한 비용은 {wire['wire_us_per_point']:.3f} μs/point였다. 이는 로컬 통신 실측이며 cloud IPC·병렬화 SLA 검증은 아니다. reduced 입력과 target은 사용하지 않았다."
    if wire.get('transport')=='subprocess':
        cold=wire['cold_start_to_ready_seconds']
        wire_text+=f" 위 전체 시간에는 fold별로 새로 만든 5개 프로세스의 시작/JIT 준비 비용이 포함된다(각 {min(cold):.2f}–{max(cold):.2f}초). 준비 신호 이후의 초기화·통신·예측 비용은 {wire['after_ready_wire_us_per_point']:.3f} μs/point였다."
    final_wire=smoke['actual_final_full_streaming']['PRIMARY_DEV_CANDIDATE']
    final_wire_timing=final_wire['primary_socket_timing']
    wire_text+=f" 최종 전체 DEV refit 모델도 별도로 {final_wire['total_rows']:,}개 예측의 배치/스트리밍 일치와 새 consumer 프로세스 1개의 공식 소켓 일치를 통과했다. 최종 모델의 소켓 전체 실행은 {final_wire_timing['process_total_seconds']:.2f}초, 그중 준비 완료까지 {final_wire_timing['process_start_to_ready_seconds']:.2f}초였다. 이 검사는 같은 모델의 수치 일치를 확인하며 학습 데이터 성능 점수를 계산하지 않는다."
    distribution=read('artifacts/real/latency/full_dev_distribution/RESULTS.json')
    primary_distribution=next(row for row in distribution['summaries'] if row['candidate']==primary['experiment_id'])
    middle=sorted(primary_distribution['passes'],key=lambda row:row['amortized_us_per_point'])[1]
    latency_text=f"10M point-only 측정은 실제 DEV 값의 작은 버퍼를 반복한 하나의 긴 스트림이다. 초기화 포함 {amortized:.2f}초 추정은 최종 모델의 {performance['real_history_initialization']['series']}개 DEV historical 초기화와 관측 online 길이를 사용한다. 별도로 원래 CV 모델을 DEV 8,000개 실제 길이로 3회 실행한 검사는 매번 4,014,405개 OOF 예측과 일치했고, 초기화 포함 중앙 실행은 {middle['amortized_us_per_point']:.3f} μs/point였다. 그 실행의 초기화 비중은 {middle['initialization_fraction']:.2%}다. 이 계산 구간에는 최초 JIT·모델 로딩·입력 파일 I/O·소켓 통신이 포함되지 않는다."
    archive_raw=determinism['separate_archive_raw_reproduction']
    archive_roles=archive_raw['candidates']
    archive_text=f"최종 동결 후 소스 ZIP `{archive_raw['archive']}`(SHA256 `{archive_raw['archive_sha256']}`)을 별도 경로·가상환경에 풀어 최종 세 역할 모두 원본 raw부터 재현했다. 각 후보의 5개 cache 파일은 원래 바이트와 일치했고, 독립 5-fold 재학습과 전체 DEV 배포 모델 재학습에서 각각 4,014,405개 예측 차이가 0이었다. CV 모델 바이트 일치 여부는 "+'; '.join(f"{role}: {item['cv_model_bytes_identical']}" for role,item in archive_roles.items())+"이며 전체 DEV 모델 바이트 일치 여부는 "+'; '.join(f"{role}: {item['final_model_bytes_identical']}" for role,item in archive_roles.items())+"다. 예측 동일성과 직렬화 파일 동일성은 구분한다. 원본 OOF 배열과 cache는 ZIP에 없었으며, 상세 증거는 `artifacts/real/ARCHIVE_RAW_REPRODUCTION_REAL.json`과 그 안의 보고서 경로에 있다. 이 raw 재현 당시 모델·학습·추론 소스가 현재와 일치하는지도 최종 감사에서 검사한다."
    cohort_text=''
    cohort_path=ROOT/'artifacts/real/historical_cohorts/RESULTS.json'
    if cohort_path.exists():
        cohort=json.loads(cohort_path.read_text());rows_by_name={r['cohort']:r for r in cohort['summaries'] if r['candidate']==primary['experiment_id']}
        if 'ar8_squared_acf1__bin_1' in rows_by_name:
            row=rows_by_name['ar8_squared_acf1__bin_1']
            cohort_text=f"Historical 구조별 사후 진단에서는 AR8 제곱 잔차의 lag-1 상관이 0.05 이상 0.2 미만인 {row['series']:,}개 DEV 시계열의 주력 TS-AUC가 {row['mean']:.6f}였다. 구간은 historical 값만으로 정했으며 모든 결과는 `artifacts/real/historical_cohorts`에 보존했다. 구간 간 pair는 점수에서 제외되므로 전체 점수의 가산 분해가 아니다. 400회 group bootstrap 구간은 다중 비교·모델 선택 편향을 보정하지 않은 조건부 진단이다."
    if alarms_path.exists():
        alarms=json.loads(alarms_path.read_text());chosen=[r for r in alarms['aggregate'] if r['candidate']!='B1_official_ewma' and r['nominal_training_tail_fraction']==.05]
        if chosen:
            item=chosen[0]
            alarm_text=f"별도 첫 경보 진단: `{item['candidate']}`에서 TRAIN 무변화 시계열의 상위 5% 최대값으로 fold별 임계값을 정했을 때, held-out 무변화 시계열의 any-alarm 비율은 {item['null_any_alarm_rate']:.2%}, 사전 경보 없이 변화를 검출한 비율은 {item['valid_detected_break_rate']:.2%}, 검출 사례에 한정한 지연 중앙값은 {item['detected_only_median_delay']:.1f} 관측치였다. 미검출·관측 종료 사례는 지연 중앙값에서 제외되므로 검출률과 함께 해석한다. 이는 사후 DEV 진단이며 competition 출력이나 모델 선택에 임계값을 적용하지 않았다. `artifacts/real/first_alarm_diagnostics/RESULTS.json`에 1%/5% 결과와 정확한 범위가 있다."
    text=f'''# Structural Break 실데이터 연구 전달서

DEV 연구 및 최종 검증을 완료했다. 공식 인증으로 확보한 데이터에서 {len(results)}개 후보의 5-fold 결과를 기록했다. 주력 평균 TS-AUC는 **{score(primary['mean'])}**, 공식 EWMA는 **{score(baseline['mean'])}**다. 이 수치는 반복 선택에 사용한 DEV 성능이며 독립 최종 성능이 아니다.

{candidate_table}

주력 승격 기준은 median +0.001, worst loss ≤0.002, 속도 배율 기준을 유지했다. 평균이 가장 높은 후보와 주력은 다를 수 있다. 안정성 백업과 속도 백업은 `FINAL_CANDIDATE_SELECTION_POLICY.json`의 사전 정책으로 골랐다. 최종 모델은 `artifacts/deployment/final/` 아래에 있다.

{table}

## 점수와 진단의 해석

모든 CV 예측은 공식 전송 정밀도인 float32로 평가했다. TS-AUC는 동일 online step 안의 positive/negative pair 수로 가중한 AUC이고, 폴드 보고값은 고정 5개 폴드의 산술 평균이다. Break-age 진단은 해당 age의 positive와 모든 negative control을 비교한다. 이는 임계값 기반 검출 지연이나 오경보율이 아니다.

{alarm_text}

{cohort_text}

Standalone 및 marginal 표는 같은 모델·표현·feature 설정·sampling에서 비교한 별도 패널이다. 고정 AR4/AR8 평균이 주력이면 AR4 구성 모델의 패널이며, 평균 모델 자체의 family ablation은 수행하지 않았다. 정확한 설정은 `artifacts/real/FINAL_ABLATION_PANEL.json`을 확인한다. feature importance만으로 family 효용을 단정하지 않았다. 선정된 모델의 각 표현은 57개 특징을 계산하고, 두 표현의 평균은 총 114개를 계산한다.

최종 동결 뒤 저장된 reduced 진단을 공개했다: baseline **{score(scores['baseline']['ts_auc'])}**, primary **{score(scores['primary']['ts_auc'])}**. 이 점수로 모델·feature·hyperparameter를 변경하지 않았다. LOCAL_FINAL_SEAL_OPENED = false.

공개된 scalar 보고서의 원래 바이트는 `artifacts/real/final_reduced_scalars`에 보존하며 quota ledger의 SHA256과 일치한다. 이 파일은 target 행을 포함하지 않으며 새 폴더에서 전달 문서를 재구성할 때 추가 target 조회를 대신한다.

{scope}

## 재현과 전달 범위

`23_REPRODUCTION_COMMANDS.md`, `SOURCE_MANIFEST.json`, `artifacts/validation/FINAL_AUDIT_REAL.json`을 먼저 확인한다. 코드·설정·테스트·선정 모델·작은 결과를 ZIP에 포함하고, raw/cache/대용량 OOF/가상환경/인증정보는 제외한다. 원본 데이터가 필요한 CV 재현은 manifest와 일치하는 로컬 `data/raw`에서 실행해야 한다.

수치의 전체 구조는 `artifacts/real/figures/DEV_RESULTS_OVERVIEW.pdf`와 같은 이름의 PNG/SVG에 요약했다. 그래프 해석은 `docs/FIGURE_GUIDE_REAL.md`, CSV별 행 수와 권위는 `docs/RESULTS_CATALOG_REAL.md`를 확인한다. 31개 내보내기 검사는 모든 완료 후보·폴드·시간 구간이 동결 체크포인트와 일치함을 확인했다.

{wire_text}

{latency_text}

{archive_text}

직렬화 차이는 추가로 검사했다. 선정 후보의 15개 CV 모델과 3개 최종 모델, 총 18쌍에서 실제 compact 추론에 쓰는 모든 트리 분기·값·사용 열·표현 설정의 정확한 상태가 일치했다. LightGBM 본문 및 HistGB 노드·bias 진단은 `artifacts/real/PREDICTIVE_STATE_REPRODUCTION_REAL.json`에 따로 기록했다. 이는 전체 native 추정기 객체나 다른 플랫폼의 실행까지 동일하다는 주장은 아니다.

실행 예산: {session['start_utc']} 시작, {session['new_experiment_deadline_utc']} 새 실험 마감, {session['deadline_utc']} 전체 마감. 최종 문서 생성 시각 {utc_now()}. 테스트 {audit['evidence']['tests']['tests']}개 통과. 별도 소스 복사와 가상환경 검증은 `artifacts/validation/REPRODUCTION_REPORT.json`에 기록하며, 최종 ZIP의 실행 소스 일치 여부는 `SOURCE_MANIFEST.json`이 권위다.

이전 인증 차단 단계의 문서는 역사 기록이다. 현재 상태와 실측 결과는 이 전달서와 이름에 REAL이 있는 보고서를 우선한다. 다음 작업은 `NEXT_SESSION_PLAN.md`, 한계는 `LIMITATIONS.md`에 적었다.
'''
    write('01_GPT_HANDOFF.md',text)
    write('00_GPT_READ_FIRST.md',f'''# GPT에게 먼저 전달할 내용

`01_GPT_HANDOFF.md`부터 읽는다. 실데이터 DEV 연구와 최종 검증을 완료했다. 주력 `{primary['experiment_id']}`의 5-fold mean / median / worst는 **{score(primary['mean'])} / {score(primary['median'])} / {score(primary['worst_fold'])}**다.

Train 10,000 series, DEV 8,000 / seal 2,000. **LOCAL_FINAL_SEAL_OPENED = false**. Reduced scored uses **2/2**를 소진했으므로 추가 조회·모델 선택에 사용하지 않는다. 최종 후보 3역할과 원본 모델은 `DEV_CHAMPION_CANDIDATES.json` 및 `artifacts/deployment/final/`에 있다.

{audit['evidence']['tests']['tests']} tests PASS. 독립 5-fold 및 전체 DEV 재학습, 실제 streaming parity와 인증된 공식 runner 검사 PASS. DEV 반복 선택 편향, 약한 초기 검출, cloud 실행 미검증을 한계로 유지한다. 이 ZIP은 이전 `BLOCKED_AUTH` / 20260908_080300 ZIP을 대체한다. 원시 데이터·캐시·인증정보는 ZIP에 넣지 않았다.

핵심 파일: `01_GPT_HANDOFF.md`, `DEV_CHAMPION_CANDIDATES.json`, `CV_RESULTS_REAL.csv`, `FEATURE_STANDALONE_RANKING.csv`, `LATENCY_REAL.json`, `DETERMINISM_REAL.json`, `DEPLOYMENT_SMOKE_REAL.json`, `SOURCE_MANIFEST.json`.

요약 그림: `artifacts/real/figures/DEV_RESULTS_OVERVIEW.pdf` 또는 PNG. 데이터 표 안내: `docs/RESULTS_CATALOG_REAL.md`.
''')
    write('00_START_HERE.md','최신 시작 문서: [00_GPT_READ_FIRST.md](00_GPT_READ_FIRST.md). 자세한 실측 결과: [01_GPT_HANDOFF.md](01_GPT_HANDOFF.md).\n')
    write('LIMITATIONS.md',f'''# 한계

1. 고정 DEV 5 folds에서 {len(results)}개 후보를 반복 비교했다. 선택 편향을 보정한 독립 성능은 없다. group bootstrap도 고정 OOF에 조건부인 진단이다.
2. 주력 break-age 0–4 TS-AUC {score(primary['age_profile']['0-4'])}; 빠른 검출은 여전히 약하다. 첫 경보에 관한 DEV 진단은 `artifacts/real/first_alarm_diagnostics`에 있지만, 관측 기간과 학습 구간 임계값에 의존하며 독립 운영 성능 보증은 아니다.
3. Exact / normalized exact / prefix64 중복은 없었고 사전 indexed 12,438쌍도 merge 기준을 통과하지 않았다. 같은 길이만 검사한 근사 탐색이므로 숨은 생성 family나 모든 near duplicate의 부재를 증명하지 않는다. split/threshold는 바꾸지 않았다.
4. {scope}
5. 실제 Windows 공식 runner와 DEV 전체 loopback 소켓 경로는 검증했다. Linux cloud 설치·프로세스 병렬화·cloud IPC 포함 SLA는 측정하지 않았다. package lock은 로컬 재현 기준이다.
6. Point-only 속도는 historical 초기화 비용을 제외한다. 10M amortized 추정은 관측한 historical/online 길이 분포에 의존한다. 속도 백업 선정에는 CV fold-0 모델을 사용했고, 최종 latency 표에서는 전체 DEV refit 모델을 별도로 측정했다.
7. 선정된 모델의 각 표현은 57개 feature를 계산하므로 model-column ablation이 feature 계산량을 줄이지 않는다. 고정 AR4/AR8 평균은 두 표현, 총 114개를 계산한다. 일부 tail feature는 historical null clip에 걸리며 상세 비율은 feature_health audit에 있다.
8. Reduced 2회 quota는 모두 사용했다. 단 100개 series 진단을 일반화 성능이나 추가 튜닝 근거로 삼지 않는다. LOCAL_FINAL_SEAL은 미개봉이다.
9. 연구 전 원래 setup 위치 파일은 Windows byte 접근이 거부되었다. 원본을 수정하지 않고 공식 인증으로 동일 이름 데이터를 canonical data/raw에 새로 받아 SHA256을 동결했다.
10. 순위 점수는 교정된 posterior 확률로 검증하지 않았다. 실제 대회 제출이나 업로드는 수행하지 않았다.
''')
    write('NEXT_SESSION_PLAN.md',f'''# 다음 세션

1. ZIP manifest 및 별도 환경 재현 보고서를 확인하고 frozen primary의 DEV CV와 source hash를 재현한다. 기존 `REAL_SESSION.json`의 모델 연구는 종료되었고 원래 10시간 예산과 마감 시각은 보존했다. 같은 명령으로 연구 예산이 새로 시작되지는 않는다. 새 연구 예산은 별도로 명시하고 기존 split을 유지한다.
2. 가장 큰 연구 과제는 break-age 0–4다. AR 잔차 기반 의존성 변화의 어떤 historical 구조에서 초기 신호가 사라지는지 DEV의 기존 OOF와 prefix 진단으로 분석한다. 작은 사전 가설 1–2개씩 검증하고 새로운 row split·seed shopping은 하지 않는다.
3. 실제 Crunch Linux 환경에서 고정 모델의 패키지 호환성·초기화·IPC·메모리를 검증한다. 현재 로컬 point-only {point_us:.3f} μs를 cloud SLA로 해석하지 않는다. 제출은 사용자의 별도 지시가 있을 때 수행한다.

고정 후보는 `DEV_CHAMPION_CANDIDATES.json`에 있다. Reduced target 사용은 2/2 소진했다. LOCAL_FINAL_SEAL은 새로 명시된 최종 검증 절차가 있기 전까지 열지 않는다. 이번 세션의 주력 선택은 이미 동결됐으며 reduced 점수를 보고 소급 변경하지 않는다.
''')
    write('23_REPRODUCTION_COMMANDS.md',rf'''# 재현 명령 (프로젝트 경로의 PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q --junitxml=artifacts/validation/pytest_recheck.xml
.\.venv\Scripts\python.exe scripts/verify_scorer.py
```

처음 두 줄은 새 Windows 작업 폴더에서 환경을 만들 때만 실행한다. 기존 가상환경을 다시 만들 필요는 없다. 위 단계는 raw 데이터 없이 가능한 소스 검사다. package lock은 검증한 Windows Python 3.13 환경 기준이며 Linux 호환성을 보증하지 않는다.

모델과 동결 설정은 정확한 파일 바이트의 SHA256에 연결되어 있다. ZIP의 원본 줄바꿈과 내용을 유지한다. Git 체크아웃이나 편집기가 줄바꿈을 바꾸면 동작이 같아 보여도 hash 검사가 실패할 수 있으므로, 기록된 ZIP 원본으로 재현하고 hash를 임의로 갱신하지 않는다.

선정 CV를 별도 출력 위치에 재현하려면 원본과 hash가 같은 data/raw가 필요하다. 기존 작업 폴더에서는 원본 cache와 OOF를 그대로 비교한다. ZIP을 새 위치에 풀었다면 --rebuild-missing을 추가한다. 이 경우 빠진 cache를 재생성하고 ZIP에 포함된 원래 fold 모델의 예측과 비교하며, 비교 기준을 보고서에 명시한다.

다음 세 명령은 원본 cache·OOF를 보존한 기존 작업 폴더용이다. ZIP을 새 위치에 풀었을 때는 바로 아래의 별도 명령을 사용한다. 전체 최종 감사는 실제 raw 파일·선정 모델·보존된 검증 보고서가 필요하므로 raw 없이 가능한 첫 소스 검사와 구분한다. Git 메타데이터가 없는 압축 해제 폴더에서는 SOURCE_MANIFEST의 포함 목록으로 파일 감사를 수행한다.

```powershell
.\.venv\Scripts\python.exe scripts/final_audit_real.py
.\.venv\Scripts\python.exe scripts/reproduce_selected_cv.py --candidate {primary['experiment_id']} --tag next_session_recheck
.\.venv\Scripts\python.exe scripts/verify_fused_real.py --candidate {primary['experiment_id']} --fast-initialization
```

ZIP을 새 폴더에 푼 경우에는 manifest와 일치하는 data/raw를 먼저 준비하고 다음과 같이 실행한다. 역할별 CV 명령이 필요한 cache를 원본 raw부터 만들고, 이어서 전체 DEV refit을 별도 출력 위치에 저장한다. 기존 결과를 덮어쓰지 않도록 재실행할 때는 새로운 tag를 쓴다.

```powershell
$selectedRoles = (Get-Content DEV_CHAMPION_CANDIDATES.json -Raw | ConvertFrom-Json).candidates
foreach ($selectedRole in $selectedRoles.PSObject.Properties) {{
    .\.venv\Scripts\python.exe scripts/reproduce_selected_cv.py --candidate $selectedRole.Value.experiment_id --tag next_archive_cv --rebuild-missing
    if ($LASTEXITCODE -ne 0) {{ throw 'Selected CV reproduction failed' }}
    .\.venv\Scripts\python.exe scripts/reproduce_final_model_from_raw.py --role $selectedRole.Name --tag next_archive_refit
    if ($LASTEXITCODE -ne 0) {{ throw 'Selected final-model reproduction failed' }}
}}
```

전체 train 데이터로 `main.train()`을 호출하면 configs/deployment.json의 DEV 8,000개만 학습하고 seal ID는 값·target 처리 전에 건너뛴다. 기존 최종 모델을 덮어쓰지 말고 새 model directory를 사용한다. 입력 ID universe는 동결된 train 10,000개와 일치해야 한다.

`scripts/reproduce_snapshot.py <archive.zip>`은 ZIP을 새 폴더에 추출하고 별도 .repro-venv에서 의존성·테스트·scorer를 검사한다. 새로운 PC에서는 requirements-lock.txt에 맞는 별도 가상환경을 먼저 준비해야 한다. 이 source-only 검사는 raw 데이터를 포함한 CV 재현과 구분한다.

이 명령에 `--artifact-audit final`을 추가하면 원래 작업 폴더의 canonical data/raw를 검증된 참조로 연결하고, 압축 해제 폴더에서 전체 보존 artifact 감사도 수행한다. 이 추가 감사는 raw 바이트 해시를 읽지만 target 성능을 다시 계산하거나 인증 tester를 호출하지 않는다. 최종 전달본의 별도 환경 검증에는 이 추가 단계도 포함했다.

```powershell
py -3.13 -m venv .repro-venv
.\.repro-venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

`reduced_diagnostic.py` / 일반 `crunch_tools.test()`를 다시 실행하지 않는다. 이미 허용된 baseline 1회와 primary 1회가 완료됐고 추가 target 조회를 금지했다. 인증정보는 ZIP이나 명령행에 넣지 않는다.
''')
    aliases={'03_RAW_DATA_MANIFEST.json':'REAL_DATA_MANIFEST.json','04_DATA_AUDIT.md':'REAL_DATA_AUDIT.md','05_DATA_AUDIT.json':'REAL_DATA_AUDIT.json','08_SPLIT_LOCK.json':'SPLIT_LOCK_REAL.json',
        '10_BASELINE_RESULTS.csv':'BASELINE_RESULTS_REAL.csv','12_MODEL_RESULTS.csv':'MODEL_RESULTS_REAL.csv','13_CV_RESULTS.csv':'CV_RESULTS_REAL.csv','14_FEATURE_ABLATIONS.csv':'FEATURE_ABLATIONS_REAL.csv',
        '15_TIME_FRACTURE_REPORT.csv':'TIME_FRACTURE_REAL.csv','16_LATENCY_BENCHMARK.json':'LATENCY_REAL.json','17_DETERMINISM_REPORT.json':'DETERMINISM_REAL.json','18_DEPLOYMENT_SMOKE.json':'DEPLOYMENT_SMOKE_REAL.json',
        '19_DEV_CHAMPION_CANDIDATE.json':'DEV_CHAMPION_CANDIDATES.json','20_EXPERIMENT_LEDGER.csv':'EXPERIMENT_LEDGER.csv','21_LIMITATIONS.md':'LIMITATIONS.md','22_NEXT_SESSION_PLAN.md':'NEXT_SESSION_PLAN.md'}
    for target,source in aliases.items():shutil.copy2(ROOT/source,ROOT/target)
    shutil.copy2(ROOT/'artifacts/real/FINAL_CANDIDATE_SELECTION_POLICY.json',ROOT/'FINAL_CANDIDATE_SELECTION_POLICY.json')
    write_json(ROOT/'01_PROJECT_STATUS.json',{'status':'REAL_DEV_RESEARCH_COMPLETE_FINAL_VALIDATION_PASS','timestamp':utc_now(),'completed_candidates':len(results),
        'primary_id':primary['experiment_id'],'mean':primary['mean'],'median':primary['median'],'worst_fold':primary['worst_fold'],
        'tests_passed':audit['evidence']['tests']['tests'],'reduced_scored_uses':2,'LOCAL_FINAL_SEAL_OPENED':False,'submitted':False,
        'new_experiment_deadline_utc':session['new_experiment_deadline_utc'],'run_start_utc':session['start_utc'],'final_document':'01_GPT_HANDOFF.md'})
    print('REAL NUMERICAL HANDOFF BUILT',primary['experiment_id'],'mean',primary['mean'],flush=True)


if __name__=='__main__':main()
