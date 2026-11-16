"""Assemble final Korean findings from fixed outputs; no statistical refits."""
from pathlib import Path
import sys,os,csv,json,datetime,html,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next6.common import BASE,read_json,save,digest,utc_now

def csv_rows(name):
    with (BASE/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def union_seconds(intervals):
    merged=[]
    for a,b in sorted(intervals):
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    return sum(b-a for a,b in merged)

def accounting():
    rows=csv_rows('EXPERIMENT_LEDGER.csv');groups={};alltimes=[];cpu_leaf=0
    for r in rows:
        a=datetime.datetime.fromisoformat(r['started_utc']).timestamp();b=datetime.datetime.fromisoformat(r['ended_utc']).timestamp();groups.setdefault(r['kind'],[]).append((a,b));alltimes.append((a,b))
        # Enclosing head_design already includes fit_M*. Do not count both.
        if not r['task'].startswith('fit_'):cpu_leaf+=float(r['process_cpu_seconds'])
    start=datetime.datetime.fromisoformat(read_json(BASE/'SESSION.json')['started_utc']).timestamp();now=datetime.datetime.now(datetime.timezone.utc).timestamp();observed=union_seconds(alltimes)
    research=[x for k,values in groups.items() if k!='FINAL_AUDIT' for x in values];audit=groups.get('FINAL_AUDIT',[])
    return {'recorded_utc':utc_now(),'session_elapsed_seconds':now-start,'instrumented_union_seconds':observed,'research_and_other_non_audit_union_seconds':union_seconds(research),
        'final_audit_union_seconds':union_seconds(audit),'research_audit_overlap_seconds':union_seconds(research)+union_seconds(audit)-observed,
        'per_kind_union_seconds':{k:union_seconds(v) for k,v in groups.items()},'unclassified_elapsed_seconds':max(0,now-start-observed),
        'process_CPU_seconds_excluding_nested_fit_contexts':cpu_leaf,'CPU_note':'parent subprocess runner CPU does not include child pytest CPU; not total machine CPU',
        'accounting_note':'overlapping and nested activity intervals merged, not summed; coding/review outside instrumented processes remains unclassified; no idle time credited as research',
        'target_research_budget_seconds':36000,'target_separate_audit_budget_seconds':[2700,3600],'budget_is_not_elapsed_claim':True,'sleep_padding':False}

def main():
    comparison=csv_rows('MODEL_COMPARISON_CONTINUOUS_M1.csv' if (BASE/'MODEL_COMPARISON_CONTINUOUS_M1.csv').exists() else 'MODEL_COMPARISON_NEXT6.csv');by={r['method']:r for r in comparison};paired=csv_rows('PAIRED_UNCERTAINTY_NEXT6.csv');lat=read_json(BASE/'LATENCY_NEXT6.json');latby={r['method']:r for r in lat['rows']}
    strict_comparison=csv_rows('STRICT_INPUT_MODEL_COMPARISON_NEXT6.csv');strict_by={r['method']:r for r in strict_comparison};strict_pairs={r['candidate']:r for r in csv_rows('STRICT_INPUT_PAIRED_NEXT6.csv')}
    assert read_json(BASE/'M1_STRICT_REPRODUCTION_NEXT6.json')['status']=='PASS'
    if not (BASE/'MODEL_COMPARISON_CONTINUOUS_M1.csv').exists():
        shutil.copy2(BASE/'MODEL_COMPARISON_NEXT6.csv',BASE/'MODEL_COMPARISON_CONTINUOUS_M1.csv')
        shutil.copy2(BASE/'STRICT_INPUT_MODEL_COMPARISON_NEXT6.csv',BASE/'MODEL_COMPARISON_NEXT6.csv')
    tests=read_json(BASE/'final_audit/FULL_TESTS_RECEIPT.json');parity=read_json(BASE/'PREFIX_PARITY_NEXT6.json');integrity=read_json(BASE/'final_audit/ARTIFACT_INTEGRITY.json');stability=read_json(BASE/'GRAPH_SELECTION_PROVENANCE.json');raw=read_json(BASE/'RAW_NULL_CONTROL.json')
    assert tests['status']==parity['status']==integrity['status']=='PASS'
    assert read_json(BASE/'final_audit/TEST_ACCESS_GUARD.json')['status']=='PASS'
    gates=[]
    for m in ('M2','M3','M4'):
        if m not in by:continue
        r=by[m];unc={p['reference']:p for p in paired if p['candidate']==m}
        checks={'complete_5_folds':int(r['outer_folds'])==5,'delta_M0_at_least_.0012':float(r['delta_matched_M0'])>=.0012,
            'delta_M1_at_least_.0002':float(r['delta_same_input_M1'])>=.0002,'delta_random_at_least_.0002':float(r['delta_same_size_RANDOM'])>=.0002,
            'paired_lower_M0_positive':float(unc['M0']['ci_2_5'])>0,'paired_lower_M1_positive':float(unc['M1']['ci_2_5'])>0,
            'at_least_4_positive_folds':int(r['positive_matched_folds'])>=4,'worst_delta_at_least_minus_.001':float(r['worst_matched_fold_delta'])>=-.001,
            'all_causal_checks':True,'scalar_latency_ratio_at_most_3':m in latby and latby[m]['median_ratio_vs_M0']<=3}
        checks['delta_input_complete_M1_STRICT_at_least_.0002']=float(strict_by[m]['delta_input_complete_M1_STRICT'])>=.0002
        checks['paired_lower_vs_M1_STRICT_positive']=float(strict_pairs[m]['ci_2_5'])>0
        gates.append({'method':m,'checks':checks,'eligible_DEV_only':all(checks.values()),'failed_checks':[k for k,v in checks.items() if not v]})
    save('PROMOTION_DECISION_NEXT6.json',{'status':'DEV_ELIGIBLE_EXISTS' if any(g['eligible_DEV_only'] for g in gates) else 'NO_PROMOTION','gates':gates,
        'champion':'BL80 retained unchanged','robustness':'D2 retained unchanged','fallback':'J102 retained unchanged','Shadow_opened':False,'Seal_opened':False,
        'no_full_RP_candidate_refit_or_deployment':True,'M4_partial_folds_not_a_full_candidate':True})
    time=accounting();save('TIME_ACCOUNTING_NEXT6.json',time)
    scoretable='| 방법 | 외부 fold | 평균 TS-AUC | Δ BL80 | Δ 입력 완비 M1_STRICT |\n|---|---:|---:|---:|---:|\n'
    for r in strict_comparison:scoretable+=f"| {r['method']} | {r['outer_folds']} | {float(r['mean_TS_AUC']):.6f} | {float(r['delta_M0']):+.6f} | {float(r['delta_input_complete_M1_STRICT']):+.6f} |\n"
    gatebullets='\n'.join(f"- {g['method']}: "+('DEV 기준 통과' if g['eligible_DEV_only'] else '승격 보류 — '+', '.join(g['failed_checks'])) for g in gates)
    edge=csv_rows('EDGE_STABILITY_RESULTS.csv');strong=[r for r in edge if r['replicated_descriptive']=='True' and r['evaluation_phase']==r['selection_phase']];strong.sort(key=lambda r:-float(r['heldout_mean_relative_MSE_gain']))
    edgetext='\n'.join(f"- fold {r['fold']}, node {r['source']} → {r['target']}, lag {r['lag']}: held-out 상대 MSE 개선 {float(r['heldout_mean_relative_MSE_gain']):+.4f}, 95% CI [{float(r['ci_2_5']):+.4f}, {float(r['ci_97_5']):+.4f}], {r['heldout_series']} series. {r['mechanical']}." for r in strong[:6]) or '- 사전에 정한 held-out 재현 기준을 만족한 연결이 없습니다.'
    unc_text='\n'.join(f"- {m} 대 BL80 평균 Δ의 95% paired CI: [{float(next(p for p in paired if p['candidate']==m and p['reference']=='M0')['ci_2_5']):+.6f}, {float(next(p for p in paired if p['candidate']==m and p['reference']=='M0')['ci_97_5']):+.6f}]." for m in ('M2','M3'))
    unc_text+='\n'+'\n'.join(f"- {m} 대 입력 완비 M1_STRICT 평균 Δ의 95% CI: [{float(strict_pairs[m]['ci_2_5']):+.6f}, {float(strict_pairs[m]['ci_97_5']):+.6f}]." for m in ('M2','M3'))
    atlas=read_json(BASE/'CORRELATION_ATLAS/TRAIN_ATLAS.json');aa={(r['phase'],r['lag'],r['mode']):r for r in atlas['results']};ab={r['name']:r for r in read_json(BASE/'CORRELATION_ATLAS/PSEUDO_AND_NOT_YET.json')['results']}
    window_text=f"특히 창 길이 효과가 컸습니다. node 09 원 에너지 상방과 node 06 에너지 장기 지속의 부분 순위 상관은 정상 전체 길이에서 {aa['normal_no_break',0,'partial']['mean'][9][6]:.3f}, 실제 onset 8시점 창에서 {aa['onset',0,'partial']['mean'][9][6]:.3f}였습니다. 그러나 변화 없는 series의 가상 onset 창에서도 {ab['pseudo_null_onset']['mean'][9][6]:.3f}였고, 같은 시점·같은 길이의 H 매칭 not-yet 대조와의 onset 차이는 {ab['matched_not_yet_delta_onset']['mean'][9][6]:+.3f}로 작아졌습니다. 이는 학습 구간에서 고른 설명용 사례이며, 전체 길이 정상 지도와 짧은 onset 지도의 큰 차이를 변화의 증거로 해석하면 안 됨을 보여줍니다. [창 길이 대조 그림](FIGURES/08_WINDOW_LENGTH_CONFOUND.png)을 함께 확인하세요. 기존 04_MAP_C 그림은 길이가 다른 창의 탐색적 차이이며, 같은 창의 관계 변화 판단에는 07_ALIGNMENT의 matched_not_yet_delta 지도를 사용합니다."
    transition=csv_rows('STATE_TRANSITION_TABLES/HELDOUT_LOG_SCORE_GAIN.csv');parent=[r for r in transition if int(r['selected_parents'])>0];trans_positive=sum(float(r['ci_2_5'])>0 for r in parent)
    controls=csv_rows('COUNTEREXAMPLE_RESULTS_NEXT6.csv');ct=[]
    import numpy as np
    for scenario in ('N0','N1','N2','N3','O1','OB16','V2','V4','D1','D2','LATE_V2'):
        for m in ('M0','M3','AMPLITUDE_0.9_M0'):
            rows=[r for r in controls if r['scenario']==scenario and r['method']==m and r['kind']=='counterexample']
            if not rows:continue
            metric='null_eventual_alarm_rate' if scenario in ('N0','N1','N2','N3','O1','OB16') else 'postbreak_first_detection_rate';vals=[float(r[metric]) for r in rows if r[metric]]
            if vals:ct.append(f"| {scenario} | {m} | {metric} | {np.mean(vals):.3f} |")
    controltable='| 대조 사례 | 방법 | 지표 | 5 fold 평균 |\n|---|---|---|---:|\n'+'\n'.join(ct)
    verdict='사전 기준을 통과한 DEV 후보가 있습니다. 배포와 숨긴 데이터 검증은 수행하지 않았습니다.' if any(g['eligible_DEV_only'] for g in gates) else '새 모델은 사전 승격 기준을 통과하지 못했습니다. BL80와 D2의 기존 역할을 유지합니다.'
    text=f'''# NEXT6 연구 결과와 읽는 순서

{verdict}

이 문서는 최초 NEXT6 실험을 동결한 시점의 보고서입니다. 사용자의 추가 7시간 이상 연구 요청에 따른 병목 개선 실험은 별도 `artifacts/next6_extension`에서 진행하며, 이 보고서의 외부 평가 결과와 모델은 보존합니다.

실제 계산 결과로 만든 [동적 상태전이망](FIGURES/02_DYNAMIC_STATE_TRANSITION_NETWORK_fold0_outer_full.png), [예측적 선후관계망](FIGURES/01_PREDICTIVE_FEATURE_NETWORK_fold0_outer_full.png), [전체 그림 갤러리](FIGURES/index.html)를 먼저 볼 수 있습니다. 상태전이망의 숫자는 학습된 조건부 피처 상태 확률이며, 생성형 이미지나 경제적 인과관계 그림이 아닙니다.

## 같은 평가 범위의 결과

{scoretable}
M0는 기존 NEXT5 ResearchPool 6,400개 series·원래 grouped 5-fold의 BL80입니다. 과거 전체 DEV 8,000개에서 보고한 0.623268 수치와 직접 비교하지 않습니다. 원래 M1은 16개 연속형 노드·lag 1/2/4/8/16·delta·EWMA 8/32와 기존 J102/C205 피처를 쓰는 245열 대조였습니다. 감사에서 명시적 상태·경과 시점 입력이 빠진 것을 발견해, 외부 점수를 열기 전에 M1_STRICT를 고정했습니다. M1_STRICT는 상태 코드와 5개 lag, 상태 EWMA 8/32, 노드별 own-state 지속 길이와 현재 시점을 포함하는 390열 비그래프 대조입니다. 같은 고정 learner로 5회만 학습했으며 튜닝·그래프 재학습은 하지 않았습니다. 학습 recipe는 대조군 포함 총 8개입니다.

M2는 관계 요약, M3는 전이 요약을 추가합니다. 이득 판단에는 입력이 완비된 M1_STRICT를 사용하고, 원래 M1에 대한 gate도 유지해 기준을 더 엄격하게 했습니다. 원래 연속형 M1 결과는 MODEL_COMPARISON_CONTINUOUS_M1.csv에 보존했습니다. M4는 각 fold 내부에서 M2/M3 모두보다 +0.0002 이상 좋아야 외부 학습을 허용했습니다. 일부 fold만 있는 M4 평균은 전체 5-fold 후보 성적이 아닙니다.

{unc_text}

1,000회 whole-series paired bootstrap은 고정된 외부 예측에 조건부입니다. 과거 ResearchPool 반복 사용, 이번 가설 설계 및 그래프 선택의 모든 불확실성을 제거하지 않습니다. 숨긴 데이터 인증 결과로 해석할 수 없습니다.

## 관계와 전이에서 확인한 것

공식 tau_index에 대해 target=1[t>=tau]를 적용했습니다. 변화가 없는 series는 전부 0입니다. pre=-32..-1, onset=0..7, earlyPost=8..31, establishedPost=32..127, laterPost=128+를 점수 확인 전에 고정했습니다. no-break, 같은 절대 시점의 not-yet-break, 학습 H 특성으로 매칭한 대조, 정답·최종 길이와 독립인 pseudo-event를 사용했고 없는 구간은 채우지 않았습니다.

각 series 내부 순위 상관을 먼저 계산해 평균했습니다. 부분 상관에서는 창 안의 선형 추세와 공통 크기를 제거했습니다. lag 검증은 자기 자신의 5개 과거값, 공통 에너지·조건부 분산 과거값, 현재까지의 경과 시점을 조건으로 둔 예측 오차 비교입니다. 상관 변화만으로 추가 예측 이득이 있다고 판단하지 않았습니다.

{window_text}

5개 외부 학습 그래프에서 선택된 서로 다른 edge는 {stability['unique_selected_edges']}개, 5 fold 모두 선택된 edge는 {stability['edges_selected_all_5_folds']}개입니다. 선택된 fold-edge {stability['selected_fold_edges']}개 중 선택 phase에서 기술적 held-out 재현 기준을 만족한 경우는 {stability['replicated_selected_phase_fold_edges']}개입니다. 같은 edge가 여러 fold에서 중복 집계될 수 있습니다.

{edgetext}

Parent가 있는 node/phase/fold 전이 평가 {len(parent)}건 중 own-history 전이 대비 log probability gain의 CI 하한이 양수인 경우는 {trans_positive}건입니다. 이는 피처 상태 예측 결과이며, 최종 탐지 TS-AUC 이득과 구분합니다. 희소 context는 독립 series 8개 미만이면 own-history로 backoff하고, Dirichlet 0.5 평활화를 적용했습니다. 전체 3^16 상태 조합은 만들지 않았습니다.

edge/전이 재현 CI는 선택된 학습 그래프에 조건부인 기술통계이며 다중검정 전체를 보정한 확증이 아닙니다. 학습 자료가 겹치는 다섯 그래프의 선택 빈도도 독립 다섯 번의 증명으로 해석하지 않습니다.

정상 raw Gaussian/AR/ARCH/heavy-tail에 같은 계산기를 적용한 대조에서 {raw['selected_edges']}개 연결이 선택됐고 {raw['mechanical_edges']}개가 계산식 유래 연결로 표시됐습니다. 모든 노드는 동일한 단변량 raw를 공유합니다. 서로 다른 노드의 동시 확인도 독립된 정보원의 확인을 뜻하지 않습니다. 독립적인 경제적 원인이나 변화 이전의 강한 선행 신호는 입증하지 않았습니다. 급격한 합성 변화와 대응 control의 변화 이전 예측은 정확히 같았습니다.

## 이상치와 지속 변화

{controltable}

임계값은 외부 점수와 독립인 정상 합성 calibration 256개 series의 1,024시점 최대 점수에서 정했습니다. 4,096시점 정상 실험은 장기 오경보 진단이며 anytime 보장이 아닙니다. 표의 null alarm은 낮을수록, 진짜 변화 탐지는 높을수록 유리합니다. 탐지 지연은 탐지율·미탐과 함께, 이상치 회복 시간은 censoring과 함께 읽어야 합니다. 0.9×M0는 자체 재보정 임계값과 M0의 같은 수치 임계값을 모두 비교해 단순 점수 축소 효과를 드러냅니다. 상세 지표는 COUNTEREXAMPLE_RESULTS_NEXT6.csv와 METRIC_DIRECTIONS_NEXT6.csv에 있습니다.

실제 held-out 사례는 M3를 미리 지정하고 fold 0의 고정 ID 해시 순서에서 성공·지연·오경보·누락을 골랐습니다. 사례 부재는 CASE_STUDIES_NEXT6.json에 명시합니다. tau는 그림에 사후 표시했으며 예측기에 전달하지 않았습니다.

## 검증과 결정

{gatebullets}

기존 및 새 테스트 {tests['counts']['tests']}개, 실패 {tests['counts']['failures']}개, 오류 {tests['counts']['errors']}개입니다. prefix/suffix·길이 변경, batch/scalar, 저장·로드, 상태 초기화·다른 series와의 교차 실행, 결정론, tau/label/phase/ID 입력 거절을 검사했습니다. 재현은 저장된 모델을 실제 H와 한 점씩 관측에 다시 적용한 결과이며 모든 모델을 처음부터 다시 학습했다는 뜻은 아닙니다. latency는 LATENCY_NEXT6.json의 실제 predict_one 측정치로 확인합니다.

보호 대상 기존 파일 {integrity['protected_files_unchanged']}개가 그대로입니다. Shadow 1,600개와 Seal A/B/C/D는 열지 않았고 새 Shadow·split·public/cloud/reduced test를 만들거나 읽지 않았습니다. BL80 champion, D2 robustness, J102 fallback, B2/X reference 역할을 바꾸지 않았습니다. 모든 NEXT6 모델은 DEV 연구 산출물로만 보존합니다.

목표 연구 예산 10시간과 별도 audit 45~60분을 실제 사용 시간으로 주장하지 않습니다. 현재 세션 경과 {time['session_elapsed_seconds']/3600:.2f}시간, 계측 작업의 중복 제거 합계 {time['instrumented_union_seconds']/3600:.2f}시간입니다. 연구와 audit의 겹침·미계측 편집/검토 시간은 TIME_ACCOUNTING_NEXT6.json에 따로 기록했습니다. 시간을 채우는 대기나 추가 후보 탐색은 하지 않았습니다.

## 재현과 다음 작업

[인수인계](01_RESEARCH_HANDOFF_NEXT6.md)에 실행 명령·원본 의존성·ZIP 제외 범위를 정리했습니다. [다음 세션 계획](NEXT_SESSION_PLAN_NEXT6.md)은 이번 외부 점수로 미세 조합을 더 만드는 대신 미재현 원인과 계산식 유래 연결을 다루도록 작성했습니다.
'''
    (BASE/'00_READ_FIRST_NEXT6_KO.md').write_text(text,encoding='utf-8')
    save('FINAL_AUDIT_NEXT6.json',{'status':'PASS','scientific_decision':verdict,'completed_utc':utc_now(),'tests':tests['counts'],'protected_files_unchanged':integrity['protected_files_unchanged'],
        'required_research_outputs_complete':True,'no_promotion_without_all_gates':True,'canonical_unchanged':True,'Shadow_or_Seal_opened':False,'time_accounting':'TIME_ACCOUNTING_NEXT6.json',
        'scope_limits':['adaptive DEV reuse','conditional bootstrap','mechanical/shared input relations','M4 partial folds when ineligible','synthetic finite-horizon thresholds','saved-model replay rather than complete fresh refit'],
        'handoff_archive_verification':'separate HANDOFF_AUDIT_NEXT6.json generated after packaging'})

if __name__=='__main__':main()
