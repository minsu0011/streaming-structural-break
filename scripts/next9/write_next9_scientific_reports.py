"""Korean evidence-linked interpretation of frozen A/B/C results."""
from pathlib import Path
import sys
import csv
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now,csv_write

def rows(name):
    with (BASE/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def number(value):return 'NA' if value is None or value in ('','None','nan') else f'{float(value):.12f}'

def main():
    with activity('Write evidence-linked Korean17field featurecards and20question scientific handoff','SCIENTIFIC_REPORTING','B2_NO_REPLICATED_OPPOSITE_ORIENTATION'):
        import numpy as np
        from src.next9.feature_meanings import MEANINGS,NORMALIZATION_KO,STATE_KO,LIMIT_KO
        science=read_json(BASE/'SCIENTIFIC_RESULTS_NEXT9.json');cert=read_json(BASE/'VARIABLE_CERTIFICATION_NEXT9.json')
        graph=read_json(BASE/'PREDICTIVE_FACTOR_GRAPH_NEXT9.json')
        candidates=sorted(r['feature'] for r in cert['feature_classification'] if r['outer_A_selected_count']>=4)
        normal=science['C']['M7_vs_BL80']['NORMAL'];early=science['C']['M7_vs_BL80']['EARLY']
        comparison=rows('MODEL_COMPARISON_NEXT9.csv');orientation=rows('CONDITIONAL_ORIENTATION_RESULTS.csv')
        common=rows('COMMON_RANGE_RESULTS.csv');self_rows=rows('SELF_TRANSITION_RESULTS.csv')
        centers=[]
        for part in range(4):centers.extend(rows(f'center_decomposition/PART_{part}.csv'))
        csv_write('CENTER_DECOMPOSITION_ALL_NEXT9.csv',centers)
        supported=defaultdict(set);raw=defaultdict(set)
        for r in common:
            if r['mode']=='NORMAL' and r['domain_name']=='ALL' and r['pair_index']=='0' and r['status']=='COMPLETE':supported[int(r['feature'])].add(int(r['fold']))
        for r in orientation:
            if r['mode']=='NORMAL' and r['domain_name']=='ALL' and r['orientation_pattern'] not in ('0','','None'):
                raw[(int(r['feature']),int(r['pair_index']),int(r['orientation_pattern']))].add(int(r['fold']))
        support4=sum(len(v)>=4 for v in supported.values())
        raw4=sum(len(v)>=4 for v in raw.values())
        center_summary=[]
        for j in candidates:
            for mode in ['NORMAL','EARLY_LATE','EARLY_SELECT']:
                for h in [1,4,16]:
                    rr=[r for r in centers if int(r['feature'])==j and r['inner']=='' and r['mode']==mode and int(r['horizon'])==h]
                    assert len(rr)==15
                    center_summary.append(dict(feature=j,mode=mode,horizon=h,internal_panels=15,
                        positive_ci_panels=sum(float(r['TRAINconstant_to_D1_ci_low'])>0 for r in rr),
                        mean_TRAINconstant_to_D1_gain=float(np.mean([float(r['TRAINconstant_to_D1_gain']) for r in rr])),
                        mean_Hmedian_to_TRAINconstant_gain=float(np.mean([float(r['Hmedian_to_TRAINconstant_gain']) for r in rr])),
                        overlapping_internal_panels_not_independent=True,no_candidate_reselection=True))
        csv_write('CENTER_DECOMPOSITION_SUMMARY_NEXT9.csv',center_summary)
        pathmeta=read_json(BASE/'figure_data/DONE.json');assert digest(BASE/'figure_data/TRAJECTORIES.npz')==pathmeta['files_sha256']['TRAJECTORIES.npz']
        from src.next9.transition_inputs import path_features
        pathnames=['state_now','direction_run_capped15','previous_issued_surprise','reversal_probability','continuation_probability','entropy','reversal_age_capped15','signed_slope_over_epsilon']
        pathrows=[]
        with np.load(BASE/'figure_data/TRAJECTORIES.npz') as z:
            for j in pathmeta['candidates']:
                for t,values in zip(z['time'],z[f'F{j}_path']):
                    assert len(values)==8
                    pathrows.append(dict(dataset_id=pathmeta['trajectory_dataset_id'],fold=0,mode='NORMAL',feature=j,time=int(t),
                        **dict(zip(pathnames,map(float,values))),diagnostic_only=True))
        csv_write('TRANSITION_PATH_FEATURES.csv',pathrows)
        cards=['# NEXT9 피처 카드\n\n고정된 A 후보 전체를 설명한다. 후보라는 말은 조건부 방향 반전 인증이나 배포 승격을 뜻하지 않는다.\n']
        for j in candidates:
            title,meaning,lineage=MEANINGS[j]
            rules=[r for r in cert['certified_rules'] if r['feature']==j]
            edges=[r for r in graph['certified_edges'] if r['destination']==j]
            a=[r for r in center_summary if r['feature']==j and r['mode']=='NORMAL']
            transitions=[r for r in self_rows if int(r['feature'])==j and r['mode']=='NORMAL' and r['domain']=='ALL' and r['contrast']=='T2_SELF_RUN_SLOPE_H__vs__T1_SELF_MARKOV']
            transition_text='; '.join(f"fold{r['fold']} 로그손실 이득 {number(r['point'])} [{number(r['ci_low'])}, {number(r['ci_high'])}]" for r in transitions)
            conditional='인증 규칙: '+str(rules) if rules else '현재 F와 break의 방향이 과거 상태에 따라 뒤집힌다고 인증하지 못했다. 상태 이름만으로 매수·매도 또는 break 방향을 정하지 않는다.'
            fields=[('피처 이름',f'F{j:03d}: {title}'),('무엇을 측정하는가',meaning),('H 기준 의미',NORMALIZATION_KO),
                ('과거 movement',STATE_KO),('UP에서 현재 F의 의미','해당 피처가 직전까지 증가했다는 뜻이다. '+conditional),
                ('DOWN에서 현재 F의 의미','해당 피처가 직전까지 감소했다는 뜻이다. '+conditional),
                ('FLAT의 의미','H 전용 임계값 이내의 변화다. break 없음이나 현재 F가 0이라는 뜻이 아니다.'),
                ('reversal 뒤 의미','직전 16개 F 안에서 관측한 비평탄 방향 전환이다. 긴 FLAT 구간과 창 밖 사건은 놓칠 수 있고 실제 구조변화 시점이 아니다.'),
                ('자기 과거 예측력','정답을 쓰지 않은 A 기준으로 선정되었다. T2 대 T1의 held-out 상태전이 비교: '+transition_text),
                ('H-center 대비 추가 정보','H 중앙값 기준의 이득과 TRAIN 평균 상수 기준의 이득을 분리했다. '+ '; '.join(f"h{r['horizon']}: D1이 TRAIN 상수를 이긴 CI 패널 {r['positive_ci_panels']}/15, 평균 이득 {number(r['mean_TRAINconstant_to_D1_gain'])}" for r in a)),
                ('다른 피처의 추가 정보',f'기계적 계보 {lineage}. D5 비교와 parent 제거 비교가 필요하다. 인증된 유입 parent는 {len(edges)}개다.'),
                ('cross parent',str(edges) if edges else '없음. 예측 그래프의 빈 자리에 edge를 만들지 않았다.'),
                ('시간 일반화','EARLY 모델은 t≤199에서 따로 학습하고 다른 held-out series의 t≥400에서 평가했다. OUTER_EARLY_LATE_MOVEMENT_RESULTS.csv 및 CONDITIONAL_ORIENTATION_RESULTS.csv 참고. 인증 규칙의 시간 재현은 '+('통과.' if rules else '미확인.')),
                ('H 일반화','H 길이·의존성·꼬리는 H에서 정해지는 series 배경이다. series 내부 동적 전환과 구분한다. 조건부 방향의 H 재현은 '+('인증 규칙에 기록.' if rules else '미확인.')),
                ('BL80 대비 추가 정보',f"전체 M7의 NORMAL 평균 Δ={number(normal['point'])}. 개별 F의 BL80 기여도로 해석하지 않는다."),
                ('사용할 수 없는 상황',LIMIT_KO+' H<160 또는 마지막16F 유효성 부족이면 movement UNKNOWN. 공통 범위 밖에서 인증된 방향을 외삽하지 않는다.'),
                ('fallback','내부 M7 적격성 미통과 시 해당 excluded-fold BL80를 정확히 사용한다. 배포 primary는 BL80다. B가 비어도 M1~M3의 사전등록 진단은 남는다.')]
            assert len(fields)==17
            cards.append(f'\n## F{j:03d} — {title}\n\n'+'\n\n'.join(f'{k+1}. **{name}**: {value}' for k,(name,value) in enumerate(fields)))
        atomic(BASE/'FEATURE_CARDS_NEXT9_KO.md','\n'.join(cards)+'\n')
        normal_m7=next(r for r in comparison if r['mode']=='NORMAL' and r['method']=='M7')
        normal_m0=next(r for r in comparison if r['mode']=='NORMAL' and r['method']=='M0')
        answers=[f'153개 중 반복 선정된 movement 후보는 {len(candidates)}개다: '+', '.join(f'F{j:03d}' for j in candidates),
            f'운영 정의인 H 중앙값과 현재 F persistence를 함께 이긴 A 후보는 {len(candidates)}개다. TRAIN 평균 상수 대비 이득은 CENTER_DECOMPOSITION_SUMMARY_NEXT9.csv에 따로 기록했다.',
            f'엄격한 EARLY 후보 선정과 원래 EARLY→LATE 보조 예측 기준을 통과한 후보 수는 각 candidates/outer_*/FROZEN.json에 고정되어 있다. 합집합을 전체 fold 공통 통과로 바꾸지 않는다.',
            f'기본 S1 UP/DOWN 공통 범위와 held-out 지원 조건을 NORMAL 4/5 fold 이상 충족한 후보는 {support4}개다.',
            f'대조모형 검정을 무시한 원시 반대 방향 패턴이 같은 피처·상태쌍·방향으로 4/5 fold 이상 나타난 조합은 {raw4}개다. 이것만으로 가변 피처를 인증하지 않는다.',
            f'모든 조건을 통과한 최종 가변 피처는 {cert["certified_variable_count"]}개다.',
            '자기 상태전이 예측은 SELF_TRANSITION_RESULTS.csv의 T2 대 T0/T1 및 같은 입력 T3 비교로 판단한다. B 미통과로 M5 분기가 빠진 결과를 전이 모델의 용량 부족으로 해석하지 않는다.',
            f'인증된 cross parent edge는 {graph["cross_edge_count"]}개다.',
            'state-only·고정 비선형 O1을 모두 이겨야 variable 효과다. 후보별 원시 방향 차이와 이 추가효과는 CONDITIONAL_ORIENTATION_RESULTS.csv에서 분리했다.',
            'M6 대 TABULAR는 입력이 같다. M5 대 TABULAR는 M6=M5인 경우에만 같은 입력이다. 비유의는 동등성 증거가 아니며, 연구 signal 비열등 한계는 사전고정 .001이다.',
            f'M7 평균 TS-AUC={normal_m7["mean_TS_AUC"]}, BL80={normal_m0["mean_TS_AUC"]}. 점수 gate 결과: {science["C"]["promotion_score_gates"]}. 최종 runtime·누수·재현 승인은 별도다.',
            f'정확한 NORMAL 평균 TS-AUC 차이: {number(normal["point"])}.',
            'NORMAL fold별 Δ: '+', '.join(number(v) for v in normal['fold_deltas'])+'.',
            f'전체 원래 group을 2,000회 재표본화하고 5개 fold를 동일 가중했다. 95% CI=[{number(normal["ci_low"])}, {number(normal["ci_high"])}]. 고정 학습 절차 조건부 불확실성이며 RP 반복 사용 전체를 보정한 독립 확인이 아니다.',
            f'EARLY→LATE 평균 M7−BL80 Δ={number(early["point"])}. TIME/H의 지원 부족은 NA로 남기며 TIME_GENERALIZATION_NEXT9.csv, H_GENERALIZATION_NEXT9.csv와 점수 gate를 함께 본다.',
            '현재 primary=BL80, robustness=D2, distilled=J102. canonical 역할은 변경하지 않았다.',
            '새 연구 역할은 B·같은 입력 비교·누수 감사가 모두 통과할 때만 부여한다. VARIABLE_CERTIFICATION_NEXT9.json과 FINAL_AUDIT_NEXT9.json을 함께 확인한다.',
            '허용된 기존 RP 6,400 series만 사용했다. Shadow/Seal 관측값·정답·예측·점수 및 reduced/public/cloud 신규 사용은 0이다. 최종 접근 감사는 LEAKAGE_AUDIT_NEXT9.json에 기록한다.',
            '핵심 병목: 조건부 반대 방향의 반복 검증, 자기 예측력과 필터 기억의 구분, 반복 사용한 RP에서 미세 점수 차이를 해석하는 한계.',
            '다음 가설: 동일 F 범위에서 movement가 state-only보다 추가 정보를 주는가; 긴 FLAT 구간의 관측 한계가 reversal 해석을 바꾸는가; TRAIN 평균과 같은 계보 통제 뒤에도 자기 예측력이 남는가. 새 후보 탐색은 다음 세션의 사전등록 뒤 수행한다.']
        assert len(answers)==20
        atomic(BASE/'SCIENTIFIC_READ_FIRST_NEXT9_KO.md','# NEXT9 과학 결과: 20개 질문\n\n최종 감사·전달 확인은 00_READ_FIRST_NEXT9_KO.md를 기준으로 한다.\n\n'+'\n\n'.join(f'{i+1}. {answer}' for i,answer in enumerate(answers))+'\n')
        atomic(BASE/'NEXT_SESSION_PLAN_NEXT9.md','# 다음 세션\n\n'+answers[-1]+'\n\n현재 과학 판정: '+science['B']['status']+'\n\n기존 RP는 반복 개발에 사용됐다. 현재 결과를 독립 hidden 확인으로 바꾸어 표현하지 않는다. Shadow/Seal 접근과 canonical 변경은 이번 세션에서 하지 않았다.\n')
        atomic(BASE/'SCIENTIFIC_REPORTS_DONE_NEXT9.json',dict(status='COMPLETE',timestamp=utc_now(),feature_cards=len(candidates),fields_per_card=17,read_first_questions=20,
            source_science_sha256=digest(BASE/'SCIENTIFIC_RESULTS_NEXT9.json'),final_audit_pending=True,
            files_sha256={name:digest(BASE/name) for name in ['FEATURE_CARDS_NEXT9_KO.md','SCIENTIFIC_READ_FIRST_NEXT9_KO.md','NEXT_SESSION_PLAN_NEXT9.md','TRANSITION_PATH_FEATURES.csv','CENTER_DECOMPOSITION_SUMMARY_NEXT9.csv']}))

if __name__=='__main__':main()
