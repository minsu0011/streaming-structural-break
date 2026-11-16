"""Write the Korean45-question guide and concrete GPT handoff from final evidence."""
from pathlib import Path
import csv
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.next7.common import BASE,read_json,save,digest,utc_now,activity,note


def rows(name):
    with (BASE/name).open(newline='',encoding='utf-8-sig') as handle:return list(csv.DictReader(handle))


def table(headers,records):
    def cell(value):return str(value).replace('|',' / ').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(cell,headers))+' |','| '+' | '.join('---' for _ in headers)+' |']+
        ['| '+' | '.join(map(cell,row))+' |' for row in records])


def pct(value):return f'{100*float(value):.2f}%'
def dec(value):return '해당 없음' if value in ('',None) else f'{float(value):+.6f}'
def hours(value):return f'{float(value)/3600:.3f}시간 ({float(value):.1f}초)'


def choose(data,**keys):
    found=[r for r in data if all(str(r.get(k))==str(v) for k,v in keys.items())]
    assert len(found)==1,(keys,len(found))
    return found[0]


def main():
    final=read_json(BASE/'FINAL_AUDIT_NEXT7.json')
    assert final['status']=='PASS' and final['decision']=='NEXT7_NO_PROMOTION'
    assert final['productive_research_seconds']>=36000 and final['final_audit_seconds']>0
    assert final['final_audit_target_seconds']==3600 and final['final_audit_target_is_approximate']
    comparison=rows('MODEL_COMPARISON_NEXT7.csv')
    m={r['method']:r for r in comparison}
    best=m['I4_EVENT_SCALE_TREE']
    assert final['best_NEXT7_research_candidate']['method']==best['method']
    shock=rows('SHOCK_EVENT_CONTAMINATION_NEXT7.csv')
    scale=rows('SCALE_UP_DOWN_RESULTS_NEXT7.csv')
    null=rows('LONG_NULL_RESULTS_NEXT7.csv')
    classifier=rows('CLASSIFIER_STRESS_RESULTS_NEXT7.csv')
    classifier_paired=rows('CLASSIFIER_STRESS_PAIRED_NEXT7.csv')
    h_transport=rows('H_LENGTH_CALIBRATION_NEXT7.csv')
    crossed_null=rows('CONDITIONAL_NULL_RESULTS_NEXT7.csv')
    crossed_h=rows('CROSSED_H_LENGTH_CONTRASTS_NEXT7.csv')
    timing=rows('TIME_FRACTURE_NEXT7.csv');age=rows('BREAK_AGE_NEXT7.csv');regime=rows('REGIME_RESULTS_NEXT7.csv')
    warm=rows('LATENCY_NEXT7.csv');throughput=rows('DIRECT_10M_THROUGHPUT_NEXT7.csv')
    influence=read_json(BASE/'SHOCK_INFLUENCE_AUDIT_DONE.json')
    assert influence['status']=='COMPLETE' and influence['original_pairs_checked']==1536
    ceiling=read_json(BASE/'OBSERVED_SCALE_ADMISSION_DONE_NEXT7.json')
    assert ceiling['status']=='COMPLETE' and ceiling['RP_rows_checked']==3203157
    scale_paths=read_json(BASE/'SCALE_EVIDENCE_PATH_AUDIT_DONE_NEXT7.json')
    assert scale_paths['status']=='COMPLETE' and scale_paths['original_series_replayed']==2048
    scale_first=rows('SCALE_FIRST_EVENT_FAILURE_PARTITION_NEXT7.csv')
    questions=[]
    def qa(number,title,answer):
        assert number==len(questions)+1
        questions.append(f'### {number}. {title}\n\n{answer}')
    def state_value(data,case,method,metric,horizon=1024):
        return choose(data,case=case,method=method,metric=metric,horizon=horizon)['estimate']
    def classifier_value(case,method,horizon,metric='any_alarm'):
        return choose(classifier,case=case,method=method,horizon=horizon,metric=metric)['mean_across_fixed_folds']
    def shock_table(case):
        return table(['기억 방식','1,024개까지 한 번 이상 기억','대조에는 없는 추가 기억','4,096개까지 한 번 이상 기억'],[
            [label,pct(state_value(shock,case,method,'any_latch')),pct(state_value(shock,case,method,'excess_latch_vs_paired_control')),
             pct(state_value(shock,case,method,'any_latch',4096))]
            for method,label in [('NEXT6_HARD4_EVENT','기존 네 블록 확인'),('G1_RAW_EVENT','G1 원시 사건 기억'),('G2_QUALIFIED_EVENT','G2 H 보정 사건 기억')]])
    def classifier_shock_table(case):
        def display(method,horizon,metric):
            value=classifier_value(case,method,horizon,metric)
            return '미측정' if value in ('',None) else pct(value)
        return table(['최종 분류기','관측 길이','충격 경로 경보','대조 경로 경보','충격 경로에만 경보'],[
            [label,horizon,display(method,horizon,'any_alarm'),display(method,horizon,'control_any_alarm'),
             display(method,horizon,'treatment_alarm_control_never')]
            for horizon in (1024,4096)
            for method,label in [('M0','BL80'),('M3','이전 NEXT6 M3'),('I4_EVENT_SCALE_TREE','I4'),
                                 ('G2_FULL','G2 전체'),('G3_PAIRWISE_FULL','G3')]])
    def temporary_change_table():
        records=[]
        for case,label in [('TEMP_SCALE_PULSE','32개 관측 뒤 회복'),('RETURN_NORMAL','256개 관측 뒤 회복')]:
            for horizon in (1024,4096):
                paired=choose(classifier_paired,case=case,horizon=horizon,candidate='G3_PAIRWISE_FULL',
                    reference='M0',metric='postbreak_first_detection')
                records.append([label,horizon,pct(classifier_value(case,'M0',horizon,'postbreak_first_detection')),
                    pct(classifier_value(case,'G3_PAIRWISE_FULL',horizon,'postbreak_first_detection')),
                    f"{100*float(paired['mean_delta']):+.2f}%p",
                    f"[{100*float(paired['CI_low']):+.2f}, {100*float(paired['CI_high']):+.2f}]%p"])
        return table(['변화 지속 길이','관측 길이','BL80 첫 검출','G3 첫 검출','대응 차이','원래 95% 구간'],records)
    qa(1,'실제 productive research 시간은?',hours(final['productive_research_seconds'])+
       '. 실행 중인 계산과 실제 작업이 확인된 시간의 합집합이다. 대기·모니터링만 한 시간과 병렬 작업의 중복 시간은 더하지 않았다. 최종 감사 '+hours(final['final_audit_seconds'])+'는 별도다.')
    qa(2,'wall-clock은?',hours(final['wall_seconds_at_computation_audit'])+
       '. 최종 계산 감사 기록 시점까지의 경과 시간이다. 압축·해제 재현·바탕화면 복사가 끝난 시점의 wall-clock은 `FINAL_DELIVERY_AUDIT_NEXT7.json`에 별도로 기록한다.')
    qa(3,'실험 수는?',f"구조 계열 6개, screening 후보 7개, 전체 5-fold 후보 15개다. 감사 시점 실행 장부의 작업 단위는 {final['ledger_work_units_at_snapshot']:,}개지만, 시드·H·fold별 작업과 검증을 별도 실험 후보로 세지 않았다. 학습된 모델 파일은 inner 기준 모델 30개와 보정 head 25개, 합계 55개다.")
    qa(4,'full candidate 수는?','15개다. 허용된 16개 이내이며 학습 보정 recipe 5개를 사용했다. 좋은 숫자를 찾기 위해 남은 한 칸을 채우지 않았다.')
    qa(5,'시작 primary는?','기존 BL80이다. J102 80%와 C205 20%의 고정 결합이다. 이번 RP6400의 원래 grouped 5-fold 기준 평균 TS-AUC는 '+m['M0']['mean_TS_AUC']+'다. 과거 DEV 점수와 섞어 비교하지 않는다.')
    qa(6,'최종 primary는?','동일한 BL80이다. 기존 D2·J102 참조도 보존했다. 새로운 NEXT7 운영 역할은 선택하지 않았다.')
    qa(7,'exact TS-AUC delta는?',f"실제 primary 교체의 Δ는 **0**이다. NEXT7 중 가장 높은 I4의 평균 TS-AUC는 **{best['mean_TS_AUC']}**, 원래 fold끼리 짝지은 평균 Δ는 **{best['mean_matched_delta_M0']}**다. 이는 개선이 아니라 감소다.")
    qa(8,'confirmation-only 구조는 동작했는가?',
       '확인 후에만 사건 기억을 갱신하는 규칙은 동작했다. 원시 관측 16개로 새 블록을 만들고, 네 완성 블록의 반복 증거가 확인 기준을 넘을 때만 갱신한다. 현재 상태가 회복돼도 사건 기록은 남는다. 다만 엄격한 원시 입력 비중복 요구는 새 scale 가지에서 충족하고, 승계한 core 가지에서는 미충족이다. core의 lag 특징은 인접 블록 사이에 원시 관측 최대 4개가 겹친다. 기존 구현의 보존이 이 요구를 면제하지 않으며 전체 비중복 계약이 통과했다고 주장하지 않는다. 설령 원시 관측이 겹치지 않더라도 통계적 독립을 뜻하지 않는다. `FRESH_BLOCK_REQUIREMENT_AUDIT_NEXT7.json`에 구분했다. “확인”은 정해진 증거 기준을 통과했다는 뜻이며 실제 구조변화가 확실하다는 뜻은 아니다.')
    qa(9,'한 번의 shock가 event memory를 오염시키는 비율은?',shock_table('O1')+
       '\n\n첫 열의 기억률에는 원래 정상 경로에서도 생겼을 오류가 포함된다. 추가 기억은 같은 H·난수의 대조 경로에는 없는 기억의 비율이다. 최종 분류기의 경보율은 다음과 같이 별도다.\n\n'+classifier_shock_table('O1')+
       '\n\n각 분류기의 고정된 별도 기준을 적용했다. 확인 후에만 기억을 갱신하는 규칙이 분류기 경보까지 확인 이후로 제한하는 것은 아니다. 기존 화면에 고정된 O1 seed 0 예제는 원시·보정 사건 기억이 모두 1,024개 관측 내내 0이어도 BL80과 G3가 경보를 냈고, 대조 경로에는 그 경보가 없었다. 한 예제를 전체 경로의 원인 비율로 확대하지 않는다. M3의 과거 paired-control 미측정 칸은 채우지 않았다. 전체 방법의 대응 구간은 `CLASSIFIER_STRESS_PAIRED_NEXT7.csv`에 있다.')
    qa(10,'16-point shock 결과는?',shock_table('OB16')+
       '\n\n단일 극단값을 막는 성질이 16개 연속 오염까지 자동으로 보호하지는 않는다. 분류기 경보는 별도로 비교했다.\n\n'+classifier_shock_table('OB16'))
    qa(11,'repeated shock 결과는?',shock_table('REPEATED_SHOCK')+
       '\n\n128개 간격으로 네 번 가한 기존 고정 충격이다. 원래 1,536개 충격 쌍 전체에서 분석한 결과, 국소 증거와 현재 상태는 유한한 영향 범위 뒤 대조 경로와 비트 단위로 같아졌다. 사건 기억의 차이는 별도로 남을 수 있다. `SHOCK_PERSISTENT_MEMORY_DIFFERENCES_NEXT7.csv`에 그 비율과 기존 2,000회 구간을 기록했다.\n\n'+classifier_shock_table('REPEATED_SHOCK')+
       '\n\n대조에 없는 경보만 줄었다고 강건성 향상으로 해석하면 안 된다. 4,096개에서 G3는 BL80보다 충격 경로 전체 경보가 0.390625%p 늘고, 대조 경로 경보는 2.2265625%p 늘었다. 두 경로 모두 경보를 내는 경우가 늘면서 “충격 경로에만 경보”는 1.8359375%p 줄었다. 이 endpoint 감소는 전체 경보 감소가 아니다. 원래 대응 구간과 두 경로의 결합 확률을 함께 보존했다.')
    qa(12,'event memory는 recovery 후에도 유지되는가?',
       '그렇다. RETURN_NORMAL 예제 seed 0에서는 현재 상태가 다시 정상이어도 사건 기억과 사후 정답 1이 유지된다. 화면의 현재 경보 여부는 별도 표시한다. 기억을 유지하는 계약을 구현한 것과, 오기억까지 포함한 전체 RP 순위를 개선하는 것은 별개의 검증이다.\n\n'+temporary_change_table()+
       '\n\n두 조건 모두 같은 기존 N2 배경에서 관측값을 √2배했다가 되돌리는 실제 일시 변화이며, 처음 변한 시점부터 정답은 계속 1이다. 첫 검출은 변화 전에 경보하지 않고 변화 이후 처음 경보한 비율이다. 256개 관측의 변화에서는 G3가 개선됐지만 32개 관측의 짧은 변화에서는 개선을 확인하지 못했다. 이 표의 검출률은 기억 유지율이나 회복 속도가 아니다. 전체 방법과 원래 대응 구간은 `CLASSIFIER_STRESS_RESULTS_NEXT7.csv`, `CLASSIFIER_STRESS_PAIRED_NEXT7.csv`에 있다.')
    qa(13,'false latch long-null 결과는?',table(['배경','G1: 1,024 / 4,096 / 16,384','G2: 1,024 / 4,096 / 16,384'],[
        [case,*[' / '.join(pct(state_value(null,case,method,'any_latch',h)) for h in (1024,4096,16384))
                for method in ('G1_RAW_EVENT','G2_QUALIFIED_EVENT')]] for case in ('N0','N1','N2','N3')])+
        '\n\n원시 G1의 긴 정상 경로 오기억은 여러 배경에서 50%를 넘었다. H 보정이 이를 줄여도 정상상태에서 한 번 잘못 붙은 기억이 계속 남는 문제가 해결되지는 않았다. 이 표는 사건 기억률이다.')
    qa(14,'1,024 / 4,096 / 16,384 결과는?',table(['배경','BL80 경보율: 1,024 / 4,096 / 16,384','I4 경보율: 같은 길이','G2 전체 경보율: 같은 길이'],[
        [case,*[' / '.join(pct(classifier_value(case,method,h)) for h in (1024,4096,16384))
                for method in ('M0','I4_EVENT_SCALE_TREE','G2_FULL')]] for case in ('N0','N1','N2','N3')])+
        '\n\n각 모델의 별도 고정 경보 기준을 적용한 분류기 결과다. N0·N1과 N2·N3의 양상이 다르다. M0에 0.9를 곱하고 경보 기준도 0.9배하면 경보는 정확히 같다. 기준을 그대로 두고 경보만 줄어든 현상을 개선으로 세지 않았다. 전체 8개 방법과 대응 구간은 `CLASSIFIER_LONG_NULL_RESULTS_NEXT7.csv`, 네 배경 비교 이미지는 `visuals/classifier_nulls/four_backgrounds_001/`에 있다.')
    def scale_table(cases):
        return table(['상황','기존 core 첫 검출','G1 첫 검출','G2 첫 검출'],[
            [case,*[pct(state_value(scale,case,method,'first_detection')) for method in
                    ('NEXT6_HARD4_EVENT','G1_RAW_EVENT','G2_QUALIFIED_EVENT')]] for case in cases])
    qa(15,'scale expansion 성능은?',scale_table(('V1','V2'))+
       '\n\n1,024개까지의 첫 사건 검출률이다. 변화 전에 이미 오기억이 생긴 경로는 올바른 최초 검출로 세지 않는다. 실제 분류기의 고정 기준 검출률·지연과 전체 비교는 `SCALE_EXPANSION_CONTRACTION_RESULTS.csv`에 함께 두되 record_type으로 구분했다.')
    qa(16,'scale contraction 성능은?',scale_table(('V3','V4'))+
       '\n\nV3는 분산0.7배, V4는 분산0.5배다. 표준편차는 각각 약0.837배,0.707배이므로 V4를 표준편차 절반으로 읽지 않는다. block log-MAD의 부호 있는 대칭 증거를 사용해 큰 축소 V4를 더 잘 드러냈다. 작은 축소와 전체 분류 순위의 어려움은 남았다.')
    qa(17,'V3/V4는 개선됐는가?',
       '기존 core에 비해 V4의 사건 검출은 개선됐다. G2의 H 보정은 G1의 검출을 일부 줄였다. RP의 사후 scale-decrease 구간에서는 G2 전체 보정의 이득이 관찰됐지만, 그 구간은 전체 RP의 일부이고 사후 진단이다. 이 결과를 전체 TS-AUC 개선이나 새 robust 역할로 승격하지 않았다.\n\n'+
       table(['1,024개까지의 G2 최초 사건 결과','V3','V4'],[
           [label,*[pct(choose(scale_first,case=case,horizon=1024,category=category)['fraction']) for case in ('V3','V4')]]
           for category,label in [('correct_first_qualification','올바른 최초 확인'),('prebreak_qualified_memory','변화 전 오기억'),
                                 ('no_postbreak_raw_confirmation','변화 뒤 원시 확인 없음'),('postbreak_raw_confirmation_rejected','원시 확인 뒤 보정에서 모두 거절')]])+
       '\n\nscale 증거 상한 때문에 공통 확인 문턱을 절대 넘지 못하는 조건도 있다. 그러나 V4의 1,024개 내 최초 검출 실패 332개 중 해당 조건과 겹친 경로는 11개뿐이다. 이것만을 누락의 주원인으로 보지 않는다. RP에서는 지원 가능한 원시 scale 확인 2,275건 중 456건이 상한 조건에, 596건이 상한보다 낮은 문턱에 걸렸다. 이는 반복 블록 수이며 독립 표본 수나 검출률이 아니다. core가 대신 확인할 수도 있다.')
    qa(18,'H 길이별 calibration은 안정적인가?',table(['정상 배경','H1000','H1500','H2048','H3000','H5000'],[
        [case,*[pct(choose(h_transport,case=case,H_length=length,horizon=16384,method='G2_QUALIFIED_EVENT',metric='any_latch')['estimate'])
                for length in (1000,1500,2048,3000,5000)]] for case in ('N0','N1','N2','N3')])+
        '\n\n공통 H5000의 고정 접미부와 같은 난수를 사용한 H 길이 이동 검사다. 길이를 늘릴수록 좋아진다는 단조 관계는 없었다. 아래는 별도로 완료한 12개 512×512 교차 검사의 16,384개 관측 시점 G2 오기억률이다.\n\n'+
        table(['배경','H1000','H2048','H5000','H5000−H2048 대응 차이 / 95% 구간 (%p)'],[
            [case,*[pct(choose(crossed_null,case=case,H_length=length,horizon=16384,metric='G2_QUALIFIED_EVER_LATCH')['mean_crossed_rate'])
                    for length in (1000,2048,5000)],
             ' / '.join(f"{100*float(choose(crossed_h,case=case,H_candidate=5000,H_reference=2048,horizon=16384,metric='G2_QUALIFIED_EVER_LATCH')[key]):+.2f}"
                        for key in ('mean_rate_delta','CI_low_approx_pointwise','CI_high_approx_pointwise'))]
            for case in ('N0','N1','N2','N3')])+
        '\n\nH별 위험 차이, 경험적 재표본 지지집합 밖의 초과, 지지집합 안의 꼬리 초과, 이른 확인이 남은 기억을 분리했다. 예를 들어 N1에서 H5000은 H2048보다 지지집합 밖의 초과가 줄어도 범위 안의 초과가 늘어 전체 오기억이 증가했다. 이는 H 길이 외에도 정규화 기준과 보정 임계값이 함께 바뀐 고정 절차의 비교이며, 순수한 표본 수 증가의 인과 효과가 아니다. 262,144개 셀을 독립 표본으로 세지 않았고 구간은 항목별 근사 구간이다. 고정 길이 보정을 조건부 5% 또는 모든 시점의 5% 보장이라고 부르지 않는다.')
    qa(19,'current state score는?',f"현재 상태 단독 순위 점수 CURRENT_ONLY의 평균 TS-AUC는 {m['CURRENT_ONLY']['mean_TS_AUC']}다. 상태 화면의 색이나 증거 크기는 보정된 변화 확률이 아니다.")
    qa(20,'event memory score는?',f"EVENT_ONLY의 평균 TS-AUC는 {m['EVENT_ONLY']['mean_TS_AUC']}다. 기억을 유지한다는 성질만으로 분류 순위가 높아지지는 않았다.")
    qa(21,'BL80 + state는?',f"고정 M_CURRENT: {m['M_CURRENT']['mean_TS_AUC']}, 평균 대응 Δ {m['M_CURRENT']['mean_matched_delta_M0']}. 학습 I3(current+event): {m['I3_CURRENT_EVENT']['mean_TS_AUC']}다.")
    qa(22,'BL80 + event는?',f"원시 M_EVENT: {m['M_EVENT']['mean_TS_AUC']}. H 보정 G2_EVENT: {m['G2_EVENT']['mean_TS_AUC']}. 잘못 붙은 음성 기억을 줄이는 이득과, 실제 변화의 양성 기억을 잃는 손해가 함께 나타났다.")
    qa(23,'BL80 + state + event + scale은?',table(['방법','평균 TS-AUC','BL80 대비 대응 Δ'],[
        [method,m[method]['mean_TS_AUC'],m[method]['mean_matched_delta_M0']]
        for method in ('M_FULL_FIXED','M_FULL','I4_EVENT_SCALE_TREE','G2_FULL','G3_PAIRWISE_FULL')])+
        '\n\nG3는 같은 입력에서 학습 목적만 같은 시점의 양성·음성 쌍 비교로 바꾼 것이다. 전체 성능은 개선되지 않았다.')
    qa(24,'5-fold scores는?',table(['원래 fold','BL80','I4','G2 전체','G3','I4 대응 Δ'],[
        [fold,*[f"{float(m[method][f'fold_{fold}_TS_AUC']):.9f}" for method in
               ('M0','I4_EVENT_SCALE_TREE','G2_FULL','G3_PAIRWISE_FULL')],dec(best[f'fold_{fold}_matched_delta_M0'])]
        for fold in range(5)])+'\n\n원래 동일한 그룹 fold끼리 비교했다. 행을 무작위로 섞는 CV를 사용하지 않았다.')
    qa(25,'matched delta는?',best['mean_matched_delta_M0']+' (I4). 각 원래 fold에서 I4−BL80을 먼저 계산한 뒤 평균했다. 다른 실험의 가장 높은 fold끼리 짝지은 값이 아니다.')
    qa(26,'median은?',f"I4의 **대응 fold 차이의 중앙값**은 {best['median_matched_fold_delta_M0']}다. 별개 통계량인 **각 방법 fold 점수 중앙값의 차이**는 {best['difference_of_fold_score_medians_M0']}다. 뒤의 양수를 전자의 개선처럼 쓰면 안 된다. `MEDIAN_ESTIMAND_CLARIFICATION_NEXT7.json`에서 구분했다.")
    qa(27,'worst fold는?',f"I4의 최악 대응 fold 차이는 {best['worst_matched_fold_delta_M0']}다. 가장 낮은 I4 fold 점수 자체는 {best['worst_fold_TS_AUC']}다. robust 역할의 ‘최악 대응 fold가 0 이상’ 기준을 통과하지 못했다.")
    qa(28,'bootstrap CI는?',f"I4의 평균 대응 Δ 95% 구간은 [{best['mean_delta_CI_low']}, {best['mean_delta_CI_high']}], Δ가 양수인 재표본 비율은 {best['probability_mean_delta_positive']}다. 원래 fold 안에서 전체 시계열 단위의 동일한 2,000회 재표본을 사용했다. RP6400은 실제로 6,400개 서로 다른 역사 그룹이라 이 자료에서 시계열 단위와 그룹 단위가 일치한다. 추가 draw로 결론을 뒤집지 않았다.")
    qa(29,'ALT는?','미실행이다. 사전 strong-candidate 조건을 통과한 후보가 없었다. ALT 점수 칸을 0으로 채우지 않았고 새로운 ALT seed를 고르지 않았다.')
    regime_rows=[r for r in regime if r['candidate']=='G2_FULL' and r['reference']=='M0' and r['scope'].startswith('regime_')]
    qa(30,'H regime 결과는?',table(['기존 H regime','G2 전체 대응 Δ','95% 구간'],[
        [r['scope'],dec(r['estimate_mean_matched_delta']),f"[{dec(r['mean_delta_ci_2_5'])}, {dec(r['mean_delta_ci_97_5'])}]"] for r in regime_rows])+
        '\n\n기존 H-regime 변환에는 NEXT5 이전 전체 DEV H로 만들어진 역사적 transductive 요소가 있다. 여기서는 고정된 사후 묶음별 진단으로만 썼으며, outer 학습으로 검증된 새 입력이나 독립적인 외부 검증으로 보지 않는다. 이번 세션에서 Shadow 관측값을 새로 읽은 것은 아니다.')
    def fracture_table(data):
        selected=[r for r in data if r['candidate']=='G2_FULL' and r['reference']=='M0']
        return table(['구간','G2 전체 대응 Δ','95% 구간'],[[r['scope'],dec(r['estimate_mean_matched_delta']),
            f"[{dec(r['mean_delta_ci_2_5'])}, {dec(r['mean_delta_ci_97_5'])}]"] for r in selected])
    qa(31,'time fracture는?',fracture_table(timing)+
       '\n\n같은 관측 시점의 양성과 음성을 비교한다. 특정 시간대의 양수를 전체 평균의 개선으로 해석하지 않았다. 구간별 95% 구간은 전체 비교에 대한 동시 보장이 아니다.')
    qa(32,'break-age fracture는?',fracture_table(age)+
       '\n\n양성의 변화 후 경과 시간만 제한하고, 해당 관측 시점의 음성 비교 대상은 모두 유지했다. 사후 정답 tau를 detector에 전달하지 않았다.')
    qa(33,'latency는?',table(['실제 scalar 방법','중앙값 μs','p95 μs','p99 μs','M0 대비 중앙값'],[
        [r['method'],f"{float(r['median_us']):.3f}",f"{float(r['p95_us']):.3f}",f"{float(r['p99_us']):.3f}",f"{float(r['median_ratio_vs_M0']):.3f}배"]
        for r in warm if r['kind']=='CLASSIFIER'])+
        '\n\n다른 연구 계산이 끝난 뒤 측정했다. H 초기화와 cold start는 별도 표에 있다. 7개 분류기 각각에 긴 단일 상태 1천만 개와 새 상태를 반복하는 1천만 개를 실제 처리해 총 1억4천만 scalar 호출을 확인했다. batch 속도를 scalar 속도로 제시하지 않았고, 빠르다는 이유로 성능 승격을 허용하지 않았다.')
    qa(34,'state memory는?',table(['방법','관측한 숫자 배열 bytes','상태 변경 배열 bytes','측정 범위'],[
        [r['method'],r['reachable_numeric_array_bytes_excluding_models'],r['bounded_mutable_stream_or_added_branch_bytes'],r['memory_byte_scope']]
        for r in warm if r['method'] in ('M0','I4_EVENT_SCALE_TREE','G2_FULL','G3_PAIRWISE_FULL')])+
        '\n\nPython 객체·불변 모델 트리는 별도다. NEXT7 방법의 변경 배열 값은 추가 상태 분기의 크기이므로 전체 프로세스 메모리 상한으로 읽지 않는다. 직접 1천만 개 처리 중 배열 크기가 늘지 않았다. G2는 16,384개 이후 새 qualified 확인을 보류하고 이전 기억을 유지한다.')
    qa(35,'leakage는?','최종 감사 통과. RP 회원·raw hash·원래 grouped split, 30개 inner 모델의 제외 그룹, 25개 head의 학습 경로와 원래 OOF 혼합을 확인했다. tau·미래 접미부·미래 전체 길이·dataset ID를 scalar 인터페이스에 넣는 시도는 거부됐다. 이 결과는 검토한 실행 경로와 애플리케이션 guard에 대한 감사이며 OS 수준 정보흐름 증명은 아니다.')
    qa(36,'determinism은?','고정 90개 경로의 scalar 결과, 역순 reset, 독립 상태의 교차 실행, 저장·복원 뒤 이어 처리가 원래 fixture와 비트 단위로 같았다. 별도 batch/scalar 비교의 허용 오차는 1e-12이고 실제 오차를 개별 기록했다. 다른 하드웨어나 라이브러리 버전의 비트 단위 동일성을 주장하지 않는다.')
    qa(37,'fresh reproduction은?','새 NEXT7 역할이 없으므로 raw RP부터 옛 모델을 의미 없이 다시 학습하지 않았다. 분리된 새 source/model 출력 폴더와 새 프로세스에서 5개 fold × 3개 고정 예제 × 6개 방법, 92,160개 scalar 출력을 정확히 재현했다. 별도로 기존 BL80·D2·J102의 실제 공개 infer 진입점 9개 경로, 9,216개 출력도 새 폴더에서 비트 단위로 재현했다. 이 기존 모델들의 재현은 과거 학습된 파일의 실행 검사이며 RP OOF 점수의 재평가가 아니다. 원래 프로젝트의 source/data/artifact 읽기는 차단하고 기존 Python 환경의 라이브러리는 사용했다. 최종 전달도 실제 ZIP 해제 후 두 재현 검사를 통과해야만 허용되며, 그 ZIP 해시와 결과는 외부 `FINAL_DELIVERY_AUDIT_NEXT7.json`으로 묶는다.')
    qa(38,'새 generation인가?','아니다. 평균 +0.0025, 양의 불확실성 구간, ALT 일관성 등 기존 조건을 충족하지 못했다.')
    qa(39,'새 robustness model인가?','아니다. 가장 좋은 I4도 최악 대응 fold가 음수였다. 여러 정상 배경에서 오경보가 증가했고, 축소 검출 일부의 이득으로 전체 강건성 기준을 대신하지 않았다.\n\n사전 설계의 한계도 있다. 사용자의 “대폭 감소”를 에이전트가 10%p 절대 감소로 정량화했는데, 기존 오기억률이 약5%인 일부 비교에서는 수학적으로 달성할 수 없는 조건이었다. 기준 모델·오류 endpoint·길이도 역할 필드에 더 분명히 명시할 필요가 있었다. 이를 `ROLE_GATE_EMPIRICAL_FEASIBILITY_NEXT7.csv`와 별도 연구 노트에 기록했다. 이번에 기준을 완화하거나 유리한 endpoint를 고르지는 않았다. 모든15개 후보는 별도로 명시된 최악 fold 비악화 조건도 실패하므로 이 설계 한계와 독립적으로 승격하지 않는다.')
    qa(40,'왜 이런 판정을 했는가?','모든 15개 NEXT7 후보의 평균 대응 Δ가 음수였다. 확인·회복·사건 기록을 분리하는 구현과 시각화는 완성했지만, 참 사건을 살리는 이득보다 잘못된 사건 기억이 순위를 해치는 손해를 충분히 줄이지 못했다. 결과는 E(실패 위치를 좁힘)와 F(검토한 구조의 강한 negative result)에 해당한다.')
    qa(41,'Shadow opened?','NO. 이번 세션에서 Shadow 관측값의 해독·학습·평가를 하지 않았다. 경계 확인용 회원 ID와 잠금 해시는 확인했다.')
    qa(42,'Seal A/B/C/D opened?','모두 NO. Seal 관측값을 해독하거나 평가하지 않았다. raw 파일 전체의 바이트 해시 검사는 데이터 무결성 감사이며 hidden 구간의 관측값 해독과 구분했다.')
    qa(43,'reduced/public usage?','모두 0. reduced 실험, public 평가, 외부 대회 데이터, cloud submission을 사용하지 않았다.')
    qa(44,'현재 가장 큰 병목 3개는?','1. H 보정용 기준과 실제 처리 기준이 달라지고, 유한 경험적 지지집합이 장기 정상 경로의 극단을 충분히 다루지 못하는 문제.\n2. 실제 변화의 기억을 보존할수록 잘못 붙은 음성 기억도 오래 남는 순위 손익 문제.\n3. 축소·초기 변화 등 일부 구간의 이득이 전체 RP의 일관된 이득으로 이어지지 않는 문제.')
    qa(45,'다음 연구 우선순위 3개는?','1. 엄격한 원시 입력 비중복을 충족하는 블록 대조를 설계하고 H 보정과 실제 처리의 기준을 일치시킨다. serial dependence·관측 길이에 대해 어떤 오류 보장을 목표로 하는지도 먼저 명시한다.\n2. 사건 기록의 보존과 그 기록을 분류 순위에 반영하는 신뢰도를 분리하는 작은 구조 가설을 새 프로토콜로 고정한다.\n3. 약한 축소 V3와 강한 축소 V4의 민감도·오경보·지연을 함께 검증한다. 큰 graph, 경계 미세탐색, 이미 실패한 recipe의 반복으로 우회하지 않는다.')
    assert len(questions)==45
    intro='''# NEXT7 먼저 읽기

**결론: NEXT7_NO_PROMOTION. 기존 BL80을 유지한다.** 현재 상태와 과거 사건 기억을 분리하는 동적 상태망은 구현했지만, 새 분기의 전체 RP 순위 성능은 개선되지 않았다. 실패 위치와 장기 오기억의 구조를 좁힌 결과를 남긴다.

바탕화면 결과 폴더에서는 **18번 HTML**을 열면 상태망을 직접 움직일 수 있다. **19번 PNG**는 현재 상태가 회복돼도 기억이 남는 예제다. **20번 ZIP**에는 전체 보고서·코드·검증표·그림·연구용 재현 bundle이 있다. ZIP의 `artifacts/next7/visuals/classifier_stress/full_suite_001/`에는 23개 시나리오 전체의 오경보·첫 검출 비교 그림도 있다. 기존 NEXT6 결과는 그대로 보존한다.

상태망의 선은 실제 관측된 상태 전이와 계산 관계를 보여준다. 학습된 인과관계나 새로운 거래 의사결정을 뜻하지 않는다. 기본 화면은 선택한 시점까지 표시하고, 미래 전체 경로와 정답은 별도 사후 보기로 구분했다.

표의 **사건 기억률**, **고정 기준 경보율**, **TS-AUC 순위 성능**은 서로 다른 값이다. 아래 45개 답변은 이 구분을 유지한다.
'''
    read_first=intro+'\n\n'+'\n\n'.join(questions)+'\n'
    handoff='''# NEXT7 GPT handoff

The completed decision is **NEXT7_NO_PROMOTION**. Preserve BL80 as primary and preserve existing D2/J102 references. The15 candidate results are completed negative experiments. Do not treat them as software failures to retry until positive. No NEXT7 deployment role was selected.

## Start here

Read `00_READ_FIRST_NEXT7_KO.md` for all45 user questions, `FINAL_AUDIT_NEXT7.json` for computation acceptance, and the external `FINAL_DELIVERY_AUDIT_NEXT7.json` for the exact delivered ZIP and20 desktop copies. The delivery receipt is outside the ZIP to avoid a self-hash cycle. `MODEL_COMPARISON_NEXT7.csv` preserves its pre-audit generation flags; final candidate decisions are in `FINAL_CANDIDATE_DECISIONS_NEXT7.csv`.

## Reproduction

From the extracted ZIP root, use a Python environment matching `artifacts/next7/ENVIRONMENT_NEXT7.json` and run:

```powershell
python -B scripts/next7/reproduce_research_bundle.py --output-directory demo_output
```

`demo_output` must be fresh. The exact fixture binds the reproducer source. Expected result:90 paths,92,160 scalar predictions; no model fitting and no raw competition data. All seven feature configurations are included. A new output directory and process were tested using the existing interpreter and libraries; this is not a clean-machine dependency-installation test.

The historical BL80, D2 and J102 tree-used models include their original public entries and required model descriptors. Their separate inference check is:

```powershell
python -B scripts/next7/reproduce_frozen_references.py --output-directory reference_output
```

Expected result:9 public-entry paths and9,216 bitwise-identical scalar predictions. The three preexisting references retain their historical training provenance, which can predate the NEXT5 RP/Shadow split; they are not newly fitted or newly evaluated on RP. This fixture checks portable inference only. BL80/J102 model.json files and D2 RESULTS.json are necessary loader dependencies. The D2 descriptor was outside the initial protected inventory; its current bytes are explicitly bound and its model hash matches the independently protected D2 model. The descriptors do not contain raw competition observations. The actual final ZIP must pass this replay as well as the90-path research replay.

These two commands are the portable, raw-free replay entry points. Research-training scripts, the full historical test suite, and the complete original stress/runtime drivers can require the authorized original workspace, raw RP data or legacy diagnostic artifacts intentionally excluded from the ZIP. Their source and completed audit evidence are included for review; do not infer that every historical driver can run from this handoff alone.

The five research fold bundles retain only I4, G2_FULL and G3 for diagnosis, plus their common base components. Their `RESEARCH_FOLD_ONLY_NOT_PROMOTED` role is enforced. They are not25 production choices and not all15 rejected candidates. Their default is M0; cross-fold recipe assignment is rejected.

## Fixed inference contracts

Inputs are H and the current finite float32 scalar. Tau, future suffix, future total length and dataset ID are excluded. H actual length routing is allowed. The current state may recover while the target1[t>=tau] and confirmed event record persist.

The core uses eight bounded rank channels,16 current observations per block, and four-block same-sign confirmation. Lag4 creates up to four raw points of overlap across neighboring blocks; the blocks are not independent. No RP graph edges were learned. The visual network is operational/debug evidence, not a causal graph.

The user's strict all-raw-input nonoverlap requirement is **not met by the inherited core**. New signed-scale raw blocks are disjoint. Distinct current timestamps, finite local shock influence and confirmation-only updates do not waive or prove the stricter requirement. `FRESH_BLOCK_REQUIREMENT_AUDIT_NEXT7.json` records the unmet requirement explicitly; final computation acceptance is an audit of the negative research and handoff, not approval of a strict-disjoint production detector.

S1 is signed log-MAD from each16-point raw block with an H-fitted robust reference. S2 is the registered H-PIT scale representation. C0/C1/C2 are the three tested bounded rules. S1 C1 and C2 produce identical confirmed scale events when their common strict cutoff is at least2: exactly two positive values bounded by4 cannot make the four-value mean exceed2. This applies to the fixed H1000/1500/2048 S1 cuts; it does not prove current-score or whole-prediction identity, nor all other H bins or S2.

G1 records confirmation-only maximum strength and capped episode/age/direction features. G2 replaces raw memory with H-qualified memory while preserving raw current-state inputs. Its nine admission horizons end at1024 blocks (16,384 points); subsequent new qualification abstains while prior memory and BL80 continue. G3 preserves G2 inputs/standardization and changes only the correction loss to within-time pairwise logistic loss.

## Validation design and its limits

RP6400 and the original PRIMARY grouped5-fold assignment are unchanged. Each outer training set has three inner group folds and30 total base reference fits. Each correction sees inner OOF BL80 predictions; outer validation is excluded. All five outer predictions are frozen before the one score release for each generation. Final lineage audit rechecks every inner row index, exact BL80 mixture and25 head artifacts.

The15 full candidates span six structural families and five learned correction recipes. The7 screens,30 inner references and55 total fitted models are different accounting units. Limits were not relaxed. ALT was not opened because no candidate met the fixed strong-candidate trigger. No new draws were added after inspecting intervals.

The agent's preregistered0.10 absolute shock-risk reduction quantified the user's nonnumeric request for a large reduction. It is unattainable for empirical references below0.10 even if the candidate risk is zero. The compact role field also needed more explicit reference, endpoint and horizon definitions. `ROLE_GATE_EMPIRICAL_FEASIBILITY_NEXT7.csv` and `research_notes/ROBUSTNESS_GATE_DESIGN_LIMITATION_KO.md` expose these design limitations without selecting a favorable endpoint or relaxing the gate. All15 candidates independently fail the user's explicit worst-fold-nonworse requirement, so this limitation does not determine the no-promotion outcome.

RP uncertainty uses the original2,000 paired whole-series draws within each original outer fold. All6,400 RP series have distinct historical groups. Median matched-fold delta and difference of score medians are reported separately. Pointwise intervals do not correct the whole adaptive research history.

Time and positive-age fractures retain all negative comparisons at the same original online time. Unsupported pair draws remain missing; sparse mechanism categories report usable draws. Observed-mechanism signatures are posthoc descriptions, not known DGP labels or training inputs. The historical H-regime transform has a pre-NEXT5 whole-DEV transductive history; its fixed RP group summaries are not independent external validation or outer-fitted predictive features.

Synthetic stress uses the original23 cases,512 fixed seeds, paired controls and registered H lengths. State memory and classifier alarms are separate tables. Classifier calibration is frozen per method/fold on the separate original calibration partition before evaluation. 0.9M0 with its own0.9 cutoff produces exactly M0's alarms; keeping the unscaled M0 cutoff only demonstrates an amplitude effect.

The12 crossed null grids use512 original H histories by512 original innovation paths. They do not contain262,144 independent observations. The same seed multiplicities weight both axes in each of the existing2,000 draws. This is an approximate same-seed-pair product bootstrap, not an invocation of an independent-factor pigeonhole-bootstrap theorem.

`CROSSED_RESAMPLING_EXACT_MOMENTS_NEXT7.csv` audits the exact finite-multinomial mean and variance of that frozen quadratic functional across all12 complete cohorts. It compares the original2,000 draws with the exact moments and provides an algebraic independent-axis variance comparator. No extra draws or replacement intervals were generated. Exact finite resampling arithmetic does not establish population coverage.

## Failure location

The G2 pseudo-online reference is initialized from H's first60%, while actual online normalization uses all H. Center/spread and rank-map transport differ. The complete23-covariate tables retain every planned diagnostic; correlations are posthoc associations and do not identify a causal contribution.

The rank-second-moment covariate is nearly deterministic at fixed fit sizes. Exact ties and floating-point reduction explain its tiny variation; it is not a new distributional-risk feature. The rank-tail IQR is quantized on a1/64 lattice before scaling/flooring. These arithmetic facts were checked over all6,144 H histories without online points.

The exact empirical bootstrap support envelope was enumerated with the original physical four-slot order, and all6,144 admission-cut reconstructions match the original bootstrap bitwise. At short H the long terminal cut already reaches that empirical envelope. More draws under the same support/order-statistic construction cannot push it beyond the envelope. This is a scoped obstruction, not a universal impossibility theorem. Partition long false memory into beyond-support, within-support-above-terminal-cut, and earlier-retained-only components. Within-support exceedance is not automatically Monte Carlo error.

The original1,536 paired shocks also receive a finite-influence audit. After the last changed point, lag4 and four completed blocks give an explicit local-evidence equality bound; two more recovery-block steps settle the current automata. Memory has no corresponding erasure requirement. Equal current evidence does not imply equal classifier predictions.

Confirmation-only memory does not make the integrated classifier a confirmation-only alarm system. In the complete O1 cohort at1024, BL80 any-alarm probability is.79140625 and G3 is.782421875; their paired difference interval crosses zero. In the already fixed O1 seed0 viewer example, both raw and qualified event memory stay exactly zero throughout1024 points, yet BL80 and G3 cross their own cutoffs and their paired controls do not. All eight exact alarm arrays were checked against the reviewed viewer. This demonstrates a separation of contracts, not a whole-cohort causal allocation of classifier errors; see `research_notes/CLASSIFIER_SHOCK_WITHOUT_EVENT_MEMORY_KO.md`.

The one-way endpoint P(treatment alarm and no control alarm) also differs from both the treatment alarm probability and their marginal difference. In repeated shock at4096, G3 treatment alarm probability rises by.00390625 versus M0, control alarm probability rises by.022265625, but the one-way endpoint falls by.018359375. In that comparison control-only alarm probability is zero in both methods, so the decrease is exactly the larger increase in simultaneous alarms. `SHOCK_CLASSIFIER_JOINT_PROBABILITIES_NEXT7.csv` and `SHOCK_CLASSIFIER_ENDPOINT_DECOMPOSITION_NEXT7.csv` audit all48 original shock/method/horizon rows and42 method comparisons against the existing means. Six missing M3 joint rows remain missing. These are arithmetic identities with recorded1e-12 tolerance, with no new intervals or claims of shock robustness.

Bounded scale evidence creates a separate admission obstruction: its raw ratio cannot exceed4/scale_cut, while G2 uses a strict shared core/scale cutoff. At or above that ceiling only core can win a new qualification, although earlier memory remains. This is not the dominant explanation of V4 misses by1024: only11 of332 missed original paths intersect the ceiling condition. On RP,2,275 supported raw scale confirmations partition into1,223 scale-cut passes,596 subceiling rejections and456 hard-ceiling rejections;322 of the1,052 scale rejections still qualify through core. These are dependent block-event counts, not detection rates. All3.2M cached RP cut/support/fresh-decision rows match their original H calibrations exactly. The separate original2,048 V1-V4 replays partition first-event errors and fully postbreak scale evidence without threshold changes. Opposing signed blocks lower the four-block mean even when three signs agree.

G3 exact full-training-pair gradients are evaluated at ZERO, G2_PARENT and frozen G3 coefficients in all five folds. They diagnose sampled-objective approximation without retraining. Strong-convexity objective/distance bounds apply only to the fixed regularized training objective, never to held-out AUC.

## Visual review

`visuals/v4/DYNAMIC_STATE_TRANSITION_NEXT7.html` is the final standalone/offline viewer. It preserves all eight examples and classifier curves from v3 and adds an exact confirmation explanation: recent scale signs, raw confirmation, shared H admission, core fallback, fresh qualification and retained memory are separate. Flags come from the already exported exact synthetic state arrays, before rounding; they use no future truth. All64 scenario/classifier combinations and extra boundary/initial/final explanation frames pass browser checks. Desktop and narrow views pass reset/playback/jump/prefix/paired controls with no external requests. The original v3 remains included. PNG/SVG counterparts and complete diagnostic figures are included.

`visuals/classifier_scale/original_cases_002/` compares all four original variance changes at1,024/4,096 points, with the original paired classifier intervals at1,024. These are own-calibrated first classifier alarms, distinct from event-memory admission. The displayed variance factors are not standard-deviation factors.

## Engineering and handoff boundaries

All6,333 prior protected authority files are rehashed in the final audit. The original appended NEXT6 delivery ledger row is explicitly reconciled with the archive. Seven failed activity-ledger work units and the repaired BOM queue quarantine remain visible, along with separate development/preflight errata; negative experiments are not erased. One-retry/dead-owner/checkpoint/quarantine behavior was exercised on owned processes. That is not a claim that every possible30-minute stall was reproduced.

One targeted unit-test invocation refreshed derived Numba caches outside NEXT7 because its cache environment was omitted. Source/model authority bytes were unchanged; the run was repeated with the NEXT7 cache and guarded receipt. Derived caches are not shipped. See `audit/TARGETED_TEST_CACHE_LOCATION_NOTE_001.json`.

Runtime is measured after competing research jobs finish, using actual scalar loops, separate cold/H initialization measurements and two direct10M workloads for each of seven classifiers. State-only10M estimates are explicitly marked as estimates. Array byte counts exclude Python overhead/immutable model arrays and distinguish added-branch from whole-state scope.

The ZIP excludes raw competition observations, hidden values, credentials, large feature caches, full real point OOF, individual RP event/tau tables, old archives, browser profiles, Numba caches and reproduction staging trees. It includes aggregate RP scores/uncertainty, selected research models, source/configs, synthetic diagnostic summaries and the fixed synthetic replay fixture. This can reproduce inference and review recorded uncertainty; complete RP refitting/scoring still requires the authorized original RP workspace.

## Next session

Do not reopen Shadow or Seal from this handoff. Both recommendations are NO. Read `NEXT_SESSION_PLAN_NEXT7.md` before designing a new recipe. Prioritize a consistent historical calibration target, the event-admission versus ranking tradeoff, and weak/strong scale contraction under the same grouped design. Do not resume edge searches or threshold micro-tuning.
'''
    limitations='''# NEXT7 한계

- 모든15개 후보의 평균 대응 Δ가 음수다. 국소 검출 이득, 시각화 완성도, 실행 속도는 운영 승격의 근거를 대신하지 않는다.
- 같은 RP에서 적응적으로 가설을 바꾼 개발 연구다.2,000회 구간은 전체 탐색의 선택 편향을 제거하지 않는다. ALT·Shadow·Seal은 새 검증 근거로 사용하지 않았다.
- 실제 상태와 통계적 독립성을 구분한다.16개 새 관측으로 만든 core 블록에도 lag4 원시 영향이 겹친다. S1 MAD 블록은 현재16개 원시값만 사용한다.
- 사건 기억 점수는 보정된 구조변화 확률이 아니다. 오확인이 기억에 들어가면 현재 상태가 정상으로 돌아와도 남을 수 있다. G2는16,384개 이후 새 확인을 보류하지만 기존 오기억이나 최종 분류기의 이후 경보를 없애지 않는다.
- 확인 후에만 사건 기억을 갱신한다는 규칙은 최종 분류기의 경보를 확인 이후로 제한하지 않는다. 단일 충격에서 사건 기억이 0인 채로 BL80과 G3가 경보를 내는 고정 예제를 확인했다. 기억 보호를 분류기의 충격 강건성으로 대신 주장하지 않는다.
- 충격 경로에만 있는 경보 확률은 전체 충격 경보율이나 두 경로의 주변확률 차이와 다르다. 모델별 대조 경보가 늘면 이 단방향 확률은 줄 수도 있다. 반복 충격에서 실제로 확인했으므로 결합 확률과 전체 경보를 함께 보고한다.
- 보정용 prefix-H와 실제 full-H의 기준이 다르다. 상관관계·분산 분해·IQR 격자와 경험적 지지집합 분석은 이 문제의 범위를 좁히며, 실데이터에서의 원인 비율을 증명하지 않는다.
- 고정 horizon 최대값 calibration은 조건부5%나 모든 시점의5% 보장이 아니다. AR/ARCH와 참조 재사용의 의존성을 독립 표본처럼 취급하지 않았다.
- scale 증거 상한이 공통 확인 문턱보다 낮을 때는 scale 단독 새 확인이 불가능하다. 실제 V4에서 이 조건과 실패가 겹친 경로는 일부에 그치며, 이를 전체 축소 실패의 주원인으로 단정하지 않는다. 반복 블록 건수는 독립 표본 수나 검출률이 아니다.
- 교차 격자의 두 축에 같은 seed 가중치를 쓴 근사 bootstrap은 독립 요인 재표본 정리와 다르다.512×512를262,144개 독립 표본으로 보고하지 않는다.
- H-regime은 NEXT5 이전 전체 DEV H로 만든 고정 변환의 역사적 transductive 요소가 있다. 이번에는 RP 사후 설명에만 사용했다. 관측 mechanism signature 역시 사후 설명이며 정답 DGP 이름이나 학습 입력이 아니다.
- 희소 mean-change 등 일부 구간은 유효 재표본이 적다. 없는 AUC나 구간을0 또는0.5로 채우지 않았다. 구간별 구간은 동시 보장이 아니다.
- 드문 사건에서 bootstrap 구간이[0,0]이면 원래 표본에 관찰된 사건이 없었다는 뜻이다. 모집단 위험0의 보증으로 해석하지 않는다. 충격 감사의 모든 역사 좌표 차이에는 시각·경과 시간도 포함되므로 사건 오기억률과 구분한다.
- 에이전트가 사전에 정한 충격 위험 10%p 절대 감소 조건은 기존 위험이 그보다 작은 비교에서 달성할 수 없다. 기준 모델·endpoint·길이 정의도 더 구체적이어야 했다. 이는 사용자에게 귀속할 숫자가 아닌 실험 설계 한계다. 모든15개 후보는 별도의 최악 fold 비악화 조건도 실패했으며, 이번 결과를 보고 기준을 완화하지 않았다.
- 엄격한 원시 입력 비중복은 새 scale 가지에서 충족하고 승계한 core의 lag4 가지에서는 미충족이다. 원래 core 보존이나 유한 영향 검사가 이 요구를 면제하지 않는다. 최종 감사 통과는 이 한계를 포함한 negative research 기록의 검증이며 전체 요구를 충족한 운영 모델의 승인이 아니다.
- G3의 전체 쌍 gradient와 강볼록성 경계는 고정된 정규화 학습 목적에만 적용된다. 검증 TS-AUC의 최적성 경계가 아니다.
- 재현은 같은 기록된 Python/라이브러리 환경의 새 프로세스에서 수행했다. 새 컴퓨터 설치, 다른 하드웨어의 비트 동일성이나 같은 지연시간을 보장하지 않는다. 애플리케이션 guard는 OS sandbox가 아니다.
- 숫자 배열 크기는 Python 객체·모델·프로세스 전체 메모리와 다르다. 반복 새 상태의 H 초기화 비용과 아주 긴 한 상태의 비용도 분리했다.
- 원래 실패·복구 기록을 보존했다. 한 번의 test cache 경로 누락은 유도 캐시 파일을 갱신했지만 protected source/model byte는 바꾸지 않았고 최종 ZIP에서 캐시는 제외한다.
- ZIP에는 raw RP와 full real point OOF가 없으므로 이를 요구하는 전체 학습·대회 scoring을 독립 실행할 수는 없다. 제공된 원본 고정 synthetic fixture와 연구 bundle의 scalar 재현은 별도로 가능하다.
'''
    next_plan='''# 다음 연구의 우선순위

현재 판정은 NO_PROMOTION이다. BL80·D2·J102의 기존 역할을 유지한다. 다음 실행은 새로운 사전 프로토콜을 먼저 잠그고 시작한다. 이번 수치를 보고 실패 후보의 임계값을 조금씩 바꾸는 작업은 하지 않는다.

1. **원시 입력 비중복, H 기준의 일치와 오류 목표를 먼저 정한다.** 기존 core의 lag4 경계 중복은 미충족 요구로 남았으므로, 새 후보에서는 각 블록의 원시 입력 범위를 명시하고 엄격히 겹치지 않는 대조를 설계한다. 구체적인 미평가 설계안은 블록 b의 새 관측을 16b부터 16b+15까지만 허용하고, lag1 곱은 블록 내부 위치 1~15의 15쌍, lag4 곱은 위치 4~15의 12쌍으로 계산하는 것이다. 각 곱의 두 관측이 같은 블록 안에 있으므로 새 raw 입력 범위가 이웃 블록과 겹치지 않는다. 대신 유효 쌍 수와 통계량 분포가 달라지므로 기존 16쌍 기반 임계값을 재사용해서는 안 된다. 이 설계는 이번에 구현·평가한 후보가 아니며 개선을 주장하지 않는다. 원래 reference는 보존한다. 가상 보정과 실제 처리에 같은 참조 추정 절차를 사용하고, H 전체를 무조건 더 길게 쓰는 것이 해답이라고 가정하지 않는다. conditional·marginal·fixed-horizon·average-run-length·anytime 중 무엇을 주장할지, serial dependence에 어떤 가정을 둘지 구분한다. 같은 유한 경험적 지지집합의 bootstrap 반복 수만 늘리는 후보는 제외한다.
2. **과거 기록과 현재 순위의 신뢰도를 분리한다.** 사건이 확인됐다는 기록은 보존하되, 잘못된 확인이 모든 이후 음성을 높이는 비용과 참 양성의 이득을 동시에 검증한다. 기억이 0인 상태에서도 발생하는 최종 분류기 경보를 별도 endpoint로 둔다. O1의 기존 고정 예제와 전체 코호트가 이 계약 차이를 보여주지만 새로운 veto의 효과까지 검증한 것은 아니다. 작은 구조 가설 한 개와 대응 baseline을 고정하고, 동일 grouped inner OOF를 유지한다. 사후 정답의 시간대나 mechanism을 예측 gate로 넣지 않는다.
3. **약한 축소와 큰 축소를 함께 다룬다.** V3 민감도, V4 민감도, single·burst·repeated shock, 긴 null 오기억, 검출 지연과 전체 RP TS-AUC를 하나의 등록된 판정표에 둔다. 충격 감소의 비교 기준 모델, 상태 기억 또는 분류기 경보, 전체 발생률 또는 대조 대비 추가 발생률, 관측 길이를 먼저 명시한다. 요구한 절대 감소량이 기준 위험보다 커서 원천적으로 달성 불가능한지 설계 단계에서 확인하고, 평가 결과를 본 뒤 상대 감소율이나 유리한 endpoint로 바꾸지 않는다. 관측 sign의 대칭성만으로 순위의 대칭적인 이득을 기대하지 않는다.

G3의 큰 전체 쌍 진단은 반복하지 않는다. 이미 고정 계수에 대한 모든 원래 쌍의 gradient를 계산했다. 다음 목적함수 변경은 그 결과로 설명할 수 있는 별도 가설이 있을 때만 허용한다. 큰 graph나 새 edge/lag 검색은 이번 병목을 해결한다는 근거가 없다.

Shadow·Seal 개방 권고는 NO다. 이번 결과를 정당화하려고 열지 않는다. 향후 세션의 사용자 지시와 미리 고정한 승격 조건이 충족될 때 별도로 판단한다.
'''
    with activity('Write final45-question Korean guide and GPT handoff','FINAL_AUDIT','FINAL_AUDIT'):
        save('00_READ_FIRST_NEXT7_KO.md',read_first)
        save('01_GPT_HANDOFF_NEXT7.md',handoff)
        save('LIMITATIONS_NEXT7.md',limitations)
        save('NEXT_SESSION_PLAN_NEXT7.md',next_plan)
        for name in ('SEAL_OPEN_RECOMMENDATION_NEXT7.md','SHADOW_OPEN_RECOMMENDATION_NEXT7.md'):
            save(name,'# 개방 권고: NO\n\nNEXT7에서 기존 승격 기준을 통과한 후보가 없다. 실제 관측값 접근·평가를 수행하지 않았다. 이번 negative result를 뒤집기 위한 개방을 권고하지 않는다. 기존 잠금 상태를 유지한다.\n')
        save('audit/FINAL_DOCUMENTS_DONE.json',{'status':'COMPLETE','created_utc':utc_now(),'source_sha256':digest(__file__),
            'answered_user_questions':45,'all_answers_derived_after_final_computation_audit':True,
            'final_audit_sha256':digest(BASE/'FINAL_AUDIT_NEXT7.json'),
            'files_sha256':{name:digest(BASE/name) for name in ('00_READ_FIRST_NEXT7_KO.md','01_GPT_HANDOFF_NEXT7.md',
                'LIMITATIONS_NEXT7.md','NEXT_SESSION_PLAN_NEXT7.md','SEAL_OPEN_RECOMMENDATION_NEXT7.md','SHADOW_OPEN_RECOMMENDATION_NEXT7.md')},
            'final_archive_delivery_receipt_separate':True})
    note('Final documents written','All45 user questions answered; no promotion, no hidden opening','FINAL_AUDIT',phase='FINAL_AUDIT')


if __name__=='__main__':main()
