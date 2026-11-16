"""Review complete O1 classifier results and the already fixed viewer example."""
from pathlib import Path
import csv
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from src.next7.common import BASE,read_json,save,digest,utc_now,activity,require_guard


def rows(path):
    with path.open(newline='',encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def main():
    require_guard()
    cohort=BASE/'classifier_statistics/O1'
    complete=read_json(cohort/'DONE.json')
    assert complete['status']=='COMPLETE' and complete['draws']==2000 and complete['fixed_folds']==5
    for name in ('SUMMARY.csv','PAIRED_CONTRASTS.csv'):
        assert digest(cohort/name)==complete['files_sha256'][name]
    directory=BASE/'visuals/v4'
    built=read_json(directory/'BUILD_DONE.json')
    for name in ('STATE_NETWORK_TRACES.json','EXACT_SYNTHETIC_CURVES.npz'):
        assert digest(directory/name)==built['files_sha256'][name]
    data=read_json(directory/'STATE_NETWORK_TRACES.json')
    trace=next(t for t in data['traces'] if t['id']=='single_shock_unlatched')
    assert trace['case']=='O1' and trace['seed']==0 and trace['horizon']==1024
    summary=[r for r in rows(cohort/'SUMMARY.csv') if r['metric'] in ('any_alarm','treatment_alarm_control_never')]
    paired=[r for r in rows(cohort/'PAIRED_CONTRASTS.csv') if r['reference']=='M0' and r['metric'] in ('any_alarm','treatment_alarm_control_never')]
    assert len(summary)==32
    examples=[]
    with activity('Review classifier shock alarms despite unlatched event memory',
                  'COMPLETED_COHORT_FORENSIC','F4_NULL_GROWTH_CORRECTION'):
        with np.load(directory/'EXACT_SYNTHETIC_CURVES.npz',allow_pickle=False) as arrays:
            state=arrays[trace['id']+'_qualified_state']
            assert state.shape==(1024,45) and np.all(state[:,3]==0) and np.all(state[:,30]==0)
            for method in data['classifier_methods']:
                scores=arrays[trace['id']+'_'+method+'_score']
                control=arrays[trace['id']+'_'+method+'_control']
                cut=trace['classifiers'][method]['cut']
                alarm=scores>cut
                control_alarm=control>cut
                assert np.array_equal(alarm,trace['classifiers'][method]['alarm'])
                assert np.array_equal(control_alarm,trace['classifiers'][method]['control_alarm'])
                indices=np.flatnonzero(alarm)
                examples.append({'method':method,'fixed_cut':cut,'first_alarm_index':int(indices[0]) if len(indices) else None,
                    'alarm_timepoints':int(alarm.sum()),'control_alarm_timepoints':int(control_alarm.sum()),
                    'maximum_score':float(scores.max()),'maximum_score_index':int(scores.argmax()),
                    'raw_and_qualified_event_memory_always_zero':True})
        assert all(r['control_alarm_timepoints']==0 for r in examples)
        assert all(next(r for r in examples if r['method']==method)['alarm_timepoints']>0
                   for method in ('M0','G3_PAIRWISE_FULL'))
        inputs=[cohort/'DONE.json',cohort/'SUMMARY.csv',cohort/'PAIRED_CONTRASTS.csv',
                directory/'BUILD_DONE.json',directory/'STATE_NETWORK_TRACES.json',directory/'EXACT_SYNTHETIC_CURVES.npz']
        save('audit/O1_CLASSIFIER_MEMORY_BOUNDARY_REVIEW_001.json',{'status':'PASS','created_utc':utc_now(),
            'source_sha256':digest(__file__),'input_sha256':{p.relative_to(BASE).as_posix():digest(p) for p in inputs},
            'whole_completed_cohort_summary':summary,'whole_completed_cohort_paired_comparisons':paired,
            'example_id':trace['id'],'example_original_selection':trace['selection'],
            'example_fitted_fold':data['classifier_fitted_fold'],'example_scheduled_intervention_index':trace['info']['scheduled'],
            'example_rows':examples,'exact_alarm_arrays_match_reviewed_viewer':True,
            'conclusion':'Confirmation-only event-memory updates do not make the integrated classifier a confirmation-only alarm system.',
            'scope':'The fixed example proves that classifier alarms can occur with zero event memory. It does not estimate the whole-cohort fraction caused by any particular feature or head coordinate.',
            'new_fit_or_seed_or_draw_or_threshold_or_candidate':False})
        save('research_notes/CLASSIFIER_SHOCK_WITHOUT_EVENT_MEMORY_KO.md','''# 사건 기억이 없어도 발생하는 분류기 경보

완료된 O1 원래 512개 시드와 고정된 5개 fold 모델에서 1,024개 관측까지 한 번 이상 경보를 낸 비율은 BL80 79.140625%, G3 78.2421875%다. G3−BL80의 대응 차이는 −0.8984375%p, 기존 2,000회 95% 구간은 [−1.8359375, +0.0390625]%p로 0을 포함한다. 대조에는 경보가 없고 충격 경로에만 있었던 비율은 각각 76.7578125%, 75.9375%다. 이는 상태의 사건 기억률과 다른 endpoint다.

이전 NEXT6 M3의 전체 경보율은 66.5625%이고 BL80 대비 차이의 구간은 [−15.0390625, −10.1943359375]%p다. M3의 과거 paired-control 미측정 칸은 그대로 남겼다. I4의 전체 경보 감소는 1.09375%p이며 기존 구간은 [0.625, 1.5625]%p다. 이 국소 결과가 전체 RP의 승격 실패를 바꾸지는 않는다.

이미 선택되어 검토된 v4 화면의 O1 seed 0 예제에서 충격 예정 시점은 128이다. 이 경로는 1,024개 관측 내내 원시 사건 기억과 G2 사건 기억이 모두 정확히 0이다. 그럼에도 BL80은 시점 129부터 17개 시점에, G3는 시점 193부터 12개 시점에 자기 고정 기준을 넘는다. 두 방법의 대조 경로 경보는 0개다. 모든 8개 분류기의 원래 float64 곡선에서 다시 읽은 경보 boolean이 검토된 화면과 정확히 일치함을 확인했다. 경보 시점 수는 독립 표본 수가 아니다.

따라서 확인 후 기억 갱신이라는 계약과 최종 분류기의 경보 규칙은 구분해야 한다. BL80을 유지한 결합은 새로운 기억이 붙지 않아도 경보를 낼 수 있다. 이 예제가 전체 O1 경보 중 어떤 특징이 몇 %를 유발했는지 증명하지는 않는다. 그 원인 비율이나 새로운 veto의 효과를 측정했다고 주장하지 않는다.

다음 구조 연구에서는 확인 전 분류기 경보, 확인 후 사건 기억, 현재 상태의 회복을 별도 판정해야 한다. 단순 점수 축소로 경보를 줄였다는 주장을 피하고, 각 모델의 보정 기준과 RP 순위 비용을 함께 고정해야 한다. 이번에는 새 veto·head·임계값·시드를 추가하지 않았다.
''')
    print('PASS: complete O1 cohort and all8 existing viewer alarm arrays; zero event memory example',flush=True)


if __name__=='__main__':
    main()
