"""Audit paired-alarm endpoint bookkeeping from all completed shock cohorts."""
from pathlib import Path
import argparse
import csv
import math
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.next7.common import BASE,read_json,save,csv_write,digest,utc_now,activity

CASES=('O1','OB16','REPEATED_SHOCK')
METHODS=('M0','M3','M_EVENT','G2_EVENT','I4_EVENT_SCALE_TREE','G2_FULL','G3_PAIRWISE_FULL','AMPLITUDE_0_9_M0')
METRICS=('any_alarm','control_any_alarm','treatment_alarm_control_never')
TOL=1e-12


def rows(path):
    with path.open(newline='',encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def choose(data,**keys):
    selected=[r for r in data if all(str(r.get(k))==str(v) for k,v in keys.items())]
    assert len(selected)==1,(keys,len(selected))
    return selected[0]


def value(raw):
    if raw in ('',None):
        return None
    result=float(raw)
    assert math.isfinite(result)
    return result


def lock():
    name='SHOCK_CLASSIFIER_JOINT_AUDIT_LOCK_NEXT7.json'
    if (BASE/name).exists():
        assert read_json(BASE/name)['source_sha256']==digest(__file__)
        return
    save(name,{'status':'FROZEN','created_utc':utc_now(),'source_sha256':digest(__file__),
        'cases':CASES,'methods':METHODS,'horizons':[1024,4096],'source_endpoints':METRICS,
        'scope':'Posthoc arithmetic audit motivated by completed repeated-shock results. Audit all three original completed shock cohorts and all eight methods; no outcome-based case or method selection.',
        'identity':'T_only=P(T)-P(T and C); C_only=P(C)-P(T and C). Therefore a fall in T_only need not imply a fall in P(T).',
        'new_draws_or_intervals_or_models_or_thresholds':False,'identity_tolerance':TOL})


def main(lock_only):
    lock()
    if lock_only:
        return
    inputs={}
    data={}
    for case in CASES:
        directory=BASE/'classifier_statistics'/case
        receipt=read_json(directory/'DONE.json')
        assert receipt['status']=='COMPLETE' and receipt['fixed_folds']==5 and receipt['draws']==2000
        inputs[(directory/'DONE.json').relative_to(BASE).as_posix()]=digest(directory/'DONE.json')
        for name in ('SUMMARY.csv','PAIRED_CONTRASTS.csv'):
            assert digest(directory/name)==receipt['files_sha256'][name]
            inputs[(directory/name).relative_to(BASE).as_posix()]=digest(directory/name)
        data[case]=(rows(directory/'SUMMARY.csv'),rows(directory/'PAIRED_CONTRASTS.csv'))
    joint=[]
    comparisons=[]
    checked=0
    with activity('Audit all paired shock alarm joint probabilities and endpoint changes',
                  'COMPLETED_COHORT_FORENSIC','F4_NULL_GROWTH_CORRECTION'):
        for case in CASES:
            summary,paired=data[case]
            for horizon in (1024,4096):
                by_method={}
                for method in METHODS:
                    values={metric:value(choose(summary,method=method,horizon=horizon,metric=metric)['mean_across_fixed_folds']) for metric in METRICS}
                    for number in values.values():
                        assert number is None or 0<=number<=1
                    missing=any(number is None for number in values.values())
                    row={'case':case,'horizon':horizon,'method':method,**values,
                        'both_alarm':None,'control_only_alarm':None,'neither_alarm':None,
                        'net_treatment_minus_control_probability':None,'joint_missing_not_imputed':missing,
                        'roundoff_clamped_to_zero':False,'new_confidence_interval':False}
                    if not missing:
                        t,c,only=(values[metric] for metric in METRICS)
                        raw={'both_alarm':t-only,'control_only_alarm':c-(t-only),
                             'neither_alarm':1-t-(c-(t-only))}
                        for key,number in raw.items():
                            assert -TOL<=number<=1+TOL,(case,horizon,method,key,number)
                            if number<0:
                                row['roundoff_clamped_to_zero']=True
                                number=0.
                            row[key]=number
                        row['net_treatment_minus_control_probability']=t-c
                        assert abs(only+row['both_alarm']+row['control_only_alarm']+row['neither_alarm']-1)<=TOL
                        assert abs(only-row['control_only_alarm']-(t-c))<=TOL
                    joint.append(row)
                    by_method[method]=row
                reference=by_method['M0']
                for method in METHODS[1:]:
                    candidate=by_method[method]
                    out={'case':case,'horizon':horizon,'candidate':method,'reference':'M0'}
                    for field in (*METRICS,'both_alarm','control_only_alarm','net_treatment_minus_control_probability'):
                        a,b=candidate[field],reference[field]
                        delta=None if a is None or b is None else a-b
                        out['delta_'+field]=delta
                        if field in METRICS:
                            original=value(choose(paired,candidate=method,reference='M0',horizon=horizon,metric=field)['mean_delta'])
                            assert (original is None)==(delta is None)
                            if delta is not None:
                                assert abs(delta-original)<=TOL
                                checked+=1
                    dt,do,db=(out['delta_'+key] for key in ('any_alarm','treatment_alarm_control_never','both_alarm'))
                    out['one_way_endpoint_decreases_without_treatment_alarm_decrease']=None
                    if do is not None:
                        assert abs(do-(dt-db))<=TOL
                        out['one_way_endpoint_decreases_without_treatment_alarm_decrease']=do < -TOL and dt >= -TOL
                    out['algebraic_point_estimate_audit_not_new_interval']=True
                    comparisons.append(out)
        assert len(joint)==48 and len(comparisons)==42
        assert sum(r['joint_missing_not_imputed'] for r in joint)==6
        example=choose(comparisons,case='REPEATED_SHOCK',horizon=4096,candidate='G3_PAIRWISE_FULL')
        assert example['one_way_endpoint_decreases_without_treatment_alarm_decrease']
        assert example['delta_control_any_alarm']>0
        csv_write('SHOCK_CLASSIFIER_JOINT_PROBABILITIES_NEXT7.csv',joint)
        csv_write('SHOCK_CLASSIFIER_ENDPOINT_DECOMPOSITION_NEXT7.csv',comparisons)
        outputs=['SHOCK_CLASSIFIER_JOINT_PROBABILITIES_NEXT7.csv','SHOCK_CLASSIFIER_ENDPOINT_DECOMPOSITION_NEXT7.csv']
        save('SHOCK_CLASSIFIER_JOINT_AUDIT_DONE_NEXT7.json',{'status':'COMPLETE','created_utc':utc_now(),
            'lock_sha256':digest(BASE/'SHOCK_CLASSIFIER_JOINT_AUDIT_LOCK_NEXT7.json'),
            'inputs_sha256':inputs,'outputs_sha256':{name:digest(BASE/name) for name in outputs},
            'complete_cases':3,'joint_rows':48,'comparison_rows':42,'missing_joint_rows_preserved':6,
            'original_paired_mean_deltas_checked':checked,'identity_tolerance':TOL,
            'example':example,'new_draws_or_intervals_or_models_or_thresholds':False,
            'claim':'A method-dependent paired-control endpoint can decrease because simultaneous alarms increase, while treatment alarm probability itself increases.'})
        save('research_notes/PAIRED_CONTROL_ALARM_ENDPOINT_INTERPRETATION_KO.md','''# 대조에 없는 경보가 줄어도 전체 경보가 줄지는 않는다

T를 충격 경로의 경보, C를 같은 H·난수의 대조 경로 경보라고 하자. 이번 `treatment_alarm_control_never`는 P(T이고 C가 아님)이다. 이는 P(T)−P(C)가 아니며, 두 경로의 경보 규칙도 비교하는 모델마다 달라진다.

완료된 반복 충격의 4,096개 관측에서 BL80은 P(T)=98.671875%, P(C)=9.0625%, P(T이고 C가 아님)=89.609375%다. G3는 각각 99.0625%, 11.2890625%, 87.7734375%다. 이 비교에서는 C에만 경보가 있는 확률이 두 방법 모두 0이다.

따라서 G3의 충격 경로 전체 경보는 0.390625%p 늘고, 대조 경로와 동시 경보는 2.2265625%p 늘었다. 그 차이로 충격 경로에만 남는 경보가 1.8359375%p 줄었다. 원래 2,000회 대응 구간은 전체 경보 변화 [+0.1171875, +0.7041015625]%p, 대조 경보 변화 [+1.25, +3.28125]%p, 충격 경로에만 있는 경보 변화 [−2.8916015625, −0.78125]%p다. 마지막 구간만 보고 충격 강건성이 개선됐다고 결론 내리면 안 된다.

O1·OB16·REPEATED_SHOCK 전체 완료 코호트, 모든 8개 방법, 두 관측 길이의 48행을 동일하게 계산했다. 알려진 주변확률과 단방향 확률로 네 결합 확률을 복원하고, M0 대비 42개 비교의 원래 평균 차이와 1e-12 허용오차로 일치하는지 확인했다. 이는 기존 결과의 산술 해석이며 새 구간이나 새 효과 추정 실험이 아니다. 확률과 구간은 고정된 5개 모델의 평균이며 fold를 독립 표본으로 세지 않는다.

M3의 결합 확률을 복원할 수 없는 6행은 비워 두었다. 미측정을 0으로 채우거나 다른 방법의 대조를 대신 쓰지 않았다. 전체 경보·대조 경보·처리에만 경보·대조에만 경보를 함께 보존하고, 운영 판정에서 어떤 endpoint를 요구하는지 미리 정의해야 한다. 새 후보, 임계값, 시드, random draw는 추가하지 않았다.
''')
    print('PASS:48 joint rows,42 comparisons,6 missing rows preserved;',checked,'original mean deltas checked',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--lock-only',action='store_true')
    main(parser.parse_args().lock_only)
