"""Diagnostic full-cadence self-loss comparison; no certification or selection."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,csv_write,activity,utc_now
import numpy as np
from src.next9.self_cadence import VARIANTS
from src.next9.inference import weights_for,interval


def main():
    results=[];contrasts=[];sources={};fold_samples={};runtime=[]
    for fold in range(5):
        for mode in ['NORMAL','EARLY']:
            with activity(f'F{fold} {mode} original phase0 recovery and paired full-cadence self-law comparison',
                          'SELF_LAW_FULL_CADENCE','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'):
                ids=[];losses=[];features=None
                for part in range(2):
                    root=BASE/'self_cadence'/f'outer_{fold}'/mode/f'part_{part}'
                    done=read_json(root/'DONE.json');assert done['status']=='PASS'
                    runtime.append(done);sources[str((root/'DONE.json').relative_to(BASE))]=digest(root/'DONE.json')
                    if features is None:features=done['features']
                    else:assert features==done['features']
                    for name,sha in done['batch_receipts_sha256'].items():
                        p=BASE/name;assert digest(p)==sha;r=read_json(p)
                        data=p.parent/'GROUP_LOSSES.npz';assert digest(data)==r['group_losses_sha256']
                        with np.load(data) as z:
                            np.testing.assert_array_equal(z['features'],features)
                            ids.extend(z['group_id']);losses.extend(z['loss'])
                ids=np.asarray(ids,dtype=np.int64);loss=np.asarray(losses);assert len(ids)==len(np.unique(ids))
                order=np.argsort(ids);ids=ids[order];loss=loss[order]
                eligible=np.isfinite(loss).all(axis=(1,2,3));ids=ids[eligible];loss=loss[eligible]
                assert len(ids)>=64
                for col,j in enumerate(features):
                    weights,resampling=weights_for(ids,f'OUTER_T|outer_{fold}|{mode}|F{j}|ALL')
                    point=loss[:,:,col].mean(axis=0)
                    boots=np.einsum('ng,gcm->ncm',weights.astype(float),loss[:,:,col])/weights.sum(axis=1)[:,None,None]
                    original=BASE/'outer_self_transition_evaluation'/f'outer_{fold}'/mode/f'F{j:03d}'/'ALL'
                    old=read_json(original/'DONE.json');assert old['groups']==len(ids)
                    assert old['resampling']['weight_sha256']==resampling['weight_sha256']
                    np.testing.assert_allclose(point[0],[old['mean_logloss'][v] for v in VARIANTS],rtol=0,atol=1e-12)
                    assert digest(original/'BOOTSTRAP.npz')==old['bootstrap_sha256']
                    with np.load(original/'BOOTSTRAP.npz') as z:
                        np.testing.assert_allclose(boots[:,0],z['values'],rtol=0,atol=1e-12)
                    for cadence,name in enumerate(['PHASE0_ORIGINAL','ALL_POINTS']):
                        for k,variant in enumerate(VARIANTS):
                            results.append(dict(fold=fold,mode=mode,feature=j,cadence=name,variant=variant,
                                original_groups=len(ids),mean_logloss=float(point[cadence,k])))
                        for reference in [0,1,3]:
                            p=point[cadence,reference]-point[cadence,2]
                            b=boots[:,cadence,reference]-boots[:,cadence,2]
                            key=(mode,j,name,VARIANTS[reference]);fold_samples.setdefault(key,[]).append((fold,p,b))
                            contrasts.append(dict(fold=fold,mode=mode,feature=j,cadence=name,contrast='T2__vs__'+VARIANTS[reference],
                                **interval(p,b),exploratory_unadjusted_no_certification=True))
                    # Direct cadence effect on the same fixed T2 and same groups.
                    contrasts.append(dict(fold=fold,mode=mode,feature=j,cadence='ALL_MINUS_PHASE0',contrast='T2_LOSS_CHANGE',
                        **interval(point[1,2]-point[0,2],boots[:,1,2]-boots[:,0,2]),exploratory_unadjusted_no_certification=True))
    combined=[]
    for (mode,j,cadence,reference),entries in fold_samples.items():
        assert sorted(r[0] for r in entries)==list(range(5))
        p=float(np.mean([r[1] for r in entries]));b=np.mean([r[2] for r in entries],axis=0)
        combined.append(dict(mode=mode,feature=j,cadence=cadence,contrast='T2__vs__'+reference,
            **interval(p,b),positive_folds=sum(r[1]>0 for r in entries),
            positive_CI_folds=sum(interval(r[1],r[2])['ci_low']>0 for r in entries),
            original_equal_fold_weight=True,exploratory_unadjusted_no_certification=True))
    csv_write('SELF_CADENCE_MODEL_LOSSES_NEXT9.csv',results)
    csv_write('SELF_CADENCE_FOLD_CONTRASTS_NEXT9.csv',contrasts)
    csv_write('SELF_CADENCE_FIVE_FOLD_NEXT9.csv',combined)
    for mode in ['NORMAL','EARLY']:
        rec=[r for r in runtime if r['mode']==mode]
        ids=[i for r in rec for i in r['series_ids']]
        assert len(ids)==len(set(ids))==6400 and sum(r['points'] for r in rec)==3203157
    text=['# NEXT9 자기 전이 모델 전체 시점 진단','',
        '각 외부 TRAIN에서 고정한 자기 전이 모델을 원래 제외된 시계열에서 관측 순서대로 실행했다. NORMAL과 EARLY 각각 6,400개 시계열, 총 6,406,314개 관측을 검증했다.',
        '현재 F를 입력하기 전에 상태 예측과 8개 경로 요약을 발행했다. 학습 파라미터 불변, 첫 surprise=0, 저장·복원 후 일치, 배치 계산과의 오차를 검사했다.',
        'B 인증 피처가 없어 이 자기 전이 요약은 이번 M5에 사용되지 않았다. 여기서 실행한 것은 보조 모델의 배포 가능성과 학습 표본 간격에 대한 진단이며 M5나 B를 변경하지 않았다.',
        '기존 8시점 간격의 평균 로그손실과 2000회 bootstrap도 1e-12 이내로 복구했다. 전체 시점 비교는 같은 시계열 집합과 같은 고정 모델을 사용한다.',
        '추가 신뢰구간은 사후 탐색적 진단으로 다중비교 보정이나 새 인증에 사용하지 않는다. EARLY 평가는 기존대로 t>=400이다.','',
        '|모드|표본|비교|9개 피처 평균 로그손실 이득|4/5 폴드 CI 양수 피처 수|',
        '|---|---|---|---:|---:|']
    for mode in ['NORMAL','EARLY']:
        for cadence in ['PHASE0_ORIGINAL','ALL_POINTS']:
            for reference in [VARIANTS[k] for k in [0,1,3]]:
                rs=[r for r in combined if r['mode']==mode and r['cadence']==cadence and r['contrast']=='T2__vs__'+reference]
                text.append(f"|{mode}|{cadence}|T2 vs {reference}|{np.mean([r['point'] for r in rs]):.10f}|{sum(r['positive_CI_folds']>=4 for r in rs)}|")
    text+=['','서로 상관된 피처들의 평균은 설명용이며 독립 실험 9회의 평균으로 해석하지 않는다. 양수 이득은 기준 모델보다 낮은 로그손실을 뜻한다.']
    atomic(BASE/'SELF_CADENCE_FINDING_NEXT9_KO.md','\n'.join(text)+'\n')
    atomic(BASE/'SELF_CADENCE_DONE_NEXT9.json',dict(status='PASS',timestamp=utc_now(),
        outer_procedures=10,original_series_per_mode=6400,stream_observations=6406314,
        source_sha256=sources,original_phase0_losses_and_bootstrap_recovered_within_1e_minus12=True,
        maximum_probability_error=max(r['maximum_probability_error'] for r in runtime),
        maximum_path_error=max(r['maximum_path_error'] for r in runtime),
        no_parameter_updates_current_F_at_issuance_or_break_labels=True,
        inactive_auxiliary_branch_audited_separately=True,scientific_rules_and_M_predictions_unchanged=True))


if __name__=='__main__':main()
