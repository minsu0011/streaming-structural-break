from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,csv_write,activity,utc_now
import numpy as np
from src.next9.sampling_phase import FAMILIES,WIDTHS,OFFSETS

with activity('Summarize all-point sampling phase support across original outer TRAIN and held-out groups',
              'SAMPLING_PHASE_SUPPORT','B9_NORMALIZATION_STATE_AVAILABILITY'):
    combined={};ids=[];points=0;sources={}
    for part in range(4):
        root=BASE/'sampling_phase'/f'part_{part}';r=read_json(root/'DONE.json')
        assert r['status']=='COMPLETE' and digest(root/'AGGREGATES.npz')==r['aggregates_sha256']
        ids+=r['series_ids'];points+=r['points'];sources[str((root/'DONE.json').relative_to(BASE))]=digest(root/'DONE.json')
        with np.load(root/'AGGREGATES.npz') as z:
            for name in z.files:combined[name]=combined.get(name,0)+z[name]
    assert len(ids)==len(set(ids))==6400 and points==3203157
    sums=combined['equal_group_frequency_sum'];counts=combined['complete_phase_groups'];records=[]
    for fold in range(5):
        for cohort in ['TRAIN','HELDOUT']:
            use=[k for k in range(5) if k!=fold] if cohort=='TRAIN' else [fold]
            n=counts[use].sum(axis=0);total=sums[use].sum(axis=0)
            freq=np.divide(total,n[:,None,None,None],out=np.zeros_like(total),where=n[:,None,None,None]>0)
            for d,domain in enumerate(['ALL']+[f'TIME_{i}' for i in range(7)]):
                if not n[d]:continue
                for j in range(153):
                    for family,offset,width in zip(FAMILIES,OFFSETS,WIDTHS):
                        full=freq[d,8,j,offset:offset+width]
                        np.testing.assert_allclose(full.sum(),1,atol=1e-12)
                        for phase in range(8):
                            sampled=freq[d,phase,j,offset:offset+width];difference=sampled-full
                            records.append(dict(fold=fold,cohort=cohort,domain=domain,feature=j,family=family,phase=phase,
                                original_groups=int(n[d]),TV_distance=float(.5*np.abs(difference).sum()),
                                largest_absolute_state_frequency_change=float(np.abs(difference).max()),
                                sampled_reversal_frequency=float(sampled[2:4].sum()) if family=='S4_REVERSAL' else None,
                                full_reversal_frequency=float(full[2:4].sum()) if family=='S4_REVERSAL' else None,
                                excluded_incomplete_phase_groups=int(combined['excluded_incomplete_phase_groups'][use,d].sum())))
    csv_write('SAMPLING_PHASE_ALL153_NEXT9.csv',records)
    candidates=[72,81,89,59,85,77,64,51,1]
    selected=[r for r in records if r['feature'] in candidates and r['phase']==0]
    csv_write('SAMPLING_PHASE_SELECTED9_NEXT9.csv',selected)
    summaries=[]
    for cohort in ['TRAIN','HELDOUT']:
        for family in FAMILIES:
            for domain in ['ALL']+[f'TIME_{i}' for i in range(7)]:
                group=[r for r in selected if r['cohort']==cohort and r['family']==family and r['domain']==domain]
                if group:summaries.append(dict(cohort=cohort,family=family,domain=domain,
                    mean_feature_fold_TV=float(np.mean([r['TV_distance'] for r in group])),
                    maximum_feature_fold_TV=max(r['TV_distance'] for r in group),feature_fold_panels=len(group)))
    csv_write('SAMPLING_PHASE_SUMMARY_NEXT9.csv',summaries)
    text=['# NEXT9 학습 표본의 시점 간격 진단','',
          '원래 6,400개 시계열의 모든 3,203,157개 시점에서 153개 피처의 고정된 과거 상태를 계산했다. 라벨은 읽지 않았다.',
          '학습에 사용한 t mod 8 = 0 표본과 전체 시점의 상태 빈도를 비교했다. 각 시간 구간에서 8개 나머지 위상 모두가 있는 동일 시계열 집합에 한정하고, 시계열마다 같은 가중치를 주었다.',
          'TV는 두 상태 빈도 분포 차이의 절반 합이다. 아래 수치는 사후 공변량 진단이며 성능 저하의 원인이나 통계적 유의성을 뜻하지 않는다. 위상·표본 간격·모델을 새로 선택하지 않았다.','',
          '|집합|상태|전체 시간 평균 TV|최대 피처·폴드 TV|','|---|---|---:|---:|']
    for r in summaries:
        if r['domain']=='ALL':text.append(f"|{r['cohort']}|{r['family']}|{r['mean_feature_fold_TV']:.8f}|{r['maximum_feature_fold_TV']:.8f}|")
    text+=['','TRAIN 폴드 집합은 서로 겹친다. 45개 피처·폴드를 독립 반복 표본으로 해석하면 안 된다.',
           '전체 8개 위상·153개 피처·시간대별 자료와 짧은 꼬리 구간의 제외 개수는 CSV에 보존했다.']
    atomic(BASE/'SAMPLING_PHASE_FINDING_NEXT9_KO.md','\n'.join(text)+'\n')
    atomic(BASE/'SAMPLING_PHASE_DONE_NEXT9.json',dict(status='PASS',timestamp=utc_now(),series=6400,points=points,features=153,
        rows=len(records),source_sha256=sources,phase_optimized=False,labels_or_fits_used=False,
        diagnostic_only_no_scientific_rule_changes=True,summary=summaries))
